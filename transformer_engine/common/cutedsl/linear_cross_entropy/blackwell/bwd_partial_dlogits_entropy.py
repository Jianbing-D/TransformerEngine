# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional, Tuple, Type
from functools import partial
import math

import cuda.bindings.driver as cuda  # type: ignore
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline  # type: ignore
import cutlass.utils as utils  # type: ignore
import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
from cutlass.cute.nvgpu import cpasync, tcgen05

from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from transformer_engine.common.cutedsl.linear_cross_entropy.scheduler import (
    StaticPersistentScheduler,
    TileSchedulerParams,
    ParamsBase,
)
from transformer_engine.common.cutedsl.linear_cross_entropy import ptx

SM100_TMEM_CAPACITY_COLUMNS: int = 512


def make_thread_cooperative_group(size: int, alignment: Optional[int] = None):
    """
    Create a thread cooperative group.
    """
    return pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size, alignment=alignment if alignment is not None else size
    )


class BwdPartialDlogitsEntropy:
    """
    This class implements the backward kernel for partial d_logits.
    """

    def __init__(
        self,
        reduction: int,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = True,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        vocab_per_split: int = 512,
    ):
        self.REDUCTION: cutlass.Constexpr[cutlass.Int32] = cutlass.const_expr(reduction)
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler = (*mma_tiler_mn, 1) if use_2cta_instrs else (mma_tiler_mn[0] // 2, mma_tiler_mn[1], 1)
        self.vocab_per_split = vocab_per_split

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.threads_per_warp: int = 32

        self.epi_warp_ids = (0, 1, 2, 3)
        self.load_warp_ids = 4
        self.mma_warp_ids = 5
        self.store_warp_ids = 6
        self.empty_warp_ids = (7,)

        self.threads_per_cta: int = self.threads_per_warp * len(
            (*self.epi_warp_ids, self.load_warp_ids, self.mma_warp_ids, self.store_warp_ids, *self.empty_warp_ids)
        )
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )

        self.buffer_align_bytes: int = 1024
        self.num_regs_other: int = 32
        self.num_regs_epi: int = 192
        self.epilog_sync_bar_id: int = 2

        self.LOG2_E: float = math.log2(math.e)

    def _compute_grid(
        self,
        problem_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        cta_tiler: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        cluster_shape_mnk = (*cluster_shape_mn, 1)

        grid = cute.round_up(
            (
                cute.ceil_div(problem_mnk[0], cta_tiler[0]),
                cute.ceil_div(self.vocab_per_split, cta_tiler[1]),
                1,
            ),
            cluster_shape_mnk,
        )
        return grid

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
    ):
        # make sure it takes all TMEM columns
        num_acc_stage = SM100_TMEM_CAPACITY_COLUMNS // mma_tiler[1]
        num_ab_stage = 4
        # make sure each stage process 64 elements.
        num_epi_stage_per_tile = cute.ceil_div(mma_tiler[1], 64)
        num_write_stage = 3
        return num_acc_stage, num_ab_stage, num_epi_stage_per_tile, num_write_stage

    def _setup_attributes(
        self,
        tiled_mma: cute.TiledMma,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
    ):
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,)
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        # it requires k-mode to be 128B aligned
        mma_inst_tile_k: int = 4
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * mma_inst_tile_k)

        self.num_acc_stage, self.num_ab_stage, self.num_epi_stage_per_tile, self.num_write_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, a_dtype, b_dtype
        )
        self.tmem_alloc_cols = self.num_acc_stage * self.mma_tiler[1]
        assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS
        # when tmem_alloc_cols == SM100_TMEM_CAPACITY_COLUMNS, no need to allocate tmem
        self.do_tmem_alloc: bool = self.tmem_alloc_cols < SM100_TMEM_CAPACITY_COLUMNS

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

    @cute.kernel
    def kernel(
        self,
        split_idx: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC: cute.Tensor,
        mLabels: cute.Tensor,
        mDlogprobs: cute.Tensor,
        mDentropy: cute.Tensor,
        mAccu: cute.Tensor,
        mEntropyB: cute.Tensor,
        mDlogits_partial: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
        ignore_index: cutlass.Int64,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        problem_mnk: Tuple[int, int, int],
        rank: cutlass.Int32,
        inv_temperature: cutlass.Float32,
        scheduler_params: ParamsBase,
    ) -> None:
        """
        The backward kernel for partial d_logits (persistent scheduler version).
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # CTA rank within cluster (0 or 1 for 2-CTA mode)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = True
        if cutlass.const_expr(self.use_2cta_instrs):
            is_leader_cta = mma_tile_coord_v == 0

        # prefetch tma descriptors
        if warp_idx == self.load_warp_ids:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_ab_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_ids])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            tx_count=self.tma_copy_ab_bytes,
            barrier_storage=storage.load_ab_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_ab_stage
        )
        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )

        num_mma_consumer_threads = self.threads_per_warp * len(self.epi_warp_ids)
        if cutlass.const_expr(self.use_2cta_instrs):
            num_mma_consumer_threads *= 2
        mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.num_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            consumer_group=make_thread_cooperative_group(num_mma_consumer_threads),
            barrier_storage=storage.mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        store_c_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_write_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.epi_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.store_warp_ids])
            ),
            barrier_storage=storage.write_c_mbar_ptr.data_ptr(),
        )
        store_c_producer_state = pipeline.PipelineState(
            self.num_write_stage,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
        )
        store_c_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_write_stage
        )

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        if cutlass.const_expr(self.do_tmem_alloc):
            if warp_idx == self.empty_warp_ids[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, self.threads_per_warp * len(self.epi_warp_ids)
                    )
                    cute.arch.mbarrier_init_fence()

        if cutlass.const_expr(self.use_2cta_instrs):
            # Cluster barrier sync after barrier init
            pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # -------- Fixed tensor partitions (tile-independent) ------------ #
        # swizzle o [(tileM, tileK), loopM, loopK, stage]
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # swizzle o [(tileN, tileK), loopN, loopK, stage]
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # swizzle o [(subtileM, subtileN), stage]
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # [MMA, loopM, loopK, stage]
        tCsA = thr_mma.make_fragment_A(sA)
        # [MMA, loopN, loopK, stage]
        tCsB = thr_mma.make_fragment_B(sB)

        # Multicast masks for TMA loads (needed for 2-CTA barrier accounting)
        a_mcast_mask = None
        b_mcast_mask = None
        if cutlass.const_expr(self.use_2cta_instrs):
            a_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # [vocab_per_split, dim] — depends on split_idx (constant per kernel call)
        mB_n = cute.local_tile(
            mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (split_idx, 0)
        )

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

        if cutlass.const_expr(self.use_2cta_instrs):
            # Cluster wait before TMEM alloc
            pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ------ Allocate TMEM (once, reused across tiles) ------ #
        tmem_ptr = None
        if cutlass.const_expr(self.do_tmem_alloc):
            tmem_holding_buf = storage.tmem_holding_buf
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.alloc_tmem(
                    self.tmem_alloc_cols, tmem_holding_buf, is_two_cta=self.use_2cta_instrs
                )
            self.cta_sync_barrier.arrive_and_wait()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
        else:
            self.cta_sync_barrier.arrive_and_wait()
            tmem_ptr = cute.make_ptr(self.acc_dtype, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)

        tmem_shape = (128, self.tmem_alloc_cols)
        acc_shape = thr_mma.partition_shape_C(tmem_shape)
        tCtC_fake = thr_mma.make_fragment_C(acc_shape)
        # [(tileM, tileN), loopM, loopN]
        tCtC = cute.make_tensor(tmem_ptr, tCtC_fake.layout)

        # Scheduler factory (shared across warp groups)
        TileSchedulerCls = partial(
            StaticPersistentScheduler.create, scheduler_params,
            cluster_m_size=self.cluster_m_size,
        )

        # C TMA partition for TMA S2G store (shared across warps)
        tCgC = thr_mma.partition_C(mC)
        tCgC_t = transform_partitioned_tensor_layout(tCgC)
        tCgC_epi = cute.flat_divide(tCgC_t, self.epi_subtile)
        bSG_sC_tma, bSG_gC_all = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )

        # ------ Empty ------ #
        if warp_idx in self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ------ Load (persistent) ------ #
        if warp_idx == self.load_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Pre-compute all-tile GMEM partitions outside the loop
            gA_all = cute.local_tile(
                mA, (self.mma_tiler[0], self.cta_tile_shape_mnk[2]), (None, None)
            )
            gB_all = cute.local_tile(
                mB_n, (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]), (None, None)
            )
            tCgA_all = thr_mma.partition_A(gA_all)
            tCgB_all = thr_mma.partition_B(gB_all)
            tTMAsA, tTMAgA_all = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA_all, 0, 3),
            )
            tTMAsB, tTMAgB_all = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB_all, 0, 3),
            )

            scheduler = TileSchedulerCls()
            work_tile = scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                pidm, pidn = work_tile.tile_idx

                for k in cutlass.range(self.num_k_tiles):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tTMAgA_all[(None, pidm, k)],
                        tTMAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tTMAgB_all[(None, pidn, k)],
                        tTMAsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

                scheduler.advance_to_next_work()
                work_tile = scheduler.get_current_work()

        # ------ MMA (persistent, leader CTA only) ------ #
        if warp_idx == self.mma_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            if is_leader_cta:
                scheduler = TileSchedulerCls()
                work_tile = scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    mma_pipeline.producer_acquire(mma_producer_state)

                    for k in cutlass.range(self.num_k_tiles):
                        ab_pipeline.consumer_wait(ab_consumer_state)

                        for kblock_idx in cutlass.range(
                            cute.size(tCsA, mode=[2]), unroll_full=True
                        ):
                            cute.gemm(
                                tiled_mma,
                                cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                                tCsA[(None, None, kblock_idx, ab_consumer_state.index)],
                                tCsB[(None, None, kblock_idx, ab_consumer_state.index)],
                                cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()

                    mma_pipeline.producer_commit(mma_producer_state)
                    mma_producer_state.advance()

                    scheduler.advance_to_next_work()
                    work_tile = scheduler.get_current_work()

        # ------ EPI (persistent) ------ #
        if warp_idx in self.epi_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

            # Fixed epilogue setup (tile-independent)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                self.use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(
                tCtC[((None, None), 0, None)],
                (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTMEM_load_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTMEM_load_tAcc = cute.group_modes(tTMEM_load_tAcc, 3, cute.rank(tTMEM_load_tAcc) - 1)

            cAcc = cute.make_identity_tensor(self.mma_tiler[:2])
            tCcAcc = thr_mma.partition_C(cAcc)
            tCcAcc_epi = cute.flat_divide(
                tCcAcc[((None, None), 0, None)],
                (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
            )
            tTMEM_load_cAcc = thr_copy_t2r.partition_D(tCcAcc_epi)
            tTMEM_load_cAcc_shape = cute.select(tTMEM_load_cAcc.shape, mode=[0, 1, 2])
            tTMEM_load_rAcc = cute.make_fragment(tTMEM_load_cAcc_shape, self.acc_dtype)

            # Per-CTA identity for label/accu loading (epi_tile[0] M-rows per CTA)
            cAcc_cta = cute.make_identity_tensor((self.epi_tile[0], self.mma_tiler[1]))

            copy_atom_g2r_int64 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mLabels.element_type
            )
            copy_atom_g2r_fp32 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mDlogprobs.element_type
            )
            epilogue_thread_layout = cute.make_layout(
                (self.epi_tile[0], 1), stride=(1, 1)
            )
            tiled_copy_g2r_int64 = cute.make_tiled_copy_tv(
                copy_atom_g2r_int64, epilogue_thread_layout, cute.make_layout((1, 1))
            )
            tiled_copy_g2r_fp32 = cute.make_tiled_copy_tv(
                copy_atom_g2r_fp32, epilogue_thread_layout, cute.make_layout((1, 1))
            )
            # Clamp thread index for g2r copies: for 2-CTA, only epi_tile[0] threads
            # participate in label/accu loading; excess threads map to last valid slot
            g2r_tidx = cutlass.min(tidx, cutlass.Int32(self.epi_tile[0] - 1))
            thr_copy_g2r_int64 = tiled_copy_g2r_int64.get_slice(g2r_tidx)
            thr_copy_g2r_fp32 = tiled_copy_g2r_fp32.get_slice(g2r_tidx)

            tMCAcc = thr_copy_g2r_int64.partition_S(cAcc_cta)[(None, None, 0)]

            # R2S copy setup for TMA store
            tTR_rC = cute.make_rmem_tensor(tTMEM_load_rAcc.shape, mDlogits_partial.element_type)
            tiled_copy_r2s, tRS_rC, tRS_sC = utils.epilog_smem_copy_and_partition(
                utils.LayoutEnum.from_tensor(mDlogits_partial),
                mDlogits_partial.element_type,
                self.acc_dtype,
                tiled_copy_t2r,
                tTR_rC,
                tidx,
                sC,
            )

            # Pre-compute all-tile partitions outside the loop
            gLabels_all = cute.local_tile(mLabels, (self.epi_tile[0],), (None,))
            gAccu_all = cute.local_tile(mAccu, (self.epi_tile[0],), (None,))
            gDentropy_all = cute.local_tile(mDentropy, (self.epi_tile[0],), (None,))
            gEntropyB_all = cute.local_tile(mEntropyB, (self.epi_tile[0],), (None,))
            tMgLabels_all = thr_copy_g2r_int64.partition_S(cute.append_ones(gLabels_all))
            tMgAccu_all = thr_copy_g2r_fp32.partition_S(cute.append_ones(gAccu_all))
            tMgDentropy_all = thr_copy_g2r_fp32.partition_S(cute.append_ones(gDentropy_all))
            tMgEntropyB_all = thr_copy_g2r_fp32.partition_S(cute.append_ones(gEntropyB_all))
            if cutlass.const_expr(self.REDUCTION == 0):
                gDlogprobs_all = cute.local_tile(mDlogprobs, (self.epi_tile[0],), (None,))
                tMgDlogprobs_all = thr_copy_g2r_fp32.partition_S(cute.append_ones(gDlogprobs_all))

            # Allocate register fragments once (reused across tiles)
            # tMgLabels_all has 4 modes: ((val_m,val_n), rest_m, num_tiles, rest_appended)
            tMrLabels = cute.make_fragment(
                tMgLabels_all[(None, None, 0, None)].shape, tMgLabels_all.element_type
            )
            tMrAccu = cute.make_fragment(
                tMgAccu_all[(None, None, 0, None)].layout, tMgAccu_all.element_type
            )
            tMrDlogprobs = cute.make_fragment(
                tMgAccu_all[(None, None, 0, None)].layout, mDlogprobs.element_type
            )
            tMrDentropy = cute.make_fragment(
                tMgAccu_all[(None, None, 0, None)].layout, mDentropy.element_type
            )
            tMrEntropyB = cute.make_fragment(
                tMgAccu_all[(None, None, 0, None)].layout, mEntropyB.element_type
            )
            tMCAcc_mask = cute.make_fragment(tMCAcc.shape, cutlass.Boolean)
            tMCAcc_mask = cute.append_ones(tMCAcc_mask)

            if cutlass.const_expr(self.REDUCTION == 2):
                num_valid_tokens = cute.make_tensor(scalarNumValidTokens, layout=(1,))

            # Persistent epilogue loop
            scheduler = TileSchedulerCls()
            work_tile = scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                pidm, pidn = work_tile.tile_idx

                # Per-CTA M index: pidm is 128-row tile, each CTA handles 64 rows
                pidm_cta = pidm * self.cluster_m_size + mma_tile_coord_v

                tMCAcc_mask[0] = (
                    cute.elem_less(tidx, self.epi_tile[0])
                    and cute.elem_less(
                        pidm_cta * self.epi_tile[0] + tidx, cute.size(mA, mode=[0])
                    )
                )

                cute.copy(tiled_copy_g2r_int64, tMgLabels_all[(None, None, pidm_cta, None)], tMrLabels, pred=tMCAcc_mask)
                cute.copy(tiled_copy_g2r_fp32, tMgAccu_all[(None, None, pidm_cta, None)], tMrAccu, pred=tMCAcc_mask)
                cute.copy(tiled_copy_g2r_fp32, tMgDentropy_all[(None, None, pidm_cta, None)], tMrDentropy, pred=tMCAcc_mask)
                cute.copy(tiled_copy_g2r_fp32, tMgEntropyB_all[(None, None, pidm_cta, None)], tMrEntropyB, pred=tMCAcc_mask)

                # scale accu (lse) to log2 scale
                tMrAccu[0] *= self.LOG2_E

                if cutlass.const_expr(self.REDUCTION == 2):
                    tMrDlogprobs[0] = mDlogprobs[0] / num_valid_tokens[0].to(cutlass.Float32)
                elif cutlass.const_expr(self.REDUCTION == 1):
                    tMrDlogprobs[0] = mDlogprobs[0]
                else:
                    cute.copy(tiled_copy_g2r_fp32, tMgDlogprobs_all[(None, None, pidm_cta, None)], tMrDlogprobs, pred=tMCAcc_mask)

                tMrDlogprobs[0] *= inv_temperature
                tMrDentropy[0] *= inv_temperature
                tMrDlogprobs[0] *= tMrLabels[0] != ignore_index

                block_vocab_left_idx: cutlass.Int64 = (
                    split_idx * self.vocab_per_split + pidn * self.epi_tile[1]
                )
                block_vocab_right_idx: cutlass.Int64 = min(
                    split_idx * self.vocab_per_split + (pidn + 1) * self.epi_tile[1],
                    min((split_idx + 1) * self.vocab_per_split, problem_mnk[1]),
                )
                num_n_subtiles: cutlass.Int64 = cute.ceil_div(
                    (block_vocab_right_idx - block_vocab_left_idx),
                    cute.size(tTMEM_load_rAcc, mode=[0]),
                )

                mma_pipeline.consumer_wait(mma_consumer_state)
                for n_subtile in cutlass.range(num_n_subtiles):
                    # T2R: load accumulator from TMEM to registers
                    cute.copy(
                        tiled_copy_t2r,
                        tTMEM_load_tAcc[(None, None, None, n_subtile, mma_consumer_state.index)],
                        tTMEM_load_rAcc,
                    )

                    # Per-element: softmax gradient computation + entropy gradient
                    pos_start: cutlass.Int64 = (
                        rank * problem_mnk[1]
                        + split_idx * self.vocab_per_split
                        + pidn * self.epi_tile[1]
                        + n_subtile * cute.size(tTMEM_load_rAcc, mode=[0])
                    )
                    for idx in cutlass.range_constexpr(cute.size(tTMEM_load_rAcc, mode=[0])):
                        logit_val = tTMEM_load_rAcc[idx]
                        tTMEM_load_rAcc[idx] = ptx.fma(tTMEM_load_rAcc[idx], self.LOG2_E * inv_temperature,  -tMrAccu[0])
                        tTMEM_load_rAcc[idx] = cute.math.exp2(tTMEM_load_rAcc[idx], fastmath=True)
                        # tTMEM_load_rAcc[idx] is now softmax_v = exp(z_v - LSE)
                        softmax_val = tTMEM_load_rAcc[idx]

                        pos: cutlass.Int64 = pos_start + idx
                        mask: cutlass.Boolean = (
                            pos == tMrLabels[0] and tMrLabels[0] != ignore_index
                        )
                        # CE gradient: dlogprobs * (softmax - one_hot)
                        tTMEM_load_rAcc[idx] = ptx.fma(softmax_val, tMrDlogprobs[0], mask * -tMrDlogprobs[0])
                        # Entropy gradient: d_entropy * (-softmax) * (z - entropy_b)
                        tTMEM_load_rAcc[idx] += -tMrDentropy[0] * softmax_val * (logit_val * inv_temperature - tMrEntropyB[0])

                    # R2S: retile, convert FP32→output dtype, store to SMEM
                    acc_vec = tiled_copy_r2s.retile(tTMEM_load_rAcc).load()
                    acc_vec = acc_vec.to(mDlogits_partial.element_type)
                    tRS_rC.store(acc_vec)
                    store_c_pipeline.producer_acquire(store_c_producer_state)
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, store_c_producer_state.index)],
                    )

                    # Fence: ensure R2S visible to TMA
                    cute.arch.fence_proxy("async.shared", space="cta")
                    store_c_pipeline.producer_commit(store_c_producer_state)
                    store_c_producer_state.advance()

                mma_pipeline.consumer_release(mma_consumer_state)
                mma_consumer_state.advance()

                scheduler.advance_to_next_work()
                work_tile = scheduler.get_current_work()

        # ------ Write C (persistent) ------ #
        if warp_idx == self.store_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_other)

            scheduler = TileSchedulerCls()
            work_tile = scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                pidm, pidn = work_tile.tile_idx

                block_vocab_left_idx: cutlass.Int64 = (
                    split_idx * self.vocab_per_split + pidn * self.epi_tile[1]
                )
                block_vocab_right_idx: cutlass.Int64 = min(
                    split_idx * self.vocab_per_split + (pidn + 1) * self.epi_tile[1],
                    min((split_idx + 1) * self.vocab_per_split, problem_mnk[1]),
                )
                num_n_subtiles: cutlass.Int64 = cute.ceil_div(
                    (block_vocab_right_idx - block_vocab_left_idx),
                    self.epi_subtile[1],
                )

                n_subtiles_per_tile: int = self.epi_tile[1] // self.epi_subtile[1]
                for n_subtile in cutlass.range(num_n_subtiles):
                    n_subtile_global: cutlass.Int32 = pidn * n_subtiles_per_tile + n_subtile

                    cute.arch.cp_async_bulk_wait_group(self.num_write_stage - 1, read=True)
                    store_c_pipeline.consumer_release(store_c_consumer_state)
                    store_c_pipeline.consumer_wait(store_c_consumer_state)
                    cute.copy(
                        tma_atom_c,
                        bSG_sC_tma[(None, store_c_consumer_state.index)],
                        bSG_gC_all[(None, pidm, n_subtile_global)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    store_c_consumer_state.advance()

                scheduler.advance_to_next_work()
                work_tile = scheduler.get_current_work()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        if cutlass.const_expr(self.use_2cta_instrs):
            # make sure CGA is alive
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        if cutlass.const_expr(self.do_tmem_alloc):
            # ------ Deallocate TMEM ------ #
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols, is_two_cta=self.use_2cta_instrs)

    @cute.jit
    def __call__(
        self,
        split_idx: cutlass.Int32,
        hidden: cute.Tensor,
        weight: cute.Tensor,
        labels: cute.Tensor,
        dlogprobs: cute.Tensor,
        dentropy: cute.Tensor,
        accu: cute.Tensor,
        entropy_b: cute.Tensor,
        dlogits_partial: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
        ignore_index: cutlass.Int64,
        rank: cutlass.Int32,
        inv_temperature: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        a_dtype: Type[cutlass.Numeric] = hidden.element_type
        b_dtype: Type[cutlass.Numeric] = weight.element_type

        if cutlass.const_expr(hidden.element_type != weight.element_type):
            raise RuntimeError(
                f"data type don't match: {hidden.element_type} v.s. {weight.element_type}"
            )
        if cutlass.const_expr(hidden.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise RuntimeError("hidden can only be FP16 or BF16")
        if cutlass.const_expr(hidden.layout.shape[1] != weight.layout.shape[1]):
            raise RuntimeError("K dimension doesn't match")

        problem_mnk = (hidden.layout.shape[0], weight.layout.shape[0], hidden.layout.shape[1])
        if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 16 != 0):
            raise RuntimeError(f"K dimension is not 16B aligned: {problem_mnk[2]}")
        if cutlass.const_expr((problem_mnk[2] * b_dtype.width // 8) % 128 != 0):
            raise RuntimeError(f"K dimension is not 128B aligned: {problem_mnk[2]}")

        num_m_tiles = cute.ceil_div(problem_mnk[0], self.mma_tiler[0])
        num_n_tiles = cute.ceil_div(self.vocab_per_split, self.mma_tiler[1])
        sched_params_init = TileSchedulerParams(
            num_tiles_M=cutlass.Int32(num_m_tiles),
            num_tiles_N=cutlass.Int32(num_n_tiles),
        )
        sched_params = StaticPersistentScheduler.to_underlying_arguments(sched_params_init)
        self.cluster_m_size = self.cluster_shape_mn[0]
        grid = StaticPersistentScheduler.get_grid_shape(
            sched_params, cluster_m_size=self.cluster_m_size, occupancy=1
        )

        a_major_mode = utils.LayoutEnum.from_tensor(hidden).mma_major_mode()
        b_major_mode = utils.LayoutEnum.from_tensor(weight).mma_major_mode()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, a_major_mode, b_major_mode, self.acc_dtype, self.cta_group, self.mma_tiler[:2]
        )
        self._setup_attributes(tiled_mma, a_dtype, b_dtype)

        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.num_k_tiles = cute.ceil_div(problem_mnk[2], self.cta_tile_shape_mnk[2])
        self.epi_subtile = (
            self.epi_tile[0],
            self.epi_tile[1] // self.num_epi_stage_per_tile,
        )

        # C staging buffer for TMA S2G store
        output_dtype = dlogits_partial.element_type
        output_layout = utils.LayoutEnum.from_tensor(dlogits_partial)
        c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            output_dtype, output_layout, self.epi_subtile, self.num_write_stage
        )
        c_smem_layout_one_stage = cute.slice_(c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dlogits_partial,
            c_smem_layout_one_stage,
            self.epi_subtile,
        )

        # Swizzle o [(tileM, tileK), loopM, loopK, stage]
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_ab_stage
        )
        # Swizzle o [(tileN, tileK), loopN, loopK, stage]
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_ab_stage
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        tma_load_op_a = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        tma_load_op_b = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )

        # Swizzle o [(tileM, tileK), loopM, loopK]
        a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op_a,
            hidden,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # Swizzle o [(tileN, tileK), loopN, loopK]
        b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op_b,
            weight,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        self.tma_copy_ab_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        @cute.struct
        class SharedStorage:
            """
            The shared storage for the backward kernel.
            """

            load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            write_c_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_write_stage * 2]

            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

            sA: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[output_dtype, cute.cosize(c_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            split_idx,
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            labels,
            dlogprobs,
            dentropy,
            accu,
            entropy_b,
            dlogits_partial,
            scalarNumValidTokens,
            ignore_index,
            a_smem_layout_staged,
            b_smem_layout_staged,
            c_smem_layout_staged,
            self.cluster_layout_vmnk,
            problem_mnk,
            rank,
            inv_temperature,
            sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )
