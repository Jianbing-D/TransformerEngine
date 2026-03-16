# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional, Tuple, Type, Callable
from functools import partial

import cuda.bindings.driver as cuda  # type: ignore
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline  # type: ignore
import cutlass.utils as cutlass_utils  # type: ignore
import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
from cutlass.cute.nvgpu import cpasync, tcgen05

from transformer_engine.common.cutedsl.linear_cross_entropy import utils
from transformer_engine.common.cutedsl.linear_cross_entropy import scheduler
from transformer_engine.common.cutedsl.linear_cross_entropy import ptx

def next_power_of_2(x: int) -> int: 
    if x <= 0: 
        return 1 
    return 1 << (x - 1).bit_length()


SM100_TMEM_CAPACITY_COLUMNS: int = 512

def make_thread_cooperative_group(size: int, alignment: Optional[int] = None):
    """
    Create a thread cooperative group.
    """
    return pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size, alignment=alignment if alignment is not None else size
    )

class BwdDHiddenDWeight:
    """
    This class implements the backward kernel for d_hidden and d_weight.
    """
    def __init__(
        self,
        reduction: utils.EntropyReductionEnum,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
    ):
        self.REDUCTION: cutlass.Constexpr = cutlass.const_expr(reduction)
        self.acc_dtype = acc_dtype
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)

        self.smem_capacity = cutlass_utils.get_smem_capacity_in_bytes("sm_100")

        self.threads_per_warp: int = 32
        self.warps_per_warpgroup: int = 4
        self.threads_per_warpgroup: int = self.threads_per_warp * self.warps_per_warpgroup

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.epilog_warp_ids = (4, 5, 6, 7)
        self.load_warp_ids = 8
        self.mma_warp_ids = 9
        self.store_warp_ids = 10
        self.empty_warp_ids = (11, 12, 13, 14, 15)

        self.warps_per_cta: int = len(
            (*self.softmax_warp_ids,
             *self.epilog_warp_ids,
             self.load_warp_ids,
             self.mma_warp_ids,
             self.store_warp_ids,
             *self.empty_warp_ids)
        )

        self.threads_per_cta: int = self.threads_per_warp * self.warps_per_cta

        self.buffer_align_bytes: int = 1024
        
        self.num_regs_load: int = 32
        self.num_regs_mma: int = 64
        self.num_regs_store: int = 32
        self.num_regs_softmax: int = 192
        self.num_regs_epilog: int = 80
        self.num_regs_empty: int = 24

        self.scheduler_cls = scheduler.StaticPersistentScheduler

    def _compute_grid(
        self,
        problem_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        mma_tiler: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int], scheduler.ParamsBase]:
        cluster_shape_mnk = (*cluster_shape_mn, 1)

        scheduler_params = scheduler.TileSchedulerParams(
            num_tiles_M=cute.ceil_div(problem_mnk[1], mma_tiler[0]),
            num_tiles_N=1,
        )
        params = self.scheduler_cls.to_underlying_arguments(scheduler_params)

        grid = self.scheduler_cls.get_grid_shape(params)
        grid = cute.round_up(
            grid, cluster_shape_mnk
        )
        return grid, params

    def _compute_stages(
        self,
    ):
        self.num_p_stage = 1
        self.num_ab_stage = 2
        self.num_epi_stage = 4

        self.num_lse_stage = 1

        self.tmem_cols_logits = self.logits_mma_tiler[1] * self.num_p_stage
        self.tmem_cols_dH = self.dH_mma_tiler[1] * 1
        self.tmem_cols_dW = self.dW_mma_tiler[1] * 1
        self.tmem_cols_p = (self.logits_mma_tiler[1] * self.num_p_stage) // 2

        self.tmem_offset_logits = 0
        self.tmem_offset_dH = self.tmem_offset_logits + self.tmem_cols_logits
        self.tmem_offset_dW = self.tmem_offset_dH + self.tmem_cols_dH
        self.tmem_offset_p = self.tmem_offset_dW + self.tmem_cols_dW

        self.tmem_alloc_cols = self.tmem_cols_logits + self.tmem_cols_dH + self.tmem_cols_dW + self.tmem_cols_p
        self.tmem_alloc_cols = next_power_of_2(self.tmem_alloc_cols)

        self.real_tmem_alloc: bool = (self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS)
        assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(
        self,
        logits_tiled_mma: cute.TiledMma,
        hidden_dtype: Type[cutlass.Numeric],
        weight_dtype: Type[cutlass.Numeric],
    ):
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (logits_tiled_mma.thr_id.shape,)
        )

        mma_inst_shape_k = cute.size(logits_tiled_mma.shape_mnk, mode=[2])
        # it requires k-mode to be 128B aligned
        mma_inst_tile_k = 128 // mma_inst_shape_k
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # otherwise, weight for dH should use a separate TMA descriptor
        assert self.mma_tiler[1] == self.mma_tiler[2]
        self.logits_mma_tiler = self.mma_tiler
        self.dH_mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_tiler[1],
        )
        self.dW_mma_tiler = (
            self.mma_tiler[1],
            self.mma_tiler[0],
            self.mma_tiler[0],
        )
        print(f"logits_mma_tiler: {self.logits_mma_tiler}, dH_mma_tiler: {self.dH_mma_tiler}, dW_mma_tiler: {self.dW_mma_tiler}")

        self.tmem_load_elts: int = 32

        self._compute_stages()

    @cute.jit
    def __call__(
        self,
        hidden: cute.Tensor,
        weight: cute.Tensor,
        labels: cute.Tensor,
        dlogprobs: cute.Tensor,
        lse: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
        dHidden: cute.Tensor,
        dWeight: cute.Tensor,
        ignore_index: cutlass.Int64,
        rank: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        hidden_dtype: Type[cutlass.Numeric] = hidden.element_type
        weight_dtype: Type[cutlass.Numeric] = weight.element_type

        if cutlass.const_expr(hidden_dtype != weight_dtype):
            raise ValueError(f"Hidden and weight must have the same dtype, but got {hidden_dtype} and {weight_dtype}")
        if cutlass.const_expr(hidden_dtype not in [cutlass.Float16, cutlass.BFloat16]):
            raise ValueError(f"Hidden must be FP16 or BF16, but got {hidden_dtype}")

        dHidden_dtype: Type[cutlass.Numeric] = dHidden.element_type
        if cutlass.const_expr(dHidden_dtype != cutlass.Float32):
            raise ValueError(f"dHidden must be FP32, but got {dHidden_dtype}")
        dWeight_dtype: Type[cutlass.Numeric] = dWeight.element_type
        if cutlass.const_expr(dWeight_dtype != cutlass.Float32):
            raise ValueError(f"dWeight must be FP32, but got {dWeight_dtype}")

        self.element_type = hidden_dtype

        # NOTE: assume vocab_size >> seqlen
        problem_mnk = (hidden.shape[0], weight.shape[0], hidden.shape[1])
        print(f"[INFO]: Problem shape: {problem_mnk}")
        if cutlass.const_expr((problem_mnk[2] * hidden_dtype.width // 8) % 128 != 0):
            raise ValueError(f"K dimension is not 128B aligned: {problem_mnk[2]}")

        grid, params = self._compute_grid(
            problem_mnk,
            self.cluster_shape_mn,
            self.mma_tiler,
        )
        cute.printf("grid: {}", grid)

        hidden_major_mode = cutlass_utils.LayoutEnum.from_tensor(hidden).mma_major_mode()
        weight_major_mode = cutlass_utils.LayoutEnum.from_tensor(weight).mma_major_mode()
        print(f"hidden_major_mode: {hidden_major_mode}, weight_major_mode: {weight_major_mode}")

        logits_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            hidden_dtype,
            weight_major_mode,
            hidden_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        print(f"logits_tiled_mma: {logits_tiled_mma}")
        self._setup_attributes(logits_tiled_mma, hidden_dtype, weight_dtype)

        dW_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            hidden_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dW_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        dH_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            weight_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dH_mma_tiler[:2],
        )

        hidden_smem_layout_staged = sm100_utils.make_smem_layout_b(
            logits_tiled_mma,
            self.logits_mma_tiler,
            hidden_dtype,
            self.num_ab_stage,
        )
        print(f"hidden_smem_layout_staged: {hidden_smem_layout_staged}")
        weight_smem_layout_staged = sm100_utils.make_smem_layout_a(
            logits_tiled_mma,
            self.logits_mma_tiler,
            weight_dtype,
            self.num_ab_stage,
        )
        print(f"weight_smem_layout_staged: {weight_smem_layout_staged}")

        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            dW_tiled_mma,
            self.dW_mma_tiler,
            hidden_dtype,
            self.num_p_stage,
        )
        print(f"p_tmem_layout_staged: {p_tmem_layout_staged}")

        hiddenT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dW_tiled_mma,
            self.dW_mma_tiler,
            hidden_dtype,
            self.num_ab_stage,
        )
        print(f"hiddenT_smem_layout_staged: {hiddenT_smem_layout_staged}")

        pT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dH_tiled_mma,
            self.dH_mma_tiler,
            weight_dtype,
            self.num_p_stage,
        )
        print(f"pT_smem_layout_staged: {pT_smem_layout_staged}")

        weightT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dH_tiled_mma,
            self.dH_mma_tiler,
            weight_dtype,
            self.num_ab_stage,
        )
        print(f"weightT_smem_layout_staged: {weightT_smem_layout_staged}")
        
        # [lse, dlogprobs]
        lse_smem_layout_staged = cute.make_layout(
            (self.mma_tiler[0], 2)
        )
        labels_smem_layout_staged = cute.make_layout(
            (self.mma_tiler[0], 1)
        )
        print(f"lse_smem_layout_staged: {lse_smem_layout_staged}")

        dH_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                (self.dH_mma_tiler[0], 32), # 32 * FP32 will form a 128B swizzle atom
            ),
            self.acc_dtype,
        )
        print(f"dH_smem_layout_atom: {dH_smem_layout_atom}")
        dH_smem_layout_staged = cute.tile_to_shape(
            dH_smem_layout_atom,
            (self.dH_mma_tiler[0], 32, self.num_epi_stage),
            order=(1, 0, 2),
        )
        print(f"dH_smem_layout_staged: {dH_smem_layout_staged}")

        dW_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                (self.dW_mma_tiler[0], 32), # 32 * FP32 will form a 128B swizzle atom
            ),
            self.acc_dtype,
        )
        dW_smem_layout_staged = cute.tile_to_shape(
            dW_smem_layout_atom,
            (self.dW_mma_tiler[0], 32, self.num_epi_stage),
            order=(1, 0, 2),
        )
        print(f"dW_smem_layout_staged: {dW_smem_layout_staged}")

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp(
            reduction_kind=cpasync.ReductionOp.ADD,
        )

        tma_atom_hidden, mHidden = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            hidden,
            cute.select(hidden_smem_layout_staged, mode=[0, 1, 2]),
            self.logits_mma_tiler,
            logits_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        print(f"tma_atom_hidden: {tma_atom_hidden}")
        print(f"mHidden: {mHidden}")

        tma_atom_weight, mWeight = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            weight,
            cute.select(weight_smem_layout_staged, mode=[0, 1, 2]),
            self.logits_mma_tiler,
            logits_tiled_mma,
            self.cluster_layout_vmnk.shape
        )
        print(f"tma_atom_weight: {tma_atom_weight}")
        print(f"mWeight: {mWeight}")

        # NOTE: since mma_tiler[1] == mma_tiler[2], we can use the same TMA descriptor for dH and dW
        self.tma_copy_hidden_bytes = cute.size_in_bytes(hidden_dtype, cute.select(hidden_smem_layout_staged, mode=[0, 1, 2]))
        self.tma_copy_weight_bytes = cute.size_in_bytes(weight_dtype, cute.select(weight_smem_layout_staged, mode=[0, 1, 2]))
        print(f"tma_copy_hidden_bytes: {self.tma_copy_hidden_bytes}, tma_copy_weight_bytes: {self.tma_copy_weight_bytes}")

        tma_atom_dW, mDWeight = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dWeight,
            cute.select(dW_smem_layout_staged, mode=[0, 1]),
            (self.dW_mma_tiler[0], 32),
        )
        print(f"tma_atom_dW: {tma_atom_dW}")
        print(f"mDWeight: {mDWeight}")
        tma_atom_dH, mDHidden = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dHidden,
            cute.select(dH_smem_layout_staged, mode=[0, 1]),
            (self.dH_mma_tiler[0], 32),
        )
        print(f"tma_atom_dH: {tma_atom_dH}")
        print(f"mDHidden: {mDHidden}")

        @cute.struct
        class SharedStorage:
            load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            logits_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_p_stage * 2]
            p_tmem_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_p_stage * 2]
            p_smem_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_p_stage * 2]
            dH_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            dW_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            load_softmax_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_p_stage * 2]
            load_H_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            load_W_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]

            store_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_epi_stage * 2]

            tmem_holding_buf: cutlass.Int32
            
            sHidden: cute.struct.Align[
                cute.struct.MemRange[hidden_dtype, cute.cosize(hidden_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sWeight: cute.struct.Align[
                cute.struct.MemRange[weight_dtype, cute.cosize(weight_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[weight_dtype, cute.cosize(p_tmem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(lse_smem_layout_staged)],
                16, # 16bytes alignment
            ]
            sLabels: cute.struct.Align[
                cute.struct.MemRange[labels.element_type, cute.cosize(labels_smem_layout_staged)],
                16, # 16bytes alignment
            ]

            # sdH & sdW use the same buffer
            sdH: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(dH_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage
        print(f"shared_storage: {self.shared_storage.size_in_bytes()} bytes, capacity: {self.smem_capacity} bytes")
        sWeight_bytes = cute.size_in_bytes(weight_dtype, weight_smem_layout_staged)
        sHidden_bytes = cute.size_in_bytes(hidden_dtype, hidden_smem_layout_staged)
        sP_bytes = cute.size_in_bytes(weight_dtype, p_tmem_layout_staged)
        sLSE_bytes = cute.size_in_bytes(self.acc_dtype, lse_smem_layout_staged)
        sdH_bytes = cute.size_in_bytes(self.acc_dtype, dH_smem_layout_staged)
        in_total_bytes = sWeight_bytes + sHidden_bytes + sP_bytes + sLSE_bytes + sdH_bytes
        print(f"in_total_bytes: {in_total_bytes}")
        print(f"sWeight_bytes: {sWeight_bytes}, sHidden_bytes: {sHidden_bytes}, sP_bytes: {sP_bytes}, sLSE_bytes: {sLSE_bytes}, sdH_bytes: {sdH_bytes}")

        self.kernel(
            mHidden,
            mWeight,
            labels,
            dlogprobs,
            lse,
            scalarNumValidTokens,
            mDHidden,
            mDWeight,
            tma_atom_hidden,
            tma_atom_weight,
            tma_atom_dW,
            tma_atom_dH,
            logits_tiled_mma,
            dW_tiled_mma,
            dH_tiled_mma,
            hidden_smem_layout_staged,
            hiddenT_smem_layout_staged,
            weight_smem_layout_staged,
            weightT_smem_layout_staged,
            p_tmem_layout_staged,
            pT_smem_layout_staged,
            lse_smem_layout_staged,
            labels_smem_layout_staged,
            dH_smem_layout_staged,
            dW_smem_layout_staged,
            self.cluster_layout_vmnk,
            params,
            ignore_index,
            rank,
            weight.shape[0], # NOTE: this is the local vocab_size
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mHidden: cute.Tensor,
        mWeight: cute.Tensor,
        labels: cute.Tensor,
        dlogprobs: cute.Tensor,
        lse: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
        mDHidden: cute.Tensor,
        mDWeight: cute.Tensor,
        tma_atom_hidden: cute.CopyAtom,
        tma_atom_weight: cute.CopyAtom,
        tma_atom_dW: cute.CopyAtom,
        tma_atom_dH: cute.CopyAtom,
        logits_tiled_mma: cute.TiledMma,
        dW_tiled_mma: cute.TiledMma,
        dH_tiled_mma: cute.TiledMma,
        hidden_smem_layout_staged: cute.ComposedLayout,
        hiddenT_smem_layout_staged: cute.ComposedLayout,
        weight_smem_layout_staged: cute.ComposedLayout,
        weightT_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        pT_smem_layout_staged: cute.ComposedLayout,
        lse_smem_layout_staged: cute.Layout,
        labels_smem_layout_staged: cute.Layout,
        dH_smem_layout_staged: cute.ComposedLayout,
        dW_smem_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        scheduler_params: scheduler.ParamsBase,
        ignore_index: cutlass.Int64,
        rank: cutlass.Int32,
        local_vocab_size: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_hidden)
            cpasync.prefetch_descriptor(tma_atom_weight)
            cpasync.prefetch_descriptor(tma_atom_dW)
            cpasync.prefetch_descriptor(tma_atom_dH)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # declare pipelines
        load_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_ab_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_ids])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            barrier_storage=storage.load_ab_mbar_ptr.data_ptr(),
            tx_count=self.tma_copy_hidden_bytes + self.tma_copy_weight_bytes,
        )
        load_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_ab_stage
        )
        load_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )
        logits_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.num_p_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            barrier_storage=storage.logits_mma_mbar_ptr.data_ptr(),
        )
        logits_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_p_stage
        )
        logits_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_p_stage
        )
        p_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.num_p_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            barrier_storage=storage.p_tmem_mbar_ptr.data_ptr(),
        )
        p_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_p_stage
        )
        p_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_p_stage
        )
        pT_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.num_p_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            barrier_storage=storage.p_smem_mbar_ptr.data_ptr(),
        )
        dW_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.epilog_warp_ids)
            ),
            barrier_storage=storage.dW_mma_mbar_ptr.data_ptr(),
        )
        dW_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        dW_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        dH_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.epilog_warp_ids)
            ),
            barrier_storage=storage.dH_mma_mbar_ptr.data_ptr(),
        )
        dH_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        dH_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        load_softmax_pipeline = pipeline.PipelineCpAsync.create(
            num_stages=self.num_p_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.load_warp_ids])
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            barrier_storage=storage.load_softmax_mbar_ptr.data_ptr(),
        )
        load_softmax_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_p_stage
        )
        load_softmax_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_p_stage
        )

        load_H_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_ab_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_ids])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            barrier_storage=storage.load_H_mbar_ptr.data_ptr(),
            tx_count=self.tma_copy_hidden_bytes
        )
        load_W_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_ab_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_ids])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_ids])),
            barrier_storage=storage.load_W_mbar_ptr.data_ptr(),
            tx_count=self.tma_copy_weight_bytes
        )
        load_HW_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_ab_stage
        )
        load_HW_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )

        store_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_epi_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.epilog_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.store_warp_ids])
            ),
            barrier_storage=storage.store_mbar_ptr.data_ptr(),
        )
        store_producer_state = pipeline.PipelineState(
            self.num_epi_stage,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
        )
        store_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_epi_stage
        )

        cute.arch.mbarrier_init_fence()
        
        tmem_holding_buf = storage.tmem_holding_buf
        if cutlass.const_expr(self.real_tmem_alloc):
            if warp_idx == 0:
                cute.arch.alloc_tmem(
                    self.tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=False,
                )
        cute.arch.sync_threads()
        # a fake pointer, we already know it starts from 0
        tmem_ptr = cute.make_ptr(self.acc_dtype, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        if cutlass.const_expr(self.real_tmem_alloc):
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

        # get SMEM tensors
        sHidden = storage.sHidden.get_tensor(hidden_smem_layout_staged.outer, swizzle=hidden_smem_layout_staged.inner)
        print(f"sHidden: {sHidden}")
        sWeight = storage.sWeight.get_tensor(weight_smem_layout_staged.outer, swizzle=weight_smem_layout_staged.inner)
        print(f"sWeight: {sWeight}")
        sHiddenT_ptr = cute.recast_ptr(sHidden.iterator, hiddenT_smem_layout_staged.inner)
        sHiddenT = cute.make_tensor(sHiddenT_ptr, hiddenT_smem_layout_staged.outer)
        print(f"sHiddenT: {sHiddenT}")
        sWeightT_ptr = cute.recast_ptr(sWeight.iterator, weightT_smem_layout_staged.inner)
        sWeightT = cute.make_tensor(sWeightT_ptr, weightT_smem_layout_staged.outer)
        print(f"sWeightT: {sWeightT}")
        sP = storage.sP.get_tensor(p_tmem_layout_staged.outer, swizzle=p_tmem_layout_staged.inner)
        print(f"sP: {sP}")
        sPT_ptr = cute.recast_ptr(sP.iterator, pT_smem_layout_staged.inner)
        sPT = cute.make_tensor(sPT_ptr, pT_smem_layout_staged.outer)
        print(f"sPT: {sPT}")

        # [lse, dlogprobs]
        sLSE_and_dlogprobs = storage.sLSE.get_tensor(lse_smem_layout_staged)
        print(f"sLSE_and_dlogprobs: {sLSE_and_dlogprobs}")
        sLSE = cute.append_ones(sLSE_and_dlogprobs[None, 0])
        sdlogprobs = cute.append_ones(sLSE_and_dlogprobs[None, 1])
        print(f"sLSE: {sLSE}, sdlogprobs: {sdlogprobs}")
        sLabels = storage.sLabels.get_tensor(labels_smem_layout_staged)
        print(f"sLabels: {sLabels}")

        sdH = storage.sdH.get_tensor(dH_smem_layout_staged.outer, swizzle=dH_smem_layout_staged.inner)
        print(f"sdH: {sdH}")
        sdW_ptr = cute.recast_ptr(sdH.iterator, dW_smem_layout_staged.inner)
        sdW = cute.make_tensor(sdW_ptr, dW_smem_layout_staged.outer)
        print(f"sdW: {sdW}")

        # slice GMEM tensors
        gHidden = cute.local_tile(
            mHidden,
            (self.mma_tiler[1], self.mma_tiler[2]),
            (None, None)
        )
        print(f"gHidden: {gHidden}")
        gWeight = cute.local_tile(
            mWeight,
            (self.mma_tiler[0], self.mma_tiler[2]),
            (None, None)
        )
        print(f"gWeight: {gWeight}")
        gLabels = cute.local_tile(
            labels,
            (self.mma_tiler[1],),
            (None,)
        )
        print(f"gLabels: {gLabels}")
        gLSE = cute.local_tile(
            lse,
            (self.mma_tiler[1],),
            (None,)
        )
        print(f"gLSE: {gLSE}")

        # 32xFP32 will form a 128B swizzle atom
        gDHidden = cute.local_tile(
            mDHidden,
            (self.dW_mma_tiler[0], 32),
            (None, None)
        )
        print(f"gDHidden: {gDHidden}")
        gDWeight = cute.local_tile(
            mDWeight,
            (self.dH_mma_tiler[0], 32),
            (None, None)
        )
        print(f"gDWeight: {gDWeight}")

        # get TMEM tensors
        thr_logits_mma = logits_tiled_mma.get_slice(0) # 1 CTA
        tLogits_shape = (self.logits_mma_tiler[0], self.tmem_cols_logits)
        tCtLogits_shape = thr_logits_mma.partition_shape_C(tLogits_shape)
        tCtLogits_fake = thr_logits_mma.make_fragment_C(tCtLogits_shape)
        tCtLogits = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_offset_logits, dtype=self.acc_dtype),
            tCtLogits_fake.layout,
        )
        print(f"tCtLogits: {tCtLogits}")

        thr_dW_mma = dW_tiled_mma.get_slice(0) # 1 CTA
        tDWeight_shape = (self.dW_mma_tiler[0], self.tmem_cols_dW)
        tCtDWeight_shape = thr_dW_mma.partition_shape_C(tDWeight_shape)
        tCtDWeight_fake = thr_dW_mma.make_fragment_C(tCtDWeight_shape)
        tCtDWeight = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_offset_dW, dtype=self.acc_dtype),
            tCtDWeight_fake.layout,
        )
        print(f"tCtDWeight: {tCtDWeight}")

        thr_dH_mma = dH_tiled_mma.get_slice(0) # 1 CTA
        tDHidden_shape = (self.dH_mma_tiler[0], self.tmem_cols_dH)
        tCtDHidden_shape = thr_dH_mma.partition_shape_C(tDHidden_shape)
        tCtDHidden_fake = thr_dH_mma.make_fragment_C(tCtDHidden_shape)
        tCtDHidden = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_offset_dH, dtype=self.acc_dtype),
            tCtDHidden_fake.layout,
        )
        print(f"tCtDHidden: {tCtDHidden}")

        tP_ptr = cute.recast_ptr(tmem_ptr + self.tmem_offset_p, dtype=sHidden.element_type)
        tP_fake = cute.make_tensor(tP_ptr, p_tmem_layout_staged.outer)
        tCtP_fake = thr_dW_mma.make_fragment_A(tP_fake)
        tCtP = cute.make_tensor(tP_ptr, tCtP_fake.layout)
        print(f"tCtP: {tCtP}")


        TileSchedulerCls = partial(self.scheduler_cls.create, scheduler_params)

        if warp_idx in self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if warp_idx == self.load_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            self.load(
                TileSchedulerCls,
                thr_logits_mma,
                thr_dW_mma,
                thr_dW_mma,
                gHidden,
                gWeight,
                sHidden,
                sWeight,
                tma_atom_weight,
                tma_atom_hidden,
                load_pipeline,
                load_producer_state,
                cute.size(gHidden, mode=[2]),
                cute.size(gHidden, mode=[3]),
                sLSE,
                gLSE,
                load_softmax_pipeline,
                load_softmax_producer_state,
                sdlogprobs,
                dlogprobs,
                sLabels,
                gLabels,
                load_H_pipeline,
                load_W_pipeline,
                load_HW_producer_state,
                sHiddenT,
                sWeightT,
            )

        if warp_idx == self.mma_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)

            self.mma(
                TileSchedulerCls,
                logits_tiled_mma,
                dW_tiled_mma,
                dH_tiled_mma,
                load_pipeline,
                load_consumer_state,
                sHidden,
                sWeight,
                tCtLogits,
                cute.size(gHidden, mode=[2]),
                cute.size(gHidden, mode=[3]),
                logits_mma_pipeline,
                logits_mma_producer_state,
                p_pipeline,
                p_consumer_state,
                pT_pipeline,
                dW_mma_pipeline,
                dW_mma_producer_state,
                dH_mma_pipeline,
                dH_mma_producer_state,
                load_H_pipeline,
                load_W_pipeline,
                load_HW_consumer_state,
                sHiddenT,
                sWeightT,
                tCtDWeight,
                tCtDHidden,
                tCtP,
                sPT,
            )

        if warp_idx in self.softmax_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            self.softmax(
                TileSchedulerCls,
                cute.size(gHidden, mode=[2]),
                logits_mma_pipeline,
                logits_mma_consumer_state,
                p_pipeline,
                p_producer_state,
                pT_pipeline,
                load_softmax_pipeline,
                load_softmax_consumer_state,
                tCtLogits,
                thr_logits_mma,
                sLSE,
                sdlogprobs,
                dlogprobs,
                sLabels,
                rank,
                local_vocab_size,
                ignore_index,
                tCtP,
                sP,
                scalarNumValidTokens,
            )

        if warp_idx in self.epilog_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_epilog)

            self.epilog(
                TileSchedulerCls,
                cute.size(gHidden, mode=[2]),
                cute.size(gHidden, mode=[3]),
                dW_mma_pipeline,
                dW_mma_consumer_state,
                dH_mma_pipeline,
                dH_mma_consumer_state,
                tCtDWeight,
                tCtDHidden,
                sdH,
                sdW,
                thr_dW_mma,
                store_pipeline,
                store_producer_state,
            )

        if warp_idx == self.store_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_store)

            self.store(
                TileSchedulerCls,
                store_pipeline,
                store_consumer_state,
                cute.size(gHidden, mode=[2]),
                cute.size(gHidden, mode=[3]),
                sdW,
                sdH,
                gDWeight,
                gDHidden,
                tma_atom_dW,
                tma_atom_dH,
            )

        if cutlass.const_expr(self.real_tmem_alloc):
            cute.arch.sync_threads()
            if warp_idx == 0:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(
                    tmem_ptr,
                    self.tmem_alloc_cols,
                )


    @cute.jit
    def load(
        self,
        TileSchedulerCls: Callable,
        thr_mma_logits: cute.ThrMma,
        thr_mma_dW: cute.ThrMma,
        thr_mma_dH: cute.ThrMma,
        gHidden: cute.Tensor,
        gWeight: cute.Tensor,
        sHidden: cute.Tensor,
        sWeight: cute.Tensor,
        tma_atom_weight: cute.CopyAtom,
        tma_atom_hidden: cute.CopyAtom,
        load_pipeline: pipeline.PipelineTmaUmma,
        load_producer_state: pipeline.PipelineState,
        numN: cutlass.Int32,
        numK: cutlass.Int32,
        sLSE: cute.Tensor,
        gLSE: cute.Tensor,
        load_softmax_pipeline: pipeline.PipelineCpAsync,
        load_softmax_producer_state: pipeline.PipelineState,
        sdlogprobs: cute.Tensor,
        dlogprobs: cute.Tensor,
        sLabels: cute.Tensor,
        gLabels: cute.Tensor,
        load_H_pipeline: pipeline.PipelineTmaUmma,
        load_W_pipeline: pipeline.PipelineTmaUmma,
        load_HW_producer_state: pipeline.PipelineState,
        sHiddenT: cute.Tensor,
        sWeightT: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()

        tAgWeight = thr_mma_logits.partition_A(gWeight)
        print(f"tAgWeight: {tAgWeight}")
        tTMAsWeight, tTMAgWeight = cpasync.tma_partition(
            tma_atom_weight,
            0,
            cute.make_layout(1),
            cute.group_modes(sWeight, 0, 3),
            cute.group_modes(tAgWeight, 0, 3),
        )
        print(f"tTMAsWeight: {tTMAsWeight}, tTMAgWeight: {tTMAgWeight}")

        tBgHidden = thr_mma_logits.partition_B(gHidden)
        print(f"tBgHidden: {tBgHidden}")
        tTMAsHidden, tTMAgHidden = cpasync.tma_partition(
            tma_atom_hidden,
            0,
            cute.make_layout(1),
            cute.group_modes(sHidden, 0, 3),
            cute.group_modes(tBgHidden, 0, 3),
        )
        print(f"tTMAsHidden: {tTMAsHidden}, tTMAgHidden: {tTMAgHidden}")

        tBgWeightT = thr_mma_dH.partition_B(gWeight)
        print(f"tBgWeightT: {tBgWeightT}")
        tTMAsWeightT, tTMAgWeightT = cpasync.tma_partition(
            tma_atom_weight,
            0,
            cute.make_layout(1),
            cute.group_modes(sWeightT, 0, 3),
            cute.group_modes(tBgWeightT, 0, 3),
        )
        print(f"tTMAsWeightT: {tTMAsWeightT}, tTMAgWeightT: {tTMAgWeightT}")

        tBgHiddenT = thr_mma_dW.partition_B(gHidden)
        print(f"tBgHiddenT: {tBgHiddenT}")
        tTMAsHiddenT, tTMAgHiddenT = cpasync.tma_partition(
            tma_atom_hidden,
            0,
            cute.make_layout(1),
            cute.group_modes(sHiddenT, 0, 3),
            cute.group_modes(tBgHiddenT, 0, 3),
        )
        print(f"tTMAsHiddenT: {tTMAsHiddenT}, tTMAgHiddenT: {tTMAgHiddenT}")

        async_g2s_fp32_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Float32,
            num_bits_per_copy=32, # 16bytes per copy
        )
        print(f"async_g2s_fp32_atom: {async_g2s_fp32_atom}")
        async_g2s_int64_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int64,
            num_bits_per_copy=64, # 16Bytes per copy
        )
        print(f"async_g2s_int64_atom: {async_g2s_int64_atom}")

        tiled_copy_g2s_fp32 = cute.make_tiled_copy_tv(
            async_g2s_fp32_atom,
            cute.make_layout(self.threads_per_warp), # 32-threads per warp
            cute.make_layout(sLSE.shape[0] // self.threads_per_warp), # 1 value per thread
        )
        tiled_copy_g2s_int64 = cute.make_tiled_copy_tv(
            async_g2s_int64_atom,
            cute.make_layout(self.threads_per_warp), # 32-threads per warp
            cute.make_layout(4), # 2 value per thread
        )
        thr_copy_g2s_fp32 = tiled_copy_g2s_fp32.get_slice(lane_idx)
        thr_copy_g2s_int64 = tiled_copy_g2s_int64.get_slice(lane_idx)
        
        tsLSE = thr_copy_g2s_fp32.partition_D(sLSE)
        print(f"tsLSE: {tsLSE}")
        tgLSE = thr_copy_g2s_fp32.partition_S(gLSE)
        print(f"tgLSE: {tgLSE}")

        tsDlogprobs = None
        tgDlogprobs = None
        if cutlass.const_expr(self.REDUCTION == utils.EntropyReductionEnum.kNone):
            gDlogprobs = cute.local_tile(
                dlogprobs,
                (self.mma_tiler[1],),
                (None,)
            )
            tsDlogprobs = thr_copy_g2s_fp32.partition_D(sdlogprobs)
            print(f"tsDlogprobs: {tsDlogprobs}")
            tgDlogprobs = thr_copy_g2s_fp32.partition_S(gDlogprobs)
            print(f"tgDlogprobs: {tgDlogprobs}")

        tsLabels = thr_copy_g2s_int64.partition_D(sLabels)
        print(f"tsLabels: {tsLabels}")
        tgLabels = thr_copy_g2s_int64.partition_S(gLabels)
        print(f"tgLabels: {tgLabels}")

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            midx, _ = work_tile.tile_idx

            for n in cutlass.range(numN):
                # load data for logits MMA
                for k in cutlass.range(numK):
                    load_pipeline.producer_acquire(load_producer_state)

                    cute.copy(
                        tma_atom_weight,
                        tTMAgWeight[None, midx, k],
                        tTMAsWeight[None, load_producer_state.index],
                        tma_bar_ptr=load_pipeline.producer_get_barrier(load_producer_state)
                    )
                    cute.copy(
                        tma_atom_hidden,
                        tTMAgHidden[None, n, k],
                        tTMAsHidden[None, load_producer_state.index],
                        tma_bar_ptr=load_pipeline.producer_get_barrier(load_producer_state)
                    )

                    load_pipeline.producer_commit(load_producer_state)
                    load_producer_state.advance()
                # load data for softmax
                load_softmax_pipeline.producer_acquire(load_softmax_producer_state)
                cute.copy(
                    async_g2s_fp32_atom,
                    tgLSE[None, 0, n],
                    tsLSE[None, 0, load_softmax_producer_state.index],
                )
                if cutlass.const_expr(self.REDUCTION == utils.EntropyReductionEnum.kNone):
                    cute.copy(
                        async_g2s_fp32_atom,
                        tgDlogprobs[None, 0, n],
                        tsDlogprobs[None, 0, load_softmax_producer_state.index],
                    )
                cute.copy(
                    async_g2s_int64_atom,
                    tgLabels[None, 0, n],
                    tsLabels[None, 0, load_softmax_producer_state.index],
                )
                load_softmax_pipeline.producer_commit(load_softmax_producer_state)
                load_softmax_producer_state.advance()

                # load data for dH and dW
                for k in cutlass.range(numK):
                    load_H_pipeline.producer_acquire(load_HW_producer_state)
                    cute.copy(
                        tma_atom_hidden,
                        tTMAgHiddenT[None, n, k],
                        tTMAsHiddenT[None, load_HW_producer_state.index],
                        tma_bar_ptr=load_H_pipeline.producer_get_barrier(load_HW_producer_state)
                    )
                    load_H_pipeline.producer_commit(load_HW_producer_state)

                    load_W_pipeline.producer_acquire(load_HW_producer_state)
                    cute.copy(
                        tma_atom_weight,
                        tTMAgWeightT[None, midx, k],
                        tTMAsWeightT[None, load_HW_producer_state.index],
                        tma_bar_ptr=load_W_pipeline.producer_get_barrier(load_HW_producer_state)
                    )
                    load_W_pipeline.producer_commit(load_HW_producer_state)

                    load_HW_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        TileSchedulerCls: Callable,
        logits_tiled_mma: cute.TiledMma,
        dW_tiled_mma: cute.TiledMma,
        dH_tiled_mma: cute.TiledMma,
        load_pipeline: pipeline.PipelineTmaUmma,
        load_consumer_state: pipeline.PipelineState,
        sHidden: cute.Tensor,
        sWeight: cute.Tensor,
        tCtLogits: cute.Tensor,
        numN: cutlass.Int32,
        numK: cutlass.Int32,
        logits_mma_pipeline: pipeline.PipelineUmmaAsync,
        logits_mma_producer_state: pipeline.PipelineState,
        p_pipeline: pipeline.PipelineAsyncUmma,
        p_consumer_state: pipeline.PipelineState,
        pT_pipeline: pipeline.PipelineAsyncUmma,
        dW_mma_pipeline: pipeline.PipelineUmmaAsync,
        dW_mma_producer_state: pipeline.PipelineState,
        dH_mma_pipeline: pipeline.PipelineUmmaAsync,
        dH_mma_producer_state: pipeline.PipelineState,
        load_H_pipeline: pipeline.PipelineTmaUmma,
        load_W_pipeline: pipeline.PipelineTmaUmma,
        load_HW_consumer_state: pipeline.PipelineState,
        sHiddenT: cute.Tensor,
        sWeightT: cute.Tensor,
        tCtDWeight: cute.Tensor,
        tCtDHidden: cute.Tensor,
        tCtP: cute.Tensor,
        sPT: cute.Tensor,
    ):
        thr_mma_logits = logits_tiled_mma.get_slice(0) # 1 CTA
        thr_mma_dW = dW_tiled_mma.get_slice(0) # 1 CTA
        thr_mma_dH = dH_tiled_mma.get_slice(0) # 1 CTA

        tCsWeight = thr_mma_logits.make_fragment_A(sWeight)
        print(f"tCsWeight: {tCsWeight}")
        tCsHidden = thr_mma_logits.make_fragment_B(sHidden)
        print(f"tCsHidden: {tCsHidden}")

        tCsHiddenT = thr_mma_dW.make_fragment_B(sHiddenT)
        print(f"tCsHiddenT: {tCsHiddenT}")

        tCsPT = thr_mma_dH.make_fragment_A(sPT)
        print(f"tCsPT: {tCsPT}")
        tCsWeightT = thr_mma_dH.make_fragment_B(sWeightT)
        print(f"tCsWeightT: {tCsWeightT}")

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            midx, _ = work_tile.tile_idx

            for n in cutlass.range(numN):
                logits_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                logits_mma_pipeline.producer_acquire(logits_mma_producer_state)
                for k in cutlass.range(numK):
                    load_pipeline.consumer_wait(load_consumer_state)

                    for kblock in cutlass.range_constexpr(cute.size(tCsWeight, mode=[2])):
                        cute.gemm(
                            logits_tiled_mma,
                            cute.append_ones(tCtLogits[None, None, logits_mma_producer_state.index]),
                            tCsWeight[None, None, kblock, load_consumer_state.index],
                            tCsHidden[None, None, kblock, load_consumer_state.index],
                            cute.append_ones(tCtLogits[None, None, logits_mma_producer_state.index]),
                        )
                        logits_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    load_pipeline.consumer_release(load_consumer_state)
                    load_consumer_state.advance()
                logits_mma_pipeline.producer_commit(logits_mma_producer_state)
                logits_mma_producer_state.advance()

                p_pipeline.consumer_wait(p_consumer_state)
                pT_pipeline.consumer_wait(p_consumer_state)
                for k in cutlass.range(numK):
                    load_H_pipeline.consumer_wait(load_HW_consumer_state)
                    dW_mma_pipeline.producer_acquire(dW_mma_producer_state)

                    dW_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kblock in cutlass.range_constexpr(cute.size(tCsWeightT, mode=[2])):
                        cute.gemm(
                            dW_tiled_mma,
                            cute.append_ones(tCtDWeight[None, None, dW_mma_producer_state.index]),
                            tCtP[None, None, kblock, p_consumer_state.index],
                            tCsHiddenT[None, None, kblock, load_HW_consumer_state.index],
                            cute.append_ones(tCtDWeight[None, None, dW_mma_producer_state.index]),
                        )
                        dW_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    dW_mma_pipeline.producer_commit(dW_mma_producer_state)
                    load_H_pipeline.consumer_release(load_HW_consumer_state)
                    dW_mma_producer_state.advance()

                    load_W_pipeline.consumer_wait(load_HW_consumer_state)
                    dH_mma_pipeline.producer_acquire(dH_mma_producer_state)

                    dH_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kblock in cutlass.range_constexpr(cute.size(tCsHiddenT, mode=[2])):
                        cute.gemm(
                            dH_tiled_mma,
                            cute.append_ones(tCtDHidden[None, None, dH_mma_producer_state.index]),
                            tCsPT[None, None, kblock, p_consumer_state.index],
                            tCsWeightT[None, None, kblock, load_HW_consumer_state.index],
                            cute.append_ones(tCtDHidden[None, None, dH_mma_producer_state.index]),
                        )
                        dH_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    dH_mma_pipeline.producer_commit(dH_mma_producer_state)
                    load_W_pipeline.consumer_release(load_HW_consumer_state)
                    dH_mma_producer_state.advance()
                    load_HW_consumer_state.advance()

                p_pipeline.consumer_release(p_consumer_state)
                pT_pipeline.consumer_release(p_consumer_state)
                p_consumer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def softmax(
        self,
        TileSchedulerCls: Callable,
        numN: cutlass.Int32,
        logits_mma_pipeline: pipeline.PipelineUmmaAsync,
        logits_mma_consumer_state: pipeline.PipelineState,
        p_pipeline: pipeline.PipelineAsyncUmma,
        p_producer_state: pipeline.PipelineState,
        pT_pipeline: pipeline.PipelineAsyncUmma,
        load_softmax_pipeline: pipeline.PipelineCpAsync,
        load_softmax_consumer_state: pipeline.PipelineState,
        tCtLogits: cute.Tensor,
        thr_mma_logits: cute.ThrMma,
        sLSE: cute.Tensor,
        sdlogprobs: cute.Tensor,
        dlogprobs: cute.Tensor,
        sLabels: cute.Tensor,
        rank: cutlass.Int32,
        local_vocab_size: cutlass.Int32,
        ignore_index: cutlass.Int64,
        tCtP: cute.Tensor,
        sP: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            self.threads_per_warp * len(self.softmax_warp_ids)
        )
        tlogits_load_elts: int = 64
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(
                tcgen05.copy.Repetition(tlogits_load_elts),
            ),
            self.acc_dtype,
        )
        tLogits_load = cute.flat_divide(
            tCtLogits[(None, None,), 0, None],
            (self.logits_mma_tiler[0], tlogits_load_elts),
        )
        print(f"tLogits_load: {tLogits_load}")
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r,
            tLogits_load[None, None, 0, 0, 0]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tCtLogits_load = thr_copy_t2r.partition_S(tLogits_load)
        print(f"tCtLogits_load: {tCtLogits_load}")

        cLogits = cute.make_identity_tensor(self.logits_mma_tiler[:2])
        tCcLogits = thr_mma_logits.partition_C(cLogits)
        cLogits_load = cute.flat_divide(
            tCcLogits[(None, None), 0, None],
            (self.logits_mma_tiler[0], tlogits_load_elts),
        )
        tCcLogits_load = thr_copy_t2r.partition_D(cLogits_load)
        print(f"tCcLogits_load: {tCcLogits_load}")

        tCrLogits_load = cute.make_rmem_tensor(
            cute.select(tCcLogits_load.shape, mode=[0, 1, 2]),
            self.acc_dtype,
        )
        print(f"tCrLogits_load: {tCrLogits_load}")
        tCrP_store = cute.make_rmem_tensor(
            tCrLogits_load.shape,
            self.element_type,
        )
        print(f"tCrP_store: {tCrP_store}")

        copy_atom_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(tlogits_load_elts // 2),
            ),
            self.element_type,
        )
        def _get_tP_store(tP: cute.Tensor) -> cute.Tensor:
            layout = cute.flatten(tP.layout)
            mode1 = cute.make_layout(
                cute.get(layout, mode=[1]).shape * cute.get(layout, mode=[3]).shape,
                stride=cute.get(layout, mode=[1]).stride,
            )
            shape = (
                cute.get(layout, mode=[0]).shape,
                mode1.shape,
                cute.get(layout, mode=[2]).shape,
                cute.get(layout, mode=[4]).shape,
                cute.get(layout, mode=[5]).shape,
            )
            stride = (
                cute.get(layout, mode=[0]).stride,
                mode1.stride,
                cute.get(layout, mode=[2]).stride,
                cute.get(layout, mode=[4]).stride,
                cute.get(layout, mode=[5]).stride,
            )
            layout = cute.make_layout(shape, stride=stride)
            return cute.make_tensor(tP.iterator, layout)

        tP_store = _get_tP_store(tCtP)
        print(f"tP_store: {tP_store}")
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            copy_atom_r2t,
            tP_store[None, None, 0, 0, 0],
        )
        thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

        tCtP_store = thr_copy_r2t.partition_D(tP_store)
        print(f"tCtP_store: {tCtP_store}")

        sP_cpy = cute.composition(
            sP,
            cute.make_ordered_layout(
                (self.logits_mma_tiler[0], self.logits_mma_tiler[1], self.num_p_stage),
                (0, 1, 2)
            )
        )
        print(f"sP_cpy: {sP_cpy}")
        thr_sP_cpy = cute.flatten(sP_cpy[tidx, None, None])
        print(f"thr_sP_cpy: {thr_sP_cpy}")


        dlogprob: cutlass.Float32 = 0.0
        num_valid_tokens = None
        if cutlass.const_expr(self.REDUCTION == utils.EntropyReductionEnum.kMean):
            num_valid_tokens = cute.arch.rcp_approx(
                cute.make_tensor(scalarNumValidTokens, 
                                layout=(1,)).to(cutlass.Float32)
            )
            dlogprob = dlogprobs[0] * num_valid_tokens
        elif cutlass.const_expr(self.REDUCTION == utils.EntropyReductionEnum.kSum):
            dlogprob = dlogprobs[0]

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            midx, _ = work_tile.tile_idx

            position: cutlass.Int64 = (
                rank * local_vocab_size
                + midx * self.logits_mma_tiler[0]
                + tidx
            )

            for n in cutlass.range(numN):
                logits_mma_pipeline.consumer_wait(logits_mma_consumer_state)
                load_softmax_pipeline.consumer_wait(load_softmax_consumer_state)
                p_pipeline.producer_acquire(p_producer_state)
                pT_pipeline.producer_acquire(p_producer_state)
                for nblock in cutlass.range(self.logits_mma_tiler[1] // tlogits_load_elts):
                    # t2r
                    cute.copy(
                        tiled_copy_t2r,
                        tCtLogits_load[None, None, None, 0, nblock, logits_mma_consumer_state.index],
                        tCrLogits_load
                    )
                    for idx in cutlass.range_constexpr(tlogits_load_elts):
                        # softmax
                        tCrLogits_load[idx] = cute.exp(tCrLogits_load[idx] - sLSE[tlogits_load_elts * nblock + idx])
                        
                        label: cutlass.Int64 = sLabels[tlogits_load_elts * nblock + idx]
                        mask: cutlass.Boolean = (label != ignore_index and position == label)

                        if cutlass.const_expr(self.REDUCTION == utils.EntropyReductionEnum.kNone):
                            dlogprob = sdlogprobs[tlogits_load_elts * nblock + idx]

                        tCrLogits_load[idx] = ptx.fma(
                            dlogprob,
                            tCrLogits_load[idx],
                            -dlogprob * mask
                        )
                    # type conversion
                    tCrP_store.store(tCrLogits_load.load().to(self.element_type))
                    # r2t
                    cute.copy(
                        tiled_copy_r2t,
                        tCrP_store,
                        tCtP_store[None, None, None, 0, nblock, p_producer_state.index],
                    )
                    # r2s
                    cute.autovec_copy(
                        tCrP_store,
                        thr_sP_cpy[None, nblock, p_producer_state.index],
                    )
                load_softmax_pipeline.consumer_release(load_softmax_consumer_state)
                load_softmax_consumer_state.advance()

                logits_mma_pipeline.consumer_release(logits_mma_consumer_state)
                logits_mma_consumer_state.advance()
                
                p_pipeline.producer_commit(p_producer_state)
                pT_pipeline.producer_commit(p_producer_state)
                p_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilog(
        self,
        TileSchedulerCls: Callable,
        numN: cutlass.Int32,
        numK: cutlass.Int32,
        dW_mma_pipeline: pipeline.PipelineUmmaAsync,
        dW_mma_consumer_state: pipeline.PipelineState,
        dH_mma_pipeline: pipeline.PipelineUmmaAsync,
        dH_mma_consumer_state: pipeline.PipelineState,
        tCtDWeight: cute.Tensor,
        tCtDHidden: cute.Tensor,
        sdH: cute.Tensor,
        sdW: cute.Tensor,
        thr_mma_dW: cute.ThrMma,
        store_pipeline: pipeline.PipelineAsync,
        store_producer_state: pipeline.PipelineState,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            self.threads_per_warp * len(self.epilog_warp_ids)
        )
        # 32xFP32 will form 128B swizzle pattern
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(
                tcgen05.copy.Repetition(self.tmem_load_elts),
            ),
            self.acc_dtype,
        )
        tdW_load = cute.flat_divide(
            tCtDWeight[(None, None), 0, None],
            (self.dW_mma_tiler[0], self.tmem_load_elts),
        )
        print(f"tdW_load: {tdW_load}")
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r,
            tdW_load[None, None, 0, 0, 0],
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        tCtdW_load = thr_copy_t2r.partition_S(tdW_load)
        print(f"tCtdW_load: {tCtdW_load}")

        tdH_load = cute.flat_divide(
            tCtDHidden[(None, None), 0, None],
            (self.dH_mma_tiler[0], self.tmem_load_elts),
        )
        tCtdH_load = thr_copy_t2r.partition_S(tdH_load)
        print(f"tCtdH_load: {tCtdH_load}")

        cdW = cute.make_identity_tensor(self.dW_mma_tiler[:2])
        tCcdW = thr_mma_dW.partition_C(cdW)
        cdW_load = cute.flat_divide(
            tCcdW[(None, None), 0, None],
            (self.dW_mma_tiler[0], self.tmem_load_elts),
        )
        tCcdW_load = thr_copy_t2r.partition_D(cdW_load)
        print(f"tCcdW_load: {tCcdW_load}")

        tCrO_load = cute.make_rmem_tensor(
            cute.select(tCcdW_load.shape, mode=[0, 1, 2]),
            self.acc_dtype,
        )

        sdH_cpy = cute.composition(
            sdH,
            cute.make_ordered_layout(
                (self.dH_mma_tiler[0], self.dH_mma_tiler[1], 1),
                (0, 1, 2),
            )
        )
        print(f"sdH_cpy: {sdH_cpy}")
        thr_sdH_cpy = cute.flatten(sdH_cpy[tidx, None, None])
        print(f"thr_sdH_cpy: {thr_sdH_cpy}")

        sdW_cpy = cute.composition(
            sdW,
            cute.make_ordered_layout(
                (self.dW_mma_tiler[0], self.dW_mma_tiler[1], 1),
                (0, 1, 2),
            )
        )
        print(f"sdW_cpy: {sdW_cpy}")
        thr_sdW_cpy = cute.flatten(sdW_cpy[tidx, None, None])
        print(f"thr_sdW_cpy: {thr_sdW_cpy}")
    
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            midx = work_tile.tile_idx

            for n in cutlass.range(numN):
                for k in cutlass.range(numK):
                    dW_mma_pipeline.consumer_wait(dW_mma_consumer_state)
                    for nblock in cutlass.range(self.dW_mma_tiler[1] // self.tmem_load_elts):
                        # t2r
                        cute.copy(
                            tiled_copy_t2r,
                            tCtdW_load[None, None, None, 0, nblock, dW_mma_consumer_state.index],
                            tCrO_load
                        )
                        # r2s
                        store_pipeline.producer_acquire(store_producer_state)
                        cute.autovec_copy(
                            tCrO_load,
                            thr_sdW_cpy[None, nblock, store_producer_state.index],
                        )
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        store_pipeline.producer_commit(store_producer_state)
                        store_producer_state.advance()
                    dW_mma_pipeline.consumer_release(dW_mma_consumer_state)
                    dW_mma_consumer_state.advance()

                    dH_mma_pipeline.consumer_wait(dH_mma_consumer_state)
                    for nblock in cutlass.range(self.dH_mma_tiler[1] // self.tmem_load_elts):
                        # t2r
                        cute.copy(
                            tiled_copy_t2r,
                            tCtdH_load[None, None, None, 0, nblock, dH_mma_consumer_state.index],
                            tCrO_load
                        )
                        # r2s
                        store_pipeline.producer_acquire(store_producer_state)
                        cute.autovec_copy(
                            tCrO_load,
                            thr_sdH_cpy[None, nblock, store_producer_state.index],
                        )
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        store_pipeline.producer_commit(store_producer_state)
                        store_producer_state.advance()
                    dH_mma_pipeline.consumer_release(dH_mma_consumer_state)
                    dH_mma_consumer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def store(
        self,
        TileSchedulerCls: Callable,
        store_pipeline: pipeline.PipelineAsync,
        store_consumer_state: pipeline.PipelineState,
        numN: cutlass.Int32,
        numK: cutlass.Int32,
        sdW: cute.Tensor,
        sdH: cute.Tensor,
        gDWeight: cute.Tensor,
        gDHidden: cute.Tensor,
        tma_atom_dW: cute.CopyAtom,
        tma_atom_dH: cute.CopyAtom,
    ):
        tTMAsdH, tTMAgdH = cpasync.tma_partition(
            tma_atom_dH,
            0,
            cute.make_layout(1),
            cute.group_modes(sdH, 0, 2),
            cute.group_modes(gDHidden, 0, 2),
        )
        print(f"tTMAsdH: {tTMAsdH}, tTMAgdH: {tTMAgdH}")

        print(f"tma_atom_dH: {tma_atom_dH}")
        print(f"tma_atom_dW: {tma_atom_dW}")

        tTMAsdW, tTMAgdW = cpasync.tma_partition(
            tma_atom_dW,
            0,
            cute.make_layout(1),
            cute.group_modes(sdW, 0, 2),
            cute.group_modes(gDWeight, 0, 2),
        )
        print(f"tTMAsdW: {tTMAsdW}, tTMAgdW: {tTMAgdW}")

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            midx, _ = work_tile.tile_idx

            num_iters: int = self.dW_mma_tiler[1] // self.tmem_load_elts

            for n in cutlass.range(numN):
                for k in cutlass.range(numK):
                    for nblock in cutlass.range(num_iters):
                        cute.arch.cp_async_bulk_wait_group(self.num_epi_stage - 1, read=True)
                        store_pipeline.consumer_release(store_consumer_state)
                        store_pipeline.consumer_wait(store_consumer_state)
                        # do tma store
                        cute.copy(
                            tma_atom_dW,
                            tTMAsdW[None, store_consumer_state.index],
                            tTMAgdW[None, midx, k * num_iters + nblock],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        store_consumer_state.advance()

                    for nblock in cutlass.range(num_iters):
                        cute.arch.cp_async_bulk_wait_group(self.num_epi_stage - 1, read=True)
                        store_pipeline.consumer_release(store_consumer_state)
                        store_pipeline.consumer_wait(store_consumer_state)
                        cute.copy(
                            tma_atom_dH,
                            tTMAsdH[None, store_consumer_state.index],
                            tTMAgdH[None, n, k * num_iters + nblock],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        store_consumer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

if __name__ == "__main__":
    # python -m transformer_engine.common.cutedsl.linear_cross_entropy.blackwell.bwd_dHdW

    import torch
    from cutlass.cute.runtime import from_dlpack
    from ..utils import str_to_reduction_enum

    torch.manual_seed(1111)

    batchsize = 1
    seqlen = 128
    vocabsize = 128
    dim = 128

    # batchsize = 4
    # seqlen = 2035
    # vocabsize = 152063
    # dim = 4096
    dtype = torch.bfloat16
    reduction = "none"
    ignore_index = -100
    rank = 0

    hidden = (
        torch.empty((batchsize, seqlen, dim), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_(True)
    )
    weight = (
        torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_(True)
    )
    # hidden = torch.ones((batchsize, seqlen, dim), dtype=dtype, device="cuda")
    # weight = torch.ones((vocabsize, dim), dtype=dtype, device="cuda")
    labels = torch.randint(0, vocabsize, (batchsize, seqlen), dtype=torch.long, device="cuda")

    num_valid_tokens = torch.sum(labels != ignore_index)

    dlogprobs = None
    if reduction == "none":
        dlogprobs = torch.randn((batchsize, seqlen), dtype=torch.float32, device="cuda")
    elif reduction == "sum":
        dlogprobs = torch.randn(1, dtype=torch.float32, device="cuda")
    elif reduction == "mean":
        dlogprobs = torch.randn(1, dtype=torch.float32, device="cuda")
    

    def obtain_accumulate_and_maximum(
        hidden: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = hidden.to(torch.float32) @ weight.to(torch.float32).T
        maximum, _ = torch.max(logits, dim=-1, keepdim=False)
        accumulate = torch.sum(torch.exp(logits - maximum.unsqueeze(-1)), dim=-1, keepdim=False)
        return maximum.view(-1), accumulate.view(-1)

    maximum, accumulate = obtain_accumulate_and_maximum(hidden, weight)

    # convert accumulate to LSE
    accumulate = torch.log(accumulate) + maximum

    dHidden = torch.zeros_like(hidden, dtype=torch.float32)
    dWeight = torch.zeros_like(weight, dtype=torch.float32)

    hidden_packed = from_dlpack(
        hidden.view(-1, dim).detach(),
        assumed_align=128
    ).mark_compact_shape_dynamic(mode=0)
    weight_packed = from_dlpack(
        weight.detach(),
        assumed_align=128
    )
    labels_packed = from_dlpack(
        labels.view(-1).detach(),
        assumed_align=8
    ).mark_compact_shape_dynamic(mode=0)

    accumulate_packed = from_dlpack(
        accumulate.detach(),
        assumed_align=4
    ).mark_compact_shape_dynamic(mode=0)
    scalarNumValidTokens_packed = cute.runtime.make_ptr(
        cutlass.Int64,
        num_valid_tokens.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=8
    )

    dlogprobs_packed = from_dlpack(
        dlogprobs.view(-1).detach(),
        assumed_align=4
    ).mark_compact_shape_dynamic(mode=0)

    dHidden_packed = from_dlpack(
        dHidden.view(-1, dim).detach(),
        assumed_align=128
    ).mark_compact_shape_dynamic(mode=0)
    dWeight_packed = from_dlpack(
        dWeight.detach(),
        assumed_align=128
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = BwdDHiddenDWeight(
        reduction=str_to_reduction_enum(reduction),
    )

    kernel_compiled = cute.compile(
        kernel,
        hidden_packed,
        weight_packed,
        labels_packed,
        dlogprobs_packed,
        accumulate_packed,
        scalarNumValidTokens_packed,
        dHidden_packed,
        dWeight_packed,
        ignore_index,
        rank,
        stream,
        # options="--generate-line-info"
    )

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record(stream=torch.cuda.current_stream())
    kernel_compiled(
        hidden_packed,
        weight_packed,
        labels_packed,
        dlogprobs_packed,
        accumulate_packed,
        scalarNumValidTokens_packed,
        dHidden_packed,
        dWeight_packed,
        ignore_index,
        rank,
        stream
    )
    stop.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(stop)
    print(f"[INFO]: Kernel elapsed time: {elapsed_time:.4f} ms")
    
    dHidden = dHidden.type_as(hidden)
    dWeight = dWeight.type_as(weight)

    # def torch_backward(
    #     hidden: torch.Tensor,
    #     weight: torch.Tensor,
    #     labels: torch.Tensor,
    #     dlogprobs: torch.Tensor,
    #     reduction: str,
    #     num_valid_tokens: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     logits = hidden.to(torch.float32) @ weight.to(torch.float32).T
    #     logits_view = logits.view(-1, weight.shape[0])
    #     one_hot = torch.zeros_like(logits_view)
    #     one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)
    #     pd = torch.nn.functional.softmax(logits_view, dim=-1)
    #     d_logits = (pd - one_hot)
    #     if reduction in ["none", "sum"]:
    #         d_logits *= dlogprobs.view(-1).unsqueeze(-1)
    #     elif reduction == "mean":
    #         d_logits *= (dlogprobs.view(-1).unsqueeze(-1) / num_valid_tokens.to(d_logits.dtype))
    #     d_logits = d_logits.to(hidden.dtype)

    #     d_hidden = d_logits @ weight
    #     d_weight = d_logits.T @ hidden.view(-1, dim)
    #     return d_hidden.view(hidden.shape), d_weight.view(weight.shape)

    # start.record(stream=torch.cuda.current_stream())
    # torch_d_hidden, torch_d_weight = torch_backward(hidden, weight, labels, dlogprobs, reduction, num_valid_tokens)
    # stop.record(stream=torch.cuda.current_stream())
    # torch.cuda.synchronize()
    # elapsed_time = start.elapsed_time(stop)
    # print(f"[INFO]: Torch backward elapsed time: {elapsed_time:.4f} ms")
    # # print("torch_d_hidden:\n", torch_d_hidden)
    # # print("kernel_d_hidden:\n", dHidden)
    # torch.testing.assert_close(dHidden, torch_d_hidden)
    # print("[PASSED] dHidden is close to that of PyTorch Operation")

    def torch_native_backward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        dlogprobs: torch.Tensor,
        reduction: str,
        start: torch.cuda.Event,
        stop: torch.cuda.Event,
    ):
        start.record(stream=torch.cuda.current_stream())
        logits = hidden.to(torch.float32) @ weight.to(torch.float32).T
        logits_view = logits.view(-1, weight.shape[0])
        ce = torch.nn.functional.cross_entropy(logits_view, labels.view(-1), reduction=reduction)
        stop.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(stop)
        print(f"[INFO]: Torch native forward elapsed time: {elapsed_time:.4f} ms")

        start.record(stream=torch.cuda.current_stream())
        d_hidden, d_weight = torch.autograd.grad((ce,), (hidden, weight), (dlogprobs.view(ce.shape),), retain_graph=False)
        stop.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(stop)
        print(f"[INFO]: Torch native backward elapsed time: {elapsed_time:.4f} ms")

        return d_hidden.view(hidden.shape), d_weight.view(weight.shape)

    d_hidden_native, d_weight_native = torch_native_backward(hidden, weight, labels, dlogprobs, reduction, start, stop)
    
    print(f"d_hidden_native:\n{d_hidden_native}")
    print(f"dHidden:\n{dHidden}")
    
    
    torch.testing.assert_close(dHidden, d_hidden_native)
    print("[PASSED] dHidden is close to that of PyTorch Native Operation")