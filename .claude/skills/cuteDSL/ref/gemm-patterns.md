# CuTeDSL GEMM Patterns by Architecture

## Ampere (SM80) — SIMT GEMM

**Reference example**: `ampere/sgemm.py`

```python
class SGemm:
    def __init__(self, cta_tiler=(128, 128, 8), num_stages=3, num_threads=256):
        self.cta_tiler = cta_tiler
        self.num_stages = num_stages
        self.num_threads = num_threads

    @cute.jit
    def __call__(self, mA, mB, mC, M, N, K, stream=None):
        bM, bN, bK = self.cta_tiler
        # Grid: ceil(M/bM) x ceil(N/bN)
        grid = ((M + bM - 1) // bM, (N + bN - 1) // bN, 1)

        # Smem layouts for A and B (with multi-stage buffering)
        sA_layout = cute.make_layout((bM, bK, self.num_stages))
        sB_layout = cute.make_layout((bN, bK, self.num_stages))

        self.kernel(mA, mB, mC, sA_layout, sB_layout, M, N, K).launch(
            grid=grid, block=(self.num_threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, mA, mB, mC, sA_layout, sB_layout, M, N, K):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # 1. Allocate shared memory
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(mA.element_type, sA_layout)
        sB = alloc.allocate_tensor(mB.element_type, sB_layout)

        # 2. Tile global tensors: (bM, bK, k) and (bN, bK, k)
        gA = cute.local_tile(mA, tiler=(bM, bN, bK), coord=(bidx, bidy, 0), proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=(bM, bN, bK), coord=(bidx, bidy, 0), proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=(bM, bN, bK), coord=(bidx, bidy, 0), proj=(1, 1, None))

        # 3. Setup MMA
        mma_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        tiled_mma = cute.make_tiled_mma(mma_op, atom_layout, permutation_mnk=(...))
        thr_mma = tiled_mma.get_slice(tidx)

        # 4. Setup async copy (G->S)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mA.element_type,
            num_bits_per_copy=mA.element_type.width * vector_size
        )

        # 5. Multi-stage pipeline mainloop
        for stage in range(num_stages - 1):
            cute.copy(tiled_copy_a, gA[..., stage], sA[..., stage])
            cute.copy(tiled_copy_b, gB[..., stage], sB[..., stage])
            cute.arch.cp_async_commit_group()

        tCrC.fill(0.0)
        for k in range(K_tiles):
            cute.arch.cp_async_wait_group(num_stages - 2)
            cute.arch.sync_threads()
            cute.gemm(tiled_mma, tCrC, tCrA[..., stage], tCrB[..., stage], tCrC)
            # Load next stage
            cute.copy(tiled_copy_a, gA[..., next_k], sA[..., next_stage])
            cute.arch.cp_async_commit_group()

        # 6. Epilogue: store C
        cute.copy(copy_atom_store, tCrC, tCgC)
```

## Ampere (SM80) — Tensor Core GEMM (FP16)

**Reference example**: `ampere/tensorop_gemm.py`

Key differences from SIMT:
- Uses `cute.nvgpu.MmaAtom` with FP16 MMA shapes (e.g., m16n8k16)
- `tiled_mma.make_fragment_A/B/C()` for operand fragments
- Higher throughput via Tensor Cores

```python
mma_atom = cute.make_mma_atom(cute.nvgpu.MmaAtom.from_SM80())
tiled_mma = cute.make_tiled_mma(mma_atom, layout_mnk, permutation_mnk)
tCrA = tiled_mma.make_fragment_A(tCsA[..., 0])
tCrB = tiled_mma.make_fragment_B(tCsB[..., 0])
tCrC = tiled_mma.make_fragment_C(tCgC)
```

## Hopper (SM90) — TMA + WGMMA

**Reference example**: `hopper/dense_gemm.py`, `hopper/dense_gemm_persistent.py`

Key features:
- **TMA** (Tensor Memory Accelerator) for global->smem loads — offloads address computation
- **WGMMA** (Warpgroup MMA) — 4-warp cooperative MMA
- **Cluster launch** — distributed shared memory across CTAs
- **Persistent kernels** with tile schedulers

```python
from cutlass.cute.nvgpu import cpasync
import cutlass.utils.hopper_helpers as sm90_utils

# TMA atom setup
tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    a_op, a_tensor, a_smem_layout, mma_tiler, tiled_mma, cluster_layout.shape
)

# Prefetch TMA descriptors
cpasync.prefetch_descriptor(tma_atom_a)

# Pipeline for async TMA loads
pipe = pipeline.PipelineTmaAsync(num_stages, ...)

# WGMMA-based compute
tiled_mma = sm90_utils.make_tiled_mma(...)
cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)

# Persistent kernel with tile scheduler
scheduler = cutlass.utils.DynamicPersistentTileScheduler(M, N, tile_m, tile_n)
while scheduler.has_work():
    tile_idx = scheduler.get_current_work()
    # ... compute tile ...
    scheduler.advance()
```

## Blackwell (SM100) — TMA + tcgen05 + TMEM

**Reference example**: `blackwell/dense_gemm.py`, `blackwell/dense_gemm_persistent.py`

Key features:
- **tcgen05.mma** — next-gen MMA with tensor memory (TMEM)
- **2-CTA instructions** — two CTAs cooperate on one MMA
- **TMEM** — dedicated tensor memory for accumulators
- **Block-scaled GEMM** — native FP8/FP4 with per-block scaling

```python
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05

# MMA setup
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a_dtype, a_major_mode, b_major_mode, acc_dtype, cta_group, mma_tiler[:2]
)

# TMEM allocation for accumulators
tmem_alloc = cutlass.utils.TmemAllocator()
acc = tmem_alloc.allocate(...)

# tcgen05-based compute
cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)

# Pipeline
pipe = pipeline.PipelineUmmaAsync(num_stages, ...)

# Block-scaled GEMM (FP8 with scaling factors)
cute.gemm(tiled_mma, acc, [tCrA, scale_A], [tCrB, scale_B], acc)
```

## Rubin (SM107) — Extends Blackwell with B-reuse + Lamport

**Reference example**: `rubin/dense_gemm_persistent.py` (internal)

Key features:
- Inherits all Blackwell patterns; Rubin kernels extend Blackwell kernel classes
- **B-keep/B-reuse** — reuses B matrix across two MMA ops, reducing SMEM traffic
- **LDTM.SPARSIFY** — hardware 2:4 sparsification in TMEM
- **Lamport sync** — data-based producer-consumer synchronization
- **K=64 MMA** — supports MMA K dimension of 32 or 64 (SM100 only 32)
- **328 KiB SMEM**, 576 TMEM columns (288 KiB)

```python
import cutlass.utils.rubin_helpers as sm107_utils

# Standard MMA (same as Blackwell but with sm107 helper)
tiled_mma = sm107_utils.make_trivial_tiled_mma(
    a_dtype, b_dtype, a_major_mode, b_major_mode, acc_dtype,
    cta_group, mma_inst_shape,  # inst_shape is 3D (M, N, K) — K can be 32 or 64
)

# B-keep/B-reuse pattern (when mma_tiler[0] // mma_inst_shape[0] == 2):
tiled_mma_bkeep = sm107_utils.make_trivial_tiled_mma(
    ..., b_collector_op=tcgen05.CollectorOp.FILL)      # keep B in collector
tiled_mma_breuse = sm107_utils.make_trivial_tiled_mma(
    ..., b_collector_op=tcgen05.CollectorOp.LASTUSE)    # reuse and release B

# Block-scaled
sm107_utils.make_blockscaled_trivial_tiled_mma(
    a_dtype, b_dtype, a_major_mode, b_major_mode, acc_dtype,
    sf_dtype, sf_vec_size, cta_group, mma_inst_shape, ...
)

# Sparse
from cutlass.utils.sm107 import make_sparse_trivial_tiled_mma
tiled_mma = make_sparse_trivial_tiled_mma(
    a_raw_dtype, a_major_mode, b_major_mode, acc_dtype, cta_group, mma_tiler
)
```

## Feynman (SM140) — SPMEM + tcgen06 (pre-release)

**Reference example**: `internal/feynman/sm140_dense_gemm.py` (internal)

Radical departure — SPMEM replaces SMEM, tcgen06 replaces tcgen05:
```
GMEM -> SPMEM (4MB, via DPC TMA) -> TMEM (via DLCCP) -> RMEM (via tcgen06.Ld)
```

Key features:
- **SPMEM** (Scratchpad Memory) — 4 MiB on-chip buffer with layout tags
- **tcgen06** MMA operations (new tensor core generation)
- **DPC (Data Path Controller)** — 3D cluster model `dpc_shape_mnk`
- **Temporal split-K** — tiles K across subtiles, reuses A across N-blocks
- **Global split-K** — partitions K across DPC groups

```python
from cutlass.cute.nvgpu import tcgen06
from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileG2SpOp  # GMEM -> SPMEM

# SPMEM layout tags
spmem_iter = tcgen06.make_spmem_iter(ptr, tcgen06.SP_LAYOUT_4x256B)

# DPC shape (replaces cluster shape)
dpc_shape_mnk = (8, 1, 1)  # or (4, 2, 1)
```

## Architecture selection guide

| Scenario | Recommended | Why |
|----------|------------|-----|
| FP32 GEMM, any GPU | Ampere SIMT | Widest compatibility |
| FP16/BF16 GEMM, SM80+ | Ampere Tensor Core | Good throughput, simple code |
| FP16/BF16 GEMM, SM90 | Hopper TMA+WGMMA | Best throughput on Hopper |
| FP8 GEMM, SM90 | Hopper with FP8 atoms | Native FP8 support |
| Any GEMM, SM100 | Blackwell tcgen05 | Highest available throughput |
| FP8/FP4 block-scaled, SM100 | Blackwell | Native block-scaled support |
| FP8 GEMM, SM107 | Rubin tcgen05 + B-reuse | Higher throughput via B-reuse |
| Block-scaled + LUT-B, SM107 | Rubin LUT-B GEMM | LUT-encoded B matrix |
| Sparse GEMM, SM107 | Rubin LDTM.SPARSIFY | Hardware 2:4 sparsification |
| Any GEMM, SM140 | Feynman tcgen06 + SPMEM | Next-gen memory hierarchy |
| Large batch GEMM | Persistent kernel | Amortizes launch overhead |
| MoE / variable sizes | Grouped GEMM | Batches different sizes |
| Dual output (D + Aux) | EFC epilogue fusion | Fused activation + dual store |
| Conv (fprop) | Implicit GEMM | TMA im2col, maps conv to GEMM |
| Low-latency (small N) | TGV GEMM | Small tiles, PDL overlap |
