# CuTeDSL Tutorial Progressions

## Blackwell FP16 GEMM Tutorial (fp16_gemm_0 through fp16_gemm_6)

Progressive optimization of a Blackwell GEMM kernel. Each step adds one optimization.

| Step | File | What it adds | Key new APIs |
|------|------|-------------|-------------|
| 0 | `fp16_gemm_0.py` | Baseline single-CTA GEMM | `tcgen05.MmaF16BF16Op`, `TmemAllocator`, `PipelineTmaUmma`, `PipelineUmmaAsync` |
| 1 | `fp16_gemm_1.py` | 2CTA MMA + TMA multicast | `CtaGroup.TWO`, `CopyBulkTensorTileG2SMulticastOp`, VMNK layout, `create_tma_multicast_mask` |
| 2 | `fp16_gemm_2.py` | Warp specialization + TMA store | Dedicated TMA/MMA/epilogue warps, `PipelineTmaStore`, `NamedBarrier`, `fence_view_async_shared` |
| 3 | `fp16_gemm_3.py` | Static persistent tile scheduler | `StaticPersistentTileScheduler`, `pipeline_init_arrive/wait`, work tile loop |
| 3.1 | `fp16_gemm_3_1.py` | CLC dynamic scheduler | `ClcDynamicPersistentTileScheduler`, `PipelineClcFetchAsync`, dedicated scheduler warp |
| 4 | `fp16_gemm_4.py` | Preferred/fallback clusters | Two cluster shapes, runtime selection based on SM availability |
| 5 | `fp16_gemm_5.py` | TMA prefetch (L2 priming) | `cute.prefetch(tma_atom, gmem_slice)`, rolling prefetch ahead of loads |
| 6 | `fp16_gemm_6.py` | PDL (kernel overlap) | `griddepcontrol_launch_dependents/wait`, `use_pdl=True`, CUDA graph capture |

### Step 0 key structure (baseline to understand)
```python
@cute.kernel
def kernel(mA, mB, mC, tma_a, tma_b, ...):
    # 1. Allocate TMEM for accumulator
    tmem = TmemAllocator()
    acc = tmem.allocate(512)
    tmem.wait_for_alloc()

    # 2. Create pipelines
    ab_pipe = PipelineTmaUmma(num_stages, ...)
    acc_pipe = PipelineUmmaAsync(acc_stages, ...)

    # 3. Mainloop (warp 0 does everything in baseline)
    if warp_idx == 0:
        # Prologue: fill pipeline
        for s in range(num_stages - 1):
            handle = ab_pipe.producer_acquire_and_advance()
            cute.copy(tma_a, src_a[..., s], smem_a[..., handle.index])
            cute.copy(tma_b, src_b[..., s], smem_b[..., handle.index])
            handle.release()

        # Mainloop: load + compute
        for k in range(K_tiles):
            ab_pipe.consumer_wait_and_advance()
            tiled_mma.set(ACCUMULATE, k > 0)
            cute.gemm(tiled_mma, acc, smem_a_part, smem_b_part, acc)
            ab_pipe.consumer_release()
            # Load next
            handle = ab_pipe.producer_acquire_and_advance()
            cute.copy(tma_a, ...)
            handle.release()

    # 4. Epilogue: TMEM -> RMEM -> GMEM
    acc_pipe.consumer_wait_and_advance()
    tmem_copy = tcgen05.make_tmem_copy(Ld32x32bOp(Repetition.x64), acc)
    cute.copy(tmem_copy, acc, rmem_frag)
    # Convert fp32 -> fp16, then autovec_copy to GMEM
```

## NVFP4 Block-Scaled GEMM Tutorial (nvfp4_gemm_0 through nvfp4_gemm_5)

| Step | What it adds | Key insight |
|------|-------------|-------------|
| 0 | Baseline NVFP4 GEMM | `MmaMXF4NVF4Op`, scale factors in TMEM pool, `S2T copy` (SMEM->TMEM) |
| 1 | 2CTA + multicast | Same as FP16 step 1, applied to FP4 |
| 2 | Warp spec + TMEM early release | Release TMEM after epilogue reads — allows overlap with next MMA |
| 3 | TMEM ping-pong | Two overlapping TMEM buffers; reverse iteration order maximizes reuse |
| 4 | TMA store epilogue | RMEM->SMEM->GMEM via TMA store |
| 5 | 256-bit vectorized store | Avoids TMA load/store pipeline interference from ping-pong |

### TMEM ping-pong key insight (step 3)
```
Buffer 0: TMEM cols [0, 255]    — epilogue processes subtiles LAST to FIRST
Buffer 1: TMEM cols [208, 463]  — epilogue processes subtiles FIRST to LAST
SFA/SFB:  TMEM cols [464, 511]  — fixed, never overlaps

Reverse iteration maximizes the gap between when a TMEM column is freed
and when it's reused by the other buffer's MMA.
```

### 256-bit store insight (step 5)
TMA load and TMA store share the same instruction pipeline. When using TMEM ping-pong,
this creates contention. 256-bit `st.global` uses a different pipeline, avoiding stalls.
Requires 32-byte alignment of output tensor.

## TMA Tutorial (tma_v0 through tma_v2)

| Step | What it demonstrates |
|------|---------------------|
| v0 | TMA load/store basics: `tma_partition`, manual mbarrier, GMEM->SMEM->GMEM copy |
| v1 | Matrix transpose: TMA load (row-major) -> register -> TMA store (col-major), swizzled SMEM |
| v2 | Multi-stage pipeline transpose with persistence: `PipelineTmaAsync`, persistent tile scheduler |

### tma_partition explained (v0)
```python
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileG2SOp(), tensor, smem_layout, cta_tiler
)
# tma_tensor has shape: ((tile_modes), (rest_modes))
# Index with: tma_tensor[(None, bidx, bidy)] to select per-CTA tile
```

## Block API Tutorial (higher-level abstraction)

| Step | What it adds |
|------|-------------|
| fp16_gemm_0 | Simplest Blackwell GEMM using block-level utilities |
| fp16_gemm_1 | 2CTA MMA + TMA multicast via `tma_multicast={"cluster_shape":..., "multicast_dim":"N"}` |
| fp16_gemm_2 | Full warp specialization + TMA store + `NamedBarrier` |

Block API simplifies common patterns:
```python
# Instead of manual TMA atom creation:
utils.block_copy(tma_atom, src, dst, group_modes=...)
# Instead of manual TMEM management:
pool = TmemBufferPool(cols)
acc = pool.allocate_tensor(layout, dtype)
```
