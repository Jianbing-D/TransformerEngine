# Blackwell (SM100) Kernel Architecture

## CTA (Thread Block) Layout — 8 Warps
All SM100 kernels (FwdMainLoop, BwdPartialDlogits) use the same layout:
- **Warps 0-3** (warpgroup): Epilogue — reads from TMEM, computes softmax / d_logits, writes to GMEM
- **Warp 4**: Loader — issues TMA G2S copies for `hidden` and `weight`
- **Warp 5**: MMA — issues tcgen05 GEMM instructions, reads from SMEM, writes to TMEM
- **Warps 6-7**: Empty — reduce register pressure via `warpgroup_reg_dealloc`

## CTA Layout — 16 Warps (BwdDHiddenDWeight only)
- **Warps 0-3**: Softmax warpgroup (num_regs=192)
- **Warps 4-7**: Epilog warpgroup (num_regs=80)
- **Warp 8**: Load (num_regs=32)
- **Warp 9**: MMA (num_regs=64)
- **Warp 10**: Store (num_regs=32)
- **Warps 11-15**: Empty (num_regs=24)

## Pipeline Stages
- **AB pipeline** (`PipelineTmaUmma`): 4 stages, producer=load warp, consumer=MMA warp
  - TMA copies one tile of A and B per stage (both share the same pipeline/barrier)
- **MMA pipeline** (`PipelineUmmaAsync`): 2 stages (FwdMainLoop) or 1 stage (BwdPartialDlogits)
  - Producer=MMA warp (fills TMEM), consumer=epilogue warpgroup

## TMEM (Tensor Memory) Usage
TMEM is a fast on-chip memory exclusive to SM100, capacity = 512 columns × 128 rows.
- **FwdMainLoop**: `tmem_alloc_cols = num_acc_stage * mma_tiler[1] = 2 * 256 = 512` (full capacity)
- **BwdPartialDlogits**: `tmem_alloc_cols = 1 * 256 = 256`
- **BwdDHiddenDWeight**: uses multiple TMEM regions: logits, dH, dW, p (all together, next power-of-2 rounded)

## TMA (Tensor Memory Accelerator)
- `make_tiled_tma_atom_A/B` creates TMA descriptors for bulk async copy G→S
- `cpasync.CopyBulkTensorTileG2SOp` for loading (G2S)
- `cpasync.CopyReduceBulkTensorTileS2GOp(ADD)` for atomic-add write-back in bwd (S2G with reduction)
- Alignment requirements: 16B minimum (`assumed_align`), 128B preferred for performance

## Key API Patterns

**Kernel compilation and caching:**
```python
# Compile once per (vocab_size, dim, dtype) combination
key = f"vocab_size:{vocab_size}+dim:{dim}+dtype:{hidden_view.dtype}"
if cache.get(key) is None:
    kernel = MyKernel(...)
    compiled = cute.compile(kernel, *sample_tensors, ...)
    cache[key] = compiled
compiled = cache[key]
compiled(*runtime_tensors, ...)  # re-use with different num_tokens
```

**Dynamic shape marking:**
```python
tensor_packed = from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0)
# mode=0: leading dimension (num_tokens) is dynamic; others are static
```

**SMEM struct definition:**
```python
@cute.struct
class SharedStorage:
    load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_stages * 2]
    sA: cute.struct.Align[cute.struct.MemRange[dtype, size], align_bytes]
    sB: cute.struct.Align[cute.struct.MemRange[dtype, size], align_bytes]
```

---

## Scheduler: `StaticPersistentScheduler`

**File:** `transformer_engine/common/cutedsl/linear_cross_entropy/scheduler.py`

A persistent (wave-style) tile scheduler:
- Grid size = min(SM_count × occupancy, total_tiles)
- Each CTA processes one tile, then advances by `grid_dim.x` until all tiles are done
- `WorkTileInfo.tile_idx` is 1D; `divmod` by `num_tiles_N` gives `(m_block, n_block)`
- Used by `BwdDHiddenDWeight` to amortize launch overhead across many vocab tiles
