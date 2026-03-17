# Lessons Learned and Task-2 Insights

## Core Lessons

### L1: `accumulate` is LSE, not raw sum
The tensor named `accumulate` (and `_accu`) that is saved for backward is converted to LSE (`log(sum_exp) + max`) **by the Triton epilogue** before being stored. The backward kernel `BwdPartialDlogits` directly uses it as `exp(logit - LSE)` for the softmax computation. Any future kernel must respect this convention.

### L2: `_logprobs` in TP mode stores the label logit, not the cross entropy
In TP forward, `_logprobs[token]` accumulates `logit_at_label` across ranks via `all_reduce(SUM)`. The actual NLL is computed as `LSE - logit_at_label` in the update_logprobs Triton kernel. This two-step design avoids a second pass over the vocab.

### L3: Kernel JIT specialization on vocab_size and dim
The cuteDSL JIT specializes on `(vocab_size, dim, dtype)` — these become compile-time constants. Only `num_tokens` varies at runtime. This means the first call for a new model shape will incur JIT compilation latency.

### L4: `BwdDHiddenDWeight` is WIP and not connected
`bwd_dHdW.py` contains a more advanced fused backward kernel, but the entry point (`linear_cross_entropy_entry.py`) still uses `kDlogitsSplitN` (the split-N approach with separate cuBLAS GEMM calls). The fused path would eliminate the temporary `_d_logits_partial` tensor.

### L5: Two-stream pattern for TP forward
The TP forward uses a dedicated CUDA stream to overlap `all_reduce(_logprobs)` with the `forward_tp_epilogue` kernel that processes max/accumulate. Events synchronize the two streams before `forward_tp_epilogue_update_logprobs`.

---

## Static Persistent Scheduler (Task-2 Insights)

### Overview

`StaticPersistentScheduler` in `scheduler.py` replaces the naive `(num_m_tiles, num_splits, 1)` grid with a 1D grid capped at `min(SM_count × occupancy, total_tiles)`. Each CTA loops over multiple tiles until all tiles are exhausted.

### API Summary

```python
# Host side (inside @cute.jit)
sched_params_init = TileSchedulerParams(
    num_tiles_M=Int32(num_m_tiles),
    num_tiles_N=Int32(num_splits),
)
sched_params = StaticPersistentScheduler.to_underlying_arguments(sched_params_init)
grid = StaticPersistentScheduler.get_grid_shape(sched_params, occupancy=1)
# Pass sched_params to kernel as scheduler_params: tile_scheduler.ParamsBase

# Device side (inside @cute.kernel)
TileSchedulerCls = partial(StaticPersistentScheduler.create, scheduler_params)
scheduler = TileSchedulerCls()
work_tile = scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    m_idx, n_idx = work_tile.tile_idx
    ...
    scheduler.advance_to_next_work()
    work_tile = scheduler.get_current_work()
```

### Linearization

Tiles are linearized as `tile_idx = m_idx * num_splits + n_idx`. Decomposition uses `FastDivmodDivisor(num_splits)` for efficient integer division: `(m_idx, n_idx) = divmod(tile_idx, num_splits_divmod)`. `num_splits` must be compile-time static (it is — derived from `vocab_size` which is JIT-specialized).

### CuteDSL Scoping Rule for Multi-Warp Persistent Kernels

**Critical constraint**: CuteDSL (MLIR-based) generates a separate IR region for each `if warp_idx == X:` or `if warp_idx in (...)` block. Variables defined inside one conditional block are NOT visible in any other conditional block, even a sibling.

**Pattern**: Each warp group must contain its entire work (both fixed setup AND the persistent while loop) within a **single** `if warp_idx == X:` block. Never split setup and loop body into two separate conditional blocks.

```python
# WRONG: setup in first block, loop in second block
if warp_idx in self.epi_warp_ids:
    thr_copy = ...  # defined here

if warp_idx in self.epi_warp_ids:  # NEW MLIR REGION: thr_copy not visible!
    while work.is_valid_tile:
        cute.copy(thr_copy, ...)  # DSLRuntimeError: name 'thr_copy' not defined

# CORRECT: everything in one block
if warp_idx in self.epi_warp_ids:
    thr_copy = ...  # setup
    while work.is_valid_tile:
        cute.copy(thr_copy, ...)  # visible
```

### TMEM Allocation in Persistent Kernels

TMEM is allocated once before all warp-group blocks and reused across all tiles. A `NamedBarrier` synchronizes all warps before TMEM retrieval, and another barrier at the end ensures all warp-group loops complete before TMEM deallocation.

```python
# Allocate once (before per-warp blocks)
if warp_idx == empty_warp_ids[0]:
    cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf, ...)
cta_sync_barrier.arrive_and_wait()
tmem_ptr = cute.arch.retrieve_tmem_ptr(...)

# ... all per-warp persistent loops ...

# Deallocate once (after all loops)
cta_sync_barrier.arrive_and_wait()
if warp_idx == empty_warp_ids[0]:
    cute.arch.relinquish_tmem_alloc_permit()
    cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, ...)
```

### Shared vs Per-Tile State

Variables that are compile-time constants or depend only on the kernel configuration can be computed once in shared scope before all warp-group blocks:
- `tCsA`, `tCsB`, `tCtC` (SMEM/TMEM tensor partitions)
- `num_k_tiles` (static: `ceil_div(dim, mma_tiler_k)`, `dim` is JIT-specialized)
- `TileSchedulerCls` (factory: `partial(StaticPersistentScheduler.create, scheduler_params)`)

Variables that depend on `(pidm, pidn)` are recomputed per tile inside each warp group's while loop:
- `gA`, `gB`, TMA partitions (load warp)
- `num_n_tiles` from `pidn` (all warp groups)
- Register fragments (`tR2GrMax.fill(-1e30)`, etc.) reset each tile (epilogue warpgroup)
