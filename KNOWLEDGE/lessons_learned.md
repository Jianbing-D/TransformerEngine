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

---

## Task-4 Optimization Insights (BwdPartialDlogits)

### Optimization Phase Results

| Phase | Duration | SMEM | Registers | Key Change |
|-------|----------|------|-----------|------------|
| Baseline (naive grid) | 162.43 μs | 197.63 KB | 128/thr | Static grid (32,12,1)=384 tiles |
| Phase 1: Persistent Scheduler | 163.55 μs | 197.63 KB | 128/thr | Grid→152 (1 wave), but per-tile GMEM recomputation offsets tail wave savings |
| Phase 2: TMA S2G Store | 153.70 μs | 214.02 KB | 117/thr | R2G→R2S+S2G, removes predicate registers, **5.4% speedup** |
| Phase 3: 2-CTA MMA | 154.62 μs | 148.48 KB | 119/thr | Cluster(2,1), mma_tiler=(256,256), performance neutral |

### L6: Persistent Scheduler Overhead Can Offset Tail-Wave Savings
When tile count is close to SM count (384 tiles on 132 SMs = ~3 waves), the persistent scheduler reduces wave tail (384→152 CTAs, 1 wave). However, per-tile GMEM/TMA partition recomputation inside the persistent loop adds overhead that can completely offset the savings. Net effect: **neutral**. Persistent scheduler is most beneficial when wave quantization is severe (many more tiles than SMs).

### L7: TMA S2G Store Reduces Register Pressure
Replacing predicated R2G stores with TMA S2G (via R2S→fence→barrier→S2G pattern) saved 11 registers/thread (128→117) by eliminating store predicate computation. The 5.4% speedup came from removing per-element GMEM store predication, not from TMA bandwidth advantage. **Key gotcha**: `tma_partition()` requires the TMA tensor from `make_tiled_tma_atom`, not the raw GMEM tensor.

### L8: 2-CTA MMA Doesn't Help Epilogue-Bound Kernels
2-CTA MMA reduces the number of MMA instructions (leader CTA only issues GEMM; both CTAs handle epilogue). For bwd_partial_dlogits, the epilogue (per-element softmax gradient: `exp`, subtract, multiply, mask) dominates runtime, not the GEMM. 2-CTA halves the MMA instruction overhead but doesn't reduce epilogue work — each CTA still processes its share of output elements. **Rule of thumb**: 2-CTA MMA only helps when the kernel is compute/MMA-bound, not when epilogue or memory operations dominate.

### L9: 2-CTA Pipeline Consumer Group Sizing
When enabling 2-CTA MMA, the `PipelineUmmaAsync` (accumulator pipeline) consumer group must account for **both** CTAs' epilogue warpgroups. If the consumer count only reflects one CTA, the `consumer_release` barrier fires early (after one CTA's threads arrive), allowing the MMA warp to overwrite TMEM while the other CTA is still reading. This causes silent data corruption, not a hang.

```python
# WRONG: only counts one CTA's epilogue threads
consumer_group = make_thread_cooperative_group(threads_per_warp * len(epi_warp_ids))

# CORRECT: counts both CTAs' epilogue threads
consumer_threads = threads_per_warp * len(epi_warp_ids) * (2 if use_2cta_instrs else 1)
consumer_group = make_thread_cooperative_group(consumer_threads)
```

### L10: mma_tiler_mn Must Scale with 2-CTA
With `use_2cta_instrs=True`, `cta_tile_shape_mnk[0] = mma_tiler_M // atom_thr_size`. To maintain 128 M-rows per CTA (same as 1-CTA baseline), `mma_tiler_mn` must be `(256, N)` not `(128, N)`. Using (128, N) gives only 64 M-rows per CTA, which causes incorrect results due to tile size mismatches in epilogue partitioning and TMA store addressing.

### L11: NCU Profiling with Clusters
When profiling 2-CTA kernels with NCU, the grid size doubles (`grid = tiles * cluster_m_size`). The `--launch-skip` count in the Makefile target must be recalibrated. Key NCU metrics for 2-CTA: look at "Cluster Size", "Cluster Scheduling Policy" (should be "PolicySpread"), and "Max Active Clusters" to understand SM utilization.

---

## Task-5 Bug Fix Insights

### L12: PipelineAsync Producer Phase Must Be 1 (Not 0)
`PipelineAsync.producer_acquire` calls `sync_object_empty.wait(state.index, state.phase)` which issues `mbarrier.try_wait.parity(bar, phase)`. The PTX semantics are: wait until `current_phase != phase`. Empty barriers start at phase=0. With `make_pipeline_state(Producer, ...)`, phase=1, so `0 != 1` is true → producer proceeds immediately. Manual `PipelineState(..., phase=0)` causes `0 != 0` = false → **deadlock**. Always use `make_pipeline_state` instead of manual `PipelineState` construction.

### L13: partition_C with 2-CTA Already Accounts for CTA Rank
`thr_mma = tiled_mma.get_slice(mma_tile_coord_v)` creates a CTA-specific view. `thr_mma.partition_C(mC)` produces tiles indexed by the scheduler's M tile index (`pidm`), not the global CTA-aware index (`pidm * cluster_m_size + mma_tile_coord_v`). Using `pidm_cta` to index `bSG_gC_all` causes out-of-bounds access when `pidm >= num_m_tiles / cluster_m_size`. The global M offset is already baked into the partition via `mma_tile_coord_v`. **Rule**: index TMA-partitioned GMEM tensors derived from `partition_C` with `pidm`, not `pidm_cta`. Use `pidm_cta` only for raw GMEM tensors (labels, accu) that are indexed by flat token position.

---

## Task-6 Scheduler Fix

### L14: Persistent Scheduler Must Account for Cluster Size in Grid Calculation
`get_grid_shape` computes `vacancies = sm_count * occupancy` as the max concurrent CTAs. With clusters of size K, each cluster occupies K SMs. The correct formula is `vacancies = (sm_count // cluster_m_size) * occupancy` — the number of clusters that fit, not the number of CTAs. Without this, the grid is K× too large, causing K waves instead of 1. On B200 (152 SMs) with cluster_size=2: grid went from 304→152 CTAs (152→76 clusters), duration from 154.62→136.86 μs (11.5% speedup).

---

## Testing Practices

### L15: Run GPU Unit Tests Sequentially, Never in Parallel
Multiple GPU test suites (e.g., `make unit-test-1gpu` and entropy tests) must NOT run concurrently. They contend on GPU memory and compute, causing OOM failures or incorrect results in both. Always wait for one test run to finish before starting the next.

### L16: Autograd Gradients May Be Non-Contiguous
When a custom `torch.autograd.Function` returns multiple outputs (e.g., `(logprobs, entropy)`), the gradient tensors received in `backward()` may be non-contiguous — for example, `dentropy` from `.sum()` backward produces an expanded/strided tensor. Always call `.contiguous()` on incoming gradient tensors before passing them to CUDA kernels that require contiguous memory.

### L17: Non-Scalar Loss Requires `.sum()` Before `.backward()`
When `reduction="none"`, the loss tensor (e.g., `logprobs`) has shape `(num_tokens,)`. Calling `.backward()` on a non-scalar tensor requires an explicit `gradient` argument, or the tensor must be reduced to a scalar first (e.g., `logprobs.sum() + entropy.sum()`). This applies in test code combining CE loss and entropy for backward verification.
