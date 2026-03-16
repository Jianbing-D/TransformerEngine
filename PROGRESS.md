# Progress

## Task-2: Use Static Tile Scheduler in fwd_mainloop.py
**Status: Completed**

### PLAN-Task2: Persistent Scheduler for FwdMainLoop

#### Problem
Current `FwdMainLoop` kernel uses a static grid `(ceil_div(num_tokens, 128), num_splits, 1)` — one CTA per `(m_tile, n_tile)` pair. This means the kernel may launch more CTAs than there are SMs, causing wave quantization overhead. A persistent scheduler caps the grid to SM count and has each CTA loop over multiple tiles.

#### Architecture
- **Scheduler**: `StaticPersistentScheduler` in `scheduler.py`
  - Grid = `min(SM_count × occupancy, total_tiles)`, 1D
  - Linearization: `tile_idx = m_idx * num_splits + n_idx`
  - `m_idx = tile_idx // num_splits` → `pidm`
  - `n_idx = tile_idx % num_splits` → `pidn`
  - Per-CTA: loop `while tile_idx < total_blocks`, advance by `grid_dim.x`
- **Key constraint**: `FastDivmodDivisor(num_splits)` requires `num_splits` to be compile-time static (it is — derived from static `vocab_size`). `num_m_tiles` is dynamic (varies with `num_tokens`).

#### What Changes in `fwd_mainloop.py`
1. **Imports**: Add `partial` from functools, add `tile_scheduler` import
2. **`_compute_grid`**: Use `StaticPersistentScheduler.get_grid_shape()` instead of naïve grid
3. **`__call__`**: Pass `sched_params` to kernel
4. **`kernel`**:
   - Remove `pidm, pidn = block_idx()`
   - Hoist TMEM alloc before while loop (allocate once, reuse across tiles)
   - Hoist warpgroup_reg_alloc/dealloc before while loop
   - Hoist epilogue fixed setup (copy atoms, TMEM load layouts) before while loop
   - Add persistent `while work_tile.is_valid_tile:` loop
   - Inside loop: extract `pidm, pidn` from scheduler, recompute GMEM partitions & TMA partitions
   - Move TMEM dealloc after while loop

#### What Stays Fixed Across Tiles (can be hoisted)
- SMEM layouts, `sA`, `sB`
- `thr_mma`, `tCsA`, `tCsB`
- `tCtC` (TMEM tensor layout)
- Copy atoms (t2r, g2r, r2g) and tiled copies
- `tTMEM_load_tAcc`, `tTMEM_load_cAcc_shape`, `cAcc`, `tCcAcc_epi`
- Pipeline objects and initial states

#### What Changes Per Tile (inside while loop)
- `pidm, pidn` from scheduler
- `block_vocab_left/right_idx`, `num_n_tiles`
- `gA`, `gB`, `tCgA`, `tCgB`, `tTMAsA/B`, `tTMAgA/B`
- Epilogue GMEM slices (`gLabels`, `gMax`, `gAccu`, `gLogprobs`)
- Epilogue register fragments (`tR2GrMax.fill(-1e30)`, etc.) — reset each tile
- `tLabelsrLabels`, `valid_mask`

#### TODO-list
- [x] Read and understand current fwd_mainloop kernel structure
- [x] Read and understand StaticPersistentScheduler API
- [x] Write plan to PROGRESS.md
- [x] Add scheduler import and `partial` to fwd_mainloop.py
- [x] Modify `_compute_grid` to use StaticPersistentScheduler
- [x] Modify `__call__` to pass sched_params to kernel
- [x] Modify `kernel` to use persistent while loop
- [x] Run `make unit-test-1gpu` to verify correctness — 75 passed, 89 skipped
- [x] Update KNOWLEDGE.md with scheduler insights
- [x] Mark task-2 completed in TASK.yaml

---

## Task-1: Understand Linear-Cross-Entropy Fusion Code Structure
**Status: Completed**

### PLAN-Task1: Code Exploration & Documentation

#### Architecture Overview
Study and document the complete code structure of the Linear-Cross-Entropy (LCE) fusion kernel in TransformerEngine, focusing on tensor shapes, data types, and algorithm flow.

#### Files to Examine
1. `tests/pytorch/test_linear_cross_entropy.py` — test usage and problem sizes
2. `transformer_engine/pytorch/linear_cross_entropy.py` — public API + autograd Function
3. `transformer_engine/pytorch/cutedsl/linear_cross_entropy_entry.py` — host-side forward/backward orchestration
4. `transformer_engine/common/cutedsl/linear_cross_entropy/` — kernel implementations
   - `blackwell/fwd_mainloop.py` — SM100 forward GEMM + softmax epilogue
   - `blackwell/bwd_partial_dlogits.py` — SM100 backward d_logits kernel
   - `blackwell/bwd_dHdW.py` — SM100 fused d_hidden + d_weight kernel (WIP)
   - `utils.py` — enums
   - `scheduler.py` — persistent tile scheduler
   - `ptx.py` — inline PTX helpers
5. `transformer_engine/common/triton/linear_cross_entropy.py` — Triton epilogue kernels

#### TODO-list
- [x] Read and understand the public API layer
- [x] Read and understand the forward host function and tensor shapes
- [x] Read and understand FwdMainLoop (SM100 GEMM + online softmax)
- [x] Read and understand BwdPartialDlogits (SM100 d_logits kernel)
- [x] Read and understand BwdDHiddenDWeight structure (fused backward WIP)
- [x] Read and understand Triton epilogue kernels
- [x] Read and understand scheduler and ptx helpers
- [x] Write comprehensive KNOWLEDGE.md report
- [x] Update TASK.yaml to mark task-1 as completed
