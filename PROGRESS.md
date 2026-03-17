# Progress

## Task-4: Optimize the Backward Kernel bwd_partial_dlogits.py
**Status: Completed**

### PLAN-Task4: Apply GEMM Optimizations to BwdPartialDlogits

#### Current State (from NCU profiling)
- Duration: 162.43 μs per invocation (called 42× per backward = ~6.8 ms total)
- Grid: (32, 12, 1) = 384 tiles, 256 threads/CTA
- Memory throughput: 50.48%, Compute throughput: 59.68%
- **Occupancy: 12.38%** (limited by 197.63 KB SMEM per block)
- NCU estimates **33.33% speedup** from eliminating partial wave tail effect
- Load imbalance across SMs: 14-22% variance

#### Optimization Phases

##### Phase 1: Persistent Scheduler (Low risk, ~33% speedup)
Apply `StaticPersistentScheduler` to reduce wave quantization. Current grid has 384 tiles on ~132 SMs = ~3 waves with tail. Persistent grid caps at 132 CTAs, each processing ~3 tiles.

**What stays fixed (hoisted before persistent loop):**
- SMEM allocation & tensor creation (sA, sB)
- MMA fragments (tCsA, tCsB)
- TMEM allocation & retrieval
- Pipeline creation & initial states
- `mB_n` (depends on split_idx, constant per kernel call)
- `a_cta_layout`, `b_cta_layout`, `block_in_cluster_coord_vmnk`
- Epilogue copy atoms (t2r, g2r, r2g), tiled copies, thread slices
- `cAcc`, `tCcAcc_epi`, `tTMEM_load_cAcc_shape`, `tTMEM_load_rAcc`
- `dLogits_half` allocation

**What changes per tile (inside while loop):**
- Load warp: `gA`, `gB`, `tCgA`, `tCgB`, `tTMAsA/B`, `tTMAgA/B`
- MMA warp: just `accumulate` reset (SMEM reads are tile-independent)
- Epilogue warp: `gLabels`, `gAccu`, `gDlogits_partial`, all label/mask/pred computation, `block_vocab` bounds

**Pipeline state handling:** States wrap naturally across tiles because both producer and consumer advance the same number of times per tile. K_tiles is constant across all tiles.

##### Phase 2: TMA Store for d_logits (Medium risk)
Replace R2G epilogue store with TMA S2G. Requires:
1. Add `sC` staging buffer to `SharedStorage`
2. Create `PipelineTmaStore` for multi-stage C buffering
3. R2S copy after type conversion, then S2G TMA store
4. Only epilogue warp 0 issues TMA store

**Trade-off:** Adds SMEM pressure (sC buffer) but removes predicated store overhead and improves memory coalescing.

##### Phase 3: 2-CTA MMA Instructions (High risk)
Enable `use_2cta_instrs=True` to double the N-tile (256→512). Requires:
1. `cluster_shape_mn = (2, 1)`, `cta_group = CtaGroup.TWO`
2. Leader CTA only issues MMA
3. Multicast TMA loads
4. Grid M must be multiple of 2

#### Files to Modify
- `transformer_engine/common/cutedsl/linear_cross_entropy/blackwell/bwd_partial_dlogits.py`

#### TODO-list
- [x] Read and understand current bwd_partial_dlogits kernel
- [x] Read and understand CUTLASS GEMM example optimizations
- [x] Write KNOWLEDGE/cutlass_gemm_examples.md
- [x] Write detailed plan to PROGRESS.md
- [x] **Phase 1: Persistent Scheduler**
  - [x] Add scheduler imports to bwd_partial_dlogits.py
  - [x] Modify `_compute_grid` to use `StaticPersistentScheduler`
  - [x] Modify `__call__` to create scheduler params and pass to kernel
  - [x] Modify kernel: add scheduler_params parameter
  - [x] Modify kernel: hoist fixed state before warp specialization
  - [x] Modify kernel: add persistent while loop in load warp
  - [x] Modify kernel: add persistent while loop in MMA warp
  - [x] Modify kernel: add persistent while loop in epilogue warp
  - [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
  - [x] Run 4-GPU tests — **88 passed, 76 skipped**
  - [x] Run `make ncu-bwd-cli` — **163.55 μs** (vs 162.43 μs baseline)
  - [x] **Result**: Grid reduced from 384 to 152 (1 wave), but per-tile overhead (GMEM/TMA partition recomputation) offsets tail wave savings. Net effect: neutral. Persistent scheduler is now in place as foundation for Phase 2/3.
- [x] **Phase 2: TMA Store**
  - [x] Add sC to SharedStorage
  - [x] Create TMA store atom for dlogits_partial
  - [x] Add PipelineTmaStore pipeline
  - [x] Modify epilogue: R2S + S2G instead of R2G
  - [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
  - [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
  - [x] Run `make ncu-bwd-cli` — **153.70 μs** (vs 163.55 μs Phase 1 / 162.43 μs baseline → **5.4% speedup**)
  - [x] **Result**: TMA S2G store replaces predicated R2G stores. Key changes:
    - SMEM: 197.63 → 214.02 KB (+sC staging buffer, still fits 1 block/SM)
    - Registers: 128 → 117/thread (removed R2G predicate registers)
    - Compute throughput: 59.22% → 62.55% (now compute-dominant)
    - Duration: 162.43 → 153.70 μs (**5.4% improvement**)
    - Key insight: `tma_partition` requires TMA tensor (`mC`), not raw GMEM tensor
  - [x] Update PROGRESS.md with results
- [x] **Phase 3: 2-CTA MMA**
  - [x] Modify scheduler.py: add `cluster_m_size` param to `create`, `get_grid_shape`, `advance_to_next_work`
  - [x] Modify `__call__`: use `cluster_shape_to_tma_atom_A/B`, multiply tx_count by atom_thr_size, pass cluster_m_size to scheduler
  - [x] Modify kernel: CTA rank via `block_idx_in_cluster()`, `mma_tile_coord_v`, `is_leader_cta`
  - [x] Modify kernel: `thr_mma.get_slice(mma_tile_coord_v)` instead of `get_slice(0)`
  - [x] Modify kernel: add `cta_layout_vmnk` to pipelines, add `pipeline_init_arrive/wait`
  - [x] Modify kernel: create multicast masks, pass `mcast_mask` to TMA copy
  - [x] Modify kernel: gate MMA warp with `is_leader_cta`
  - [x] Modify kernel: non-leader MMA warp idles (no pipeline ops)
  - [x] Modify `__call__` entry point: pass `mma_tiler_mn=(256, 256)` for 2-CTA mode
  - [x] Fix: double `PipelineUmmaAsync` consumer group for 2-CTA (both CTAs' epilogue warps participate)
  - [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
  - [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
  - [x] Run `make ncu-bwd-cli` — **154.62 μs** (vs 153.70 μs Phase 2 / 162.43 μs baseline)
  - [x] **Result**: 2-CTA MMA with cluster(2,1) and mma_tiler_mn=(256,256). Key changes:
    - Grid: 304 CTAs in 152 clusters (was 152 1-CTA blocks in Phase 2)
    - SMEM: 148.48 KB/CTA (down from 214.02 KB — smaller per-CTA buffers)
    - Registers: 119/thread (from 117)
    - Cluster size: 2, Cluster Scheduling: PolicySpread
    - Compute throughput: 60.72% (from 62.55%)
    - Duration: 154.62 μs (**performance neutral** vs Phase 2)
    - Key insight: 2-CTA MMA doesn't help this kernel because it's epilogue-bound, not GEMM-bound. The per-element softmax gradient computation dominates. 2-CTA reduces the number of MMA instructions but doesn't reduce epilogue work.
    - Key bug fix: `mma_tiler_mn` must be (256, 256) for 2-CTA (not default 128,256). With (128,256) each CTA only handles 64 M rows — wrong tile size causes incorrect results.
    - Key bug fix: `PipelineUmmaAsync` consumer group must double for 2-CTA since both CTAs' epilogue warpgroups call `consumer_release`.
  - [x] Update PROGRESS.md with results

---

## Task-3: Brain Storm Better Implementations for LCE Fusion
**Status: Completed**

### PLAN-Task3: Mathematical Analysis and Implementation Brainstorm

#### Goal
Perform a rigorous analysis of the LCE forward/backward algorithm, quantify current bottlenecks using arithmetic-intensity reasoning, and enumerate concrete improvements with their trade-offs.

#### Approach
1. Derive FLOP and bandwidth counts for the reference configuration (DeepSeek: T=4096, V=129280, D=7168, BF16)
2. Classify each bottleneck (compute-bound vs. bandwidth-bound)
3. Enumerate alternative forward strategies
4. Enumerate alternative backward strategies
5. Analyze data-type precision trade-offs
6. Document all findings in KNOWLEDGE.md Section 15

#### TODO-list
- [x] Read and understand `linear_cross_entropy_entry.py` (full forward + backward)
- [x] Read and understand `bwd_partial_dlogits.py` (current backward kernel)
- [x] Read and understand `bwd_dHdW.py` (WIP fused backward)
- [x] Read and understand Triton epilogue kernels
- [x] Derive FLOP and bandwidth counts for reference config
- [x] Identify compute bottleneck vs bandwidth bottleneck
- [x] Brainstorm forward improvements (token-centric, FP8, etc.)
- [x] Brainstorm backward improvements (fused, persistent, etc.)
- [x] Analyze precision trade-offs
- [x] Write comprehensive analysis to KNOWLEDGE.md Section 15
- [x] Mark task-3 completed in TASK.yaml

---

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
