# Task-4: Optimize the Backward Kernel bwd_partial_dlogits.py
**Status: Completed**

## PLAN-Task4: Apply GEMM Optimizations to BwdPartialDlogits

### Current State (from NCU profiling)
- Duration: 162.43 μs per invocation (called 42× per backward = ~6.8 ms total)
- Grid: (32, 12, 1) = 384 tiles, 256 threads/CTA
- Memory throughput: 50.48%, Compute throughput: 59.68%
- **Occupancy: 12.38%** (limited by 197.63 KB SMEM per block)
- NCU estimates **33.33% speedup** from eliminating partial wave tail effect
- Load imbalance across SMs: 14-22% variance

### Optimization Phases

#### Phase 1: Persistent Scheduler (Low risk, ~33% speedup)
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

#### Phase 2: TMA Store for d_logits (Medium risk)
Replace R2G epilogue store with TMA S2G. Requires:
1. Add `sC` staging buffer to `SharedStorage`
2. Create `PipelineTmaStore` for multi-stage C buffering
3. R2S copy after type conversion, then S2G TMA store
4. Only epilogue warp 0 issues TMA store

**Trade-off:** Adds SMEM pressure (sC buffer) but removes predicated store overhead and improves memory coalescing.

#### Phase 3: 2-CTA MMA Instructions (High risk)
Enable `use_2cta_instrs=True` to double the N-tile (256→512). Requires:
1. `cluster_shape_mn = (2, 1)`, `cta_group = CtaGroup.TWO`
2. Leader CTA only issues MMA
3. Multicast TMA loads
4. Grid M must be multiple of 2

### Files Modified
- `transformer_engine/common/cutedsl/linear_cross_entropy/blackwell/bwd_partial_dlogits.py`

## TODO-list
- [x] Read and understand current bwd_partial_dlogits kernel
- [x] Read and understand CUTLASS GEMM example optimizations
- [x] Write KNOWLEDGE/cutlass_gemm_examples.md
- [x] Write detailed plan
- [x] **Phase 1: Persistent Scheduler**
  - [x] Add scheduler imports to bwd_partial_dlogits.py
  - [x] Modify `_compute_grid` to use `StaticPersistentScheduler`
  - [x] Modify `__call__` to create scheduler params and pass to kernel
  - [x] Modify kernel: add scheduler_params parameter
  - [x] Modify kernel: hoist fixed state before warp specialization
  - [x] Modify kernel: add persistent while loop in load/MMA/epilogue warps
  - [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
  - [x] Run 4-GPU tests — **88 passed, 76 skipped**
  - [x] Run `make ncu-bwd-cli` — **163.55 μs** (vs 162.43 μs baseline)
  - [x] **Result**: Grid reduced from 384 to 152 (1 wave), but per-tile overhead offsets tail wave savings. Net effect: neutral. Foundation for Phase 2/3.
- [x] **Phase 2: TMA Store**
  - [x] Add sC to SharedStorage, create TMA store atom, add PipelineTmaStore
  - [x] Modify epilogue: R2S + S2G instead of R2G
  - [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
  - [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
  - [x] Run `make ncu-bwd-cli` — **153.70 μs** (vs 163.55 μs Phase 1 / 162.43 μs baseline → **5.4% speedup**)
  - [x] **Result**: SMEM 197.63→214.02 KB, Registers 128→117/thread, Compute 59.22%→62.55%, Duration 162.43→153.70 μs
- [x] **Phase 3: 2-CTA MMA**
  - [x] Modify scheduler.py, `__call__`, and kernel for 2-CTA cluster support
  - [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
  - [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
  - [x] Run `make ncu-bwd-cli` — **154.62 μs** (performance neutral vs Phase 2)
  - [x] **Result**: 2-CTA MMA doesn't help — kernel is epilogue-bound, not GEMM-bound. Per-element softmax gradient dominates.
