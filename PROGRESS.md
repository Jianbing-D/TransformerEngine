# Progress

## Task-5: Fix Bugs in bwd_partial_dlogits.py
**Status: Completed**

### PLAN-Task5: Fix Store Warp Bugs

#### Bugs Identified & Fixed
1. **PipelineAsync producer phase (DEADLOCK)**: Manual `PipelineState(num_write_stage, 0, 0, 0)` used phase=0 for the store_c producer. PipelineAsync's `producer_acquire` calls `sync_object_empty.wait(state.index, state.phase)` which uses `mbarrier.try_wait.parity` — waits until `current_phase != phase`. Empty barriers start at phase=0. With phase=0, producer blocks forever (0 != 0 is false). Fix: use `make_pipeline_state(PipelineUserType.Producer, ...)` which sets phase=1.
2. **bSG_gC_all M index out of bounds (DATA CORRUPTION)**: Store warp indexed `bSG_gC_all[(None, pidm_cta, n_subtile_global)]` where `pidm_cta = pidm * cluster_m_size + mma_tile_coord_v`. But `bSG_gC_all` is derived from `thr_mma.partition_C(mC)` which already accounts for the 2-CTA split via `mma_tile_coord_v`. The M dimension has `num_m_tiles` entries (not `num_m_tiles * cluster_m_size`). Fix: use `pidm` instead of `pidm_cta`.
3. **Scope violation (from prior session)**: `bSG_sC_tma`, `bSG_gC_all` defined in epi warp block not visible in store warp block. Fix: moved C TMA partition to shared scope before warp blocks.
4. **num_c_stage mismatch (from prior session)**: `num_c_stage=1` but `num_write_stage=2`. Fix: changed to `num_c_stage=2`.
5. **Per-subtile pipeline (from prior session)**: Epi warp producer_acquire/commit moved inside subtile loop; store warp given per-subtile consume loop.

#### TODO-list
- [x] Identify all bugs
- [x] Change `num_c_stage` from 1 to 2
- [x] Move C TMA partition to shared scope
- [x] Fix epi warp: per-subtile producer cycle
- [x] Fix store warp: per-subtile consumer with proper addressing
- [x] Fix PipelineAsync producer phase (deadlock)
- [x] Fix bSG_gC_all M index (data corruption)
- [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
- [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**

---

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

#### TODO-list
- [x] Read and understand current fwd_mainloop kernel structure
- [x] Read and understand StaticPersistentScheduler API
- [x] Write plan to PROGRESS.md
- [x] Implement persistent scheduler in fwd_mainloop.py
- [x] Run `make unit-test-1gpu` — 75 passed, 89 skipped
- [x] Update KNOWLEDGE.md with scheduler insights
- [x] Mark task-2 completed in TASK.yaml

---

## Task-1: Understand Linear-Cross-Entropy Fusion Code Structure
**Status: Completed**

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
