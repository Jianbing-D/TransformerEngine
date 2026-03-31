# Task-5: Fix Bugs in bwd_partial_dlogits.py
**Status: Completed**

## PLAN-Task5: Fix Store Warp Bugs

### Bugs Identified & Fixed
1. **PipelineAsync producer phase (DEADLOCK)**: Manual `PipelineState(num_write_stage, 0, 0, 0)` used phase=0 for the store_c producer. PipelineAsync's `producer_acquire` calls `sync_object_empty.wait(state.index, state.phase)` which uses `mbarrier.try_wait.parity` — waits until `current_phase != phase`. Empty barriers start at phase=0. With phase=0, producer blocks forever (0 != 0 is false). Fix: use `make_pipeline_state(PipelineUserType.Producer, ...)` which sets phase=1.
2. **bSG_gC_all M index out of bounds (DATA CORRUPTION)**: Store warp indexed `bSG_gC_all[(None, pidm_cta, n_subtile_global)]` where `pidm_cta = pidm * cluster_m_size + mma_tile_coord_v`. But `bSG_gC_all` is derived from `thr_mma.partition_C(mC)` which already accounts for the 2-CTA split via `mma_tile_coord_v`. The M dimension has `num_m_tiles` entries (not `num_m_tiles * cluster_m_size`). Fix: use `pidm` instead of `pidm_cta`.
3. **Scope violation (from prior session)**: `bSG_sC_tma`, `bSG_gC_all` defined in epi warp block not visible in store warp block. Fix: moved C TMA partition to shared scope before warp blocks.
4. **num_c_stage mismatch (from prior session)**: `num_c_stage=1` but `num_write_stage=2`. Fix: changed to `num_c_stage=2`.
5. **Per-subtile pipeline (from prior session)**: Epi warp producer_acquire/commit moved inside subtile loop; store warp given per-subtile consume loop.

## TODO-list
- [x] Identify all bugs
- [x] Change `num_c_stage` from 1 to 2
- [x] Move C TMA partition to shared scope
- [x] Fix epi warp: per-subtile producer cycle
- [x] Fix store warp: per-subtile consumer with proper addressing
- [x] Fix PipelineAsync producer phase (deadlock)
- [x] Fix bSG_gC_all M index (data corruption)
- [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
- [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
