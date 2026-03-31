# Task-9: Apply 2-CTA MMA Instruction to fwd_mainloop.py
**Status: Completed**

## PLAN-Task9: Apply 2-CTA MMA to Forward Kernel

### Goal
Enable 2-CTA MMA instructions in `fwd_mainloop.py`, following the same pattern established in `bwd_partial_dlogits.py`. Both CTAs in a cluster cooperate on a single MMA tile, with the leader CTA issuing GEMM instructions and both CTAs executing the epilogue on their respective M-row partitions.

### Architecture
- **Files modified**:
  - `transformer_engine/common/cutedsl/linear_cross_entropy/blackwell/fwd_mainloop.py` — main kernel changes
  - `transformer_engine/pytorch/cutedsl/linear_cross_entropy_entry.py` — caller: pass `use_2cta_instrs=True, mma_tiler_mn=(256, 256)`

### Key Design Decisions
- `mma_tiler_mn=(256, 256)` for 2-CTA → each CTA handles 128 M rows (same as 1-CTA)
- Grid M = `ceil_div(M, 128)` CTAs, grouped into clusters of 2
- `pidm = bidx // 2` — both CTAs in a cluster share the same logical M tile
- `epi_tile = cta_tile_shape_mnk[:2]` = (128, 256) for per-CTA epilogue processing
- `pidm_cta = pidm * 2 + mma_tile_coord_v` for per-CTA label/aux tensor indexing

### Changes to `__call__` (JIT host)
1. Use `cluster_shape_to_tma_atom_A/B` for TMA load ops (not raw `CopyBulkTensorTileG2SOp`)
2. Multiply `tx_count` by `atom_thr_size` for 2-CTA barrier accounting
3. Set `epi_tile = cta_tile_shape_mnk[:2]` (per-CTA, not per-MMA-tile)
4. Fix grid: use per-CTA M tile size for grid calculation

### Changes to `kernel`
1. CTA rank: `block_idx_in_cluster()`, `mma_tile_coord_v`, `is_leader_cta`
2. `thr_mma = tiled_mma.get_slice(mma_tile_coord_v)` (not `get_slice(0)`)
3. Multicast masks for TMA loads
4. Pipelines: add `cta_layout_vmnk`, `defer_sync=True`; double mma_pipeline consumer group
5. `pipeline_init_arrive/wait` for cluster barrier sync
6. `pidm = bidx // cluster_shape_mn[0]`
7. MMA warp: gate with `is_leader_cta`
8. Epilogue: per-CTA M index `pidm_cta` for labels/max/accu/logprobs
9. Cluster sync (`cluster_arrive/wait`) before TMEM dealloc

## TODO-list
- [x] Write plan
- [x] Implement `__call__` changes (TMA ops, tx_count, epi_tile, grid)
- [x] Implement kernel changes (CTA rank, thr_mma, multicast, pipelines, cluster init)
- [x] Implement kernel changes (pidm, MMA gating, epilogue per-CTA indexing, cluster sync)
- [x] Update entry point to enable 2-CTA for forward
- [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
- [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
