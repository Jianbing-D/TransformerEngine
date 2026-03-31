# Task-7: Move Tensor Partitions Outside While Loops in bwd_partial_dlogits.py
**Status: Completed**

## PLAN-Task7: Hoist Tensor Partitions

### Goal
Move all tensor partition operations (local_tile, partition_A/B, tma_partition) out of the persistent while loops. Inside the loops, only use index-based access into pre-computed tensors.

### Load Warp Changes
- Before loop: `gA_all = local_tile(mA, tile, (None, None))` — keep all M tiles
- Before loop: `gB_all = local_tile(mB_n, tile, (None, None))` — keep all N tiles
- Before loop: `tCgA_all = partition_A(gA_all)`, `tCgB_all = partition_B(gB_all)`
- Before loop: `tTMAsA, tTMAgA_all = tma_partition(...)`, same for B
- Inside loop: `tTMAgA_all[(None, pidm, k)]` and `tTMAgB_all[(None, pidn, k)]`

### EPI Warp Changes
- Before loop: `gLabels_all = local_tile(mLabels, tile, (None,))` — keep all tiles
- Before loop: `gAccu_all = local_tile(mAccu, tile, (None,))`
- Before loop: `gDlogprobs_all = local_tile(mDlogprobs, tile, (None,))` (REDUCTION==0 only)
- Inside loop: index with `[(None, pidm_cta)]` then partition_S on the slice

### EPI Warp — Phase 2: Hoist partition_S and make_fragment

Operations moved out of EPI while loop:

| Line | Operation | Why it can move |
|------|-----------|-----------------|
| 515 | `tMCAcc_mask = make_fragment(...)` | Shape is tile-independent (from `tMCAcc.shape`) |
| 516 | `tMCAcc_mask = append_ones(...)` | Reshapes the above — still tile-independent |
| 524 | `tMgLabels = partition_S(append_ones(gLabels))` | Partition all tiles at once → index inside loop |
| 525 | `tMrLabels = make_fragment(...)` | Shape `(1, 1)` is tile-independent — allocate once |
| 527 | `tMgAccu = partition_S(append_ones(gAccu))` | Same approach as labels |
| 528 | `tMrAccu = make_fragment(...)` | Same as labels |
| 531 | `tMrDlogprobs = make_fragment(...)` | Same shape as tMrAccu — allocate once |
| 533 | `num_valid_tokens = make_tensor(...)` | Pointer-to-tensor wrap, constant across tiles |
| 539 | `tMgDlogprobs = partition_S(...)` | Same approach as labels (REDUCTION==0 only) |

**Key insight**: `partition_S(append_ones(gLabels_all))` where `gLabels_all` is `(epi_tile_m, num_tiles)`:
- `append_ones` → `(epi_tile_m, num_tiles, 1)` (3 modes)
- `partition_S` with copy tile `(epi_tile_m, 1)`: partitions first 2 modes, preserves mode 2 as outer
- Result: `(1, num_tiles, 1)` per thread
- Index `[(None, pidm_cta, None)]` → `(1, 1)` — matches original per-tile shape

## TODO-list
- [x] Write plan
- [x] Implement load warp changes (gA, gB, tCgA, tCgB, tTMAs, tTMAg all hoisted)
- [x] Implement EPI warp changes — Phase 1 (gLabels, gAccu, gDlogprobs local_tile hoisted)
- [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
- [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
- [x] Implement EPI warp changes — Phase 2 (partition_S, make_fragment hoisted)
  - `tMgLabels_all`, `tMgAccu_all`, `tMgDlogprobs_all` — partition_S on all-tile tensors
  - `tMrLabels`, `tMrAccu`, `tMrDlogprobs` — make_fragment allocated once
  - `tMCAcc_mask` — make_fragment + append_ones allocated once
  - `num_valid_tokens` — make_tensor hoisted (REDUCTION==2)
  - Inside loop: only scalar mask update + `cute.copy` with `[(None, None, pidm_cta, None)]` indexing
  - Key fix: `tMgLabels_all` has 4 modes `((val_m,val_n), rest_m, num_tiles, rest_appended)` — need 4-element coord
- [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
- [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
