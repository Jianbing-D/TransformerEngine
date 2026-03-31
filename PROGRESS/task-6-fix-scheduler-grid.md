# Task-6: Fix Static Scheduler Grid Size for Clusters
**Status: Completed**

## PLAN-Task6: Fix Scheduler to Launch SM_count / cluster_size Clusters

### Problem
`StaticPersistentScheduler.get_grid_shape` computes `vacancies = sm_count * occupancy` without accounting for cluster size. With `cluster_m_size=2` on 152 SMs:
- Before: grid = `min(152, total_blocks) * 2 = 304` CTAs = 152 clusters → needs 304 SMs → 2 waves
- After: grid = `min(76, total_blocks) * 2 = 152` CTAs = 76 clusters → fits in 1 wave on 152 SMs

### Fix
Changed `vacancies = sm_count * occupancy` to `vacancies = (sm_count // cluster_m_size) * occupancy` in `scheduler.py:get_grid_shape`.

### Results
- Grid: 304 → 152 CTAs (152 → 76 clusters)
- Waves: 2 → 1
- Duration: 154.62 → **136.86 μs (11.5% speedup)**
- Compute throughput: 72.80%

## TODO-list
- [x] Identify root cause in scheduler.py
- [x] Fix `get_grid_shape` to divide sm_count by cluster_m_size
- [x] Run `make unit-test-1gpu` — **75 passed, 89 skipped**
- [x] Run `make unit-test-4gpu` — **88 passed, 76 skipped**
- [x] Run `make ncu-bwd-cli` — **136.86 μs** (11.5% speedup from fixing grid)
