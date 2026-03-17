# CUTLASS Blackwell GEMM Example Optimization Techniques

## Warp Specialization Pattern (8 warps)
All SM100 GEMM kernels use the same warp assignment:
- **Warps 0-3 (epilogue warpgroup)**: Read accumulator from TMEM, apply epilogue ops, store to GMEM
- **Warp 4 (MMA)**: Issues `tcgen05` GEMM instructions, reads SMEM, writes TMEM
- **Warp 5 (TMA load)**: Issues TMA G2S copies for A and B matrices
- **Warps 6-7 (empty)**: Register pressure reduction via `warpgroup_reg_dealloc`

## Pipeline Architecture (3 pipelines)

### AB Load Pipeline (`PipelineTmaUmma`)
- Stages: typically 3-5 (computed from available SMEM)
- Producer: TMA load warp → Consumer: MMA warp
- `tx_count`: sum of A and B tile bytes per stage

### Accumulator Pipeline (`PipelineUmmaAsync`)
- Stages: typically 2 (double-buffered TMEM accumulators)
- Producer: MMA warp → Consumer: epilogue warpgroup
- Each stage = one full N-tile accumulation in TMEM

### C Store Pipeline (`PipelineTmaStore`, TMA store only)
- Stages: 2-4 (uses remaining SMEM after A/B allocation)
- Producer: epilogue warp 0 → S2G TMA store
- Fire-and-forget: no explicit consumer

## Persistent Scheduler

### Grid Calculation
```python
grid = min(SM_count × occupancy, total_tiles)
```
Each CTA pulls tiles from `StaticPersistentTileScheduler`, advances by `grid_dim.x`.

### Key Pattern: Each Warp Has Its Own Persistent Loop
```python
if warp_idx == load_warp_id:
    scheduler = TileSchedulerCls()
    work_tile = scheduler.initial_work_tile_info()
    while work_tile.is_valid_tile:
        pidm, pidn = work_tile.tile_idx
        # ... per-tile load work ...
        scheduler.advance_to_next_work()
        work_tile = scheduler.get_current_work()
```
All warps start with the same tile index (from `block_idx`) and advance in lockstep.

### Pipeline State Persistence Across Tiles
Pipeline states are created ONCE before the persistent loop. They wrap around naturally because both producer and consumer advance the same number of times per tile. No reset needed between tiles.

## TMA Store (S2G) for Output

### 4-Phase Epilogue
1. **T2R**: `tcgen05.make_tmem_copy` loads accumulator from TMEM to registers
2. **Type conversion**: FP32 accumulator → output dtype, apply epilogue op
3. **R2S**: SIMT copy from registers to SMEM staging buffer
4. **S2G**: `CopyBulkTensorTileS2GOp()` TMA store from SMEM to GMEM

### Requirements
- Needs SMEM staging buffer (`sC`) in `SharedStorage`
- Needs `PipelineTmaStore` for multi-stage buffering
- Output tiles must be aligned for TMA (128B minimum)
- Only epilogue warp 0 issues the TMA store command

### Direct R2G Alternative (No TMA Store)
- 3-phase: T2R → type conversion → R2G via `CopyUniversalOp`
- Simpler, no SMEM staging needed
- Uses predicated stores for boundary handling

## 2-CTA MMA Instructions

### Overview
Two CTAs cooperate on a single MMA tile. The `tiled_mma` operates across both CTAs' resources (shared TMEM accumulator). Each CTA handles half the M-tile rows. Only the "leader" CTA (CTA 0 in the cluster) issues actual MMA instructions.

### Configuration
```python
cta_group = tcgen05.CtaGroup.TWO
cluster_shape_mn = (2, 1)   # 2 CTAs along M dimension
cluster_shape_mnk = (2, 1, 1)
```
- `cute.size(tiled_mma.thr_id.shape) == 2` indicates 2-CTA mode
- `cta_tile_shape_mnk[0] = mma_tiler[0] // 2` — each CTA processes half the M rows

### CTA Rank and Coordination (kernel setup)
```python
# Obtain CTA's position within its cluster
cta_rank_in_cluster = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster()
)
block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

# Determine leader CTA for MMA
bidx, _, _ = cute.arch.block_idx()
mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)  # 0 or 1
is_leader_cta = mma_tile_coord_v == 0
```
- `cute.arch.block_idx_in_cluster()` — flat rank within cluster (0 or 1)
- `mma_tile_coord_v` — which "virtual CTA" within the MMA instruction

### cluster_layout_vmnk for 2-CTA
```python
cluster_layout_vmnk = cute.tiled_divide(
    cute.make_layout(cluster_shape_mnk),   # (2, 1, 1)
    (tiled_mma.thr_id.shape,)              # (2,)
)
# Result: shape ((2, 1), 1, 1) → V=2, M=1, N=1, K=1
# CTA 0: coord (0, 0, 0, 0)
# CTA 1: coord (1, 0, 0, 0)
```
The V dimension captures the 2-CTA split. M, N, K remain 1 (no additional spatial clustering).

### thr_mma Partitioning
```python
thr_mma = tiled_mma.get_slice(mma_tile_coord_v)  # NOT get_slice(0)
```
- Each CTA gets its own partition of the global tensors via `partition_A`, `partition_B`, `partition_C`
- CTA 0 gets the first half of M rows; CTA 1 gets the second half
- `make_fragment_A(sA)` and `make_fragment_B(sB)` are NOT parameterized by CTA rank (SMEM layout is shared)

### TMA Load Ops (No Multicast for (2,1) Cluster)
```python
# Determined by sm100_utils.cluster_shape_to_tma_atom_A/B:
# For cluster_shape_mn=(2,1), atom_thr_size=2:
#   A: mcast = not(cluster_shape[1]==1) = False → CopyBulkTensorTileG2SOp(CtaGroup.TWO)
#   B: mcast = not(cluster_shape[0]==atom_sm_cnt) = not(2==2) = False → CopyBulkTensorTileG2SOp(CtaGroup.TWO)
# Same op we already use! CtaGroup.TWO handles 2-CTA data organization internally.

a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
```
- Multicast (`CopyBulkTensorTileG2SMulticastOp`) only needed when cluster is larger than the MMA atom (e.g., cluster (4,1) with 2-CTA MMA → 2 clusters each needing B multicast)
- For simple `(2,1)` cluster, the `CtaGroup.TWO` param in `CopyBulkTensorTileG2SOp` suffices

### Multicast Masks (Always Created for 2-CTA)
Even without actual multicast, the CUTLASS example creates masks when `use_2cta_instrs=True`:
```python
a_mcast_mask = cpasync.create_tma_multicast_mask(
    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
)
b_mcast_mask = cpasync.create_tma_multicast_mask(
    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
)
# Pass to TMA copy:
cute.copy(tma_atom_a, src, dst, tma_bar_ptr=..., mcast_mask=a_mcast_mask)
```
- `mcast_mode=2` for A (along K dim in VMNK)
- `mcast_mode=1` for B (along N dim in VMNK)
- For `(2,1)` cluster these masks are trivial but ensure correct barrier signaling

### tx_count for 2-CTA
```python
atom_thr_size = cute.size(tiled_mma.thr_id.shape)  # 2 for 2-CTA
tma_copy_ab_bytes = (a_copy_size + b_copy_size) * atom_thr_size
```
The `tx_count` doubles because both CTAs' TMA loads contribute to the same pipeline barrier.

### Pipeline Changes for 2-CTA

Both `PipelineTmaUmma` and `PipelineUmmaAsync` accept `cta_layout_vmnk` for cross-CTA barrier management:
```python
ab_pipeline = pipeline.PipelineTmaUmma.create(
    ...,
    tx_count=tma_copy_ab_bytes,  # total across both CTAs
    cta_layout_vmnk=cluster_layout_vmnk,
    defer_sync=True,
)

mma_pipeline = pipeline.PipelineUmmaAsync.create(
    ...,
    cta_layout_vmnk=cluster_layout_vmnk,
    defer_sync=True,
)
```

Cluster-wide barrier init/sync is required:
```python
pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
# ... setup SMEM tensors ...
pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
```

### MMA Warp: Leader-Only Gating
```python
if is_leader_cta:
    # Only leader CTA issues GEMM
    ab_pipeline.consumer_wait(ab_consumer_state)
    for kblock_idx in ...:
        cute.gemm(tiled_mma, tCtC, tCsA[...], tCsB[...], tCtC)
    ab_pipeline.consumer_release(ab_consumer_state)
    mma_pipeline.producer_commit(mma_producer_state)
```
Non-leader CTA's MMA warp stays idle (no pipeline ops, no GEMM).

### Load Warp: Both CTAs Load
Both CTAs' load warps issue TMA loads independently. Each loads its own partition of A (different M rows) and B (same N rows, handled by `CtaGroup.TWO`).

### Epilogue: Both CTAs Participate
Both CTAs' epilogue warps read their portion of TMEM (determined by `thr_mma.get_slice(mma_tile_coord_v)` affecting `partition_C`) and write to their GMEM region.

### TMEM Allocation for 2-CTA
```python
cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf, is_two_cta=True)
cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=True)
```
TMEM is shared across both CTAs. Deallocation requires cross-CTA synchronization via a dedicated mbarrier.

### Grid Calculation for 2-CTA
```python
grid = cute.round_up(
    (ceil_div(M, cta_tile_M), ceil_div(N, cta_tile_N), 1),
    cluster_shape_mnk   # (2, 1, 1) — grid M must be multiple of 2
)
```
- `cta_tile_M = mma_tiler_M // 2` (64 for mma_tiler_M=128)
- 2 CTAs per cluster → each cluster covers 128 M rows

### Persistent Scheduler with 2-CTA
The scheduler must assign the same tile to both CTAs in a cluster:
- `tile_idx = block_idx.x // cluster_m_size` — both CTAs in same cluster get same tile
- `grid = min(SM_count, total_tiles) * cluster_m_size` — grid accounts for cluster size
- `advance: tile_idx += grid_dim.x // cluster_m_size` — advance by number of clusters

### Summary Table

| Aspect | 1-CTA | 2-CTA |
|--------|-------|-------|
| cta_group | `CtaGroup.ONE` | `CtaGroup.TWO` |
| thr_id shape | (1,) | (2,) |
| cta_tile_M | mma_tiler_M | mma_tiler_M / 2 |
| MMA execution | All CTAs | Leader only |
| TMA loads | Each CTA loads | Each CTA loads (CtaGroup.TWO) |
| Multicast | No | Trivial for (2,1) cluster |
| Epilogue | All CTAs | All CTAs (own partition) |
| Pipeline cta_layout | None needed | `cluster_layout_vmnk` required |
| tx_count | a+b bytes | (a+b) × 2 bytes |
| TMEM | per-CTA | shared, `is_two_cta=True` |
| cluster_init | Not needed | `pipeline_init_arrive/wait` required |

## SMEM Stage Heuristic
```
num_ab_stage = (smem_capacity / occupancy - mbar_bytes - c_stage_bytes) / ab_bytes_per_stage
```
After A/B allocation, remaining SMEM is used for C staging buffer stages.

## TMA Store Phase 2 Lessons (from bwd_partial_dlogits implementation)

### Key Pattern: R2S → fence → barrier → S2G
```python
# R2S: convert and copy to SMEM
acc_vec = tiled_copy_r2s.retile(tTMEM_load_rAcc).load()
acc_vec = acc_vec.to(output_dtype)
tRS_rC.store(acc_vec)
cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])

# Fence + barrier: ensure R2S visible to TMA engine
cute.arch.fence_proxy("async.shared", space="cta")
epilog_sync_barrier.arrive_and_wait()

# S2G: TMA store (warp 0 only)
if warp_idx == epi_warp_ids[0]:
    cute.copy(tma_atom_c, bSG_sC_tma[(None, c_buffer)], bSG_gC)
    c_pipeline.producer_commit()
    c_pipeline.producer_acquire()
epilog_sync_barrier.arrive_and_wait()
```

### Critical: tma_partition Requires TMA Tensor
`cpasync.tma_partition()` GMEM argument must be the TMA tensor (`mC` from `make_tiled_tma_atom`), NOT the raw GMEM tensor. The TMA tensor has special metadata for the TMA engine.

### SMEM Layout for TMA Store
```python
c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
    output_dtype, output_layout, epi_subtile, num_c_stage
)
c_smem_layout_one_stage = cute.slice_(c_smem_layout_staged, (None, None, 0))
tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(), dlogits_partial, c_smem_layout_one_stage, epi_subtile
)
```

### R2S Setup via `epilog_smem_copy_and_partition`
```python
tiled_copy_r2s, tRS_rC, tRS_sC = utils.epilog_smem_copy_and_partition(
    layout_enum, output_dtype, acc_dtype, tiled_copy_t2r, tTR_rC, tidx, sC
)
```
This utility creates the R2S tiled copy, partitions register and SMEM tensors. No Python examples exist for this in CUTLASS examples — the reference is in `cutlass.utils.gemm.sm100`.

## 2-CTA MMA Phase 3 Lessons (from bwd_partial_dlogits implementation)

### Critical: mma_tiler_mn Must Double M for 2-CTA
With `use_2cta_instrs=True`, use `mma_tiler_mn=(256, 256)` not `(128, 256)`. Each CTA handles `mma_tiler_M // 2` rows. With (128, 256), each CTA gets only 64 M rows — this produces incorrect results due to wrong tile size assumptions in the epilogue.

### Critical: PipelineUmmaAsync Consumer Group Must Double for 2-CTA
Both CTAs' epilogue warpgroups call `consumer_wait` and `consumer_release` on the accumulator pipeline. The consumer group size must account for both CTAs:
```python
num_mma_consumer_threads = threads_per_warp * len(epi_warp_ids)
if use_2cta_instrs:
    num_mma_consumer_threads *= 2
```
If not doubled, the barrier fires after only one CTA's threads arrive, causing the MMA warp to overwrite TMEM while the other CTA's epilogue is still reading.

### Performance Analysis: 2-CTA Neutral for Epilogue-Bound Kernels
For bwd_partial_dlogits, 2-CTA MMA achieved 154.62 μs vs 153.70 μs (1-CTA Phase 2). No speedup because:
- The kernel is epilogue-bound (per-element softmax gradient: exp, subtract, multiply)
- 2-CTA reduces MMA instruction count but doesn't reduce epilogue work
- Both CTAs still process the same total number of output elements
- SMEM per CTA decreased (148 KB vs 214 KB) but occupancy remained 1 block/SM

### Non-Leader CTA MMA Warp Behavior
In the CUTLASS reference, the non-leader CTA's MMA warp still enters the persistent loop and advances pipeline states (without calling wait/commit/release). In our kernel, the non-leader MMA warp exits immediately after `warpgroup_reg_dealloc`. Both patterns work — the pipeline state advancement on the non-leader is a no-op since only the leader CTA signals the hardware barriers.
