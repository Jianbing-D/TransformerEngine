# Architecture-Specific CuTeDSL Features

## Ampere (SM80/86/89)

### Copy mechanisms
- **Synchronous copy**: `CopyUniversalOp` — simple load/store via registers
- **Async copy (cp.async)**: `cpasync.CopyG2SOp` — global-to-shared without register staging
  ```python
  atom = cute.make_copy_atom(
      cute.nvgpu.cpasync.CopyG2SOp(), dtype,
      num_bits_per_copy=dtype.width * vector_size
  )
  cute.copy(tiled_copy, src, dst, pred=pred)
  cute.arch.cp_async_commit_group()
  cute.arch.cp_async_wait_group(n)
  ```

### MMA
- **SIMT (FMA)**: `MmaUniversalOp(cutlass.Float32)` — works for any dtype, lower throughput
- **Tensor Core**: `MmaAtom.from_SM80()` — FP16/BF16/INT8/FP64, higher throughput
  - Shapes: m16n8k8 (FP16), m16n8k16 (FP16), m16n8k32 (INT8)

### Shared memory
- Default: 48KB, Extended: up to 164KB (opt-in)
- Swizzle for bank conflict avoidance:
  ```python
  swizzle = cute.make_swizzle(3, 2, 4)  # (bits, base, shift)
  smem_layout = cute.make_composed_layout(swizzle, base_layout)
  ```

### Software pipeline pattern
```python
# Prologue: fill pipeline stages
for s in range(num_stages - 1):
    cute.copy(tiled_copy, gA[..., s], sA[..., s])
    cute.arch.cp_async_commit_group()

# Mainloop: overlap load and compute
for k in range(K_tiles):
    cute.arch.cp_async_wait_group(num_stages - 2)
    cute.arch.sync_threads()
    cute.gemm(tiled_mma, acc, sA_part[..., read_stage], sB_part[..., read_stage], acc)
    cute.copy(tiled_copy, gA[..., k + num_stages - 1], sA[..., write_stage])
    cute.arch.cp_async_commit_group()
```

---

## Hopper (SM90)

### TMA (Tensor Memory Accelerator)
```python
from cutlass.cute.nvgpu import cpasync

tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    a_copy_op, a_tensor, a_smem_layout, mma_tiler, tiled_mma, cluster_shape
)
cpasync.prefetch_descriptor(tma_atom_a)

with cute.arch.elect_one():
    cpasync.copy(tma_tiled_copy_a, tma_src_a[..., k], smem_a[..., stage])
```

### WGMMA (Warpgroup MMA)
```python
import cutlass.utils.hopper_helpers as sm90_utils
tiled_mma = sm90_utils.make_tiled_mma(dtype_a, dtype_b, acc_dtype, mma_tiler)
```

### Cluster launch & Pipeline
```python
kernel(args).launch(grid=grid, block=block, cluster=(cm, cn, 1), stream=stream)

pipe = pipeline.PipelineTmaAsync(num_stages, ...)
state = pipeline.make_pipeline_state(pipe)
```

### Persistent kernels
```python
scheduler = cutlass.utils.DynamicPersistentTileScheduler(M, N, tile_m, tile_n)
```

---

## Blackwell (SM100)

### Three-level memory hierarchy
```
GMEM -> SMEM (via TMA) -> TMEM (via tcgen05 copy) -> RMEM (via tcgen05.Ld)
```
Accumulators live in **TMEM** (tensor memory), not registers.

### tcgen05 MMA
```python
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05

# FP16/BF16
op = tcgen05.MmaF16BF16Op(io_dtype, acc_dtype, inst_shape, CtaGroup.ONE,
                           OperandSource.SMEM, K_major, K_major)
tiled_mma = cute.make_tiled_mma(op)

# Helper (simpler)
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a_dtype, a_major_mode, b_major_mode, acc_dtype, cta_group, mma_shape
)
```

### TMEM management
```python
tmem = cutlass.utils.TmemAllocator()
acc_addr = tmem.allocate(512)       # allocate 512 columns
tmem.wait_for_alloc()               # barrier until allocated
acc_ptr = tmem.retrieve_ptr(acc_dtype)  # get typed pointer
# ... use accumulator ...
tmem.relinquish_alloc_permit()      # done allocating
tmem.free(acc_ptr)                  # release

# Pool-based (for block-scaled with SFA/SFB):
pool = tmem.reserve(cols)
acc = pool.allocate_tensor(layout, dtype)
sfa = pool.allocate_tensor(sf_layout, sf_dtype)
```

### 2-CTA MMA instructions
Two CTAs cooperate on one MMA. Halves B-operand SMEM per CTA, enabling more pipeline stages.
```python
op = tcgen05.MmaF16BF16Op(..., CtaGroup.TWO, ...)
# Must launch with cluster:
kernel(args).launch(grid=grid, block=block, cluster=(2, 1, 1))
# VMNK layout for 2CTA coordination:
cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))
is_leader_cta = mma_coord_vmnk[0] == 0
```

### Warp specialization (typical 6-warp pattern)
```
Warps 0-3: Epilogue (TMEM->RMEM->SMEM->GMEM)
Warp 4:    MMA (tcgen05.mma)
Warp 5:    TMA (GMEM->SMEM loads)
```

### Pipeline abstractions
```python
# TMA load -> MMA
ab_pipe = pipeline.PipelineTmaUmma(num_stages, ...)
# MMA accumulator -> Epilogue
acc_pipe = pipeline.PipelineUmmaAsync(acc_stages, ...)
# Epilogue -> TMA store
store_pipe = pipeline.PipelineTmaStore.create(num_stages, ...)
```

### Epilogue: TMEM -> GMEM
```python
# TMEM -> RMEM
tmem_copy = tcgen05.make_tmem_copy(tcgen05.Ld32x32bOp(Repetition.x64), acc_tensor)
cute.copy(tmem_copy, tCtAcc, tTR_rAcc)
# Type convert fp32 -> fp16
# RMEM -> SMEM
cute.copy(tiled_copy_r2s, retiled_acc, smem_epi)
cute.arch.fence_view_async_shared()
# SMEM -> GMEM (TMA store)
cpasync.copy(tma_store_atom, smem_epi, gmem_out)
```

### Block-scaled GEMM (FP8/FP4 with per-block scaling)
```python
op = tcgen05.MmaMXF4NVF4Op(sf_dtype, inst_shape, CtaGroup.ONE, OperandSource.SMEM)
# Scale factors stored in TMEM:
cute.gemm(tiled_mma, acc, [tCrA, tCtSFA], [tCrB, tCtSFB], acc)
```

### Persistent tile schedulers
```python
# Static (round-robin assignment)
sched = cutlass.utils.StaticPersistentTileScheduler(params)
# CLC Dynamic (hardware-assisted, better load balancing)
sched = cutlass.utils.ClcDynamicPersistentTileScheduler(params)
```

### Preferred/fallback cluster shapes
```python
# Preferred (2,4,1) with fallback (2,1,1) for better SM utilization
# Kernel compiled for both; runtime selects based on available resources
```

### PDL (Programmatic Dependent Launch)
```python
kernel(args).launch(grid=grid, block=block, use_pdl=True)
# In kernel:
cute.arch.griddepcontrol_launch_dependents()  # signal next kernel can start
cute.arch.griddepcontrol_wait()               # wait for previous kernel
```

### Epilogue Fusion Configuration (EFC)
```python
def my_epilogue(efc_config, C, bias, alpha, beta):
    acc = efc_config.accum()          # load accumulator
    c_val = C.load()                  # load auxiliary tensor
    result = alpha * acc + beta * c_val + bias.remap_modes[0, :, :].load()
    C.store(result)                   # store output
    return efc_config.relu(result)    # fused activation
```

### Sparse GEMM (2:4 structured sparsity)
```python
# Three TMA loads: A (compressed), B (dense), E (metadata)
# tcgen05.mma sparse instructions consume compressed A + metadata
```

### Implicit GEMM (convolution)
TMA performs im2col during load — convolution mapped to GEMM automatically.

---

## Rubin (SM107)

**Architecture**: `sm_107`. Extends Blackwell with enhanced capabilities.

### Hardware resources
- SMEM: 328 KiB (vs 228 KiB on SM100)
- TMEM: 576 columns / 288 KiB (vs 512 / 256 KiB on SM100)
- MMA K dimension: supports K=32 and K=64 (SM100 only K=32)

### Key imports
```python
import cutlass.utils.rubin_helpers as sm107_utils
# or for sparse:
from cutlass.utils.sm107 import make_trivial_tiled_mma, make_sparse_trivial_tiled_mma
```

### Inheritance pattern
Rubin kernels extend Blackwell kernels, overriding arch, smem_capacity, and MMA creation:
```python
class SM107Kernel(BlackwellKernel):
    def __init__(self, ...):
        super().__init__(...)
        self.arch = "sm_107"
        self.smem_capacity = 334848  # 328 KiB
    def _create_tiled_mma(self):
        return sm107_utils.make_trivial_tiled_mma(...)
```

### B-keep / B-reuse optimization
Reuses B matrix across two MMA operations to reduce SMEM traffic:
```python
# Three TiledMMA variants:
tiled_mma_standard = sm107_utils.make_trivial_tiled_mma(...)
tiled_mma_bkeep = sm107_utils.make_trivial_tiled_mma(..., b_collector_op=CollectorOp.FILL)
tiled_mma_breuse = sm107_utils.make_trivial_tiled_mma(..., b_collector_op=CollectorOp.LASTUSE)
# First MMA fills B collector, second reuses it
```

### Lamport synchronization
Data-based producer-consumer sync — producer writes sentinel, consumer validates via TMA:
```python
# Producer initializes output buffer to -0.0 sentinel
# Consumer speculatively loads via TMA, validates in hardware
# On Rubin, validation happens as part of TMA load operation
# Benefits: reduced inter-kernel latency, per-tile readiness
```

### LDTM.SPARSIFY (hardware 2:4 sparsification)
```python
# GMEM -> RMEM -> TMEM (STTM) -> RMEM compressed+metadata (LDTM.SPARSIFY) -> GMEM
store_atom = tcgen05.St32x32bOp(tmem_repeat)          # RMEM -> TMEM
sparse_atom = tcgen05.LdSPCompress32x32bOp(repeat,
    redOp=tcgen05.TmemLoadRedOp.MAXABS)               # TMEM -> RMEM (sparse)
```

### Mixed cluster launches
Preferred cluster (e.g. 2,4,1) with fallback (e.g. 2,1,1):
```python
# Runtime selects based on SM availability. Preferred must be integer multiple of fallback.
```

### LUT-B GEMM (Look-Up Table B matrix)
B matrix stored as indices into a lookup table:
```python
import cutlass.utils.rubin_lutb_helpers as lutb_utils
tiled_mma = lutb_utils.make_blockscaled_trivial_tiled_mma(...)
```

---

## Feynman (SM140) — Pre-release

**Architecture**: `sm_140`. Radical departure from Blackwell/Rubin memory hierarchy.

### Key architectural changes
1. **SPMEM (Scratchpad Memory)** replaces SMEM as primary on-chip buffer — 4 MiB capacity
2. **SMEM** is tiny — only for barriers and small staging
3. **tcgen06** (new tensor core generation, not tcgen05)
4. **DPC (Data Path Controller)** replaces traditional cluster model
5. **DLCCP** operations for SPMEM<->TMEM data movement

### Memory hierarchy
```
GMEM -> SPMEM (via DPC TMA) -> TMEM (via DLCCP) -> RMEM (via tcgen06.Ld)
```

### Key imports
```python
from cutlass.cute.nvgpu import tcgen06
import cutlass.cute.experimental.libcute as lib
from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileG2SpOp  # GMEM -> SPMEM
```

### SPMEM layout system
```python
# Layout tags control memory organization:
tcgen06.SP_LAYOUT_4x256B, SP_LAYOUT_4x128B, SP_LAYOUT_4x64B, SP_LAYOUT_4x32B
# Selection based on K-tile bytes
spmem_iter = tcgen06.make_spmem_iter(ptr, layout_tag)
```

### DPC shape (replaces cluster shape)
```python
dpc_shape_mnk = (8, 1, 1)  # or (4, 2, 1) -- 3D with K-slice dimension
```

### Temporal split-K
Tiles K dimension into subtiles; A loaded once per K-subtile, reused across N-blocks:
```python
NUM_K_SUBTILES = 4
NUM_N_BLOCKS = 2
# Separate SPMEM/TMEM stages for A vs B
```

### Sparse GEMM on Feynman
```python
op = tcgen06.MmaSparseF8Op(dpc_shape_mnk, mma_shape_mnk, dtypes,
    sparse_metadata_format=tcgen06.SparseMetadataFormat.tid)
# Dedicated SPMEM->TMEM copy: tcgen06.Sp2TAsACopyOp, tcgen06.Sp2TAsECopyOp
```

---

## Architecture comparison

| Feature | Ampere (SM80) | Hopper (SM90) | Blackwell (SM100) | Rubin (SM107) | Feynman (SM140) |
|---------|--------------|---------------|-------------------|---------------|-----------------|
| Global->On-chip | cp.async | TMA | TMA | TMA | DPC TMA |
| On-chip buffer | SMEM (164KB) | SMEM (228KB) | SMEM (228KB) | SMEM (328KB) | SPMEM (4MB) |
| MMA unit | Warp MMA | wgmma | tcgen05 | tcgen05 | tcgen06 |
| Accumulator | Registers | Registers | TMEM (256KB) | TMEM (288KB) | TMEM (288KB/slice) |
| Multi-CTA | No | Cluster | Cluster + 2CTA | Cluster + 2CTA + B-reuse | DPC (3D) |
| Block-scaled | No | No | Native | Native + LUT-B | Native |
| Sparse | No | No | 2:4 (sw compress) | 2:4 (hw LDTM.SPARSIFY) | 2:4 (tcgen06 native) |
| Split-K | Manual | Manual | N/A | N/A | Temporal + Global |
| Persistent | Manual | Tile scheduler | Static + CLC dynamic | Same + Lamport sync | CLC + DPC |
| Kernel helpers | N/A | `hopper_helpers` | `blackwell_helpers` | `rubin_helpers` | `tcgen06` experimental |
