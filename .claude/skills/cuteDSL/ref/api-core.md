# CuTeDSL Core API Reference

## Imports

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing
```

## Decorators

### @cute.kernel — Device code
```python
@cute.kernel
def my_kernel(gA: cute.Tensor, gB: cute.Tensor, M, N):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
```

### @cute.jit — Host-side orchestration
```python
@cute.jit
def host_fn(mA, mB, mC, M, N):
    my_kernel(gA, gB, M, N).launch(grid=[grid_x, 1, 1], block=[threads, 1, 1])
```

### Calling conventions
- Python -> @jit: Yes (entry point)
- @jit -> @jit: Yes (inlined at compile time)
- @kernel -> @jit: Yes (inlined at compile time)
- Python -> @kernel: NO (must launch via @jit)
- @kernel -> @kernel: NO

### cutlass.Constexpr — Compile-time parameters
```python
@cute.jit
def fn(mA, tile_size: cutlass.Constexpr = 128, dtype: cutlass.Constexpr = cutlass.Float16):
    vector_size = tile_size // dtype.width  # resolved at compile time
```

### @cutlass.dsl_user_op — Custom operations
```python
@cutlass.dsl_user_op
def gelu(x, *, loc=None, ip=None):
    return x * (1.0 + cute.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
```

## Tensor Creation

```python
# From PyTorch
a = torch.randn(M, N, device="cuda", dtype=torch.float16)
mA = from_dlpack(a).mark_layout_dynamic()  # dynamic strides for JIT reuse
mA = from_dlpack(a, assumed_align=16)      # with alignment hint

# Dynamic shapes
mA = from_dlpack(a).mark_compact_shape_dynamic(mode=0, divisibility=16)
```

## Layouts

```python
# Basic layouts
layout = cute.make_layout((M, N), stride=(N, 1))   # row-major
layout = cute.make_layout((M, N), stride=(1, M))   # col-major
layout = cute.make_layout((M, N))                   # default: compact row-major

# Ordered layout (specify contiguity order)
thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))  # mode-1 innermost

# Thread-Value layout (central CuTe pattern)
thr_layout = cute.make_ordered_layout((thr_m, thr_n), order=(1, 0))
val_layout = cute.make_ordered_layout((val_m, vector_size), order=(1, 0))
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
```

## Tiling

```python
# Zipped divide: ((TileM, TileN), (RestM, RestN))
gA = cute.zipped_divide(mA, tiler_mn)
block_data = gA[..., bidx]  # select block's tile

# Local tile for GEMM: tile along specified dimensions
gA = cute.local_tile(mA, tiler=cta_tiler, coord=(bx, by, 0), proj=(1, None, 1))
```

## Copy Operations

### Full elementwise copy pattern
```python
# 1. Create atoms
copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), tensor.element_type)

# 2. Tile the copy
tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

# 3. Per-thread slice
thr_copy = tiled_copy.get_slice(tidx)

# 4. Partition source and destination
src_part = thr_copy.partition_S(src_tensor)
dst_part = thr_copy.partition_D(dst_tensor)

# 5. Allocate fragments
frag = cute.make_fragment_like(src_part)

# 6. Copy with predication
cute.copy(copy_atom, src_part, frag, pred=pred_tensor)

# 7. Compute
result = frag.load() * 2.0
frag.store(result)

# 8. Store back
cute.copy(copy_atom, frag, dst_part, pred=pred_tensor)
```

## Predication (boundary handling)

```python
# Create identity/coordinate tensor
idC = cute.make_identity_tensor(mC.shape)
cC = cute.zipped_divide(idC, tiler=tiler_mn)

# Partition coordinates same as data
thrCrd = thr_copy.partition_S(cC[..., bidx])

# Build predicate
frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
for i in range(cute.size(frgPred)):
    frgPred[i] = cute.elem_less(thrCrd[i], (M, N))
```

## Shared Memory

```python
@cute.kernel
def kernel(...):
    allocator = cutlass.utils.SmemAllocator()
    sA = allocator.allocate_tensor(cutlass.Float16, smem_layout, byte_alignment=128)
    # SmemAllocator auto-computes total smem size for launch
```

## Kernel Launch

```python
# Basic
my_kernel(args).launch(grid=[gx, gy, 1], block=[threads, 1, 1])

# With stream
my_kernel(args).launch(grid=grid, block=block, stream=stream)

# With cluster (Hopper+)
my_kernel(args).launch(grid=grid, block=block, cluster=(cm, cn, 1), stream=stream)

# Cooperative launch (grid-wide sync)
my_kernel(args).launch(grid=grid, block=block, cooperative=True)

# PDL (Hopper+)
my_kernel(args).launch(grid=grid, block=block, use_pdl=True)
```

## Compilation

```python
# Compile once, run many times (recommended)
compiled = cute.compile(host_fn, mA, mB, mC, stream=stream)
compiled(mA, mB, mC, stream=stream)

# Direct invocation (recompiles if args change)
host_fn(mA, mB, mC)

# With debug options
compiled = cute.compile[cute.KeepPTX, cute.GenerateLineInfo](host_fn, *args)
print(compiled.__ptx__)
print(compiled.__mlir__)
```

## Synchronization

```python
cute.arch.sync_threads()           # __syncthreads()
cute.arch.barrier()                # named barrier
cute.arch.elect_one()              # context manager: one thread executes
cute.arch.warp_idx()               # warp index
cute.arch.lane_idx()               # lane within warp
cute.arch.cp_async_commit_group()  # commit async copies
cute.arch.cp_async_wait_group(n)   # wait until <= n groups pending
```

## Data Types

```python
cutlass.Float32, cutlass.Float16, cutlass.BFloat16
cutlass.Int8, cutlass.Int32, cutlass.Uint32
cutlass.Float8E4M3FN, cutlass.Float8E5M2
cutlass.Boolean
```

## Compile-time vs Runtime

```python
print(tensor)                        # COMPILE-TIME: prints layout info
cute.printf("val=%f\n", val)         # RUNTIME: device printf
cutlass.range_constexpr(n)           # compile-time unrolled loop
range(n)                             # runtime loop (since v4.1)
cutlass.const_expr(condition)        # compile-time if
for k in range(N, unroll_full=True)  # fully unrolled runtime loop
```

## Math Operations (on TensorSSA or Float scalars)

```python
cute.math.sin(x), cute.math.cos(x), cute.math.exp(x), cute.math.log(x)
cute.math.sqrt(x), cute.math.rsqrt(x), cute.math.tanh(x), cute.math.erf(x)
# All support fastmath=True for aggressive optimization
```

## Reductions

```python
# Warp-level
val = cute.arch.warp_reduction(local_val, operator.add, threads_in_group=32)

# Block-level: warp reduce -> smem -> warp reduce across warps
```
