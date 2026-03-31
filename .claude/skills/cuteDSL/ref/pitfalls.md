# CuTeDSL Common Pitfalls & Gotchas

## 1. Compile-time vs runtime confusion

**Pitfall**: `print()` inside `@kernel`/`@jit` runs at compile time, not on the GPU.
```python
@cute.kernel
def kernel(tensor):
    print(tensor)  # Prints layout info ONCE during compilation, not per-thread
    cute.printf("thread %d: val=%f\n", tidx, val)  # Runtime, per-thread
```

**Pitfall**: `range()` is runtime since v4.1. Use `cutlass.range_constexpr()` for compile-time unrolling.
```python
# Runtime loop (generates loop IR)
for i in range(N):
    ...

# Compile-time unrolled (N must be known at compile time)
for i in cutlass.range_constexpr(N):
    ...
```

## 2. Layout and tiling mistakes

**Pitfall**: Thread-Value layout order matters for coalescing.
```python
# BAD: threads don't access contiguous memory
thr_layout = cute.make_ordered_layout((32, 4), order=(0, 1))
# GOOD: innermost thread dimension maps to contiguous memory
thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
```

**Pitfall**: `zipped_divide` tiler must match tensor's logical shape.
```python
# If mC has shape (M, N), tiler must divide (M, N)
tiler = (tile_m, tile_n)  # Must divide M and N
gC = cute.zipped_divide(mC, tiler)  # Shape: ((tile_m, tile_n), (M/tile_m, N/tile_n))
```

**Pitfall**: `local_tile` proj tuple length must match tiler tuple length.
```python
# tiler has 3 modes (bM, bN, bK)
gA = cute.local_tile(mA, tiler=(bM, bN, bK), coord=(bx, by, 0), proj=(1, None, 1))
# proj=(1, None, 1) means: take mode 0 (M), skip mode 1 (N), take mode 2 (K)
```

## 3. Predication bugs

**Pitfall**: Using tiled shape instead of original shape for boundary check.
```python
# WRONG: checks against tile size, always true
frgPred[i] = cute.elem_less(thrCrd[i], (tile_m, tile_n))

# RIGHT: check against original problem dimensions
frgPred[i] = cute.elem_less(thrCrd[i], (M, N))
```

**Pitfall**: Forgetting predication entirely when problem size isn't tile-aligned.
- Symptoms: wrong values in last row/column blocks, or illegal memory access.

## 4. Copy and fragment mismatches

**Pitfall**: Copy atom element type must match tensor element type.
```python
# WRONG: atom says Float16 but tensor is Float32
copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
cute.copy(tiled_copy, src_f32_tensor, dst)  # Type mismatch error

# RIGHT: match types
copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
```

**Pitfall**: Using `partition_S` source with `partition_D` destination from different `ThrCopy` slices.
```python
# WRONG: different slices
thr_copy_a = tiled_copy.get_slice(tidx)
thr_copy_b = tiled_copy.get_slice(tidx)  # different object!
src = thr_copy_a.partition_S(data)
dst = thr_copy_b.partition_D(frag)  # Mismatch!

# RIGHT: same slice
thr_copy = tiled_copy.get_slice(tidx)
src = thr_copy.partition_S(data)
dst = thr_copy.partition_D(frag)
```

## 5. GEMM-specific pitfalls

**Pitfall**: Forgetting to initialize accumulator.
```python
tCrC = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)  # MUST initialize! Contains garbage otherwise.
```

**Pitfall**: Wrong K-loop bound.
```python
# RIGHT: iterate based on partitioned K dimension
for k in range(cute.size(tCsA, mode=2)):
    cute.gemm(tiled_mma, tCrC, tCrA[..., k], tCrB[..., k], tCrC)
```

**Pitfall**: Missing sync between shared memory write and read.
```python
cute.copy(tiled_copy, gA, sA)  # Write to smem
cute.arch.sync_threads()        # MUST sync before reading
cute.gemm(tiled_mma, tCrC, tCsA, tCsB, tCrC)  # Read from smem
```

## 6. Shared memory issues

**Pitfall**: Exceeding shared memory limit without opt-in.
- Default limit: 48KB
- Hopper max: 228KB (requires `cudaFuncSetAttribute`)
- `SmemAllocator` auto-requests but check device capability

**Pitfall**: Missing alignment for shared memory tensors.
```python
# TMA requires 128-byte aligned smem
sA = alloc.allocate_tensor(dtype, layout, byte_alignment=128)
```

## 7. Async copy pitfalls (Ampere cpasync)

**Pitfall**: Missing `cp_async_commit_group()` after async copies.
```python
cute.copy(tiled_copy, src, dst)  # Enqueues async copy
cute.arch.cp_async_commit_group()  # MUST commit!
cute.arch.cp_async_wait_group(0)   # Wait for all
cute.arch.sync_threads()
```

**Pitfall**: Wrong wait group count in pipeline.
```python
# If num_stages=3, wait until at most (num_stages-2)=1 group pending
cute.arch.cp_async_wait_group(num_stages - 2)
```

## 8. TMA pitfalls (Hopper/Blackwell)

**Pitfall**: TMA descriptor not prefetched.
```python
cpasync.prefetch_descriptor(tma_atom)  # Do this before first TMA load
```

**Pitfall**: Mbarrier expect_tx byte count must exactly match TMA transfer size.
- Too small: hangs (barrier never completes)
- Too large: hangs (barrier never completes)

**Pitfall**: Only one thread per CTA should issue TMA loads (typically thread 0).
```python
with cute.arch.elect_one():
    cpasync.copy(tma_tiled_copy, src, dst)
```

## 9. DLPack / tensor interop

**Pitfall**: Forgetting `.mark_layout_dynamic()` — causes recompilation on every call with different strides.
```python
mA = from_dlpack(torch_tensor).mark_layout_dynamic()
```

**Pitfall**: PyTorch tensor must be contiguous or have known strides.
```python
# If torch tensor is non-contiguous, call .contiguous() first
a = a.contiguous()
mA = from_dlpack(a)
```

## 10. Control flow in kernels

**Pitfall**: Python `if` on dynamic values generates runtime branches (since v4.1). Use `cutlass.const_expr()` for compile-time branching.
```python
# Runtime branch (generates both paths in IR)
if condition:
    ...

# Compile-time branch (only selected path in IR)
if cutlass.const_expr(TILE_SIZE > 64):
    ...
```

**Pitfall**: Cannot use Python `break`/`continue` in DSL loops — use control flow primitives or restructure.
