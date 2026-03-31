# Mode: Debug a CuTeDSL Kernel

## Diagnostic workflow

### Step 1: Classify the problem

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| Compilation error / traceback in MLIR | Type mismatch, bad layout, wrong API usage | Step 2 |
| Kernel launches but produces wrong results | Predication bug, layout mismatch, race condition | Step 3 |
| Kernel hangs or deadlocks | Missing sync, barrier mismatch, TMA not arriving | Step 4 |
| Performance is poor | Wrong copy strategy, bank conflicts, low occupancy | Load [modes/optimize.md](optimize.md) |
| `cuda illegal memory access` | OOB access, missing predication, wrong pointer | Step 5 |

### Step 2: Compilation errors

**Common compilation issues:**

1. **"Type mismatch"** — Check that tensor element types match copy/MMA atom types
   ```python
   # Wrong: atom expects Float16 but tensor is Float32
   copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
   cute.copy(tiled_copy, src_f32, dst)  # ERROR
   ```

2. **"Layout rank mismatch"** — Partition/divide operations require matching ranks
   - `zipped_divide` tiler must match tensor's outermost modes
   - `local_tile` proj tuple must have same length as tiler

3. **"Cannot call @kernel from Python"** — Kernels must be launched from `@jit` functions
   ```python
   # Wrong: calling kernel directly
   my_kernel(args)
   # Right: launch from @jit
   @cute.jit
   def host(args):
       my_kernel(args).launch(grid=..., block=...)
   ```

4. **"Constexpr expected"** — Some parameters must be compile-time constants
   ```python
   # Fix: add cutlass.Constexpr annotation
   @cute.jit
   def fn(mA, tile_size: cutlass.Constexpr = 128):
   ```

5. **Unexpected compile-time prints** — `print()` in @kernel/@jit is compile-time
   ```python
   # This prints layout info at compile time, not runtime values
   print(tensor)  # compile-time
   cute.printf("val=%f\n", val)  # runtime
   ```

### Step 3: Wrong results

**Debugging strategy:**

1. **Add compile-time prints** to verify layouts:
   ```python
   @cute.kernel
   def kernel(...):
       print(sA)        # prints layout shape/stride at compile time
       print(tiled_copy) # prints copy atom configuration
   ```

2. **Add runtime prints** for small problem sizes:
   ```python
   if tidx == 0 and bidx == 0:
       cute.printf("val[0] = %f\n", frag[0])
   ```

3. **Check predication logic**:
   ```python
   # Common bug: predicate tensor shape doesn't match fragment shape
   assert cute.size(frgPred) == cute.size(frgData)

   # Common bug: using wrong shape for elem_less
   frgPred[i] = cute.elem_less(thrCrd[i], mC.shape)  # use ORIGINAL shape, not tiled
   ```

4. **Check layout consistency**:
   - Source and destination of `cute.copy()` must have compatible layouts
   - `partition_S` and `partition_D` must use the same `ThrCopy`
   - MMA partitions A/B/C must come from the same `ThrMma`

5. **Check accumulator initialization**:
   ```python
   tCrC.fill(0.0)  # Don't forget this!
   ```

6. **Check GEMM loop bounds**:
   ```python
   # K-loop must iterate over the correct number of tiles
   for k in range(cute.size(tCsA, mode=2)):  # iterate over K-tiles
       cute.gemm(tiled_mma, tCrC, tCrA[..., k], tCrB[..., k], tCrC)
   ```

### Step 4: Hangs / deadlocks

1. **Missing `sync_threads()`** after shared memory write before read
2. **Barrier count mismatch** — `mbarrier_arrive_and_expect_tx` byte count must match actual bytes copied
3. **Pipeline phase mismatch** — consumer waiting on wrong pipeline stage
4. **Cluster sync issues** — all CTAs in cluster must participate in cluster barriers
5. **Cooperative launch** — all blocks must reach `grid_sync()` if using `cooperative=True`

### Step 5: Illegal memory access

1. **Missing predication** for boundary tiles:
   ```python
   # When M or N is not divisible by tile size, must predicate
   frgPred[i] = cute.elem_less(thrCrd[i], (M, N))
   cute.copy(atom, src, dst, pred=frgPred)
   ```

2. **Wrong grid size** calculation:
   ```python
   grid_m = (M + TILE_M - 1) // TILE_M
   grid_n = (N + TILE_N - 1) // TILE_N
   ```

3. **Shared memory overflow** — verify `SmemAllocator` total doesn't exceed device limit (usually 48KB default, up to 228KB with opt-in on Hopper/Blackwell)

4. **TMA descriptor mismatch** — TMA tensor shape/stride must match actual allocation

## Quick reference: diagnostic tools

```python
# Compile-time: print layout info
print(tensor)          # shape, stride, size
print(layout)          # layout structure
print(tiled_copy)      # copy atom config
print(tiled_mma)       # MMA atom config

# Runtime: print values
cute.printf("thread %d: val = %f\n", tidx, value)

# Dump PTX for inspection
compiled = cute.compile[cute.KeepPTX](host_fn, *args)
print(compiled.__ptx__)

# Dump MLIR IR
compiled = cute.compile(host_fn, *args)
print(compiled.__mlir__)
```
