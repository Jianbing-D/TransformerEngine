# Mode: Optimize a CuTeDSL Kernel

## Optimization checklist (in priority order)

### 1. Memory access patterns

- **Vectorized loads/stores**: Use `copy_bits=128` (or 64) to maximize memory throughput
  ```python
  vector_size = copy_bits // dtype.width  # e.g., 128 // 16 = 8 for FP16
  val_layout = cute.make_ordered_layout((val_m, vector_size), order=(1, 0))
  ```

- **Coalesced access**: Ensure innermost mode of thread layout maps to contiguous memory
  ```python
  # Good: threads walk contiguously in memory (mode-1 innermost for row-major)
  thr_layout = cute.make_ordered_layout((thr_m, thr_n), order=(1, 0))
  ```

- **Shared memory bank conflicts**: Use swizzled layouts for smem
  ```python
  smem_layout = cute.composition(cute.Swizzle(bits, base, shift), base_layout)
  ```

### 2. Compute throughput

- **Use Tensor Cores** when possible (FP16/BF16/FP8/INT8 on Ampere+)
- **Hopper**: Use `wgmma` for warpgroup-level MMA (much higher throughput than warp-level)
- **Blackwell**: Use `tcgen05.mma` with TMEM for highest throughput

### 3. Latency hiding

- **Software pipelining** — overlap loads with compute:
  ```python
  # Multi-stage pipeline: load stage N+2 while computing stage N
  for stage in range(num_stages - 1):
      cute.copy(tiled_copy, src[..., stage], smem[..., stage])
  cute.arch.cp_async_commit_group()

  for k in range(K_tiles):
      cute.arch.cp_async_wait_group(num_stages - 2)
      cute.arch.sync_threads()
      # compute on current stage
      cute.gemm(...)
      # load next stage
      cute.copy(tiled_copy, src[..., next_stage], smem[..., next_stage])
  ```

- **TMA** (Hopper+) — offloads address calculation to hardware, frees SM cycles

### 4. Occupancy

- **Register pressure**: Fewer registers per thread = more concurrent warps
  - Use smaller tile sizes if register-bound
  - Check with `cute.compile[cute.KeepPTX]` and inspect register count in PTX
- **Shared memory**: Keep under architecture limit or opt-in to larger smem
  - Ampere: 48KB default, 164KB max
  - Hopper: 48KB default, 228KB max
  - Blackwell: varies by SM

### 5. Kernel-level optimizations

- **Persistent kernels** — avoid launch overhead for small problems:
  ```python
  # Use tile scheduler to dynamically assign tiles to CTAs
  scheduler = cutlass.utils.DynamicPersistentTileScheduler(...)
  ```

- **Cluster launch** (Hopper+) — enables distributed shared memory across CTAs

- **Programmatic Dependent Launch (PDL)** — overlap consecutive kernel launches:
  ```python
  kernel(args).launch(grid=grid, block=block, use_pdl=True)
  ```

### 6. Autotuning

```python
@cute.testing.autotune_jit(
    params_dict={
        "tile_m": [64, 128, 256],
        "tile_n": [64, 128, 256],
        "copy_bits": [64, 128],
        "num_stages": [2, 3, 4],
    },
    update_on_change=["M", "N", "K"],
    warmup_iterations=100,
    iterations=100,
)
@cute.jit
def gemm_kernel(mA, mB, mC, M, N, K,
                tile_m: cutlass.Constexpr = 128,
                tile_n: cutlass.Constexpr = 128,
                copy_bits: cutlass.Constexpr = 128,
                num_stages: cutlass.Constexpr = 3):
    ...
```

### 7. Benchmarking

```python
compiled = cute.compile(host_fn, *args, stream=stream)
avg_us = cute.testing.benchmark(
    compiled,
    workspace_generator=lambda: cute.testing.JitArguments(*generate_tensors()),
    workspace_count=10,
    warmup_iterations=5,
    iterations=100,
    stream=stream,
    use_cuda_graphs=True,
)
tflops = 2 * M * N * K / (avg_us * 1e-6) / 1e12
print(f"{avg_us:.1f} us, {tflops:.1f} TFLOPS")
```
