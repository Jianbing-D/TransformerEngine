# Mode: Write a CuTeDSL Kernel

## Workflow

1. **Clarify requirements** — Ask if unclear:
   - What operation? (elementwise, reduction, GEMM, attention, custom)
   - Target GPU architecture? (Ampere SM80, Hopper SM90, Blackwell SM100)
   - Data types? (FP32, FP16, BF16, FP8, INT8)
   - Problem dimensions and any dynamic shape requirements
   - Performance target or just correctness first?

2. **Choose the right pattern** — Load the relevant reference:
   - Elementwise ops: use [ref/api-core.md](../ref/api-core.md) for the copy-compute-store pattern
   - GEMM: use [ref/gemm-patterns.md](../ref/gemm-patterns.md) for architecture-specific patterns
   - Architecture-specific features: use [ref/arch-features.md](../ref/arch-features.md)

3. **Write the kernel** following this structure:

```python
# 1. Imports
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# 2. Kernel function (device code)
@cute.kernel
def my_kernel(tensor_a: cute.Tensor, tensor_b: cute.Tensor, ...):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # ... kernel body ...

# 3. Host orchestration (tiling, launch)
@cute.jit
def host_fn(mA, mB, mC, ...):
    # Tiling setup
    # Copy atom creation
    # Kernel launch
    my_kernel(args...).launch(grid=[...], block=[...])

# 4. Entry point
def run(a_torch, b_torch, c_torch, stream=None):
    mA = from_dlpack(a_torch).mark_layout_dynamic()
    mB = from_dlpack(b_torch).mark_layout_dynamic()
    mC = from_dlpack(c_torch).mark_layout_dynamic()
    compiled = cute.compile(host_fn, mA, mB, mC, stream=stream)
    compiled(mA, mB, mC, stream=stream)
```

## Kernel structure checklist

- [ ] **Imports**: Only import what you need
- [ ] **@cute.kernel**: Device code with thread/block indexing
- [ ] **@cute.jit**: Host-side tiling and launch logic
- [ ] **Tensor creation**: `from_dlpack()` with `.mark_layout_dynamic()` if shapes vary
- [ ] **Layout/tiling**: `make_layout`, `make_ordered_layout`, `make_layout_tv`, `zipped_divide`, `local_tile`
- [ ] **Copy atoms**: `make_copy_atom` + `make_tiled_copy_tv` for the right copy strategy
- [ ] **Predication**: `make_identity_tensor` + `elem_less` for boundary handling
- [ ] **Shared memory**: `SmemAllocator` if needed
- [ ] **Synchronization**: `sync_threads()`, barriers as needed
- [ ] **Launch**: Correct grid/block dims, smem size, optional cluster/stream

## Pattern selection guide

| Operation | Architecture | Key Pattern |
|-----------|-------------|-------------|
| Elementwise | Any | TV layout -> zipped_divide -> copy-compute-store |
| GEMM (simple) | Ampere | SIMT MMA + cpasync shared memory pipeline |
| GEMM (tensor core) | Ampere | MmaAtom(FP16) + smem double buffering |
| GEMM | Hopper | TMA loads + wgmma + persistent kernel |
| GEMM | Blackwell | TMA + tcgen05.mma + TMEM + 2-CTA |
| GEMM | Rubin | Extends Blackwell + B-keep/B-reuse + K=64 |
| GEMM | Feynman | SPMEM + DPC TMA + tcgen06 + temporal split-K |
| Block-scaled GEMM | Blackwell+ | MmaMXF4NVF4Op + scale factors in TMEM |
| Sparse GEMM | Blackwell | 3 TMA loads (compressed A, B, metadata E) |
| Sparse GEMM | Rubin | LDTM.SPARSIFY hardware 2:4 sparsification |
| Grouped GEMM (MoE) | Blackwell+ | Persistent + scheduler warp + torch integration |
| GEMM + epilogue | Blackwell+ | EFC framework or lambda epilogue_op |
| B2B GEMM | Blackwell | Two chained MMAs, TMEM as A-source for MMA1 |
| GEMV | Blackwell | Manual per-thread accumulation (no MMA) |
| Reduction | Any | Warp reduction -> smem -> block reduction |
| Attention (prefill) | Ampere | FlashAttention-v2 with online softmax |
| Attention (prefill) | Blackwell | TMA + persistent + tcgen05 |
| Attention (decode) | Blackwell | FMHA decode with GQA, 3-warpgroup split |
| Attention (paged KV) | Blackwell | Page table indirection + BLASST sparse skip |
| Mixed-input attention | Blackwell | Quantized KV (FP8/FP4) + FP16 Q |
| Normalization | Blackwell | Cluster reduction via distributed smem |
| Conv (fprop) | Blackwell | Implicit GEMM with TMA im2col |
| Gated dual GEMM | Blackwell | silu(A@B1) * (A@B2), fused activation |
| MLP (torch) | Any | Custom autograd.Function wrapping CuTeDSL kernel |
| Top-K selection | Blackwell | Radix-based filtered top-k |

## Class-based organization (recommended for complex kernels)

```python
class MyKernel:
    def __init__(self, tile_m=128, tile_n=128, tile_k=32, num_threads=256):
        self.tile_m = tile_m
        # ...

    @cute.jit
    def __call__(self, mA, mB, mC, stream=None):
        # Setup and launch
        self.kernel(args...).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, ...):
        # Device code
        ...
```
