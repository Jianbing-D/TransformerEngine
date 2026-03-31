# CuTeDSL Advanced Kernel Patterns

## Sparse GEMM (2:4 structured sparsity)

### Blackwell (SM100)
Three TMA loads: compressed A, dense B, metadata E.
```python
class SparsePersistentGemmKernel:
    # A is sparse_elem<2, dtype> — 2:4 compressed
    # E is sparse_elem<8, uint8> — metadata
    # B is dense
    # Warp spec: epilogue warps 0-3, MMA warp 4, TMA warp 5
    # UTCCP instruction copies metadata from SMEM -> TMEM
```

### Rubin (SM107) — Hardware sparsification via LDTM.SPARSIFY
```python
# Data flow: GMEM -> RMEM -> TMEM (STTM) -> RMEM compressed (LDTM.SPARSIFY)
store_atom = tcgen05.St32x32bOp(repeat)                        # RMEM -> TMEM
sparse_atom = tcgen05.LdSPCompress32x32bOp(repeat,
    redOp=tcgen05.TmemLoadRedOp.MAXABS)                        # TMEM -> RMEM (sparse)
sparse_type = cute.get_sparse_elem_type(num_logical, num_phys, elem_type)
```

### Feynman (SM140) — tcgen06 native sparse
```python
op = tcgen06.MmaSparseF8Op(dpc_shape_mnk, mma_shape_mnk, dtypes,
    sparse_metadata_format=tcgen06.SparseMetadataFormat.tid)
# Dedicated SPMEM->TMEM copy ops for sparse data and metadata
```

## Attention patterns

### FMHA decode (Blackwell)
```python
class FusedMultiHeadAttentionDecode:
    # Three warpgroup assignment: MMA+TMA, softmax, correction
    # Online softmax with warp-level reductions
    # Packed f32x2 math for efficiency
    # ReductionMode: Deterministic, Atomic, ClusterDeterministic
    # GQA support with register reconfiguration
```

### FMHA decode paged (KV-cache)
```python
class FusedMultiHeadAttentionDecodePaged:
    # Page table indirection for KV cache
    # BLASST: block-level attention sparse skipping
    # threshold_p for attention sparsity (skip low-attention tiles)
    # page_size must divide 128
```

### Mixed-input FMHA decode (quantized KV)
```python
class MixedInputFusedMultiHeadAttentionDecode:
    # Quantized KV (FP8/FP4) with FP16 queries
    # Per-block scale factors for K and V
    # Dedicated convert warpgroups for dequantization
```

## Gated dual GEMM (MoE gate fusion)
```python
# C = silu(A @ B1) * (A @ B2)
# NVFP4 block-scaled, two B matrices sharing layout
# silu activation fused as epilogue lambda
class Sm100BlockScaledDenseGatedDualGemmKernel:
    ...
```

## Gated Delta Net (linear attention)
Most complex kernel — 7 fused GEMMs in one kernel:
```python
# kk, qk, k*state, q*state, inverse, qkv, kv_update
# Recurrent state held in TMEM across chunks
# 12-warp assignment, 225.5KB SMEM + 256KB TMEM
# Device-side TMA descriptor updates via TensorMapManager
```

## Back-to-back GEMM (experimental)
```python
# D = epilogue(transform(A @ B0) @ B1)
# Two chained MMAs with TMEM as A-operand for second MMA
# tcgen05.OperandSource.TMEM for MMA1
# 12-warp specialization: TMA0, TMA1, MMA0, MMA1, transform, epilogue
```

## Mixed-input GEMM
```python
class MixedInputGemmKernel:
    # Narrow A (int8/uint8/int4) x wide B (fp16/bf16)
    # TransformMode.ConvertOnly: int8 -> bf16 direct
    # TransformMode.ConvertScale: int4 -> bf16 with per-element/block scale
    # Contiguous grouped variant with cumsum offsets
```

## MLP fusion (PyTorch autograd integration)
```python
class GemmReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        out = torch.empty(...)
        kernel = DenseGemmKernel(
            epilogue_op=lambda x: cute.where(x > 0, x, cute.full_like(x, 0))
        )
        kernel(from_dlpack(x), from_dlpack(w), from_dlpack(out))
        ctx.save_for_backward(x, w, out)
        return out
```

## Top-K / radix selection
```python
class FilteredTopKKernel:
    # Two-phase: coarse filter (FP16 histogram) + fine-grained refinement
    # Radix-256 histogram via shared memory
    # Vectorized 256-bit loads
    # Block-level parallel prefix sum via warp shuffle
```

## Grouped GEMM for MoE (torch integration)
```python
class GroupedGemmKernel:
    # Implements torch.nn.functional.grouped_mm interface
    # 7-warp specialization (epilogue 0-3, MMA 4, TMA 5, scheduler 6)
    # MoEStaticPersistentTileScheduler with token offset computation
    # Supports 2Dx3D (forward) and 2Dx2D (weight grad)
```

## Scaled grouped GEMM for MoE
```python
class ScaledGroupedGemmKernel:
    # Block-scaled (MXFP8/MXFP4/NVFP4) MoE grouped GEMM
    # SFA/SFB scale factors in TMEM
    # Workspace for TMA descriptors + padded scale offsets
```

## GEMV (matrix-vector multiply, NVFP4)
```python
class Sm100BlockScaledDenseGemvKernel:
    # Pure elementwise approach (no MMA atoms)
    # Manual scale factor layout conversion
    # Per-thread accumulation with FFMA in registers
    # For narrow-N problems where MMA is wasteful
```

## Implicit GEMM (convolution)
```python
class PersistentDenseGemmKernel:  # conv variant
    # TMA im2col during load — conv mapped to GEMM automatically
    # M=NxZxPxQ, N=K, K=TxRxSxC
    # Supports padding, stride, dilation
    # Both persistent and non-persistent variants
```

## Low-latency TGV GEMM
```python
class TgvGemmKernel:
    # Small tiles (CTA_M=64, CTA_N=8, CTA_K=128) for latency optimization
    # 8-warp specialization: DMA_A, DMA_B, MMA, 4x EPILOG
    # Direct TMEM->RMEM->GMEM epilogue (no TMA store)
    # PDL for kernel overlap
```

## Pointer-array batched GEMM (experimental)
```python
# Each batch has its own pointer (not contiguous)
# Per-batch pointer loading from Tensor of Int64
# Device-side TMA descriptor updates per batch
cute.make_ptr(dtype, address_as_int, mem_space=cute.AddressSpace.gmem)
```

## Lamport sync GEMM (Rubin)
```python
class SM107LamportGemmKernel:
    # KernelRole: PRODUCER writes output with -0.0 sentinel
    # KernelRole: CONSUMER speculatively loads via TMA, validates in HW
    # Reduced inter-kernel latency, per-tile readiness
    # Uses PDL for full overlap
```

## IKET instrumentation
```python
# For PIC-C profiling
cute.iket.mark("event_name", payload)            # point event
token = cute.iket.range_start("range_name")       # range start
cute.iket.range_end(token)                         # range end
cute.iket.range_push("name"); cute.iket.range_pop()  # stack-based
```
