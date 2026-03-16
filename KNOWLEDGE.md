# Knowledge Base: Linear-Cross-Entropy Fusion in TransformerEngine

---

## 1. High-Level Purpose

Linear-Cross-Entropy (LCE) fusion fuses the `lm_head` linear projection with the cross-entropy loss to avoid materializing the full `[num_tokens, vocab_size]` logits tensor in GPU memory. The equivalent PyTorch reference is:

```python
def torch_lce(hidden, weight, labels):
    logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
    logprobs = torch.nn.functional.cross_entropy(logits, labels, ...)
    return logprobs
```

By streaming over the vocab dimension in tiles, memory usage drops from O(num_tokens × vocab_size) to O(num_tokens × vocab_per_split).

---

## 2. Code Structure and File Map

```
transformer_engine/
├── pytorch/
│   ├── linear_cross_entropy.py          # Public API: linear_cross_entropy() + LinearCrossEntropy autograd
│   └── cutedsl/
│       └── linear_cross_entropy_entry.py  # Host forward/backward orchestration
└── common/
    ├── triton/
    │   └── linear_cross_entropy.py        # Triton epilogue kernels
    └── cutedsl/linear_cross_entropy/
        ├── __init__.py                    # re-exports blackwell, scheduler
        ├── utils.py                       # EntropyReductionEnum, BackwardMethodEnum
        ├── scheduler.py                   # StaticPersistentScheduler
        ├── ptx.py                         # Inline PTX: fma.rn.ftz.f32
        └── blackwell/
            ├── __init__.py               # exports FwdMainLoop, BwdPartialDlogits, BwdDHiddenDWeight
            ├── fwd_mainloop.py           # SM100 forward kernel: GEMM + online softmax epilogue
            ├── bwd_partial_dlogits.py    # SM100 backward: partial d_logits kernel
            └── bwd_dHdW.py              # SM100 fused d_hidden+d_weight kernel (WIP/not yet used)
```

---

## 3. Public API Layer

**File:** `transformer_engine/pytorch/linear_cross_entropy.py`

### Function Signature
```python
linear_cross_entropy(
    hidden: Tensor,           # (num_tokens, dim) or (batch, seqlen, dim) — FP16/BF16
    weight: Tensor,           # (local_vocab_size, dim) — FP16/BF16
    labels: Tensor,           # (num_tokens,) or (batch, seqlen) — int64
    tp_group: Optional[ProcessGroup] = None,
    reduction: Literal["none", "sum", "mean"] = "mean",
    ignore_index: int = -100,
    sequence_parallel: bool = False,
) -> Tensor                   # () scalar or (num_tokens,) — FP32
```

### Architecture Dispatch
`Implementation` is a lazy-initialized singleton. It checks GPU compute capability:
- `cc[0] == 10` (Blackwell/SM100) → uses `cutedsl` implementation
- Other architectures → raises ValueError (Triton path mentioned in comment but not implemented)

### Autograd Function: `LinearCrossEntropy`
`forward` saves `(global_hidden, weight, labels, accumulate, num_valid_tokens)` for backward.
Returns `(logprobs, accumulate, num_valid_tokens, tp_rank, tp_world_size, global_hidden)` from the underlying impl.

---

## 4. Tensor Shapes and Data Types

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `hidden` (input) | `(num_tokens, dim)` | FP16 or BF16 | Flattened internally |
| `weight` (LM head) | `(local_vocab_size, dim)` | FP16 or BF16 | Row-major; `local_vocab_size = global_vocab_size // tp_world_size` in TP |
| `labels` (input) | `(num_tokens,)` | int64 | Values in `[0, local_vocab_size)` relative to TP rank offset |
| `logprobs` (output, `reduction="none"`) | `(num_tokens,)` | FP32 | Per-token NLL |
| `logprobs` (output, `reduction="sum/mean"`) | scalar `()` | FP32 | Aggregated loss |
| `_max` (intermediate) | `(num_tokens, num_splits)` | FP32 | Per-split running max |
| `_accu` (intermediate) | `(num_tokens, num_splits)` | FP32 | Per-split `sum(exp(logit - max))` |
| `accumulate` (saved for bwd) | `(num_tokens,)` | FP32 | Converted to LSE before saving |
| `maximum` (temporary) | `(num_tokens,)` | FP32 | Global max across all splits |
| `num_valid_tokens` (scalar) | `()` | int64 | Count of tokens where `label != ignore_index` |
| `d_hidden` (gradient) | same shape as `hidden` | FP32 (→ cast to hidden dtype) | |
| `d_weight` (gradient) | same shape as `weight` | same as weight dtype | |
| `_d_logits_partial` (bwd temp) | `(num_tokens, vocab_per_split)` | same as hidden | Reused per split |

### Key Constraint
- `dim` must be 128-byte aligned (i.e., `dim * sizeof(dtype)` % 128 == 0) for Blackwell TMA.
- `hidden` and `weight` must have the same dtype.
- All input tensors must be contiguous and on the same CUDA device.

---

## 5. Vocab Splitting Strategy

Both forward and backward split the vocab dimension into `num_splits` tiles:
```python
# Forward
vocab_per_split = int(os.environ.get("LCE_FWD_VOCAB_SPLIT_SIZE", 512 * 6))  # default 3072
num_splits = ceil_div(vocab_size, vocab_per_split)

# Backward
vocab_per_split = int(os.environ.get("LCE_BWD_VOCAB_SPLIT_SIZE", 512 * 6))  # default 3072
num_splits = ceil_div(vocab_size, vocab_per_split)
```

Intermediate tensors `_max` and `_accu` have shape `(num_tokens, num_splits)` to store per-split statistics before reduction.

---

## 6. Forward Algorithm (Numerically Stable Online Softmax)

### Phase 1: FwdMainLoop cuteDSL kernel
For each split `s` of vocab:
1. **GEMM**: `logits[m, s*V:(s+1)*V] = hidden[m, :] @ weight[s*V:(s+1)*V, :].T`
2. **Online Softmax Epilogue** (in TMEM→register pipeline):
   - Load logit tile from TMEM
   - Update running max: `_max_old = _max; _max = fmax(_max, logit)`
   - Update accumulate: `_accu = exp(_max_old - _max) * _accu + exp(logit - _max)`
   - If `label` falls in this vocab range: accumulate `logit_at_label`
3. **Write** `_max[token, split]`, `_accu[token, split]`, `_logprobs[token]` to GMEM

### Phase 2: Triton Epilogue (`forward_dp_epilogue` for DP, `forward_tp_epilogue` + `forward_tp_epilogue_update_logprobs` for TP)
- Reduce `_max[token, :]` and `_accu[token, :]` across all splits using streaming max-accumulate
- Convert `accumulate` to LSE: `LSE = log(global_accu) + global_max`
- Compute `logprobs[token] = logit_at_label - LSE`
- Apply `ignore_index` mask: set logprob to 0 for ignored tokens
- Apply reduction (sum/mean using `num_valid_tokens`)

### TP Mode Epilogue
In TP mode, each rank only holds a shard of the vocab. The reductions happen across ranks:
1. `all_reduce(_max, MAX)` — global max across vocab shards
2. On a dedicated CUDA stream: `all_reduce(_logprobs, SUM)` — sum logits at label positions (only one rank contributes a non-zero value per token)
3. After all_reduce: `forward_tp_epilogue` converts per-split accu using globally-reduced max
4. `all_reduce(accumulate, SUM)` — sum the exp-accumulators
5. `forward_tp_epilogue_update_logprobs` finishes the LSE computation

---

## 7. Backward Algorithm

### Current Implementation: `kDlogitsSplitN`
For each vocab split `s` (sequential loop):
1. **`BwdPartialDlogits` kernel** computes `d_logits_partial[num_tokens, vocab_per_split]`:
   ```
   softmax[m, v] = exp(logits[m, s*V+v] - LSE[m])
   d_logits[m, v] = dlogprobs[m] * (softmax[m, v] - one_hot(label[m] == s*V+v))
   ```
   - Uses same GEMM structure as forward (hidden × weight.T)
   - `accu` (which is LSE after forward epilogue) is loaded to compute softmax
   - `dlogprobs` is scaled by `1/num_valid_tokens` if reduction == mean

2. **`cublas.addmm`** accumulates `d_hidden`:
   ```
   d_hidden += d_logits_partial @ weight[s*V:(s+1)*V, :]   (FP32 accumulation)
   ```
   - `beta=0` for first split, `beta=1` for subsequent splits

3. **`torch.matmul`** computes `d_weight`:
   ```
   d_weight[s*V:(s+1)*V, :] = d_logits_partial.T @ hidden
   ```

### Future Implementation: `kFused` (`BwdDHiddenDWeight`)
- Status: **WIP, not yet called in the entry point**
- Fuses steps 1+2+3 into a single kernel pass
- Uses persistent scheduler (`StaticPersistentScheduler`) for load-balancing across SMs
- More complex: 16-warp CTA layout with softmax warps, epilog warps, load/mma/store warps
- Uses TMEM for: logits accumulator, d_hidden accumulator, d_weight accumulator, p (probability) buffer
- Uses TMA reduce (ADD) for writing d_hidden and d_weight back to GMEM

---

## 8. Blackwell (SM100) Kernel Architecture

### CTA (Thread Block) Layout — 8 Warps
All SM100 kernels (FwdMainLoop, BwdPartialDlogits) use the same layout:
- **Warps 0-3** (warpgroup): Epilogue — reads from TMEM, computes softmax / d_logits, writes to GMEM
- **Warp 4**: Loader — issues TMA G2S copies for `hidden` and `weight`
- **Warp 5**: MMA — issues tcgen05 GEMM instructions, reads from SMEM, writes to TMEM
- **Warps 6-7**: Empty — reduce register pressure via `warpgroup_reg_dealloc`

### CTA Layout — 16 Warps (BwdDHiddenDWeight only)
- **Warps 0-3**: Softmax warpgroup (num_regs=192)
- **Warps 4-7**: Epilog warpgroup (num_regs=80)
- **Warp 8**: Load (num_regs=32)
- **Warp 9**: MMA (num_regs=64)
- **Warp 10**: Store (num_regs=32)
- **Warps 11-15**: Empty (num_regs=24)

### Pipeline Stages
- **AB pipeline** (`PipelineTmaUmma`): 4 stages, producer=load warp, consumer=MMA warp
  - TMA copies one tile of A and B per stage (both share the same pipeline/barrier)
- **MMA pipeline** (`PipelineUmmaAsync`): 2 stages (FwdMainLoop) or 1 stage (BwdPartialDlogits)
  - Producer=MMA warp (fills TMEM), consumer=epilogue warpgroup

### TMEM (Tensor Memory) Usage
TMEM is a fast on-chip memory exclusive to SM100, capacity = 512 columns × 128 rows.
- **FwdMainLoop**: `tmem_alloc_cols = num_acc_stage * mma_tiler[1] = 2 * 256 = 512` (full capacity)
- **BwdPartialDlogits**: `tmem_alloc_cols = 1 * 256 = 256`
- **BwdDHiddenDWeight**: uses multiple TMEM regions: logits, dH, dW, p (all together, next power-of-2 rounded)

### TMA (Tensor Memory Accelerator)
- `make_tiled_tma_atom_A/B` creates TMA descriptors for bulk async copy G→S
- `cpasync.CopyBulkTensorTileG2SOp` for loading (G2S)
- `cpasync.CopyReduceBulkTensorTileS2GOp(ADD)` for atomic-add write-back in bwd (S2G with reduction)
- Alignment requirements: 16B minimum (`assumed_align`), 128B preferred for performance

### Key API Patterns

**Kernel compilation and caching:**
```python
# Compile once per (vocab_size, dim, dtype) combination
key = f"vocab_size:{vocab_size}+dim:{dim}+dtype:{hidden_view.dtype}"
if cache.get(key) is None:
    kernel = MyKernel(...)
    compiled = cute.compile(kernel, *sample_tensors, ...)
    cache[key] = compiled
compiled = cache[key]
compiled(*runtime_tensors, ...)  # re-use with different num_tokens
```

**Dynamic shape marking:**
```python
tensor_packed = from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0)
# mode=0: leading dimension (num_tokens) is dynamic; others are static
```

**SMEM struct definition:**
```python
@cute.struct
class SharedStorage:
    load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_stages * 2]
    sA: cute.struct.Align[cute.struct.MemRange[dtype, size], align_bytes]
    sB: cute.struct.Align[cute.struct.MemRange[dtype, size], align_bytes]
```

---

## 9. Scheduler: `StaticPersistentScheduler`

**File:** `transformer_engine/common/cutedsl/linear_cross_entropy/scheduler.py`

A persistent (wave-style) tile scheduler:
- Grid size = min(SM_count × occupancy, total_tiles)
- Each CTA processes one tile, then advances by `grid_dim.x` until all tiles are done
- `WorkTileInfo.tile_idx` is 1D; `divmod` by `num_tiles_N` gives `(m_block, n_block)`
- Used by `BwdDHiddenDWeight` to amortize launch overhead across many vocab tiles

---

## 10. Distributed (TP/SP) Semantics

### Tensor Parallel (TP), `sequence_parallel=False`
- `hidden` and `labels`: identical on all TP ranks (replicated)
- `weight`: sharded along vocab dim → each rank holds `(local_vocab_size, dim)`
- `logprobs`: identical on all ranks after all-reduces

### Sequence Parallel (SP), `sequence_parallel=True`
- `hidden`: sharded along token dim → each rank holds `(local_num_tokens, dim)`
- An `all_gather_into_tensor` is done in forward to assemble `global_hidden` before GEMM
- `labels`: replicated (full length on all ranks)
- Backward: `d_hidden` is all-reduced across ranks, then sliced for local shard

### Key `ignore_index` Mechanic
In TP/SP mode, `_logprobs` is zero-initialized and filled atomically. Only the rank whose vocab shard contains the label contributes a non-zero value. The `all_reduce(SUM)` then gathers the true logit.

---

## 11. Utility Details

### `ptx.py`
Provides a single inline PTX function:
```python
fma(a, b, c) -> Float32  # fma.rn.ftz.f32 $0, $1, $2, $3  (round-to-nearest, flush-to-zero)
```
Used for numerically precise fused multiply-add in softmax computations.

### `utils.py`
```python
class EntropyReductionEnum(Enum):
    kNone = 0; kSum = 1; kMean = 2

class BackwardMethodEnum(Enum):
    kTwoKernels = 0   # (not used) separate kernels for d_H and d_W
    kDlogitsSplitN = 1  # (current default) partial d_logits then 2 GEMMs
    kFused = 2          # (future) single fused kernel
```

---

## 12. Test File API Reference

**File:** `tests/pytorch/test_linear_cross_entropy.py`

### Test Problem Sizes
```python
# (num_tokens_or_shape, vocab_size, dim)
(80, 125, 64)
(80, 152064, 64)
(1024, 152064, 4096)
(4096, 152063, 8192)
((1, 4096), 152064, 8192)   # 3D hidden: (batch=1, seqlen=4096, dim=8192)
((2, 4096), 152064, 8192)   # 3D hidden: (batch=2, seqlen=4096, dim=8192)
```

### Performance Benchmark Problem
```python
((1, 4096), 129280, 7168)   # DeepSeek-style: 4096 tokens, 129280 vocab, 7168 dim
```

### Correctness Tolerance
```python
torch.testing.assert_close(d_torch_hidden, d_custom_hidden, atol=1e-3, rtol=1e-3)
torch.testing.assert_close(d_torch_weight, d_custom_weight, atol=1e-3, rtol=1e-3)
```

### Test Classes
- `TestFusedLinearCrossEntropyDataParallel` — single GPU, requires `cc == 10`
- `TestFusedLinearCrossEntropyTensorParallel` — multi-GPU with `torchrun`, requires `WORLD_SIZE >= 2`

---

## 13. Lessons Learned

### L1: `accumulate` is LSE, not raw sum
The tensor named `accumulate` (and `_accu`) that is saved for backward is converted to LSE (`log(sum_exp) + max`) **by the Triton epilogue** before being stored. The backward kernel `BwdPartialDlogits` directly uses it as `exp(logit - LSE)` for the softmax computation. Any future kernel must respect this convention.

### L2: `_logprobs` in TP mode stores the label logit, not the cross entropy
In TP forward, `_logprobs[token]` accumulates `logit_at_label` across ranks via `all_reduce(SUM)`. The actual NLL is computed as `LSE - logit_at_label` in the update_logprobs Triton kernel. This two-step design avoids a second pass over the vocab.

### L3: Kernel JIT specialization on vocab_size and dim
The cuteDSL JIT specializes on `(vocab_size, dim, dtype)` — these become compile-time constants. Only `num_tokens` varies at runtime. This means the first call for a new model shape will incur JIT compilation latency.

### L4: `BwdDHiddenDWeight` is WIP and not connected
`bwd_dHdW.py` contains a more advanced fused backward kernel, but the entry point (`linear_cross_entropy_entry.py`) still uses `kDlogitsSplitN` (the split-N approach with separate cuBLAS GEMM calls). The fused path would eliminate the temporary `_d_logits_partial` tensor.

### L5: Two-stream pattern for TP forward
The TP forward uses a dedicated CUDA stream to overlap `all_reduce(_logprobs)` with the `forward_tp_epilogue` kernel that processes max/accumulate. Events synchronize the two streams before `forward_tp_epilogue_update_logprobs`.
