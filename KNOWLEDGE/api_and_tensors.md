# Public API, Tensor Shapes, and Vocab Splitting

## Public API Layer

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

## Tensor Shapes and Data Types

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

### Key Constraints
- `dim` must be 128-byte aligned (i.e., `dim * sizeof(dtype)` % 128 == 0) for Blackwell TMA.
- `hidden` and `weight` must have the same dtype.
- All input tensors must be contiguous and on the same CUDA device.

---

## Vocab Splitting Strategy

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
