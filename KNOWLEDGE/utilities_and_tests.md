# Utilities and Test Reference

## Utility Details

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

## Test File API Reference

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
