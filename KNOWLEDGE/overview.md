# Overview: Linear-Cross-Entropy Fusion

## High-Level Purpose

Linear-Cross-Entropy (LCE) fusion fuses the `lm_head` linear projection with the cross-entropy loss to avoid materializing the full `[num_tokens, vocab_size]` logits tensor in GPU memory. The equivalent PyTorch reference is:

```python
def torch_lce(hidden, weight, labels):
    logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
    logprobs = torch.nn.functional.cross_entropy(logits, labels, ...)
    return logprobs
```

By streaming over the vocab dimension in tiles, memory usage drops from O(num_tokens × vocab_size) to O(num_tokens × vocab_per_split).

---

## Code Structure and File Map

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
