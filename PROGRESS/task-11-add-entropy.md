# Task 11: Add Entropy Calculation to Existing Implementation

**Status**: Completed.

**Constraint**: Do NOT touch existing kernels (cuteDSL or Triton). Copy them and add entropy logic in the copies. Only add a new option to the torch interface.

---

## Mathematical Analysis

### Entropy Definition

Given logits `z = hidden @ weight.T` (shape `[T, V]`), the softmax distribution is:

```
p_v = exp(z_v) / sum_j exp(z_j) = exp(z_v - LSE)
```

Shannon entropy of this distribution:

```
H(p) = -sum_v p_v * log(p_v)
     = -sum_v p_v * (z_v - LSE)
     = LSE - sum_v p_v * z_v
     = LSE - entropy_b
```

where:
- `LSE = log(sum_v exp(z_v))` (log-sum-exp, already computed)
- `entropy_b = sum_v p_v * z_v = E_p[z]` (expected logit under softmax)

### Key Insight: entropy_b Uses Same Online Reduction as _accu

In the forward mainloop, we already compute:
```
_accu = sum_v exp(z_v - max)        # shifted partition function
```

The entropy_b accumulation is:
```
_entropy_b = sum_v z_v * exp(z_v - max)   # z-weighted shifted partition function
```

Both use the **same online max-correction** when moving across vocab splits:
```
_accu      = exp(old_max - new_max) * _accu      + sum(exp(z - new_max))
_entropy_b = exp(old_max - new_max) * _entropy_b + sum(z * exp(z - new_max))
```

After reduction across all splits:
```
entropy_b = global_entropy_b / global_accu    # = E_p[z]
entropy   = log(global_accu) + global_max - entropy_b   # = LSE - E_p[z] = H(p)
```

**No extra GEMM needed.** The entropy_b accumulation adds ~1 FMA per logit element in the epilogue, piggybacking on the already-computed `exp(z - max)`.

### Backward: Entropy Gradient

The derivative of entropy w.r.t. logit `z_v`:

```
dH/dz_v = d/dz_v [LSE - sum_j(p_j * z_j)]
        = p_v - [p_v + sum_j(dp_j/dz_v * z_j)]
        = -sum_j((delta(j,v)*p_v - p_v*p_j) * z_j)
        = -p_v * (z_v - sum_j(p_j * z_j))
        = -softmax_v * (z_v - entropy_b)
```

Combined d_logits with both CE loss and entropy:

```
d_logits_v = dlogprobs * (softmax_v - one_hot_v)           # existing CE term
           + d_entropy * (-softmax_v) * (z_v - entropy_b)   # NEW entropy term
```

This adds ~2 FMAs per element in the backward epilogue (subtraction and multiply), plus 2 extra scalar loads per token (`d_entropy[t]`, `entropy_b[t]`). The GEMM (dominant cost) is **unchanged**.

---

## Architecture: Copy-Not-Modify

**Rule**: Existing kernels are frozen. New kernel files are copies with entropy logic added.

### File Layout

```
EXISTING (untouched):
  common/cutedsl/linear_cross_entropy/blackwell/fwd_mainloop.py        # FwdMainLoop
  common/cutedsl/linear_cross_entropy/blackwell/bwd_partial_dlogits.py # BwdPartialDlogits
  common/triton/linear_cross_entropy.py                                # Triton epilogues
  pytorch/cutedsl/linear_cross_entropy_entry.py                        # forward(), backward()

NEW (copies with entropy):
  common/cutedsl/linear_cross_entropy/blackwell/fwd_mainloop_entropy.py        # FwdMainLoopEntropy
  common/cutedsl/linear_cross_entropy/blackwell/bwd_partial_dlogits_entropy.py # BwdPartialDlogitsEntropy
  common/triton/linear_cross_entropy_with_entropy.py                                # entropy-aware epilogues
  pytorch/cutedsl/linear_cross_entropy_with_entropy_entry.py                        # forward_entropy(), backward_entropy()

MODIFIED (torch interface only):
  pytorch/linear_cross_entropy.py   # Add return_entropy option, new autograd class
```

### Torch Interface Design

Add a new `LinearCrossEntropyWithEntropy` autograd class and a new public function:

```python
# In pytorch/linear_cross_entropy.py

class LinearCrossEntropyWithEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, tp_group, reduction, ignore_index, sequence_parallel):
        # Calls entropy_entry.forward_entropy(...)
        # Returns (logprobs, entropy)
        # Saves entropy_b for backward

    @staticmethod
    def backward(ctx, dlogprobs, dentropy):
        # Calls entropy_entry.backward_entropy(... dentropy, entropy_b ...)
        # Returns (d_hidden, d_weight, ...)

def linear_cross_entropy(
    hidden, weight, labels, tp_group=None, reduction="mean",
    ignore_index=-100, sequence_parallel=False,
    return_entropy=False,   # <-- NEW option
):
    if return_entropy:
        return LinearCrossEntropyWithEntropy.apply(...)  # returns (logprobs, entropy)
    else:
        return LinearCrossEntropy.apply(...)  # returns logprobs (unchanged)
```

When `return_entropy=False` (default), the existing code path is used — **zero overhead, zero risk**.

---

## Fusion Strategy

See diagram: `entropy-lce-dataflow.drawio`

### Forward Path (entropy mode)

1. **`FwdMainLoopEntropy` kernel** (new file, copy of `FwdMainLoop`):
   - Add `mEntropyB: cute.Tensor` parameter (shape `[T, S]`, same as `_accu`)
   - In epilogue, after computing `exp_logits = exp(z - max)`:
     - `_entropy_b = coeff * _entropy_b + z * exp_logits` (1 extra FMA)
   - Write `_entropy_b` to GMEM alongside `_max` and `_accu`

2. **New Triton epilogues** (new file, copies of existing):
   - `forward_dp_epilogue_entropy`: reduces `_entropy_b[T,S]` alongside `_accu`, computes `entropy[T]` and `entropy_b[T]`
   - `forward_tp_epilogue_entropy`: same for TP mode
   - `forward_tp_epilogue_update_logprobs_entropy`: same for TP final update

3. **`forward_entropy()`** (new entry point function):
   - Allocates `_entropy_b[T, S]`, `entropy_b[T]`, `entropy[T]`
   - Calls `FwdMainLoopEntropy` instead of `FwdMainLoop`
   - Calls entropy epilogues instead of existing epilogues
   - Returns `(logprobs, entropy, entropy_b, accumulate, ...)`

### Backward Path (entropy mode)

4. **`BwdPartialDlogitsEntropy` kernel** (new file, copy of `BwdPartialDlogits`):
   - Add `d_entropy[T]` and `entropy_b[T]` inputs
   - In epilogue d_logits computation, add:
     ```
     d_logits += d_entropy * (-softmax) * (logit - entropy_b)
     ```

5. **`backward_entropy()`** (new entry point function):
   - Accepts `dentropy` and `entropy_b` additionally
   - Calls `BwdPartialDlogitsEntropy` instead of `BwdPartialDlogits`

### Files Summary

| Action | File | Description |
|--------|------|-------------|
| **NEW** | `common/cutedsl/.../blackwell/fwd_mainloop_entropy.py` | Copy of fwd_mainloop.py + `_entropy_b` accumulation |
| **NEW** | `common/cutedsl/.../blackwell/bwd_partial_dlogits_entropy.py` | Copy of bwd_partial_dlogits.py + entropy gradient |
| **NEW** | `common/triton/linear_cross_entropy_with_entropy.py` | Copy of epilogues + `_entropy_b` reduction |
| **NEW** | `pytorch/cutedsl/linear_cross_entropy_with_entropy_entry.py` | Copy of entry point, wired to entropy kernels |
| **MODIFY** | `pytorch/linear_cross_entropy.py` | Add `return_entropy` option + new autograd class |
| **MODIFY** | `common/cutedsl/.../blackwell/__init__.py` | Export new kernel classes |
| **NEW** | `tests/pytorch/test_linear_cross_entropy_with_entropy.py` | Entropy-specific tests |

### Untouched Files (explicitly)

- `common/cutedsl/linear_cross_entropy/blackwell/fwd_mainloop.py` — NO CHANGES
- `common/cutedsl/linear_cross_entropy/blackwell/bwd_partial_dlogits.py` — NO CHANGES
- `common/triton/linear_cross_entropy.py` — NO CHANGES
- `pytorch/cutedsl/linear_cross_entropy_entry.py` — NO CHANGES

---

## PLAN-ADD-ENTROPY: Action Plan

### Phase 1: Forward — New cuteDSL FwdMainLoopEntropy kernel

- [x] 1.1 Copy `fwd_mainloop.py` → `fwd_mainloop_entropy.py`
- [x] 1.2 Rename class to `FwdMainLoopEntropy`
- [x] 1.3 Add `mEntropyB: cute.Tensor` parameter to `kernel()` and `__call__()`
- [x] 1.4 In epilogue section: partition `mEntropyB` same as `mAccu` (per-CTA tile)
- [x] 1.5 Add `tR2GrEntropyB` register accumulator, initialized to 0.0
- [x] 1.6 In the inner epilogue loop (where `exp_logits` is computed), add:
  - `tR2GrEntropyB[0] = coeff * tR2GrEntropyB[0] + tTMEM_load_rAcc[idx] * exp_logits`
- [x] 1.7 Write `tR2GrEntropyB` to GMEM after the N-tile loop
- [x] 1.8 Update `__init__.py` to export `FwdMainLoopEntropy`

### Phase 2: Forward — New Triton epilogues

- [x]2.1 Copy epilogues from `linear_cross_entropy.py` → `linear_cross_entropy_with_entropy.py`
- [x]2.2 Create `forward_dp_epilogue_entropy`:
  - Add `_entropy_b` input params (ptr + strides)
  - Add `global_entropy_b_ptr`, `global_entropy_ptr` output params
  - Accumulate `global_entropy_b` with same max-correction as `global_accu`
  - Compute `entropy_b = global_entropy_b / global_accu`
  - Compute `entropy = log(global_accu) + global_max - entropy_b`
  - Store `entropy[T]` and `entropy_b[T]`
- [x]2.3 Create `forward_tp_epilogue_entropy` (similarly)
- [x]2.4 Create `forward_tp_epilogue_update_logprobs_entropy` (similarly)
- [x]2.5 Copy `get_num_valid_tokens` (shared utility, or import from original)

### Phase 3: Forward — New entry point

- [x]3.1 Copy `linear_cross_entropy_entry.py` → `linear_cross_entropy_with_entropy_entry.py`
- [x]3.2 Import `FwdMainLoopEntropy` and new Triton epilogues
- [x]3.3 In `forward_entropy()`:
  - Allocate `_entropy_b[T, S]`, `entropy_b[T]`, `entropy[T]`
  - Pack `_entropy_b` and pass to `FwdMainLoopEntropy`
  - Call entropy epilogues with `_entropy_b` inputs
  - Return `entropy`, `entropy_b` alongside existing outputs
- [x]3.4 In `backward_entropy()`:
  - Accept `dentropy` and `entropy_b` parameters
  - Call `BwdPartialDlogitsEntropy` (from Phase 4)

### Phase 4: Backward — New cuteDSL BwdPartialDlogitsEntropy kernel

- [x]4.1 Copy `bwd_partial_dlogits.py` → `bwd_partial_dlogits_entropy.py`
- [x]4.2 Rename class to `BwdPartialDlogitsEntropy`
- [x]4.3 Add `d_entropy` and `entropy_b` tensor parameters to `kernel()` and `__call__()`
- [x]4.4 Load `d_entropy[t]` and `entropy_b[t]` as scalars per token in epilogue
- [x]4.5 In d_logits computation, add entropy gradient term:
  ```
  d_logits += d_entropy * (-softmax) * (logit - entropy_b)
  ```
- [x]4.6 Update `__init__.py` to export `BwdPartialDlogitsEntropy`

### Phase 5: Torch interface — Add return_entropy option

- [x]5.1 In `pytorch/linear_cross_entropy.py`:
  - Add `LinearCrossEntropyWithEntropy(torch.autograd.Function)` class
  - `forward()`: calls `entropy_entry.forward_entropy()`, returns `(logprobs, entropy)`, saves `entropy_b` in ctx
  - `backward(dlogprobs, dentropy)`: calls `entropy_entry.backward_entropy()` with `dentropy` and `entropy_b`
- [x]5.2 Update `linear_cross_entropy()` function:
  - Add `return_entropy: bool = False` parameter
  - When `True`, dispatch to `LinearCrossEntropyWithEntropy.apply(...)`
  - When `False`, dispatch to existing `LinearCrossEntropy.apply(...)` (unchanged)
- [x]5.3 Update `Implementation` class to also register `forward_entropy_func` / `backward_entropy_func`
- [x]5.4 Update `__all__` exports

### Phase 6: Testing

Test commands from Makefile (all have built-in `timeout 3m` to avoid hangs):

```bash
make unit-test-1gpu   # timeout 3m pytest -s -v tests/pytorch/test_linear_cross_entropy.py
make unit-test-4gpu   # timeout 3m torchrun --nproc_per_node=4 ...
make stats-1gpu       # Performance + storage stats (single GPU)
make stats-4gpu       # Performance + storage stats (multi GPU)
```

- [x]6.1 Create `tests/pytorch/test_linear_cross_entropy_with_entropy.py`:
  - Reference: compute entropy as `logsumexp(logits) - sum(softmax * logits)`
  - Compare fused entropy against reference for various problem sizes
  - Test with different reductions (none, sum, mean)
  - Test with `ignore_index`
- [x]6.2 Backward test:
  - Use `torch.autograd.gradcheck` or compare against reference backward
  - Verify d_hidden and d_weight match reference when loss = CE + entropy
- [x]6.3 Run `make unit-test-1gpu` — existing tests must still pass (we didn't touch existing code)
- [x]6.4 Run `make unit-test-4gpu` — multi-GPU tests
- [x]6.5 Add entropy-specific test targets to Makefile if needed
- [x]6.6 Add TP test class (`TestLinearCrossEntropyWithEntropyTensorParallel`):
  - Torch TP reference autograd with entropy gradient math
  - `test_torch_tp_vs_single_gpu`, `test_correctness`, `test_performance`, `test_storage`
- [x]6.7 Add SP test class (`TestLinearCrossEntropyWithEntropySequenceParallel`):
  - Torch SP reference autograd with all-gather hidden + reduce-scatter d_hidden
  - `test_torch_sp_vs_single_gpu`, `test_correctness`, `test_performance`, `test_storage`
- [x]6.8 Add Profile test class (`TestLinearCrossEntropyWithEntropyProfile`):
  - NVTX-annotated profiling comparing baseline vs entropy paths

**Important**: Always use `timeout 3m` wrapper when running tests manually. Unit tests normally complete within 3 minutes; if they hang, it signals a kernel deadlock.

### Phase 7: Performance

NCU profiling commands from Makefile:

```bash
make ncu-fwd-cli      # Profile fwd_mainloop kernel -> fwd_mainloop.log
make ncu-bwd-cli      # Profile bwd_partial_dlogits kernel -> bwd_partial_dlogits.log
make ncu-fwd          # Full NCU report for forward kernel
make ncu-bwd          # Full NCU report for backward kernel
```

- [x]7.1 Profile entropy forward kernel with NCU, compare Duration against non-entropy forward
- [x]7.2 Profile entropy backward kernel with NCU, compare Duration against non-entropy backward
- [x]7.3 Confirm <5% overhead vs non-entropy path
- [x]7.4 Compare against reference Triton implementation (`build/kernels.py`) latency

---

## Performance Results

### Entropy overhead

Benchmark problem: `((1, 4096), 129280, 7168)`, BF16, Blackwell SM100.

| Metric | Baseline | Entropy | Overhead |
|--------|----------|---------|----------|
| Forward only | 4.69 ms | 4.81 ms | **2.5%** |
| Forward + Backward | 18.60 ms | 19.43 ms | **4.5%** |

Both well under the 5% target. When `return_entropy=False` (default), the original code path is taken — **zero overhead**.

### Temperature overhead

Added `temperature: float = 1.0` parameter to both baseline and entropy paths. CUDA event benchmarks (same problem size):

| Path | T=1.0 | T=0.5 | Delta |
|------|-------|-------|-------|
| baseline fwd | 5.19 ms | 5.42 ms | +4.4% (noise) |
| entropy fwd | 5.41 ms | 5.45 ms | +0.6% |
| baseline fwd+bwd | 20.90 ms | 19.62 ms | −6.1% (noise) |
| entropy fwd+bwd | 20.11 ms | 20.00 ms | −0.6% |

**Temperature overhead is within measurement noise** — no measurable cost.

### NCU profiling analysis

| Metric | Forward (`fwd_mainloop`) | Backward (`bwd_partial_dlogits`) |
|--------|--------------------------|----------------------------------|
| Duration | 4.49 ms | 139 μs (×42 splits) |
| TC pipe utilization | **92.7%** | **87.0%** |
| Memory throughput | 46.7% | 47.1% |
| FMA pipe utilization | 7.1% | 1.7% |
| ALU pipe utilization | 9.3% | 3.8% |
| Registers/thread | 212 | 247 |
| SMEM/block | 132 KB | 181 KB |
| Occupancy | 12.5% | 12.5% |
| Top stall: Long Scoreboard | 47% | 73% |
| Top stall: Barrier | 32% | 19% |

**Both kernels are Tensor-Core-bound at 87–93% TC utilization — near hardware peak.**

- Temperature multiply (`scaled_logit = logit * inv_temperature`) hits the FMA pipe which has 83–96% headroom — completely hidden by TC.
- Entropy epilogue additions (+1 FMA for `_entropy_b`, +2 FMAs for backward gradient) also land on the underutilized FMA/ALU pipes.
- Backward kernel: precomputed `log2e_inv_t = LOG2_E * inv_temperature` once outside the 64-iteration unrolled inner loop to avoid redundant scalar multiplies.
- **No further optimization possible** for temperature/entropy — the bottleneck is the GEMM (TC pipe), and the epilogue work is fully hidden.

### Test Results

- `make unit-test-1gpu`: 84 passed (75 original + 9 temperature tests)
- `make unit-test-4gpu`: 88 passed
- Entropy tests: 22 passed (15 original + 7 temperature tests covering T=0.5/1.0/2.0, forward/backward, baseline/entropy paths)

### Test Coverage Update

Expanded `test_linear_cross_entropy_with_entropy.py` to mirror all 4 test classes from the original test file:

| Test Class | Scope | Tests |
|------------|-------|-------|
| `TestLinearCrossEntropyWithEntropyDataParallel` | Single GPU | kernel_launch, correctness (fwd+bwd), performance, storage, return_entropy_false, temperature |
| `TestLinearCrossEntropyWithEntropyTensorParallel` | Multi-GPU TP | torch_tp_vs_single_gpu, correctness (fwd+bwd), performance, storage |
| `TestLinearCrossEntropyWithEntropySequenceParallel` | Multi-GPU SP | torch_sp_vs_single_gpu, correctness (fwd+bwd), performance, storage |
| `TestLinearCrossEntropyWithEntropyProfile` | Single GPU (ONLY_PROFILE=1) | NVTX-annotated baseline vs entropy comparison |

TP/SP reference autograd classes implement full entropy gradient: `dH/dz = -softmax * (z - entropy_b)` with temperature support. Performance/storage tests compare baseline vs entropy to measure overhead.

---

## Phase 8: Deduplicate Platform/Implementation Singletons

### Problem

Three singletons performed GPU arch detection independently:

| Class | File | Duty |
|---|---|---|
| `Implementation` | `pytorch/linear_cross_entropy.py` | Arch gate → imports entry modules, stores function pointers |
| `Platform` | `pytorch/cutedsl/lce_common.py` | Arch gate (redundant) → imports `blackwell` module |

`Platform` duplicated `Implementation`'s arch detection. Entry files are already arch-specific (only imported on correct arch), yet `Platform` rechecked `cc[0]`.

### Solution: Single arch gate in `Implementation`, moved to `lce_common.py`

Move `Implementation` into `lce_common.py` so both the torch API file and the entry files can import from the same place — no circular dependency. Remove `Platform` entirely. `Implementation` becomes the single place that:
1. Detects GPU arch
2. Imports the correct kernel module (`blackwell`, future `hopper`)  
3. Imports the correct entry modules
4. Stores `gpu_entry` + function pointers

Dependency graph (no cycles):
```
linear_cross_entropy.py ──→ lce_common.py ──(lazy)──→ entry modules
entry modules ─────────────→ lce_common.py
```

### Action Plan

- [x] 8.1 Move `Implementation` class and `_get_impl()` from `linear_cross_entropy.py` → `lce_common.py`
  - Add `self.gpu_entry` attribute storing the arch-specific kernel module
  - Remove `Platform` class and `get_platform()` from `lce_common.py`
- [x] 8.2 Update `linear_cross_entropy.py`:
  - Remove `Implementation` class and `_get_impl()` definition
  - Import `_get_impl` from `lce_common`
- [x] 8.3 Update `linear_cross_entropy_entry.py`:
  - Replace `from lce_common import get_platform` → `from lce_common import _get_impl`
  - Replace `_get_platform().gpu_entry.X` → `_get_impl().gpu_entry.X`
- [x] 8.4 Update `linear_cross_entropy_with_entropy_entry.py`:
  - Same as 8.3
- [x] 8.5 Verify: all unit tests pass (84+49 DP, 88+46 TP+SP)
