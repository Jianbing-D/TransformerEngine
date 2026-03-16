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

---

## 14. Static Persistent Scheduler (Task-2 Insights)

### Overview

`StaticPersistentScheduler` in `scheduler.py` replaces the naive `(num_m_tiles, num_splits, 1)` grid with a 1D grid capped at `min(SM_count × occupancy, total_tiles)`. Each CTA loops over multiple tiles until all tiles are exhausted.

### API Summary

```python
# Host side (inside @cute.jit)
sched_params_init = TileSchedulerParams(
    num_tiles_M=Int32(num_m_tiles),
    num_tiles_N=Int32(num_splits),
)
sched_params = StaticPersistentScheduler.to_underlying_arguments(sched_params_init)
grid = StaticPersistentScheduler.get_grid_shape(sched_params, occupancy=1)
# Pass sched_params to kernel as scheduler_params: tile_scheduler.ParamsBase

# Device side (inside @cute.kernel)
TileSchedulerCls = partial(StaticPersistentScheduler.create, scheduler_params)
scheduler = TileSchedulerCls()
work_tile = scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    m_idx, n_idx = work_tile.tile_idx
    ...
    scheduler.advance_to_next_work()
    work_tile = scheduler.get_current_work()
```

### Linearization

Tiles are linearized as `tile_idx = m_idx * num_splits + n_idx`. Decomposition uses `FastDivmodDivisor(num_splits)` for efficient integer division: `(m_idx, n_idx) = divmod(tile_idx, num_splits_divmod)`. `num_splits` must be compile-time static (it is — derived from `vocab_size` which is JIT-specialized).

### CuteDSL Scoping Rule for Multi-Warp Persistent Kernels

**Critical constraint**: CuteDSL (MLIR-based) generates a separate IR region for each `if warp_idx == X:` or `if warp_idx in (...)` block. Variables defined inside one conditional block are NOT visible in any other conditional block, even a sibling.

**Pattern**: Each warp group must contain its entire work (both fixed setup AND the persistent while loop) within a **single** `if warp_idx == X:` block. Never split setup and loop body into two separate conditional blocks.

```python
# WRONG: setup in first block, loop in second block
if warp_idx in self.epi_warp_ids:
    thr_copy = ...  # defined here

if warp_idx in self.epi_warp_ids:  # NEW MLIR REGION: thr_copy not visible!
    while work.is_valid_tile:
        cute.copy(thr_copy, ...)  # DSLRuntimeError: name 'thr_copy' not defined

# CORRECT: everything in one block
if warp_idx in self.epi_warp_ids:
    thr_copy = ...  # setup
    while work.is_valid_tile:
        cute.copy(thr_copy, ...)  # visible
```

### TMEM Allocation in Persistent Kernels

TMEM is allocated once before all warp-group blocks and reused across all tiles. A `NamedBarrier` synchronizes all warps before TMEM retrieval, and another barrier at the end ensures all warp-group loops complete before TMEM deallocation.

```python
# Allocate once (before per-warp blocks)
if warp_idx == empty_warp_ids[0]:
    cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf, ...)
cta_sync_barrier.arrive_and_wait()
tmem_ptr = cute.arch.retrieve_tmem_ptr(...)

# ... all per-warp persistent loops ...

# Deallocate once (after all loops)
cta_sync_barrier.arrive_and_wait()
if warp_idx == empty_warp_ids[0]:
    cute.arch.relinquish_tmem_alloc_permit()
    cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, ...)
```

### Shared vs Per-Tile State

Variables that are compile-time constants or depend only on the kernel configuration can be computed once in shared scope before all warp-group blocks:
- `tCsA`, `tCsB`, `tCtC` (SMEM/TMEM tensor partitions)
- `num_k_tiles` (static: `ceil_div(dim, mma_tiler_k)`, `dim` is JIT-specialized)
- `TileSchedulerCls` (factory: `partial(StaticPersistentScheduler.create, scheduler_params)`)

Variables that depend on `(pidm, pidn)` are recomputed per tile inside each warp group's while loop:
- `gA`, `gB`, TMA partitions (load warp)
- `num_n_tiles` from `pidn` (all warp groups)
- Register fragments (`tR2GrMax.fill(-1e30)`, etc.) reset each tile (epilogue warpgroup)

---

## 15. Implementation Brainstorm: Better LCE Fusion (Task-3)

### Reference Configuration
DeepSeek-style: `T=4096 tokens, V=129280 vocab, D=7168 dim, BF16 (2B/element)`
- `num_splits = ceil_div(129280, 3072) = 42`
- `num_m_tiles = ceil_div(4096, 128) = 32`

---

### 15.1 Mathematical Foundation

The full computation:
```
Forward:
  z_{t,v}  = Σ_d h_{t,d} × W_{v,d}          (GEMM: T×V)
  m_t      = max_v z_{t,v}                   (online max)
  A_t      = Σ_v exp(z_{t,v} - m_t)         (partition function, shifted)
  LSE_t    = log(A_t) + m_t                  (log-sum-exp)
  NLL_t    = LSE_t - z_{t, y_t}             (cross-entropy)

Backward (chain rule):
  ∂NLL_t / ∂z_{t,v} = softmax(z_t)[v] - 1_{v=y_t}
                     = exp(z_{t,v} - LSE_t) - 1_{v=y_t}
  d_z_{t,v} = dL_t × (exp(z_{t,v} - LSE_t) - 1_{v=y_t})
  d_h_t    = Σ_v d_z_{t,v} × W_v     (d_hidden)
  d_W_v    = Σ_t d_z_{t,v} × h_t     (d_weight)
```

**Key property**: `d_z` depends on `z` (logits), which were not saved. They must be recomputed from `h` and `W`. This is the source of the 3× backward FLOPs vs forward.

---

### 15.2 FLOP and Bandwidth Accounting

#### Forward (per reference config)
| Operation | FLOPs | Bandwidth |
|-----------|-------|-----------|
| GEMM (h @ W.T) | 2×4096×129280×7168 = **7.6 TFLOPs** | W: 1.85 GB, h: 58 MB |
| Online softmax (in TMEM→reg) | negligible | _max write: 672 KB, _accu write: 672 KB |
| Triton epilogue (reduce splits) | negligible | _max+_accu read: 1.3 MB |
| **Total** | **7.6 TFLOPs** | **~1.92 GB** |

At GB200 throughput: 7.6 TFLOPs / 2000 TFLOPs/s = **3.8 ms** compute-bound lower bound.
Observed: **6.12 ms** → ~62% SM efficiency.

**Conclusion: Forward is compute-bound. Weight load (1.85 GB) is the dominant bandwidth cost.**

#### Backward — current `kDlogitsSplitN` (per split, repeated 42×)
| Operation | FLOPs | Bandwidth |
|-----------|-------|-----------|
| BwdPartialDlogits (recompute logits) | 2×4096×3072×7168 = 0.18 TFLOPs | h: 58 MB, W_s: 44 MB, _d_logits write: 25 MB |
| cuBLAS addmm (d_h += d_logits @ W_s) | 2×4096×7168×3072 = 0.18 TFLOPs | _d_logits read: 25 MB, W_s: 44 MB, d_h read+write: 117 MB |
| torch.matmul (d_W_s = d_logits.T @ h) | 2×3072×7168×4096 = 0.18 TFLOPs | _d_logits read: 25 MB, h: 58 MB, d_W_s write: 44 MB |
| **Per-split total** | **0.54 TFLOPs** | **~440 MB** |
| **42 splits total** | **22.7 TFLOPs = 3× fwd** | **~18.5 GB** |

At GB200: 22.7 TFLOPs / 2000 TFLOPs/s = **11.4 ms** compute-bound lower bound.
Observed: **17.76 ms** → ~64% SM efficiency.

**Conclusion: Backward is compute-bound. The 3× forward FLOPs is a fundamental lower bound when logits are not saved.**

---

### 15.3 Why 3× Backward FLOPs Is Unavoidable (When Logits Are Not Saved)

The backward requires computing `exp(z_{t,v} - LSE_t)` for every `(t,v)`, which requires `z_{t,v}`. Since `z_{t,v} = h_t @ W_v^T` and this was not saved, it must be recomputed. The GEMM to recompute logits has the same cost as the forward GEMM.

Then `d_h = d_z @ W` (same cost) and `d_W = d_z^T @ h` (same cost). Total = 3× forward.

**The only escape from 3× backward**: store the full logit tensor `z` (T×V×2B = 1.05 GB for reference config). This trades the extra 2× GEMM compute for 1.05 GB of HBM. The LCE fusion explicitly rejects this trade — its purpose is to avoid storing `z`.

---

### 15.4 Forward Improvement: Token-Centric Tiling

#### Current design (split-centric)
- Grid: `(num_m_tiles × num_splits, 1, 1)` tiled to SM count
- Each CTA: 1 token tile × 1 vocab split
- After all CTAs finish: Triton epilogue reduces `_max[T, num_splits]` and `_accu[T, num_splits]` to produce LSE

#### Proposed: Token-centric tiling
- Grid: `(num_m_tiles, 1, 1)` tiled to SM count
- Each CTA: 1 token tile × **all vocab splits**
- Inner loop over splits, maintaining running `max_reg[128]` and `accu_reg[128]` in registers
- Write final `LSE[T]` and `logprobs[T]` directly — no Triton epilogue

```
# Pseudocode: token-centric forward
while m_work.is_valid:
    pidm = m_work.tile_idx
    max_reg[:] = -inf
    accu_reg[:] = 0.0
    logprob_reg[:] = 0.0

    for pidn in range(num_splits):
        # GEMM: logits[128, vocab_per_split] via load/MMA pipeline
        for n in range(num_n_per_split):
            # online softmax update using logit tile from TMEM
            max_old = max_reg
            max_reg = fmax(max_reg, row_max(logit_tile))
            accu_reg = exp(max_old - max_reg) * accu_reg + sum(exp(logit_tile - max_reg))
            logprob_reg += (position == label) * logit

    # write LSE and NLL directly
    LSE = log(accu_reg) + max_reg
    write(LSE, logprob_reg - LSE, ...)
    m_work.advance()
```

**Advantages**:
- Eliminates `_max[T, num_splits]` + `_accu[T, num_splits]` intermediate tensors (~1.4 MB)
- Eliminates Triton epilogue kernel launch and synchronization overhead
- Simpler: single kernel computes final LSE and NLL

**Disadvantages**:
- Fewer tiles to fill SMs: `num_m_tiles = 32` vs `num_m_tiles × num_splits = 1344`. With 32 CTAs and ~112 SMs, only 32 of 112 SMs are busy → SM occupancy drops to 28%
- The inner N-axis loop must be sequential (each split's softmax state depends on previous splits)
- Worse for small `num_tokens` (tiny M dimension)

**When token-centric wins**: `num_tokens >> SM_count` so that `num_m_tiles >> num_splits`. E.g., T=16384 → `num_m_tiles=128 > num_splits=42` → can fill 128 SMs with no stalls.

**When split-centric wins**: Small `num_tokens` (e.g., inference with T=1) but large vocab. More tiles give better SM coverage.

---

### 15.5 Forward Improvement: Eliminating the Triton Epilogue via Atomic Write-back

An alternative approach that keeps split-centric tiling but eliminates the Triton epilogue:

Use **atomic** global reduction to merge per-split statistics:
- Each CTA writes its `(partial_max_s, partial_accu_s)` and atomically updates a global `(max, accu)` for each token using CAS or atomic max/add.
- When all splits for a token are done, the epilogue is implicitly merged.

**Challenge**: atomic max on FP32 requires careful handling; `accu` update requires `exp(partial_max_s - global_max) * partial_accu_s` which is non-trivially atomic (needs read-modify-write with correction factor). This is fundamentally non-atomic without synchronization.

**Verdict**: Not practical in this form. The Triton epilogue reduction is actually correct and efficient for the split-centric approach.

---

### 15.6 Backward Improvement: Fused `BwdDHiddenDWeight` (WIP)

The `bwd_dHdW.py` kernel fuses all three backward operations (logit recompute + d_hidden + d_weight) into a single persistent kernel.

#### 16-warp CTA layout
- **Warps 0-3 (softmax WG)**: Apply softmax to logits, compute d_logits in TMEM
- **Warps 4-7 (epilog WG)**: Write d_logits to TMEM as `p` (probability) buffer; coordinate d_H and d_W MMA
- **Warp 8 (load)**: TMA G2S loads for hidden and weight
- **Warp 9 (MMA)**: Issues tcgen05 GEMM instructions
- **Warp 10 (store)**: TMA reduce (CpReduceS2G ADD) writes d_hidden and d_weight to GMEM atomically
- **Warps 11-15 (empty)**: Register dealloc only

#### TMEM allocation
```
TMEM layout (512 columns total):
[logits_cols | d_H_cols | d_W_cols | p_cols]
= [128 | 128 | 128 | 64] = 448 → rounded up to 512 columns
```

#### Three-phase pipeline
1. **Phase 1**: GEMM: logits = W_tile @ h^T → logits in TMEM
2. **Phase 2**: Softmax WG applies `exp(logit - LSE) - 1_{label}` → `p` in TMEM
3. **Phase 3a**: `d_W_tile = p^T @ h` (d_W GEMM using p from TMEM as A, h as B)
3. **Phase 3b**: `d_h += p @ W_tile` (d_H GEMM using p from TMEM as A, W_tile as B)

**Key insight**: Phases 3a and 3b both READ `p` from TMEM (no extra GMEM read for p). Weight must still be read twice: once for Phase 1 (logits), once for Phase 3b (d_hidden). But this is done in the same kernel, potentially with better cache reuse.

**Bandwidth savings vs kDlogitsSplitN**:
- Eliminates `_d_logits[T, vocab_per_split]` writes/reads: saves 75 MB per split × 42 = 3.15 GB
- Weight still read 2× per split (same as current): no bandwidth saving on weight
- net saving: ~3.15 GB / 18.5 GB ≈ 17% bandwidth reduction

**Latency savings**:
- Eliminates 42 × 3 = 126 CUDA kernel launches (replaces with 1)
- Better SM continuity (persistent kernel keeps all SMs busy)
- No Python loop stalling CUDA stream submission

**Disadvantages**:
- Very complex kernel: 16-warp layout, 3 MMA types, multiple TMEM regions
- TMEM layout is tight: 512 columns is the exact SM100 capacity
- TMA reduce (atomic add to d_hidden) may create contention when multiple vocab tiles target the same token rows

---

### 15.7 Backward Improvement: Persistent `BwdPartialDlogits`

A lighter-weight improvement: apply the persistent scheduler to `BwdPartialDlogits` so all 42 splits run in one kernel launch, then call cuBLAS once per split from the host.

This doesn't eliminate the 126 kernel launches but does eliminate the Python-side sequential dispatch latency for the SM100 kernel. The cuBLAS calls would still serialize.

**Better approach**: Use the `StaticPersistentScheduler` inside `BwdPartialDlogits` to iterate over all splits, writing `_d_logits` to GMEM for all splits, then call cuBLAS once per split. The kernel launch overhead drops from 42 launches to 1 for the BwdPartialDlogits stage.

---

### 15.8 Data Type Precision Analysis

#### BF16 vs FP16
| | BF16 | FP16 |
|--|------|------|
| Mantissa bits | 7 | 10 |
| Exponent bits | 8 | 5 |
| Dynamic range | Same as FP32 (±3.4×10^38) | ±65504 |
| Precision | ~0.8% relative error | ~0.1% relative error |

For LCE: logits can span large dynamic range (good for BF16 exponent bits). Precision matters for gradients near zero (d_logits = softmax - one_hot, which can be tiny). BF16 is the current standard.

**FP16 risk**: For logits exceeding ±65504, FP16 overflows to inf/NaN. With logit ~ 10 × weight_norm × hidden_norm, large models can approach this. BF16 is safer.

#### FP32 accumulation in GEMM
The SM100 tcgen05 MMA accumulates in FP32 within TMEM even for BF16 inputs. This is correct — no loss compared to FP32-input GEMM.

#### The d_logits precision issue
When `BwdPartialDlogits` writes `d_logits` to GMEM, it casts from FP32 (TMEM) to BF16:
```python
dLogits_half[idx] = tTMEM_load_rAcc[idx].to(dLogits_half.element_type)
```
- `d_logits = dL × (softmax(z) - one_hot)` is computed in FP32
- Cast to BF16 for storage → 7 mantissa bits
- Then `d_hidden = d_logits @ W` (BF16 GEMM with FP32 accumulation in cuBLAS)

For tokens where `softmax(z)[label] ≈ 1.0`, `d_logits[label] ≈ dL × (1 - 1) = 0` (fine). For tokens where `softmax(z)[v] ≈ 1e-4`, `d_logits[v] ≈ dL × 1e-4` — BF16 relative error is ~1% → absolute error ~`dL × 1e-6`. This is acceptable for training.

**The `atol=1e-3` tolerance** in tests reflects this BF16 precision ceiling, not a fundamental algorithm issue.

**FP32 d_logits option**:
- Store d_logits as FP32 → `_d_logits: T×V/num_splits×4B = 50 MB` per split (vs 25 MB BF16)
- Better gradient precision for small-probability classes
- Higher bandwidth cost: adds 25 MB extra per split × 42 = 1.05 GB
- Not worth it for standard training; could matter for distillation or sparse models

#### FP8 GEMM consideration
SM100 supports E4M3 / E5M2 FP8 with 2× FLOP density over BF16.

**Risk analysis**:
- E4M3 range: ±448. Large logits (e.g., dot products of high-norm vectors) may overflow.
- FP8 requires per-tensor or per-column scaling factors → additional ops
- Softmax computation requires FP32 intermediate regardless
- `d_logits = exp(z - LSE) - 1_{label}` requires FP32 for numerical stability

**Verdict**: FP8 for the GEMM is high-risk for correctness without calibration. Not recommended without a quantization framework.

---

### 15.9 Vocab Split Size Tuning

Current default: `vocab_per_split = 3072 = 12 × 256` (12 N-tiles per MMA tiler N=256).

**Effect on forward**:
- Larger `vocab_per_split` → fewer splits → fewer tiles → less SM coverage for small T
- Smaller `vocab_per_split` → more splits → more SM coverage but more pipeline setup overhead
- With persistent scheduler, pipeline setup is amortized → larger splits preferred

**Effect on backward** (kDlogitsSplitN):
- Larger split → `_d_logits` buffer larger → more GMEM bandwidth per iteration
- Fewer splits → fewer kernel launches → less launch overhead
- Optimal balances `_d_logits` buffer size with launch overhead

**Effect on SMEM**:
- SMEM for A (hidden): `128 × 64 × 4 stages × 2B = 65 KB` (fixed by mma_tiler_k=64)
- SMEM for B (weight): `256 × 64 × 4 stages × 2B = 131 KB`
- Total: 196 KB < 256 KB SM100 limit → no SMEM constraint on split size
- Changing vocab_per_split doesn't change SMEM (only changes the outer loop count)

**Recommendation**: For the fused backward (bwd_dHdW), larger vocab_per_split reduces launch overhead further. For current kDlogitsSplitN, the Python loop makes large splits desirable too (fewer iterations).

---

### 15.10 2-CTA Instructions (use_2cta_instrs=True)

Setting `use_2cta_instrs=True` enables `tcgen05.CtaGroup.TWO` and doubles the N-dimension of the MMA tiler: `mma_tiler_mn = (128, 512)` across 2 CTAs (effective 128×512 per cluster).

**Effect**: Halves the number of N-tiles per split (3072/512 = 6 vs 3072/256 = 12). Fewer pipeline stages, better SM pairing (2 CTAs collaborate).

**Constraint**: Grid must be a multiple of cluster shape (2, 1, 1). With 112 SMs, this is fine.

**Risk**: More complex barrier synchronization between the two CTAs in a cluster. Currently `use_2cta_instrs=False` in both forward and backward kernels.

---

### 15.11 Summary and Recommendations

#### Priority 1 (High Impact, Feasible): Complete `BwdDHiddenDWeight`
- Replaces 42× (BwdPartialDlogits + cuBLAS + matmul) with 1 persistent kernel
- Eliminates 3.15 GB of intermediate tensor bandwidth
- Reduces kernel launch overhead from 126 to 1
- Expected speedup: eliminate the ~3-5ms of launch overhead + bandwidth savings
- WIP code already exists in `bwd_dHdW.py` — needs testing and integration

#### Priority 2 (Medium Impact): Token-Centric Forward for Large T
- Eliminates Triton epilogue (~0.5-1ms overhead) when `num_tokens >> SM_count`
- Reduces intermediate tensor writes (~1.4 MB)
- Best for large-batch training (T≥16384)
- For small T (inference), split-centric is better (more SM coverage)

#### Priority 3 (Low Impact): Persistent BwdPartialDlogits (without full fusion)
- Apply `StaticPersistentScheduler` to `BwdPartialDlogits` so all splits run in one kernel
- Still requires sequential cuBLAS calls afterward
- Reduces BwdPartialDlogits launch overhead from 42 to 1
- Easy to implement (same pattern as Task-2)

#### Priority 4 (Low Impact): 2-Stage TMEM in BwdPartialDlogits
- Increase `num_acc_stage` from 1 to 2
- Allows MMA to write stage 1 while epilogue reads stage 0
- Better pipeline overlap between MMA and softmax/d_logits computation
- Requires doubling TMEM to 512 columns (fills SM100 capacity)

#### Summary Table

| Improvement | Memory Δ | Latency Δ | Complexity | Status |
|-------------|----------|-----------|------------|--------|
| Token-centric fwd (no Triton epilogue) | −1.4 MB intermediate | −~1ms for large T | Medium | Not started |
| Fused backward (`bwd_dHdW`) | −3.15 GB/iteration | −~5ms (~30% bwd) | High | WIP |
| Persistent `BwdPartialDlogits` | None | −~1ms launch overhead | Low | Not started |
| 2-stage TMEM in bwd | None | −~10% bwd GEMM | Low | Not started |
| FP8 GEMM | None | −~30% GEMM compute | Very High | Risky |
| FP32 d_logits | +1.05 GB | −0ms (bandwidth ↑) | Low | Not recommended |
