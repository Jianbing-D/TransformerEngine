# Implementation Analysis: Better LCE Fusion (Task-3)

## Reference Configuration
DeepSeek-style: `T=4096 tokens, V=129280 vocab, D=7168 dim, BF16 (2B/element)`
- `num_splits = ceil_div(129280, 3072) = 42`
- `num_m_tiles = ceil_div(4096, 128) = 32`

---

## FLOP and Bandwidth Accounting

### Forward (per reference config)
| Operation | FLOPs | Bandwidth |
|-----------|-------|-----------|
| GEMM (h @ W.T) | 2×4096×129280×7168 = **7.6 TFLOPs** | W: 1.85 GB, h: 58 MB |
| Online softmax (in TMEM→reg) | negligible | `_max` write: 672 KB, `_accu` write: 672 KB |
| Triton epilogue (reduce splits) | negligible | `_max`+`_accu` read: 1.3 MB |
| **Total** | **7.6 TFLOPs** | **~1.92 GB** |

At GB200 throughput: 7.6 TFLOPs / 2000 TFLOPs/s = **3.8 ms** compute-bound lower bound.
Observed: **6.12 ms** → ~62% SM efficiency.

**Conclusion: Forward is compute-bound. Weight load (1.85 GB) is the dominant bandwidth cost.**

### Backward — current `kDlogitsSplitN` (per split, repeated 42×)
| Operation | FLOPs | Bandwidth |
|-----------|-------|-----------|
| BwdPartialDlogits (recompute logits) | 2×4096×3072×7168 = 0.18 TFLOPs | h: 58 MB, W_s: 44 MB, `_d_logits` write: 25 MB |
| cuBLAS addmm (d_h += d_logits @ W_s) | 2×4096×7168×3072 = 0.18 TFLOPs | `_d_logits` read: 25 MB, W_s: 44 MB, d_h read+write: 117 MB |
| torch.matmul (d_W_s = d_logits.T @ h) | 2×3072×7168×4096 = 0.18 TFLOPs | `_d_logits` read: 25 MB, h: 58 MB, d_W_s write: 44 MB |
| **Per-split total** | **0.54 TFLOPs** | **~440 MB** |
| **42 splits total** | **22.7 TFLOPs = 3× fwd** | **~18.5 GB** |

At GB200: 22.7 TFLOPs / 2000 TFLOPs/s = **11.4 ms** compute-bound lower bound.
Observed: **17.76 ms** → ~64% SM efficiency.

**Conclusion: Backward is compute-bound. The 3× forward FLOPs is a fundamental lower bound when logits are not saved.**

### Why 3× Backward FLOPs Is Unavoidable

The backward requires $\exp(z_{t,v} - \mathrm{LSE}_t)$ for every $(t, v)$, which requires $z_{t,v}$. Since $z_{t,v} = \mathbf{h}_t \cdot \mathbf{W}_v^\top$ was not saved, the full GEMM must be re-run at the same cost as the forward. Then:

$$\underbrace{\mathbf{z} = \mathbf{h}\mathbf{W}^\top}_{\text{recompute, } 1\times \text{fwd}} \qquad \underbrace{\frac{\partial L}{\partial \mathbf{h}} = \frac{\partial L}{\partial \mathbf{z}} \cdot \mathbf{W}}_{\text{d-hidden, } 1\times \text{fwd}} \qquad \underbrace{\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{z}}\right)^\top \mathbf{h}}_{\text{d-weight, } 1\times \text{fwd}}$$

Total backward FLOPs $= 3 \times$ forward. This is a **hard lower bound** when logits are not stored.

**The only escape**: store the full logit tensor $\mathbf{z}$ ($T \times V \times 2\,\text{B} = 1.05\,\text{GB}$ for the reference config). This trades 2× GEMM compute for 1.05 GB of HBM — exactly what LCE fusion is designed to avoid.

---

## Improvement Options

### Option 1: Token-Centric Tiling (Forward)

#### Current design (split-centric)
- Grid: `(num_m_tiles × num_splits, 1, 1)` tiled to SM count
- Each CTA: 1 token tile × 1 vocab split
- After all CTAs finish: Triton epilogue reduces `_max[T, num_splits]` and `_accu[T, num_splits]` to produce LSE

#### Proposed: Token-centric tiling
- Grid: `(num_m_tiles, 1, 1)` tiled to SM count
- Each CTA: 1 token tile × **all vocab splits**
- Inner loop over splits, maintaining running $m_t$ and $A_t$ in registers across splits
- Write final $\mathrm{LSE}_t$ and $\mathrm{NLL}_t$ directly — no Triton epilogue

The online update rule across splits is the standard two-pass softmax merger. For a new partial max $m'$ arriving from the next vocab tile:

$$m_t \leftarrow \max(m_t,\; m') \qquad A_t \leftarrow e^{m_t^{\,\mathrm{old}} - m_t} \cdot A_t + e^{m' - m_t} \cdot A'$$

```python
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

**Advantages**: Eliminates `_max[T, num_splits]` + `_accu[T, num_splits]` intermediate tensors (~1.4 MB), eliminates Triton epilogue kernel.

**Disadvantages**: Fewer tiles to fill SMs: `num_m_tiles=32` vs `1344`. With 32 CTAs and ~112 SMs, only 28% SM occupancy.

**When token-centric wins**: `num_tokens >> SM_count` so `num_m_tiles >> num_splits`. E.g., T=16384 → `num_m_tiles=128 > num_splits=42`.

---

### Option 2: Fused `BwdDHiddenDWeight` (Backward)

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

$$\underbrace{\mathbf{Z}_{\text{tile}} = \mathbf{W}_{\text{tile}}\,\mathbf{h}^\top}_{\text{Phase 1: logits GEMM}} \;\longrightarrow\; \underbrace{p_{t,v} = \exp(z_{t,v} - \mathrm{LSE}_t) - \mathbf{1}_{[v=y_t]}}_{\text{Phase 2: softmax WG}} \;\longrightarrow\; \begin{cases} \dfrac{\partial L}{\partial \mathbf{W}_{\text{tile}}} = \mathbf{p}^\top \mathbf{h} & \text{Phase 3a} \\ \dfrac{\partial L}{\partial \mathbf{h}} \mathrel{+}= \mathbf{p}\,\mathbf{W}_{\text{tile}} & \text{Phase 3b} \end{cases}$$

**Key insight**: Both Phases 3a and 3b read $\mathbf{p}$ directly from TMEM — no GMEM round-trip for the softmax probabilities.

**Bandwidth savings vs kDlogitsSplitN**: Eliminates `_d_logits[T, vocab_per_split]` writes/reads: saves 75 MB per split × 42 = **3.15 GB** (~17% bandwidth reduction).

**Latency savings**: Eliminates 42 × 3 = 126 CUDA kernel launches (replaces with 1).

---

### Option 3: Persistent `BwdPartialDlogits`

Apply `StaticPersistentScheduler` to `BwdPartialDlogits` so all splits run in one kernel launch, then call cuBLAS once per split from the host. Reduces BwdPartialDlogits launch overhead from 42 to 1.

---

### Option 4: 2-Stage TMEM in BwdPartialDlogits

Increase `num_acc_stage` from 1 to 2. Allows MMA to write stage 1 while epilogue reads stage 0. Better pipeline overlap. Requires doubling TMEM to 512 columns.

---

### Option 5: Atomic Write-back (Rejected)

Use atomic global reduction to merge per-split statistics directly. **Challenge**: the accumulator update $A_t \leftarrow e^{m_s - m_t^{\,\mathrm{global}}} \cdot A_s$ is a read-modify-write with a data-dependent scale factor — not expressible as a single atomic operation. **Verdict**: Not practical.

---

## Data Type Precision Analysis

### BF16 vs FP16
| | BF16 | FP16 |
|--|------|------|
| Mantissa bits | 7 | 10 |
| Exponent bits | 8 | 5 |
| Dynamic range | Same as FP32 (±3.4×10^38) | ±65504 |
| Precision | ~0.8% relative error | ~0.1% relative error |

**FP16 risk**: For logits exceeding ±65504, FP16 overflows to inf/NaN. BF16 is safer for large models.

### FP32 accumulation in GEMM
The SM100 tcgen05 MMA accumulates in FP32 within TMEM even for BF16 inputs. No loss compared to FP32-input GEMM.

### FP8 GEMM
SM100 supports E4M3 / E5M2 FP8 with 2× FLOP density over BF16. **Risk**: E4M3 range ±448 may overflow large logits. Requires per-tensor scaling. **Verdict**: High-risk without calibration framework.

---

## Vocab Split Size Tuning

Current default: `vocab_per_split = 3072 = 12 × 256` (12 N-tiles per MMA tiler N=256).

- Larger split → fewer splits → less SM coverage for small T, but fewer launches
- Smaller split → more splits → better SM coverage, more pipeline setup overhead
- SMEM is not the constraint: total 196 KB < 256 KB SM100 limit regardless of split size

---

## 2-CTA Instructions

Setting `use_2cta_instrs=True` enables `tcgen05.CtaGroup.TWO` and doubles the N-dimension: `mma_tiler_mn = (128, 512)` across 2 CTAs. Halves the number of N-tiles per split. Grid must be multiple of cluster shape (2,1,1). Currently `False` in all kernels — more complex barrier synchronization required.

---

## Summary and Priorities

| Improvement | Memory Δ | Latency Δ | Complexity | Status |
|-------------|----------|-----------|------------|--------|
| Token-centric fwd (no Triton epilogue) | −1.4 MB intermediate | −~1ms for large T | Medium | Not started |
| Fused backward (`bwd_dHdW`) | −3.15 GB/iteration | −~5ms (~30% bwd) | High | WIP |
| Persistent `BwdPartialDlogits` | None | −~1ms launch overhead | Low | Not started |
| 2-stage TMEM in bwd | None | −~10% bwd GEMM | Low | Not started |
| FP8 GEMM | None | −~30% GEMM compute | Very High | Risky |
| FP32 d_logits | +1.05 GB | −0ms (bandwidth ↑) | Low | Not recommended |
