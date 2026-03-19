# Task-8: Backward Pass Performance Bottleneck Analysis

## Reference Configuration

| Parameter | Value |
|-----------|-------|
| num_tokens (T) | 4096 |
| vocab_size (V) | 129280 |
| hidden_dim (D) | 7168 |
| dtype | BF16 (2 bytes) |
| vocab_per_split | 3072 (= 512 × 6) |
| num_splits | ceil(129280 / 3072) = **43** |
| Platform | GB200 (SM100), 152 SMs |
| Peak FP16/BF16 TFLOPS | ~2000 TFLOPS/s |
| HBM bandwidth | ~8 TB/s |

> Note: The last split handles only 129280 − 42×3072 = 256 vocab entries (a very small GEMM), so effectively 42 full splits + 1 tiny split.

---

## 1. Current Backward Algorithm: kDlogitsSplitN

For each split `s` in `[0, num_splits)`, three sequential operations run on the same CUDA stream:

### Operation 1: BwdPartialDlogits (cuteDSL kernel)
**What it does**: Recomputes logits via GEMM, applies softmax gradient epilogue, writes d_logits to GMEM.

```
logits[T, V_s] = hidden[T, D] × W_s[V_s, D].T       (GEMM)
softmax[t, v]  = exp(logits[t, v] − LSE[t])           (epilogue)
d_logits[t, v] = dlogprobs[t] × (softmax[t, v] − 1[v == label[t]])  (epilogue)
```

| Metric | Value |
|--------|-------|
| FLOPs | 2 × 4096 × 3072 × 7168 = **0.180 TFLOPS** |
| Reads | hidden: 4096×7168×2B = 58.7 MB, W_s: 3072×7168×2B = 44.0 MB, LSE: 4096×4B = 16 KB, labels: 4096×8B = 32 KB |
| Writes | d_logits: 4096×3072×2B = **25.2 MB** |
| Measured duration | **137 μs** (after Task-4/6/7 optimizations) |
| Compute utilization | 0.180 / (0.000137 × 2000) = **65.7%** |

### Operation 2: cuBLAS addmm (d_hidden accumulation)
**What it does**: `d_hidden[T, D] += d_logits[T, V_s] × W_s[V_s, D]` in FP32

```
d_hidden += d_logits @ W_s    (beta=0 for first split, beta=1 otherwise)
```

| Metric | Value |
|--------|-------|
| FLOPs | 2 × 4096 × 7168 × 3072 = **0.180 TFLOPS** |
| Reads | d_logits: 25.2 MB (BF16), W_s: 44.0 MB (BF16), d_hidden: 4096×7168×4B = 117.4 MB (FP32, for beta=1) |
| Writes | d_hidden: 117.4 MB (FP32) |
| Total bandwidth per call | 25.2 + 44.0 + 117.4 + 117.4 = **304 MB** (for splits 1–42, first split skips read of d_hidden) |
| Arithmetic intensity | 0.180 TFLOPS / 304 MB = **592 FLOPS/byte** |

This GEMM is compute-bound (AI > ~250 FLOPS/byte GB200 ridge point). Estimated duration at 70–80% cuBLAS efficiency: **90–103 μs**.

### Operation 3: torch.matmul (d_weight)
**What it does**: `d_weight_s[V_s, D] = d_logits[T, V_s].T × hidden[T, D]`

```
d_weight_s = d_logits.T @ hidden
```

| Metric | Value |
|--------|-------|
| FLOPs | 2 × 3072 × 7168 × 4096 = **0.180 TFLOPS** |
| Reads | d_logits: 25.2 MB, hidden: 58.7 MB |
| Writes | d_weight_s: 44.0 MB |
| Total bandwidth per call | 25.2 + 58.7 + 44.0 = **127.9 MB** |
| Arithmetic intensity | 0.180 TFLOPS / 127.9 MB = **1407 FLOPS/byte** |

Strongly compute-bound. Estimated duration at 70–80% efficiency: **90–103 μs**.

---

## 2. Total Backward Time Breakdown

### Per-split breakdown (estimated for 42 full splits)

| Component | Per-split (μs) | × 42 (ms) | % of total |
|-----------|---------------|------------|------------|
| BwdPartialDlogits | 137 | 5.75 | 37.1% |
| cuBLAS addmm | ~100 | ~4.20 | 27.1% |
| torch.matmul | ~100 | ~4.20 | 27.1% |
| Kernel launch overhead (3 launches) | ~24 | ~1.01 | 6.5% |
| Last split (tiny, V_s=256) | ~30 total | 0.03 | 0.2% |
| **Python loop + misc** | — | ~0.30 | 2.0% |
| **Total estimated** | — | **~15.49** | 100% |

### Compute breakdown

| | FLOPs per split | × 43 splits | Total |
|--|-----------------|-------------|-------|
| BwdPartialDlogits | 0.180 T | 7.74 T | 34.1% |
| cuBLAS addmm | 0.180 T | 7.74 T | 34.1% |
| torch.matmul | 0.180 T | 7.74 T | 31.8% |
| **Total** | 0.540 T | **22.7 TFLOPS** | = 3× forward |

Compute-bound lower bound: 22.7 / 2000 = **11.34 ms**

**Observed 15.49 ms → 73.2% overall SM efficiency**

### Bandwidth breakdown (main overheads)

| Traffic | Per split | × 42 | Total |
|---------|-----------|------|-------|
| d_logits write (BwdPartialDlogits) | 25.2 MB | × 42 | **1.06 GB** |
| d_logits read (cuBLAS addmm) | 25.2 MB | × 42 | 1.06 GB |
| d_logits read (torch.matmul) | 25.2 MB | × 42 | 1.06 GB |
| d_hidden read+write (cuBLAS addmm) | 234.8 MB | × 41 | **9.63 GB** |
| W_s read (BwdPartialDlogits) | 44.0 MB | × 42 | 1.85 GB |
| W_s read (cuBLAS addmm) | 44.0 MB | × 42 | 1.85 GB |
| hidden read (BwdPartialDlogits) | 58.7 MB | × 42 | 2.47 GB |
| hidden read (torch.matmul) | 58.7 MB | × 42 | 2.47 GB |
| d_weight write | 44.0 MB | × 42 | 1.85 GB |
| **Total** | | | **~23.3 GB** |

At 8 TB/s: 23.3 GB / 8 TB/s = 2.91 ms bandwidth time. Since most GEMMs are compute-bound, bandwidth is not the primary limiter, but it contributes to the efficiency gap.

---

## 3. Identified Performance Bottlenecks (Ranked)

### Bottleneck #1: Sequential 3-kernel-per-split execution (major)
Each split runs 3 kernels sequentially on the same stream. There is no overlap between BwdPartialDlogits, cuBLAS addmm, and torch.matmul. This creates pipeline bubbles where the GPU is idle between kernels.

**Impact**: 126+ kernel launches × ~8 μs launch overhead = **~1 ms** direct cost. Plus pipeline drain/fill between each kernel.

### Bottleneck #2: d_hidden repeated read-modify-write (major)
The d_hidden tensor (FP32, 117.4 MB) is read and written by cuBLAS addmm on every split (41 times with beta=1). Total d_hidden traffic: **9.63 GB** — this is **41% of all bandwidth**.

**Why it matters**: Even though individual cuBLAS calls are compute-bound, the cumulative d_hidden traffic adds up. With L2 cache (50 MB on GB200), d_hidden (117 MB) doesn't fit, causing repeated HBM round-trips.

### Bottleneck #3: d_logits materialization to GMEM (moderate)
The d_logits tensor (BF16, 25.2 MB per split) is written by BwdPartialDlogits and read twice (by cuBLAS and matmul). Total d_logits traffic: **3.18 GB**.

**Why it matters**: This intermediate tensor exists only to bridge the cuteDSL kernel and cuBLAS/matmul. If the computation were fused, this traffic is entirely eliminable.

### Bottleneck #4: Weight matrix loaded multiple times (moderate)
W_s is loaded twice per split: once by BwdPartialDlogits (for logit recompute) and once by cuBLAS addmm (for d_hidden GEMM). Total extra W traffic: **1.85 GB**.

### Bottleneck #5: Small GEMM efficiency (minor)
Each GEMM has K=3072 (BwdPartialDlogits, addmm) or K=4096 (matmul), M=4096, N=7168 or 3072. These are moderately-sized GEMMs — large enough for reasonable efficiency but not large enough for cuBLAS to hit peak throughput. The K dimension determines pipeline depth for the GEMM mainloop; K=3072 gives only 3072/64 = 48 K-tiles — adequate but not deep.

---

## 4. Alternative Design A: Two Separate Fused Kernels

### Concept
Replace the 3-kernel-per-split loop with two persistent kernels that each iterate over all splits internally:

**Kernel 1 — d_hidden**:
```
for each token tile (persistent over M-tiles):
    d_hidden_acc[m_tile, :] = 0   (FP32 in TMEM/registers)
    for each split s:
        logits[m_tile, V_s] = h[m_tile, :] × W_s.T          (GEMM, K=D)
        d_logits = softmax_grad(logits, LSE, labels)          (epilogue in TMEM)
        d_hidden_acc += d_logits × W_s                        (GEMM, K=V_s)
    TMA_reduce_add(d_hidden, d_hidden_acc)                    (write once)
```

**Kernel 2 — d_weight**:
```
for each vocab tile (persistent over V-tiles):
    d_weight_acc[V_tile, D_tile] = 0   (FP32 in TMEM)
    for each token tile t:
        logits[T_tile, V_tile] = h[T_tile, :] × W[V_tile, :].T   (GEMM, K=D)
        d_logits = softmax_grad(logits, LSE, labels)               (epilogue)
        d_weight_acc += d_logits.T × h[T_tile, :]                  (GEMM, K=T_tile)
    write(d_weight[V_tile, D_tile], d_weight_acc)
```

### FLOP Analysis

| Kernel | Logit recompute | Gradient GEMM | Total |
|--------|----------------|---------------|-------|
| d_hidden | 1× fwd (7.56 T) | 1× fwd (7.56 T) | 2× fwd |
| d_weight | 1× fwd (7.56 T) | 1× fwd (7.56 T) | 2× fwd |
| **Total** | 2× fwd | 2× fwd | **4× fwd = 30.24 TFLOPS** |

Compute-bound lower bound: 30.24 / 2000 = **15.12 ms**

### Bandwidth Analysis

| Traffic | d_hidden kernel | d_weight kernel | Total |
|---------|----------------|-----------------|-------|
| W (full, loaded once per kernel) | 1.85 GB | 1.85 GB | 3.70 GB |
| hidden (loaded per split) | 2.47 GB | 2.47 GB | 4.94 GB |
| LSE + labels | ~0.01 GB | ~0.01 GB | 0.02 GB |
| d_hidden (write once) | 0.12 GB | — | 0.12 GB |
| d_weight (write once) | — | 1.85 GB | 1.85 GB |
| d_logits (**eliminated**) | 0 | 0 | **0** |
| d_hidden repeated RMW (**eliminated**) | 0 | 0 | **0** |
| **Total** | ~4.45 GB | ~6.18 GB | **~10.6 GB** |

vs current: **23.3 GB** → **55% bandwidth reduction**.

### Kernel Launch Analysis
2 launches vs 129 launches → **saves ~1 ms** overhead.

### Verdict

| Metric | Current (kDlogitsSplitN) | Two-kernel |
|--------|--------------------------|------------|
| FLOPs | 22.7 T (3× fwd) | 30.2 T (4× fwd) |
| Bandwidth | 23.3 GB | 10.6 GB |
| Launches | 129 | 2 |
| Compute-bound floor | 11.34 ms | 15.12 ms |
| Estimated latency (75% eff) | 15.12 ms | 20.16 ms |
| Estimated latency (85% eff) | 13.34 ms | 17.79 ms |

**The two-kernel approach is fundamentally slower** because it increases compute by 33% (4× vs 3× forward). The bandwidth savings (~12.7 GB) translate to only ~1.6 ms at 8 TB/s, which does NOT compensate for the extra 7.56 TFLOPS (3.78 ms at peak).

**The extra forward-pass worth of FLOPs is the price of recomputing logits twice** — once for d_hidden, once for d_weight. This is unavoidable in the two-kernel design because d_logits cannot be shared between the two independent kernels without materializing to GMEM (which is exactly what the current design does).

### When two-kernel could win
- If the current design's efficiency is very poor (< 60%) due to launch overhead and bandwidth, AND
- the two-kernel design achieves very high efficiency (> 85%) due to perfect pipelining
- Specifically: need current_latency > 4/3 × two_kernel_compute_bound / two_kernel_efficiency
- This is unlikely on GB200 where cuBLAS already achieves ~70–80% efficiency

---

## 5. Alternative Design B: Single Fused Kernel (BwdDHiddenDWeight)

### Concept
One persistent kernel that computes both d_hidden and d_weight per split, keeping d_logits in TMEM:

```
for each tile (persistent scheduler):
    # Phase 1: logit recompute (GEMM)
    logits[m_tile, V_tile] = W_tile × h.T

    # Phase 2: softmax gradient (warpgroup in TMEM)
    p[t, v] = exp(logits[t, v] − LSE[t]) − 1[v == label[t]]

    # Phase 3a: d_weight = p.T × h  (GEMM reading p from TMEM)
    # Phase 3b: d_hidden += p × W   (GEMM reading p from TMEM)
    TMA_reduce_add(d_hidden, ...)
    TMA_reduce_add(d_weight, ...)
```

### FLOP Analysis

| Phase | FLOPs | Notes |
|-------|-------|-------|
| Logit recompute | 1× fwd = 7.56 T | Same as current |
| d_hidden GEMM | 1× fwd = 7.56 T | p × W |
| d_weight GEMM | 1× fwd = 7.56 T | p.T × h |
| **Total** | **3× fwd = 22.7 TFLOPS** | **Same as current** |

Compute-bound lower bound: **11.34 ms** (same as current)

### Bandwidth Analysis

| Traffic | Amount |
|---------|--------|
| W (loaded once across all splits) | 1.85 GB |
| hidden (loaded per split, shared by all phases) | 2.47 GB |
| LSE + labels | 0.02 GB |
| d_hidden (TMA reduce-add, write-only) | 0.12 GB |
| d_weight (TMA reduce-add, write-only) | 1.85 GB |
| d_logits (**stays in TMEM, never hits GMEM**) | **0** |
| d_hidden repeated RMW (**eliminated by TMA reduce**) | **0** |
| **Total** | **~6.3 GB** |

vs current: 23.3 GB → **73% bandwidth reduction**.

### Kernel Launch Analysis
1 launch vs 129 launches → **saves ~1 ms**.

### Verdict

| Metric | Current (kDlogitsSplitN) | Fused (BwdDHiddenDWeight) |
|--------|--------------------------|---------------------------|
| FLOPs | 22.7 T (3× fwd) | 22.7 T (3× fwd) |
| Bandwidth | 23.3 GB | 6.3 GB |
| Launches | 129 | 1 |
| Compute-bound floor | 11.34 ms | 11.34 ms |
| Estimated latency (75% eff) | 15.12 ms | 15.12 ms |
| Estimated latency (85% eff) | 13.34 ms | 13.34 ms |

**The fused kernel has the same compute-bound floor but dramatically less bandwidth.** The key question is whether it can achieve higher SM efficiency than the current approach:

- **Bandwidth advantage**: 6.3 GB vs 23.3 GB — the fused kernel has ~3.7× less memory traffic. This means less contention for HBM bandwidth, allowing compute units to run closer to peak.
- **Launch advantage**: 1 launch vs 129 — eliminates ~1 ms of pure overhead.
- **Pipeline advantage**: All three phases (logit recompute → softmax grad → d_H/d_W GEMMs) run within the same CTA, with data passing through TMEM. No GMEM round-trips between phases.

**Estimated improvement**: If the fused kernel achieves 80–85% efficiency (reasonable given the eliminated bandwidth):
- 22.7 / (2000 × 0.825) = **13.8 ms** (~11% improvement over 15.49 ms)

**Risk**: The 16-warp CTA layout (softmax WG + epilog WG + load + MMA + store) is complex. TMEM must hold logits + d_H + d_W + p simultaneously (448→512 columns, full TMEM capacity). The 3-phase pipeline has longer latency per tile, which could hurt occupancy.

---

## 6. Summary: Where the 15.49 ms Goes

```
┌─────────────────────────────────────────────────────┐
│           Backward Pass: 15.49 ms                    │
│                                                      │
│  ┌─ BwdPartialDlogits (42×137μs) ──── 5.75 ms (37%) │
│  │  • Logit recompute GEMM                           │
│  │  • Softmax gradient epilogue                      │
│  │  • d_logits write to GMEM ◄── ELIMINABLE          │
│  │                                                    │
│  ├─ cuBLAS addmm (42×~100μs) ──────── 4.20 ms (27%) │
│  │  • d_hidden += d_logits @ W_s                     │
│  │  • d_hidden RMW 117MB × 41 ◄── ELIMINABLE        │
│  │  • d_logits read from GMEM ◄── ELIMINABLE         │
│  │                                                    │
│  ├─ torch.matmul (42×~100μs) ──────── 4.20 ms (27%) │
│  │  • d_weight_s = d_logits.T @ h                    │
│  │  • d_logits read from GMEM ◄── ELIMINABLE         │
│  │                                                    │
│  ├─ Kernel launches (129×~8μs) ────── 1.03 ms (7%)  │
│  │  ◄── ELIMINABLE with fusion                       │
│  │                                                    │
│  └─ Python loop + misc ───────────── 0.31 ms (2%)   │
└─────────────────────────────────────────────────────┘
```

### Key Insight

The current backward is compute-bound at 73% efficiency. The **27% efficiency gap** comes from:
1. **d_logits materialization** (3.18 GB unnecessary GMEM traffic)
2. **d_hidden repeated RMW** (9.63 GB unnecessary GMEM traffic)
3. **129 kernel launches** (~1 ms overhead)
4. **Weight loaded twice per split** (1.85 GB redundant)

Total eliminable bandwidth: **~16.5 GB** (71% of current traffic).

---

## 7. Recommendations

| Priority | Action | Expected gain | Complexity |
|----------|--------|---------------|------------|
| **1** | Complete BwdDHiddenDWeight fused kernel | **−1.5 to −2.5 ms** (10–16%) | High (WIP exists) |
| **2** | Multi-stream: overlap d_weight with next split's BwdPartialDlogits | **−1 to −2 ms** | Medium |
| **3** | Increase vocab_per_split to reduce num_splits | **−0.5 to −1 ms** | Low |
| **Avoid** | Two separate kernels (d_hidden + d_weight) | Net negative (4× vs 3× fwd FLOPs) | High |

### Recommendation #1: Fused kernel (BwdDHiddenDWeight)
The most impactful optimization. Eliminates d_logits materialization, d_hidden repeated RMW, and 128 kernel launches. Same 3× fwd FLOPs. The `bwd_dHdW.py` skeleton already exists but is not wired into the entry point.

### Recommendation #2: Multi-stream pipelining (incremental, no new kernels)
On the current kDlogitsSplitN design, torch.matmul for split `s` can overlap with BwdPartialDlogits for split `s+1` on a separate stream (they have no data dependency — matmul reads d_logits from split s, which is already computed). This hides matmul latency behind BwdPartialDlogits compute.

```
Stream 0: BwdPD[0] → addmm[0] → BwdPD[1] → addmm[1] → ...
Stream 1:            matmul[0] →            matmul[1] → ...
```

This could save up to ~4.2 ms if matmul fully overlaps, but in practice GPU resource contention will reduce the gain to ~1–2 ms.

### Recommendation #3: Larger vocab_per_split
Increasing vocab_per_split from 3072 to 6144 halves num_splits from 43 to 22. This reduces:
- Kernel launches: 129 → 66 (saves ~0.5 ms)
- d_hidden RMW passes: 42 → 21 (saves ~0.6 ms)
- But BwdPartialDlogits duration increases (more N-tiles per kernel)

### Why the two-kernel design should be avoided
It adds 1× forward worth of extra FLOPs (7.56 TFLOPS = 3.78 ms at peak) to eliminate d_logits materialization (saves ~0.4 ms at 8 TB/s). The math strongly disfavors this trade. The bandwidth savings are real but insufficient to compensate for the compute increase.
