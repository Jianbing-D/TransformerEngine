# NCU Gotchas & Lessons Learned

Hard-won insights from real profiling sessions. These are non-obvious behaviors that can lead to misinterpretation of NCU reports. **Load this file when encountering confusing or surprising metric values.**

---

## 1. `.avg` divides by ALL hardware units, not just active ones

**The gotcha:** `.avg` rollups always divide by the total number of hardware instances (e.g., all 152 SMs on a GB200), regardless of how many CTAs were launched or how many SMs were active.

**Why it matters:** A kernel launching 1 CTA will have `sm__pipe_tc_cycles_active.avg` = total / 152, not total / 1. This makes `.avg`-based metrics look tiny even if the active SM was working hard.

**Verified empirically** with a single-CTA cuTile bf16 GEMM on a 152-SM GB200:

```
sm__cycles_elapsed.sum  = 4,464,034   (all 152 SMs clock elapsed cycles)
sm__cycles_elapsed.avg  = 29,368.6    (= sum / 152)
sm__cycles_active.sum   = 24,229      (only 1 SM was active)
sm__cycles_active.avg   = 159.4       (= 24,229 / 152, NOT 24,229 / 1)
sm__cycles_active.max   = 24,229      (the single active SM)
sm__cycles_active.min   = 0           (151 idle SMs)
```

All 152 SMs accumulate `cycles_elapsed` for the full kernel duration, even SMs that run zero CTAs.

---

## 2. `pct_of_peak_sustained_elapsed` vs `pct_of_peak_sustained_active`

**The gotcha:** These two rollups answer fundamentally different questions. Using the wrong one leads to wrong conclusions.

| Rollup | Question it answers | Denominator |
|--------|-------------------|-------------|
| `pct_of_peak_sustained_elapsed` | What fraction of the **entire GPU's** capacity was used? | `cycles_elapsed.avg` (all SMs, all time) |
| `pct_of_peak_sustained_active` | When SMs were active, what fraction of their capacity was used? | `cycles_active.avg` (active time only) |

**Real-world example:**

| | 1-CTA cuTile GEMM | 152-CTA CUTLASS GEMM |
|---|---|---|
| Grid | (1,1,1) | (152,1,1) |
| TC % **elapsed** | **0.41%** | **87.26%** |
| TC % **active** | **75.41%** | **93.60%** |

The 1-CTA kernel has 0.41% elapsed TC throughput — but the active SM was 75% TC-busy. The low elapsed number reflects GPU underutilization (1/152 SMs), not poor TC efficiency.

**Diagnostic rule:**
- `elapsed` low, `active` high → **underutilized GPU** (not enough CTAs / waves)
- Both low → **real efficiency problem** (stalls, poor IPC, wrong algorithm)
- Both high → **well-optimized kernel**

**Common pitfall:** Seeing `sm__throughput = 0.4%` and concluding "TC is barely used." Always check the `active` variant and grid size first.

---

## 3. SM Throughput = max(pipe utilizations), not sum

**The gotcha:** `sm__throughput.avg.pct_of_peak_sustained_elapsed` is the **maximum** of all `sm__pipe_*_cycles_active.avg.pct_of_peak_sustained_elapsed` values, not their sum.

**Why it matters:** If TC is at 87% and ALU is at 4%, SM Throughput is 87%, not 91%. The metric identifies the bottleneck pipe, not total utilization.

**Verified:** In the 152-CTA CUTLASS GEMM report:
```
sm__throughput.avg.pct_of_peak_sustained_elapsed              = 87.259425
sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed   = 87.259425  ← identical
```

---

## 4. High pipe duty cycle with low IPC is normal for Tensor Cores

**The gotcha:** A kernel can show 87% TC pipe utilization but only 0.26 IPC and 6% issue slot busy. This is not a contradiction.

**Why:** Tensor Core MMA instructions are high-latency, high-throughput. A single issued MMA occupies the TC pipe for many cycles. The "cycles active" counter increments every cycle the pipe is busy, so:
- From the **pipe active** perspective: high utilization (87%)
- From the **instruction issue** perspective: low rate (6%) — the scheduler only needs to issue occasionally

This is the hallmark of a well-structured TC kernel. Don't be alarmed by low IPC when TC utilization is high.

---

## 5. Idle SMs still clock `cycles_elapsed`

**The gotcha:** Even SMs that receive zero CTAs report non-zero `sm__cycles_elapsed`. The GPU-wide clock keeps running on all SMs during kernel execution.

**Consequence:**
- `sm__cycles_elapsed.min` is always > 0, even if `sm__cycles_active.min` = 0
- `sm__cycles_elapsed.avg` ≈ `sm__cycles_elapsed.max` (all SMs see roughly the same wall time)
- The "Elapsed Cycles" in the SOL header is the GPU-level wall clock, which may differ slightly from `sm__cycles_elapsed.avg`

---

## 6. Raw cycle counts may not be in the report

**The gotcha:** NCU section definitions control which rollups are collected. Many reports only contain percentage rollups (`.pct_of_peak_sustained_*`) but not the underlying absolute cycle counts (`.avg`, `.sum` without further suffix).

**Workaround:** You can derive the raw count from the percentage:
```
raw_avg = pct_elapsed × cycles_elapsed.avg / 100
```

Or re-profile with explicit metric requests:
```bash
ncu --metrics sm__pipe_tc_cycles_active.avg,sm__pipe_tc_cycles_active.sum -k <kernel> -c 1 <app>
```

---

## 7. `.min` / `.max` reveal workload imbalance

**The gotcha:** `.avg` can hide severe per-SM imbalance. Always check `.min` and `.max` when available.

**Example from 152-CTA CUTLASS GEMM:**
```
sm__pipe_tc_cycles_active.avg (elapsed) = 87.26%
sm__pipe_tc_cycles_active.max (elapsed) = 90.27%
sm__pipe_tc_cycles_active.min (elapsed) = 58.38%
```

The `.min` of 58% means at least one SM had significantly lower TC utilization — likely the last CTA with less work or delayed launch. The `.avg` of 87% masks this.

**For single-CTA kernels:**
```
sm__pipe_tc_cycles_active.max (elapsed) = 62.22%  (the one active SM)
sm__pipe_tc_cycles_active.min (elapsed) = 0%      (151 idle SMs)
```

If `.min` = 0 and `.max` >> `.avg`, the kernel doesn't fill the GPU.
