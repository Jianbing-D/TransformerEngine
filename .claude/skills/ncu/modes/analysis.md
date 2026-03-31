# NCU Report Analysis Guide

## How to receive the report

The user may provide the report in one of these ways:
- Pasted text (console output from `ncu`)
- A path to an `.ncu-rep` file — use `ncu -i <file> --page details --print-details all` to extract it
- A path to a CSV export — read it directly

If the user provides a binary `.ncu-rep` file, run:
```bash
ncu -i <file> --page details --print-details all
```
If that fails (ncu not available), ask the user to export with `ncu -i <file> --csv > report.csv`.

## Step 0: Extract and identify all metrics

Before analyzing, you MUST systematically extract every metric from the report.

### 0a. Parse all metrics from the report

Scan the entire report and collect every metric name-value pair. Metrics appear in formats like:
- `metric_name    value   unit` (tabular)
- `metric_name,value` (CSV)
- `Section: MetricName ... value` (structured text)
- Percentage bars with labels like `SM [===       ] 30.5%`

Build a complete list. Group them by section (SOL, Occupancy, Memory, Warp State, etc.).

### 0b. Look up unfamiliar metrics

For any metric you encounter that is NOT covered in [metrics.md](metrics.md), you MUST look it up:

1. **First, decode the metric name** using the naming convention in [metrics.md](metrics.md)

2. **If the metric's meaning is still unclear, search the NCU docs:**
   - Search: `"<metric_name>" site:docs.nvidia.com nsight compute`
   - Or fetch the profiling guide: `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html`
   - Or fetch the metrics reference: `https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#metrics-reference`

3. **For section-specific metrics**, fetch the section documentation (URLs in [metrics.md](metrics.md))

4. **For GPU architecture-specific behavior** (e.g., Ampere vs Hopper vs Blackwell):
   - Search: `"<architecture>" site:docs.nvidia.com nsight compute`

Do NOT skip or hand-wave any metric. If you see it in the report, you must explain it.

### 0c. Cross-reference metric values

After identifying all metrics, check for relationships:
- SOL compute % vs SOL memory % → determines kernel classification
- **SOL elapsed % vs active %** → distinguishes GPU underutilization from per-SM inefficiency (see below)
- Theoretical vs achieved occupancy → indicates scheduling efficiency
- L1 hit rate vs L2 hit rate vs DRAM throughput → shows where data comes from
- Dominant warp stall reason → confirms the bottleneck diagnosis from SOL
- Instruction mix across pipelines → reveals compute balance
- **`.min` vs `.max` rollups** → reveals per-SM imbalance (if `.min` = 0 and `.max` is high, some SMs are idle)

These cross-references are essential — a single metric in isolation can be misleading.

### 0d. Detect GPU underutilization early

Before diving into per-SM analysis, check whether the kernel fills the GPU:
- **Grid Size vs # SMs** — if Grid < #SMs, many SMs are completely idle
- **Waves Per SM** — if < 1.0, not all SMs get work

If the kernel is underutilized, SOL metrics will be misleadingly low. See [gotchas.md](gotchas.md) (especially gotchas #1 and #2) for how `elapsed` vs `active` metrics diverge and how to report both clearly.

---

## Analysis framework

Work through these steps. Be conversational — explain what each metric means as you go.

### Step 1: Kernel overview

Summarize the basics:
- Kernel name, grid dimensions, block dimensions
- GPU architecture and compute capability
- Register usage per thread, shared memory per block
- Number of launches profiled

### Step 2: Speed of Light (SOL) analysis

This is the most important section. SOL shows achieved % of theoretical peak.

| Metric | What it means |
|--------|---------------|
| SM throughput % | How well the compute units are utilized — defined as `max(sm__pipe_*_cycles_active.avg.pct_of_peak_sustained_elapsed)` across all pipes, i.e. the highest pipe utilization averaged across all SMs over elapsed cycles |
| Memory throughput % | How well memory bandwidth is utilized |

**IMPORTANT: SOL uses `pct_of_peak_sustained_elapsed`, which averages across ALL SMs on the GPU (not just active ones).** If the kernel launches fewer CTAs than SMs, SOL will be low even if the active SMs are working efficiently. Always check:
1. **Waves Per SM** and **Grid Size** — does the kernel fill the GPU?
2. If Waves < 1, also look at the **active-cycle** variants (in the Compute Workload Analysis section's "% of active cycles" table) to see per-SM efficiency

**Classify the kernel:**
- **Compute-bound**: SM % >> Memory % — the ALU/FMA/tensor pipelines are the bottleneck
- **Memory-bound**: Memory % >> SM % — bandwidth to DRAM/cache is the bottleneck
- **Latency-bound**: Both SM % and Memory % are low — the GPU is stalling, not doing useful work
- **Underutilized**: Both SM % and Memory % are low, BUT active-cycle percentages are high — the kernel doesn't launch enough CTAs to fill the GPU. This is not a per-SM efficiency issue but a parallelism issue.

Tell the user which category their kernel falls into and what that means in plain language. For underutilized kernels, make clear the distinction: "Your SM is working efficiently (X% active), but only Y out of Z SMs are doing work."

### Step 3: Occupancy analysis

Explain occupancy simply: it's how many warps (groups of 32 threads) can run simultaneously on each SM, as a fraction of the hardware maximum.

Key things to report:
- Theoretical occupancy vs achieved occupancy
- What limits occupancy (registers, shared memory, block size, barriers)
- Whether low occupancy is actually a problem here (high occupancy != high performance; it matters most for latency-bound kernels)

Use this rule of thumb:
- \>50%: Usually adequate for hiding memory latency
- 25-50%: May cause stalls — worth investigating
- <25%: Likely leaving performance on the table

### Step 4: Memory analysis

Explain the memory hierarchy: Registers -> L1/Shared Memory -> L2 Cache -> Device Memory (DRAM)

Report on:
- **L1 hit rate** — >80% is good. Low hit rate means poor spatial locality or thrashing
- **L2 hit rate** — >50% is reasonable
- **DRAM throughput** — compare to SOL %
- **Memory coalescing** — the request-to-sector ratio. Ideal is 1:1 (each warp request maps to minimal sectors). High ratios like 1:16 or 1:32 indicate uncoalesced access patterns
- **Bank conflicts** — in shared memory, bank conflicts serialize parallel accesses
- **Global memory access patterns** — stride-1 (coalesced) vs strided vs random

### Step 5: Warp stall analysis

If warp state stats are available, explain the top stall reasons:

| Stall | Plain-language meaning | What to do |
|-------|----------------------|------------|
| `long_scoreboard` | Waiting for global/local memory loads to complete | Improve memory access patterns, prefetch, increase occupancy |
| `short_scoreboard` | Waiting for shared memory or special function results | Reduce bank conflicts, reduce SFU usage |
| `barrier` | Waiting at `__syncthreads()` because other threads in the block haven't arrived | Reduce work imbalance before sync points |
| `lg_throttle` | Too many outstanding global memory requests | Reduce memory pressure per thread |
| `math_pipe_throttle` | Math pipeline is overloaded | Spread computation across cycles |
| `not_selected` | Warp was ready but another warp was picked | Not a problem — indicates healthy scheduling |
| `no_instructions` | Instruction cache miss | Kernel is very large; consider splitting |
| `mio_throttle` | Memory I/O queue full | Reduce shared memory or constant memory pressure |

Focus only on stall reasons that actually dominate. Don't enumerate stalls that are <5%.

### Step 6: Roofline model (if available)

If roofline data is present:
- Explain the axes (arithmetic intensity on X, FLOP/s on Y, both log scale)
- Where the kernel sits relative to the roofline
- Whether the kernel is below the memory bandwidth slope or the compute ceiling
- What "moving toward the roofline" would require

### Step 7: Optimization recommendations

Based on your analysis, provide **specific, actionable** suggestions. Prioritize by expected impact.

**For memory-bound kernels:**
- Improve coalescing (access consecutive addresses within a warp)
- Use shared memory as a scratchpad for reused data
- Align accesses to 128-byte cache line boundaries
- Consider data layout changes (AoS -> SoA)
- Use vectorized loads (`float4`, `int4`) when alignment permits
- Reduce unnecessary global memory traffic

**For compute-bound kernels:**
- Use faster intrinsics (`__fmaf_rn`, `__expf`, `__rsqrtf`)
- Balance work across pipelines (don't overload one pipe)
- Consider using Tensor Cores if doing matrix math
- Reduce register pressure with `__launch_bounds__`
- Use half-precision (`__half`, `__half2`) where accuracy permits

**For latency-bound kernels:**
- Increase occupancy (reduce registers via `--maxrregcount` or `__launch_bounds__`)
- Reduce shared memory per block
- Increase block size or grid size
- Restructure code to reduce synchronization
- Use instruction-level parallelism (independent operations between dependent ones)

**General suggestions:**
- Use `__restrict__` pointers to enable compiler optimizations
- Avoid warp divergence in hot loops
- Minimize thread divergence at `__syncthreads()` boundaries
- Consider kernel fusion to reduce launch overhead and memory round-trips

End with a prioritized summary: **"Top 3 things to try first"**
