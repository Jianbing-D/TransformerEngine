# NCU Metrics & Sections Reference

## Metric naming convention

Pattern: `unit__(subunit?)_(pipestage?)_quantity_(qualifiers?).rollup`

**Unit abbreviations:**
| Unit | Meaning |
|------|---------|
| `sm` | Streaming Multiprocessor |
| `smsp` | SM sub-partition |
| `l1tex` | L1/Texture cache |
| `lts` | L2 cache slice |
| `dram` | Device memory (HBM/GDDR) |
| `fbpa` | Framebuffer partition |
| `gpc` | General Processing Cluster |
| `tpc` | Thread Processing Cluster |

**Roll-up suffixes:** `.sum`, `.avg`, `.min`, `.max`, `.per_cycle_active`, `.per_cycle_elapsed`, `.per_second`, `.pct_of_peak_sustained_active`, `.pct_of_peak_sustained_elapsed`, `.peak_sustained`

**Key rollup semantics:**
- `.avg` = average across **all** instances of the unit (e.g., all 152 SMs on a GB200), **not** just active ones
- `.sum` = total across all instances
- `.pct_of_peak_sustained_elapsed` = `.avg / (peak_rate × cycles_elapsed.avg) × 100` — fraction of theoretical peak over wall-clock time
- `.pct_of_peak_sustained_active` = `.avg / (peak_rate × cycles_active.avg) × 100` — fraction of theoretical peak over active time only

**Example:** `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` = TC pipe active cycles, averaged across all SMs, as % of peak over elapsed cycles. This is how SM Throughput is computed — it's the max of all such pipe metrics.

**WARNING:** These rollups have non-obvious behaviors that can lead to misinterpretation. See [gotchas.md](gotchas.md) for critical details (e.g., `.avg` divides by ALL SMs including idle ones, `elapsed` vs `active` can differ by 200x for small kernels).

---

## Key metrics by category

### Speed of Light (SOL)
| Metric | Meaning |
|--------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute (SM) Throughput — equals `max()` of all `sm__pipe_*_cycles_active.avg.pct_of_peak_sustained_elapsed` |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 cache throughput % of peak |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache throughput % of peak |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Device memory utilization % of peak |
| `sm__cycles_elapsed.avg` | Average elapsed cycles per SM |
| `sm__cycles_elapsed.sum` | Total elapsed cycles across all SMs |
| `sm__cycles_active.avg` | Average active cycles per SM |
| `sm__cycles_active.sum` | Total active cycles across all SMs |

### Compute pipe utilization
| Metric | Meaning |
|--------|---------|
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor Core complex utilization (elapsed) |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active` | Tensor Core complex utilization (active) |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor MMA execution pipe |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor Memory (TMEM) pipe |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed` | ALU (integer/logic) pipe |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | FMA pipe |
| `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed` | FP64 pipe |
| `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed` | Shared (FP64+Tensor) pipe |

### Occupancy
| Metric | Meaning |
|--------|---------|
| `sm__maximum_warps_avg_per_active_cycle` | Theoretical active warps per SM |
| `sm__maximum_warps_per_active_cycle_pct` | Theoretical occupancy % |
| `launch__occupancy_limit_registers` | Occupancy limited by register pressure |
| `launch__occupancy_limit_shared_mem` | Occupancy limited by shared memory |
| `launch__occupancy_limit_blocks` | Occupancy limited by max blocks per SM |
| `launch__occupancy_limit_warps` | Occupancy limited by block size |

### Memory
| Metric | Meaning |
|--------|---------|
| `l1tex__t_sectors_hit.sum` | L1 cache sector hits |
| `lts__t_sectors_miss.sum` | L2 cache misses |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | Shared memory bank conflicts |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | Shared memory bank conflicts (shared mem only) |
| Request-to-sector ratio | Memory coalescing efficiency (lower = better) |

### Scheduler / Warp State
| Metric | Meaning |
|--------|---------|
| `smsp__pcsamp_warps_issue_stalled_barrier` | Stalled at `__syncthreads()` |
| `smsp__pcsamp_warps_issue_stalled_long_scoreboard` | Waiting for global/local memory (most common memory stall) |
| `smsp__pcsamp_warps_issue_stalled_short_scoreboard` | Waiting for shared memory / SFU / constant loads |
| `smsp__pcsamp_warps_issue_stalled_lg_throttle` | Local/global memory queue full |
| `smsp__pcsamp_warps_issue_stalled_math_pipe_throttle` | Math pipeline oversubscribed |
| `smsp__pcsamp_warps_issue_stalled_mio_throttle` | Memory I/O queue full |
| `smsp__pcsamp_warps_issue_stalled_tex_throttle` | Texture instruction queue full |
| `smsp__pcsamp_warps_issue_stalled_no_instructions` | Instruction cache miss |
| `smsp__pcsamp_warps_issue_stalled_not_selected` | Eligible but not selected (healthy) |
| `smsp__pcsamp_warps_issue_stalled_drain` | Waiting for memory writes after EXIT |
| `smsp__pcsamp_warps_issue_stalled_membar` | Waiting on `__threadfence()` |
| `smsp__pcsamp_warps_issue_stalled_sleeping` | Threads blocked/yielded |
| `smsp__pcsamp_warps_issue_stalled_wait` | Fixed-latency execution dependency |
| `smsp__pcsamp_warps_issue_stalled_selected` | Issued an instruction (the "good" state) |

### Instruction / Compute
| Metric | Meaning |
|--------|---------|
| `sm__inst_executed.sum` | Total instructions executed |
| `smsp__average_warp_latency.ratio` | Average warp pipeline latency |
| `sm__inst_executed_pipe_tc.avg.pct_of_peak_sustained_elapsed` | TC pipe instruction rate (low even when pipe is busy — TC instructions are high-latency) |

**Execution pipelines:** ALU, FMA, FP64, LSU, TEX, Tensor, TC, TMA, TMEM, XU (transcendental), CBU (convergence/branches), ADU, Uniform

---

## NCU section identifiers

These are the exact `--section` values accepted by `ncu`:

| `--section` identifier | Focus |
|----------------------|-------|
| `SpeedOfLight` | SOL compute vs memory % of peak |
| `SpeedOfLight_RooflineChart` | FP32/FP64 overview roofline |
| `SpeedOfLight_HierarchicalDoubleRooflineChart` | FP64 hierarchical roofline (DRAM/L2/L1) |
| `SpeedOfLight_HierarchicalHalfRooflineChart` | FP16 hierarchical roofline (DRAM/L2/L1) |
| `SpeedOfLight_HierarchicalSingleRooflineChart` | FP32 hierarchical roofline (DRAM/L2/L1) |
| `SpeedOfLight_HierarchicalTensorRooflineChart` | Tensor Core hierarchical roofline (DRAM/L2/L1) |
| `ComputeWorkloadAnalysis` | IPC, pipeline utilization breakdown |
| `MemoryWorkloadAnalysis` | High-level memory stats |
| `MemoryWorkloadAnalysis_Chart` | Memory chart visualization data |
| `MemoryWorkloadAnalysis_Tables` | Detailed per-unit memory tables |
| `SchedulerStats` | Active/eligible/issued warps per scheduler |
| `WarpStateStats` | Warp stall breakdown by reason |
| `InstructionStats` | SASS instruction mix, opcode categories |
| `LaunchStats` | Grid/block size, registers, shared mem, occupancy limits |
| `Occupancy` | Theoretical vs achieved, limiting factors |
| `SourceCounters` | Source-level metrics, warp stall sampling, branch efficiency |
| `PmSampling` | Timeline/periodic sampling |
| `NVLink_Topology` | NVLink connection info |
| `NumaAffinity` | NUMA node mapping |
| `GPUAndMemoryWorkloadDistribution` | Per-SM workload balance |

### Section sets
| `--set` | What's included |
|---------|----------------|
| `basic` | SpeedOfLight, LaunchStats, Occupancy |
| `full` | All standard sections including rooflines |
| `detailed` | Full + SourceCounters with source-level detail |

### Useful section combinations

**Quick triage:**
`--section SpeedOfLight --section LaunchStats --section Occupancy`

**Memory deep dive:**
`--section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables`

**Compute deep dive:**
`--section ComputeWorkloadAnalysis --section InstructionStats --section SchedulerStats --section WarpStateStats`

**All rooflines:**
`--section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart --section SpeedOfLight_HierarchicalSingleRooflineChart --section SpeedOfLight_HierarchicalHalfRooflineChart --section SpeedOfLight_HierarchicalDoubleRooflineChart`

---

## Roofline model interpretation

- **X-axis (log):** Arithmetic Intensity = FLOPs / bytes of memory traffic
- **Y-axis (log):** Achieved FLOP/s
- **Sloped line:** Memory bandwidth ceiling
- **Horizontal lines:** Peak compute ceilings (FP32, FP64, Tensor, etc.)
- **Ridge point:** Where bandwidth meets compute ceiling

If kernel is below the slope → memory-bound (improve access patterns or reduce traffic).
If kernel is below the horizontal ceiling → compute-bound (improve instruction efficiency).
Hierarchical roofline includes L1 and L2 bandwidth ceilings in addition to DRAM.

---

## NCU documentation URLs

### Primary references
| Topic | URL |
|-------|-----|
| Profiling Guide (main) | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html` |
| Metrics Reference | `https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#metrics-reference` |
| CLI Reference | `https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html` |

### Section-specific deep links
| Section | URL |
|---------|-----|
| Speed of Light | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#speed-of-light` |
| Roofline Charts | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts` |
| Memory Charts | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart` |
| Occupancy | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy` |
| Scheduler Stats | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#scheduler-statistics` |
| Warp State Stats | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-state-statistics` |
| Compute Workload | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#compute-workload-analysis` |
| Memory Workload | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-workload-analysis` |
| Source Counters | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#source-counters` |
| Launch Stats | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#launch-statistics` |
| Sections & Rules | `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules` |

### Search pattern for unknown metrics
Use WebSearch with: `"<metric_name>" site:docs.nvidia.com nsight compute`

---

## Metric name decoding cheat sheet

**Common quantity keywords:**
| Keyword | Meaning |
|---------|---------|
| `throughput` | Utilization rate (usually as % of peak) |
| `sectors` | 32-byte aligned memory chunks |
| `requests` | Memory access requests from warps |
| `wavefronts` | Unique parallel work packages at a memory unit |
| `bank_conflicts` | Shared memory accesses serialized due to same-bank collision |
| `inst_executed` | Instructions actually executed (not predicated off) |
| `inst_issued` | Instructions issued to a pipeline (may include replays) |
| `warps_active` | Warps resident and allocated on an SM |
| `warps_eligible` | Active warps ready to issue (no dependency stall) |
| `cycles_active` | Cycles where the unit had work to do |
| `cycles_elapsed` | Wall-clock cycles (includes idle) |
| `bytes` | Raw byte count transferred |
| `hit_rate` | Cache hit ratio (higher = better) |
| `divergent_branches` | Branches where threads in a warp took different paths |
| `predicated_on` | Instructions that were not masked by predication |

**Common subunit/pipe names:**
| Name | What it is |
|------|-----------|
| `pipe_alu` | Integer/logic ALU pipeline |
| `pipe_fma` | Fused multiply-add pipeline |
| `pipe_fp64` | Double-precision pipeline |
| `pipe_lsu` | Load/store unit pipeline |
| `pipe_tex` | Texture pipeline |
| `pipe_tensor` | Tensor Core MMA execution pipeline |
| `pipe_tc` | Tensor Core complex (union of tensor + tmem activity) |
| `pipe_tma` | Tensor Memory Accelerator pipeline (Hopper+) |
| `pipe_xu` | Transcendental (special function) pipeline |
| `mem_tensor` / `pipe_tmem` | Tensor Memory port (Blackwell) |
| `pipe_uniform` | Uniform datapath pipeline |
| `pipe_adu` | Address divergence unit |
