# NCU Profiling Guide

## Understanding the user's goal

Ask or infer:
1. **What application?** — path to executable and its arguments
2. **Which kernel(s)?** — all, by name, by regex, or by launch index
3. **What level of detail?** — quick overview vs deep dive
4. **What aspect?** — general performance, memory, compute, occupancy, roofline, or specific metrics
5. **Output format?** — interactive analysis (stdout) or saved report (.ncu-rep) for later

## Section sets

| Set | Sections collected | Overhead | When to use |
|-----|-------------------|----------|-------------|
| `basic` | SOL, Launch Stats, Occupancy | Low | Quick triage — "is this kernel fast?" |
| `full` | All standard sections + roofline | Medium | Default for serious analysis |
| `detailed` | Full + source-level counters | High | Deep dive with source correlation |

Default to `--set full` unless the user specifies otherwise or latency is a concern.

## Building the command

### Basic profiling
```bash
# Quick overview of all kernels
ncu --set full -o <output> <app> [args]

# Save report for later analysis
ncu --set full -o report <app> [args]
```

### Kernel filtering
```bash
# By exact name
ncu -k <kernel_name> --set full -o report <app> [args]

# By regex
ncu -k "regex:.*matmul.*" --set full -o report <app> [args]

# By launch index (e.g., skip warmup, profile 3rd launch)
ncu -s 2 -c 1 --set full -o report <app> [args]

# First N launches only
ncu -c 5 --set full -o report <app> [args]
```

### Specific sections
```bash
# Only occupancy and launch stats
ncu --section LaunchStats --section Occupancy -o report <app> [args]

# Memory analysis only
ncu --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables -o report <app> [args]

# Compute analysis only
ncu --section ComputeWorkloadAnalysis --section InstructionStats -o report <app> [args]

# Roofline only
ncu --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight_HierarchicalDoubleRooflineChart \
    --section SpeedOfLight_HierarchicalSingleRooflineChart \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --section SpeedOfLight_HierarchicalHalfRooflineChart \
    -o report <app> [args]
```

For exact `--section` identifiers, see [metrics.md](metrics.md).

### Specific raw metrics
```bash
# Individual metrics by name
ncu --metrics sm__pipe_tc_cycles_active.avg,sm__cycles_elapsed.avg,sm__cycles_active.avg \
    -k <kernel> -c 1 <app> [args]

# Metrics with all rollups
ncu --metrics sm__pipe_tc_cycles_active -k <kernel> -c 1 <app> [args]

# Regex metric selection
ncu --metrics "regex:sm__pipe_.*_cycles_active" -k <kernel> -c 1 <app> [args]

# Group-based selection
ncu --metrics "group:memory__dram_table" -k <kernel> -c 1 <app> [args]
```

### Metric discovery
```bash
# List all metrics available on the device
ncu --query-metrics --devices 0

# Search for metrics by keyword
ncu --query-metrics --devices 0 | grep -i "pipe_tc"

# List available sections
ncu --list-sections

# List available section sets
ncu --list-sets
```

### Output control
```bash
# Text details to stdout (no file saved)
ncu --set full --page details --print-details all -k <kernel> -c 1 <app> [args]

# CSV output for programmatic parsing
ncu --set full --csv -k <kernel> -c 1 <app> [args]

# Raw page (all raw metric names and values)
ncu --set full --page raw --csv -k <kernel> -c 1 <app> [args]

# Save to file with macros (PID, hostname, iteration)
ncu --set full -o "report_%p_%h_%i" <app> [args]
```

### Profiling controls for reproducibility
```bash
# Lock clocks for consistent measurements (IMPORTANT for benchmarking)
ncu --clock-control base --set full -o report <app> [args]

# Flush caches between replay passes
ncu --cache-control all --set full -o report <app> [args]

# Application replay (for non-deterministic kernels)
ncu --replay-mode application --set full -o report <app> [args]
```

### Multi-process / MPI profiling
```bash
# Profile all child processes
ncu --target-processes all --set full -o report <app> [args]

# MPI with per-rank reports
mpirun [mpi_args] ncu --set full -o "report_%q{OMPI_COMM_WORLD_RANK}" <app> [args]
```

### Reading back saved reports
```bash
# Text summary
ncu -i report.ncu-rep --page details --print-details all

# CSV export
ncu -i report.ncu-rep --csv > report.csv

# Raw metrics export (all raw metric names and values)
ncu -i report.ncu-rep --page raw --csv > raw.csv

# Specific section from saved report
ncu -i report.ncu-rep --page details --section SpeedOfLight

# Query specific metrics from saved report
ncu -i report.ncu-rep --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed
```

## Common profiling recipes

**"Profile my app and tell me what's slow"**
```bash
ncu --set full --clock-control base -o report <app> [args]
```

**"Compare two kernel implementations"**
```bash
ncu --set full --clock-control base -k "regex:kernel_v1|kernel_v2" -o compare <app> [args]
```

**"I only care about Tensor Core utilization"**
```bash
ncu --metrics sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__cycles_elapsed.avg,sm__cycles_active.avg \
    -k <kernel> -c 1 <app> [args]
```

**"Get raw cycle counts (not just percentages)"**
```bash
ncu --metrics sm__pipe_tc_cycles_active.avg,sm__pipe_tc_cycles_active.sum,\
sm__cycles_elapsed.avg,sm__cycles_elapsed.sum,sm__cycles_active.avg,sm__cycles_active.sum \
    -k <kernel> -c 1 <app> [args]
```

**"Check memory coalescing"**
```bash
ncu --section MemoryWorkloadAnalysis_Tables \
    --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
    -k <kernel> -c 1 <app> [args]
```

**"Source-level analysis"**
```bash
ncu --set detailed --import-source yes -k <kernel> -c 1 -o report <app> [args]
```

## After profiling

Once the profiling command completes:
1. If output was saved to a file, offer to analyze it using [analysis.md](analysis.md)
2. If output was to stdout, parse and analyze directly
3. If the user wants to refine, suggest follow-up profiling with more targeted metrics
