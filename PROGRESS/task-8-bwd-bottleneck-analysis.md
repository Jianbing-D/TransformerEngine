# Task-8: Analyze the Performance Bottleneck of the Backward Pass
**Status: Completed**

## PLAN-Task8: Backward Pass Performance Analysis

### Goal
Analyze the backward algorithm for the benchmark problem ((1, 4096), 129280, 7168), identify where the 15.49 ms is spent, and compare the current kDlogitsSplitN approach against the alternative two-kernel design (no d_logits materialization).

### Key Findings
- 15.49ms = 5.75ms BwdPartialDlogits + 4.2ms cuBLAS + 4.2ms matmul + 1ms launches
- 73% SM efficiency
- Top culprit: d_logits materialization + d_hidden repeated RMW = 12.8 GB eliminable traffic
- Best fix: fused BwdDHiddenDWeight kernel
- Detailed analysis in KNOWLEDGE/task8_bwd_bottleneck_analysis.md

## TODO-list
- [x] Read and understand backward entry point, BwdPartialDlogits kernel, and existing analysis
- [x] Compute per-split FLOP and bandwidth breakdown for all 3 operations
- [x] Estimate per-kernel latency (BwdPartialDlogits, cuBLAS addmm, torch.matmul)
- [x] Identify the dominant bottleneck (compute vs bandwidth vs launch overhead)
- [x] Analyze the two-kernel alternative (d_hidden kernel + d_weight kernel)
- [x] Analyze the single fused kernel alternative (BwdDHiddenDWeight)
- [x] Write detailed analysis report to KNOWLEDGE/task8_bwd_bottleneck_analysis.md
- [x] Update KNOWLEDGE.md index
