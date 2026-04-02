# Knowledge Base: Linear-Cross-Entropy Fusion in TransformerEngine

This file is the **index**. All detailed content lives in `KNOWLEDGE/`.

---

## Sub-Files

| File | Contents |
|------|----------|
| [KNOWLEDGE/overview.md](KNOWLEDGE/overview.md) | High-level purpose, code structure, file map |
| [KNOWLEDGE/api_and_tensors.md](KNOWLEDGE/api_and_tensors.md) | Public API signature, tensor shapes/dtypes, vocab splitting strategy |
| [KNOWLEDGE/algorithms.md](KNOWLEDGE/algorithms.md) | Forward (online softmax), backward (`kDlogitsSplitN`), math derivations |
| [KNOWLEDGE/kernel_architecture.md](KNOWLEDGE/kernel_architecture.md) | SM100 CTA layouts, pipeline stages, TMEM/TMA usage, JIT patterns, `StaticPersistentScheduler` |
| [KNOWLEDGE/distributed.md](KNOWLEDGE/distributed.md) | TP/SP semantics, `ignore_index` mechanic, two-stream TP forward |
| [KNOWLEDGE/utilities_and_tests.md](KNOWLEDGE/utilities_and_tests.md) | `ptx.py`, `utils.py` enums, test problem sizes, correctness tolerances |
| [KNOWLEDGE/lessons_learned.md](KNOWLEDGE/lessons_learned.md) | L1–L17 lessons, Task-2 persistent scheduler insights, CuteDSL scoping rules, Task-4 optimization results, Task-5 bug fixes (PipelineAsync phase semantics, partition_C 2-CTA indexing), Task-6 cluster-aware grid sizing, testing practices (sequential GPU tests, non-contiguous autograd gradients, non-scalar loss reduction) |
| [KNOWLEDGE/task3_analysis.md](KNOWLEDGE/task3_analysis.md) | FLOP/bandwidth accounting, improvement options analysis, precision analysis, priority ranking |
| [KNOWLEDGE/cutlass_gemm_examples.md](KNOWLEDGE/cutlass_gemm_examples.md) | CUTLASS Blackwell GEMM optimization techniques: persistent scheduler, TMA S2G store (R2S→fence→barrier→S2G pattern, `tma_partition` gotchas), 2-CTA MMA (full API reference: CTA rank, leader gating, `cluster_layout_vmnk`, pipeline `cta_layout_vmnk`, `tx_count` doubling, multicast masks, scheduler cluster awareness, TMEM sharing) |
| [KNOWLEDGE/task8_bwd_bottleneck_analysis.md](KNOWLEDGE/task8_bwd_bottleneck_analysis.md) | Backward pass performance bottleneck analysis: 15.49ms breakdown (37% BwdPartialDlogits, 27% cuBLAS addmm, 27% matmul, 7% launches), 73% efficiency, bottleneck ranking, two-kernel vs fused-kernel alternatives, recommendations |

---

## Quick Reference

**Key invariant**: `accumulate` saved for backward is **LSE** (not raw sum) — see L1 in lessons_learned.md.

**Current backward**: `kDlogitsSplitN` — 42 sequential splits, each with BwdPartialDlogits (2-CTA MMA, persistent scheduler, TMA S2G store) + cuBLAS + matmul.

**BwdPartialDlogits optimized**: 162.43 μs → 136.86 μs (16% total: TMA S2G store + cluster-aware grid). 2-CTA MMA neutral (epilogue-bound). See L6–L14 in lessons_learned.md.

**Backward bottleneck (Task-8)**: 15.49ms = 5.75ms BwdPartialDlogits + 4.2ms cuBLAS + 4.2ms matmul + 1ms launches. 73% SM efficiency. Top culprit: d_logits materialization + d_hidden repeated RMW = 12.8 GB eliminable traffic. Best fix: fused BwdDHiddenDWeight.

**WIP**: `BwdDHiddenDWeight` in `bwd_dHdW.py` — fused backward kernel, not yet wired into entry point.

**Architecture gate**: Blackwell/SM100 only (`cc[0] == 10`). Other GPUs raise `ValueError`.
