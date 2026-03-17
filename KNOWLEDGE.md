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
| [KNOWLEDGE/lessons_learned.md](KNOWLEDGE/lessons_learned.md) | L1–L5 lessons, Task-2 persistent scheduler insights, CuteDSL scoping rules |
| [KNOWLEDGE/task3_analysis.md](KNOWLEDGE/task3_analysis.md) | FLOP/bandwidth accounting, improvement options analysis, precision analysis, priority ranking |

---

## Quick Reference

**Key invariant**: `accumulate` saved for backward is **LSE** (not raw sum) — see L1 in lessons_learned.md.

**Current backward**: `kDlogitsSplitN` — 42 sequential splits, each with BwdPartialDlogits + cuBLAS + matmul.

**WIP**: `BwdDHiddenDWeight` in `bwd_dHdW.py` — fused backward kernel, not yet wired into entry point.

**Architecture gate**: Blackwell/SM100 only (`cc[0] == 10`). Other GPUs raise `ValueError`.
