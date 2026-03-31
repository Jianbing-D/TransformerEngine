# Progress

This file is the **index**. All detailed plans, TODO-lists, and execution records live in `PROGRESS/`.

---

## Current Status
**Active**: None — all tasks completed.

---

## Task Index

| File | Status | Summary |
|------|--------|---------|
| [PROGRESS/task-1-understand-lce.md](PROGRESS/task-1-understand-lce.md) | Completed | Initial code structure study — API, forward/backward kernels, Triton epilogues |
| [PROGRESS/task-2-static-scheduler-fwd.md](PROGRESS/task-2-static-scheduler-fwd.md) | Completed | Applied StaticPersistentScheduler to fwd_mainloop.py |
| [PROGRESS/task-3-brainstorm-lce.md](PROGRESS/task-3-brainstorm-lce.md) | Completed | Mathematical analysis, FLOP/bandwidth accounting, improvement brainstorm |
| [PROGRESS/task-4-optimize-bwd-partial-dlogits.md](PROGRESS/task-4-optimize-bwd-partial-dlogits.md) | Completed | 3-phase optimization: persistent scheduler → TMA S2G store → 2-CTA MMA. Net: 162.43→136.86 μs |
| [PROGRESS/task-5-fix-bwd-bugs.md](PROGRESS/task-5-fix-bwd-bugs.md) | Completed | Fixed 5 bugs: deadlock, OOB index, scope violation, stage mismatch, per-subtile pipeline |
| [PROGRESS/task-6-fix-scheduler-grid.md](PROGRESS/task-6-fix-scheduler-grid.md) | Completed | Fixed grid sizing for clusters: 304→152 CTAs, 11.5% speedup |
| [PROGRESS/task-7-hoist-tensor-partitions.md](PROGRESS/task-7-hoist-tensor-partitions.md) | Completed | Moved all tensor partitions outside persistent while loops in bwd_partial_dlogits |
| [PROGRESS/task-8-bwd-bottleneck-analysis.md](PROGRESS/task-8-bwd-bottleneck-analysis.md) | Completed | Backward 15.49ms breakdown: 37% BwdPartialDlogits, 27% cuBLAS, 27% matmul, 7% launches |
| [PROGRESS/task-9-2cta-fwd-mainloop.md](PROGRESS/task-9-2cta-fwd-mainloop.md) | Completed | Applied 2-CTA MMA to forward kernel, all tests passing |
| [PROGRESS/task-10-static-scheduler-fwd-v2.md](PROGRESS/task-10-static-scheduler-fwd-v2.md) | Trial-and-Quit | Static scheduler in fwd_mainloop — abandoned, no SM wave improvement found |
| [PROGRESS/task-11-add-entropy.md](PROGRESS/task-11-add-entropy.md) | Completed | Add entropy calculation to LCE fusion — 2.5% fwd overhead, 4.5% total overhead, all tests passing |
