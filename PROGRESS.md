# Progress

## Task-1: Understand Linear-Cross-Entropy Fusion Code Structure
**Status: Completed**

### PLAN-Task1: Code Exploration & Documentation

#### Architecture Overview
Study and document the complete code structure of the Linear-Cross-Entropy (LCE) fusion kernel in TransformerEngine, focusing on tensor shapes, data types, and algorithm flow.

#### Files to Examine
1. `tests/pytorch/test_linear_cross_entropy.py` — test usage and problem sizes
2. `transformer_engine/pytorch/linear_cross_entropy.py` — public API + autograd Function
3. `transformer_engine/pytorch/cutedsl/linear_cross_entropy_entry.py` — host-side forward/backward orchestration
4. `transformer_engine/common/cutedsl/linear_cross_entropy/` — kernel implementations
   - `blackwell/fwd_mainloop.py` — SM100 forward GEMM + softmax epilogue
   - `blackwell/bwd_partial_dlogits.py` — SM100 backward d_logits kernel
   - `blackwell/bwd_dHdW.py` — SM100 fused d_hidden + d_weight kernel (WIP)
   - `utils.py` — enums
   - `scheduler.py` — persistent tile scheduler
   - `ptx.py` — inline PTX helpers
5. `transformer_engine/common/triton/linear_cross_entropy.py` — Triton epilogue kernels

#### TODO-list
- [x] Read and understand the public API layer
- [x] Read and understand the forward host function and tensor shapes
- [x] Read and understand FwdMainLoop (SM100 GEMM + online softmax)
- [x] Read and understand BwdPartialDlogits (SM100 d_logits kernel)
- [x] Read and understand BwdDHiddenDWeight structure (fused backward WIP)
- [x] Read and understand Triton epilogue kernels
- [x] Read and understand scheduler and ptx helpers
- [x] Write comprehensive KNOWLEDGE.md report
- [x] Update TASK.yaml to mark task-1 as completed
