# Task-1: Understand Linear-Cross-Entropy Fusion Code Structure
**Status: Completed**

## TODO-list
- [x] Read and understand the public API layer
- [x] Read and understand the forward host function and tensor shapes
- [x] Read and understand FwdMainLoop (SM100 GEMM + online softmax)
- [x] Read and understand BwdPartialDlogits (SM100 d_logits kernel)
- [x] Read and understand BwdDHiddenDWeight structure (fused backward WIP)
- [x] Read and understand Triton epilogue kernels
- [x] Read and understand scheduler and ptx helpers
- [x] Write comprehensive KNOWLEDGE.md report
- [x] Update TASK.yaml to mark task-1 as completed
