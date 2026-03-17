# Distributed (TP/SP) Semantics

## Tensor Parallel (TP), `sequence_parallel=False`
- `hidden` and `labels`: identical on all TP ranks (replicated)
- `weight`: sharded along vocab dim → each rank holds `(local_vocab_size, dim)`
- `logprobs`: identical on all ranks after all-reduces

## Sequence Parallel (SP), `sequence_parallel=True`
- `hidden`: sharded along token dim → each rank holds `(local_num_tokens, dim)`
- An `all_gather_into_tensor` is done in forward to assemble `global_hidden` before GEMM
- `labels`: replicated (full length on all ranks)
- Backward: `d_hidden` is all-reduced across ranks, then sliced for local shard

## Key `ignore_index` Mechanic
In TP/SP mode, `_logprobs` is zero-initialized and filled atomically. Only the rank whose vocab shard contains the label contributes a non-zero value. The `all_reduce(SUM)` then gathers the true logit.

## Two-Stream Pattern for TP Forward
The TP forward uses a dedicated CUDA stream to overlap `all_reduce(_logprobs)` with the `forward_tp_epilogue` kernel that processes max/accumulate. Events synchronize the two streams before `forward_tp_epilogue_update_logprobs`.
