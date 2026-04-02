# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Triton epilogue kernels for Linear Cross Entropy with entropy calculation.
Copy of linear_cross_entropy.py epilogues with _entropy_b reduction added.
"""

import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"],
)
@triton.jit
def forward_dp_epilogue_entropy(
    num_tokens: tl.int64,
    num_splits: tl.int64,
    ignore_index: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    num_valid_tokens_ptr,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_max_ptr,
    stride_global_max: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    global_entropy_ptr,
    stride_global_entropy: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    global_logprobs_scalar_ptr,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Forward epilogue in DP mode with entropy calculation.
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_eb = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mn_mask = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits)

        _max = tl.load(
            max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n,
            mask=mn_mask, other=0.0,
        )
        _accu = tl.load(
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
            mask=mn_mask, other=0.0,
        )
        _eb = tl.load(
            entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
            mask=mn_mask, other=0.0,
        )

        # local reduction with online max-correction
        _max_old = global_max
        _local_max = tl.max(_max, axis=1, return_indices=False)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:, None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
        global_eb = _coeff * global_eb + tl.sum(_scale * _eb, axis=1)

    # store maximum
    tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)

    # compute entropy_b = global_eb / global_accu (= E_p[z])
    eb_final = tl.fdiv(global_eb, global_accu)
    tl.store(global_entropy_b_ptr + offs_m * stride_global_entropy_b, eb_final, mask=offs_m < num_tokens)

    # convert accumulate to LSE
    global_accu = tl.log(global_accu) + global_max
    # store accumulate (now LSE)
    tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)

    # compute entropy = LSE - entropy_b
    entropy = global_accu - eb_final
    tl.store(global_entropy_ptr + offs_m * stride_global_entropy, entropy, mask=offs_m < num_tokens)

    # update logprobs
    labels = tl.load(
        labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=ignore_index
    )
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    # accumulate has been converted to LSE
    global_logprobs = global_accu - global_logprobs
    label_mask = labels != ignore_index
    global_logprobs = tl.where(label_mask, global_logprobs, 0.0)

    if REDUCTION == 0:  # no-reduction
        tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)
    elif REDUCTION == 1:  # sum
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)
    elif REDUCTION == 2:  # mean
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        global_logprobs_scalar = tl.fdiv(
            tl.sum(global_logprobs, axis=0), num_valid_tokens.to(tl.float32)
        )
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"],
)
@triton.jit
def forward_tp_epilogue_entropy(
    num_tokens: tl.int64,
    num_splits: tl.int64,
    reduced_max_ptr,
    stride_reduced_max_m: tl.int64,
    stride_reduced_max_n: tl.int64,
    original_max_ptr,
    stride_original_max_m: tl.int64,
    stride_original_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_max_ptr,
    stride_global_max: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Forward epilogue in TP mode with entropy_b reduction.
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_eb = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mn_mask = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits)

        _reduced_max = tl.load(
            reduced_max_ptr + offs_m[:, None] * stride_reduced_max_m + offs_n[None, :] * stride_reduced_max_n,
            mask=mn_mask, other=0.0,
        )
        _original_max = tl.load(
            original_max_ptr + offs_m[:, None] * stride_original_max_m + offs_n[None, :] * stride_original_max_n,
            mask=mn_mask, other=0.0,
        )
        _accu = tl.load(
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
            mask=mn_mask, other=0.0,
        )
        _eb = tl.load(
            entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
            mask=mn_mask, other=0.0,
        )

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_reduced_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        _coeff = tl.exp(_max_old - global_max)
        _scale = tl.exp(_original_max - global_max[:, None])
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
        global_eb = _coeff * global_eb + tl.sum(_scale * _eb, axis=1)

    # store
    tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
    tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)
    tl.store(global_entropy_b_ptr + offs_m * stride_global_entropy_b, global_eb, mask=offs_m < num_tokens)


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 16})], key=["num_tokens"])
@triton.jit
def forward_tp_epilogue_update_logprobs_entropy(
    num_tokens: tl.int64,
    ignore_index: tl.int64,
    num_valid_tokens_ptr,
    labels_ptr,
    stride_labels: tl.int64,
    logprobs_ptr,
    stride_logprobs: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accumulate_ptr,
    stride_accumulate: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    entropy_ptr,
    stride_entropy: tl.int64,
    logprobs_scalar_ptr,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Update logprobs and compute entropy in TP mode.
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    logprobs = tl.load(logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens)
    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens)
    eb_raw = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens)

    labels = tl.load(
        labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=ignore_index
    )
    label_mask = labels != ignore_index

    # compute entropy_b = raw_eb / accumulate
    eb_final = tl.fdiv(eb_raw, accumulate)
    tl.store(entropy_b_ptr + offs_m * stride_entropy_b, eb_final, mask=offs_m < num_tokens)

    # convert accumulate to LSE
    accumulate = tl.log(accumulate) + maximum
    tl.store(accumulate_ptr + offs_m * stride_accumulate, accumulate, mask=offs_m < num_tokens)

    # compute entropy = LSE - entropy_b
    entropy = accumulate - eb_final
    tl.store(entropy_ptr + offs_m * stride_entropy, entropy, mask=offs_m < num_tokens)

    logprobs = accumulate - logprobs
    logprobs = tl.where(label_mask, logprobs, 0.0)

    if REDUCTION == 0:  # no-reduction
        tl.store(logprobs_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens)
    elif REDUCTION == 1:  # sum
        logprobs_scalar = tl.sum(logprobs, axis=0)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
    elif REDUCTION == 2:  # mean
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        logprobs_scalar = tl.fdiv(tl.sum(logprobs, axis=0), num_valid_tokens.to(tl.float32))
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
