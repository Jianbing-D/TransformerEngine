# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Shared infrastructure for Linear-Cross-Entropy entry points.

Implementation — single arch gate, resolves gpu_entry + function pointers.
FwdConfig / BwdConfig — per-entry-point compiled kernel caches.
"""

import typing
import os
from dataclasses import dataclass, field
from functools import lru_cache

import cutlass.cute as cute
import torch

from transformer_engine.common.cutedsl.linear_cross_entropy import utils


class Implementation:
    """
    Singleton that detects GPU architecture, imports the corresponding
    kernel module and entry-point functions.

    This is the single arch gate for the entire LCE stack.
    Adding a new architecture (e.g. Hopper) requires one elif branch here.

    Attributes:
        gpu_entry:             arch-specific kernel module (e.g. blackwell)
        forward_func:          baseline forward entry point
        backward_func:         baseline backward entry point
        forward_entropy_func:  entropy forward entry point
        backward_entropy_func: entropy backward entry point
    """

    _instance: typing.Optional["Implementation"] = None

    def __new__(cls) -> "Implementation":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        assert torch.cuda.is_available(), "CUDA is not available"
        device = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(device)

        if cc[0] == 10:
            from transformer_engine.common.cutedsl.linear_cross_entropy import blackwell as gpu_entry
            self.gpu_entry = gpu_entry
        else:
            raise ValueError(f"Unsupported architecture: {cc[0]}. Shall be delegated to Triton")

        # Entry modules are imported after gpu_entry is set, because they
        # may call _get_impl().gpu_entry at function-call time (not import time).
        from transformer_engine.pytorch.cutedsl import linear_cross_entropy_entry as impl
        from transformer_engine.pytorch.cutedsl import linear_cross_entropy_with_entropy_entry as entropy_impl

        self.forward_func: typing.Callable[..., typing.Any] = impl.forward
        self.backward_func: typing.Callable[..., typing.Any] = impl.backward
        self.forward_entropy_func: typing.Callable[..., typing.Any] = entropy_impl.forward
        self.backward_entropy_func: typing.Callable[..., typing.Any] = entropy_impl.backward

        self._initialized = True


@lru_cache(maxsize=1)
def _get_impl() -> Implementation:
    return Implementation()


@dataclass
class FwdConfig:
    """
    Forward pass configuration.
    Each entry point (baseline / entropy) maintains its own instance
    since they cache different compiled kernels.
    """

    _dedicated_stream: torch.cuda.Stream = field(default_factory=torch.cuda.Stream)
    _dedicated_events: typing.List[torch.cuda.Event] = field(default_factory=list)
    _initialized: bool = field(default=False)
    _fwd_mainloop_kernels: typing.Dict[str, cute.kernel] = field(default_factory=dict)
    _vocab_per_split: int = field(
        default=int(os.environ.get("LCE_FWD_VOCAB_SPLIT_SIZE", 512 * 6))
    )


@dataclass
class BwdConfig:
    """
    Backward pass configuration.
    Each entry point (baseline / entropy) maintains its own instance
    since they cache different compiled kernels.
    """

    _bwd_kernel: typing.Dict[str, cute.kernel] = field(default_factory=dict)
    _vocab_per_split: int = field(
        default=int(os.environ.get("LCE_BWD_VOCAB_SPLIT_SIZE", 512 * 7))
    )
    _backward_method: utils.BackwardMethodEnum = field(
        default=utils.BackwardMethodEnum.kDlogitsSplitN
    )
