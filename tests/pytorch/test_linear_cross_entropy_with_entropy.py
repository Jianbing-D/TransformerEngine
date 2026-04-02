# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import typing
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from transformer_engine.pytorch import linear_cross_entropy

_only_profile: bool = os.environ.get("ONLY_PROFILE", "0") == "1"


@dataclass
class DistContext:
    rank: int
    world_size: int
    group: dist.ProcessGroup
    is_chief: bool


@pytest.fixture(scope="module")
def distributed_context():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("Requires torchrun with multiple GPUs (WORLD_SIZE >= 2)")

    is_external_init = dist.is_initialized()

    if not is_external_init:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD

    print(f"[INFO]: Initialized Rank: {rank} / {world_size}")

    context = DistContext(rank=rank, world_size=world_size, group=group, is_chief=(rank == 0))

    yield context

    if not is_external_init:
        dist.destroy_process_group()


def get_device_arch_version():
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    return cc[0]


def torch_entropy_reference(hidden, weight, labels, reduction, ignore_index, temperature=1.0):
    """Compute logprobs and entropy using vanilla PyTorch."""
    logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
    scaled_logits = logits / temperature
    logprobs = torch.nn.functional.cross_entropy(
        scaled_logits.view(-1, scaled_logits.shape[-1]),
        labels.view(-1),
        reduction=reduction,
        ignore_index=ignore_index,
    )

    lse = torch.logsumexp(scaled_logits, dim=-1)
    softmax = torch.softmax(scaled_logits, dim=-1)
    entropy_b = (softmax * scaled_logits).sum(dim=-1)
    entropy = lse - entropy_b

    return logprobs.to(torch.float32), entropy.to(torch.float32)


# ===========================================================================
# Data Parallel (single GPU) tests
# ===========================================================================
@pytest.mark.skipif(_only_profile, reason="Skipping test in profile mode")
@pytest.mark.skipif(
    "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != "1",
    reason="Requires single GPU",
)
@pytest.mark.skipif(
    get_device_arch_version() != 10, reason="Requires GPU architecture = 10"
)
class TestLinearCrossEntropyWithEntropyDataParallel:

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 152063, 8192),
            ((1, 4096), 152064, 8192),
            ((2, 4096), 152064, 8192),
        ]

    @staticmethod
    def get_ignore_index():
        return [-100, 4]

    def test_kernel_launch(self):
        """Check that the entropy kernel can be launched with various sizes."""
        self.cleanup()

        num_tokens_list = [15, 26, 128, 513, 2048, 8192]
        vocab_size = 152064
        dim = 4096
        dtype = torch.bfloat16

        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda").requires_grad_()
        for num_tokens in num_tokens_list:
            hidden = torch.randn(num_tokens, dim, dtype=dtype, device="cuda").requires_grad_()
            labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda")

            logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels, reduction="mean", return_entropy=True,
            )
            assert not torch.isnan(logprobs).any(), "logprobs has NaN"
            assert not torch.isnan(entropy).any(), "entropy has NaN"
            assert entropy.shape == (num_tokens,), f"entropy shape mismatch: {entropy.shape}"

            # Test backward
            loss = logprobs + entropy.mean()
            loss.backward()
            assert hidden.grad is not None and not torch.isnan(hidden.grad).any()
            assert weight.grad is not None and not torch.isnan(weight.grad).any()
            hidden.grad = None
            weight.grad = None

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("problem", get_problems())
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("ignore_index", get_ignore_index())
    def test_correctness(self, dtype, problem, reduction, ignore_index):
        """Test that fused entropy matches PyTorch reference (forward + backward)."""
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
        if ignore_index >= 0 and ignore_index < vocabsize:
            pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            labels = pad_labels[..., 1:].contiguous()

        # Forward
        logprobs, entropy = linear_cross_entropy(
            hidden, weight, labels,
            reduction=reduction, ignore_index=ignore_index, return_entropy=True,
        )
        ref_logprobs, ref_entropy = torch_entropy_reference(
            hidden, weight, labels, reduction, ignore_index,
        )

        if reduction == "none":
            valid = labels.view(-1) != ignore_index
            torch.testing.assert_close(
                logprobs[valid], ref_logprobs[valid], atol=1e-2, rtol=1e-2,
            )
        else:
            torch.testing.assert_close(logprobs, ref_logprobs, atol=1e-2, rtol=1e-2)

        torch.testing.assert_close(entropy, ref_entropy.view(-1), atol=1e-2, rtol=1e-2)

        # Backward: use a scalar loss combining both logprobs and entropy
        if reduction == "none":
            fused_loss = logprobs.sum() + entropy.sum()
        else:
            fused_loss = logprobs + entropy.sum()
        fused_loss.backward()
        d_hidden = hidden.grad.clone()
        d_weight = weight.grad.clone()

        hidden.grad = None
        weight.grad = None
        if reduction == "none":
            ref_loss = ref_logprobs.sum() + ref_entropy.view(-1).sum()
        else:
            ref_loss = ref_logprobs + ref_entropy.view(-1).sum()
        ref_loss.backward()

        torch.testing.assert_close(d_hidden, hidden.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(d_weight, weight.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_performance(self, problem, dtype, reduction, ignore_index):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        baseline_fwd_latency = list()
        baseline_bwd_latency = list()
        entropy_fwd_latency = list()
        entropy_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            # -------- baseline (no entropy) -------- #
            start_event.record()
            baseline_logprobs = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index,
                return_entropy=False,
            )
            end_event.record()
            torch.cuda.synchronize()
            baseline_fwd_latency.append(start_event.elapsed_time(end_event))

            g_logprobs = torch.empty_like(baseline_logprobs).uniform_(-0.1, 0.1)
            start_event.record()
            torch.autograd.grad(
                (baseline_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            baseline_bwd_latency.append(start_event.elapsed_time(end_event))

            # -------- entropy -------- #
            start_event.record()
            entropy_logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index,
                return_entropy=True,
            )
            end_event.record()
            torch.cuda.synchronize()
            entropy_fwd_latency.append(start_event.elapsed_time(end_event))

            g_logprobs2 = torch.empty_like(entropy_logprobs).uniform_(-0.1, 0.1)
            start_event.record()
            loss = entropy_logprobs + entropy.mean()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            entropy_bwd_latency.append(start_event.elapsed_time(end_event))

        # Remove warmup
        baseline_fwd_latency = baseline_fwd_latency[1:]
        baseline_bwd_latency = baseline_bwd_latency[1:]
        entropy_fwd_latency = entropy_fwd_latency[1:]
        entropy_bwd_latency = entropy_bwd_latency[1:]

        print()
        print(f"[INFO]: Entropy overhead on problem {problem}, dtype {dtype}:")
        print(
            f"[INFO]: Baseline forward: {sum(baseline_fwd_latency) / len(baseline_fwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Entropy forward:  {sum(entropy_fwd_latency) / len(entropy_fwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Baseline backward: {sum(baseline_bwd_latency) / len(baseline_bwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Entropy backward:  {sum(entropy_bwd_latency) / len(entropy_bwd_latency):.2f} ms"
        )

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_storage(self, problem, dtype, reduction, ignore_index):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens
        print()
        print(f"[INFO]: Entropy storage on problem {problem}, dtype {dtype}:")

        def baseline_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            torch.cuda.reset_peak_memory_stats()
            logprobs = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index,
                return_entropy=False,
            )
            torch.cuda.synchronize()
            fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Baseline Forward peak memory: {fwd_mem:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            g_logprobs = torch.empty_like(logprobs).uniform_(-0.1, 0.1)
            torch.autograd.grad(
                (logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Baseline Backward peak memory: {bwd_mem:.2f} MB")

        def entropy_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            torch.cuda.reset_peak_memory_stats()
            logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index,
                return_entropy=True,
            )
            torch.cuda.synchronize()
            fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Entropy Forward peak memory: {fwd_mem:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            loss = logprobs + entropy.mean()
            loss.backward()
            torch.cuda.synchronize()
            bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Entropy Backward peak memory: {bwd_mem:.2f} MB")

        self.cleanup()
        baseline_storage()
        self.cleanup()
        entropy_storage()

    def test_return_entropy_false(self):
        """When return_entropy=False, behavior matches original exactly."""
        num_tokens, vocab_size, dim = 128, 1024, 256
        dtype = torch.bfloat16

        hidden = torch.randn(num_tokens, dim, dtype=dtype, device="cuda")
        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda")
        labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda")

        result = linear_cross_entropy(
            hidden, weight, labels, reduction="mean", return_entropy=False,
        )
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_temperature(self, dtype, reduction, temperature):
        """Test forward and backward with temperature scaling."""
        num_tokens, vocabsize, dim = 1024, 152064, 4096
        hidden = (
            torch.empty((num_tokens, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, (num_tokens,), dtype=torch.long, device="cuda")

        # Forward
        logprobs, entropy = linear_cross_entropy(
            hidden, weight, labels,
            reduction=reduction, return_entropy=True, temperature=temperature,
        )
        ref_logprobs, ref_entropy = torch_entropy_reference(
            hidden, weight, labels, reduction, -100, temperature=temperature,
        )

        torch.testing.assert_close(logprobs, ref_logprobs, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(entropy, ref_entropy.view(-1), atol=1e-2, rtol=1e-2)

        # Backward
        if reduction == "none":
            fused_loss = logprobs.sum() + entropy.mean()
        else:
            fused_loss = logprobs + entropy.mean()
        fused_loss.backward()
        fused_d_hidden = hidden.grad.clone()
        fused_d_weight = weight.grad.clone()

        hidden.grad = None
        weight.grad = None
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        scaled_logits = logits / temperature
        ref_lp = torch.nn.functional.cross_entropy(scaled_logits, labels, reduction=reduction)
        lse = torch.logsumexp(scaled_logits, dim=-1)
        sm = torch.softmax(scaled_logits, dim=-1)
        ref_ent = lse - (sm * scaled_logits).sum(dim=-1)
        if reduction == "none":
            ref_loss = ref_lp.sum() + ref_ent.mean()
        else:
            ref_loss = ref_lp + ref_ent.mean()
        ref_loss.backward()

        torch.testing.assert_close(fused_d_hidden, hidden.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(fused_d_weight, weight.grad, atol=5e-2, rtol=5e-2)


# ===========================================================================
# Tensor Parallel (multi-GPU) tests
# ===========================================================================
@pytest.mark.skipif(_only_profile, reason="Skipping test in profile mode")
@pytest.mark.skipif(
    ("WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2),
    reason="Requires torchrun with multiple GPUs",
)
@pytest.mark.skipif(get_device_arch_version() != 10, reason="Requires GPU architecture = 10")
@pytest.mark.usefixtures("distributed_context")
class TestLinearCrossEntropyWithEntropyTensorParallel:
    @pytest.fixture(autouse=True)
    def setup_attrs(self, distributed_context):
        self.tp_group = distributed_context.group
        self.tp_rank = distributed_context.rank
        self.tp_world_size = distributed_context.world_size
        self.is_chief = distributed_context.is_chief

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def torch_entropy_single_gpu(hidden, weight, labels, reduction="mean", temperature=1.0):
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        scaled_logits = logits / temperature
        logprobs = torch.nn.functional.cross_entropy(
            scaled_logits.view(-1, scaled_logits.shape[-1]), labels.view(-1), reduction=reduction
        )
        lse = torch.logsumexp(scaled_logits, dim=-1)
        softmax = torch.softmax(scaled_logits, dim=-1)
        entropy = lse - (softmax * scaled_logits).sum(dim=-1)
        return logprobs.to(torch.float32), entropy.to(torch.float32)

    class TorchLinearCrossEntropyWithEntropy(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden: torch.Tensor,
            weight: torch.Tensor,
            labels: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            reduction: typing.Optional[str] = "mean",
            temperature: float = 1.0,
        ):
            tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
            tp_world_size = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)

            logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)

            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            scaled_logits = whole_logits / temperature
            logprobs = torch.nn.functional.cross_entropy(
                scaled_logits.view(-1, scaled_logits.shape[-1]), labels.view(-1), reduction=reduction
            )
            lse = torch.logsumexp(scaled_logits, dim=-1)
            softmax = torch.softmax(scaled_logits, dim=-1)
            entropy = lse - (softmax * scaled_logits).sum(dim=-1)

            ctx.save_for_backward(hidden, weight, labels)
            ctx.tp_group = tp_group
            ctx.reduction = reduction
            ctx.tp_rank = tp_rank
            ctx.tp_world_size = tp_world_size
            ctx.temperature = temperature

            return logprobs.to(torch.float32), entropy.to(torch.float32)

        @staticmethod
        def backward(ctx, g_logprobs: torch.Tensor, g_entropy: torch.Tensor):
            hidden, weight, labels = ctx.saved_tensors
            tp_group = ctx.tp_group
            reduction = ctx.reduction
            tp_rank = ctx.tp_rank
            tp_world_size = ctx.tp_world_size
            temperature = ctx.temperature

            num_tokens, dim = hidden.shape

            # Re-compute whole_logits
            logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            scaled_logits = whole_logits / temperature
            softmax = torch.softmax(scaled_logits, dim=-1)
            entropy_b = (softmax * scaled_logits).sum(dim=-1)

            # CE gradient
            if reduction == "mean":
                _g_logprobs = torch.broadcast_to(g_logprobs / num_tokens, (num_tokens,))
            elif reduction == "sum":
                _g_logprobs = torch.broadcast_to(g_logprobs, (num_tokens,))
            else:
                _g_logprobs = g_logprobs

            one_hot = torch.zeros_like(scaled_logits)
            one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)

            d_scaled_logits = (softmax - one_hot) * _g_logprobs.unsqueeze(-1)

            # Entropy gradient: dH/dz_v = -softmax_v * (z_v - entropy_b)
            d_scaled_logits += g_entropy.unsqueeze(-1) * (-softmax) * (scaled_logits - entropy_b.unsqueeze(-1))

            # Chain rule for temperature: d_logits = d_scaled_logits / temperature
            d_logits = (d_scaled_logits / temperature).to(hidden.dtype)

            local_size = weight.size(0)
            local_d_logits = d_logits[:, tp_rank * local_size : (tp_rank + 1) * local_size]

            local_d_hidden = local_d_logits @ weight
            local_d_weight = local_d_logits.T @ hidden

            dist.all_reduce(local_d_hidden, op=dist.ReduceOp.SUM, group=tp_group)

            return local_d_hidden, local_d_weight, None, None, None, None

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 152063, 8192),
            ((1, 4096), 152064, 8192),
            ((2, 4096), 152064, 8192),
        ]

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", [(4096, 129280, 8192)])
    def test_torch_tp_vs_single_gpu(self, dtype, reduction, problem):
        """Validate that the torch TP reference matches single-GPU reference."""
        num_tokens, vocabsize, dim = problem
        vocabsize = vocabsize // self.tp_world_size

        hidden = (
            torch.empty((num_tokens, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, (num_tokens,), dtype=torch.long, device="cuda")

        dist.broadcast(hidden, src=0, group=self.tp_group)
        dist.broadcast(labels, src=0, group=self.tp_group)

        # Single GPU
        whole_weight = torch.empty(
            (vocabsize * self.tp_world_size, dim), dtype=dtype, device="cuda"
        )
        whole_weight_view = [
            whole_weight[i * vocabsize : (i + 1) * vocabsize, :] for i in range(self.tp_world_size)
        ]
        dist.all_gather(whole_weight_view, weight, group=self.tp_group)
        whole_weight = whole_weight.clone().requires_grad_()
        logprobs_single, entropy_single = self.torch_entropy_single_gpu(
            hidden, whole_weight, labels, reduction=reduction
        )

        # TP
        logprobs_tp, entropy_tp = self.TorchLinearCrossEntropyWithEntropy.apply(
            hidden, weight, labels, self.tp_group, reduction
        )
        torch.testing.assert_close(logprobs_single, logprobs_tp)
        torch.testing.assert_close(entropy_single, entropy_tp, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", get_problems())
    def test_correctness(self, dtype, reduction, problem):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

        dist.broadcast(hidden, src=0, group=self.tp_group)
        dist.broadcast(labels, src=0, group=self.tp_group)

        # Torch TP reference
        torch_logprobs, torch_entropy = self.TorchLinearCrossEntropyWithEntropy.apply(
            hidden.view(-1, dim), weight, labels, self.tp_group, reduction
        )

        # Custom fused
        custom_logprobs, custom_entropy = linear_cross_entropy(
            hidden, weight, labels, tp_group=self.tp_group, reduction=reduction,
            return_entropy=True,
        )

        torch.testing.assert_close(torch_logprobs, custom_logprobs)
        torch.testing.assert_close(torch_entropy, custom_entropy, atol=1e-2, rtol=1e-2)

        # Backward
        g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
        dist.broadcast(g_logprobs, src=0, group=self.tp_group)

        if reduction == "none":
            torch_loss = torch_logprobs.sum() + torch_entropy.sum()
        else:
            torch_loss = torch_logprobs + torch_entropy.sum()
        torch_loss.backward()
        torch_d_hidden = hidden.grad.clone()
        torch_d_weight = weight.grad.clone()
        hidden.grad = None
        weight.grad = None

        if reduction == "none":
            custom_loss = custom_logprobs.sum() + custom_entropy.sum()
        else:
            custom_loss = custom_logprobs + custom_entropy.sum()
        custom_loss.backward()

        torch.testing.assert_close(torch_d_hidden, hidden.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(torch_d_weight, weight.grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_performance(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        baseline_fwd_latency = list()
        entropy_fwd_latency = list()
        baseline_bwd_latency = list()
        entropy_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            # Baseline
            start_event.record()
            baseline_logprobs = linear_cross_entropy(
                hidden, weight, labels, tp_group=self.tp_group, reduction=reduction,
                return_entropy=False,
            )
            end_event.record()
            torch.cuda.synchronize()
            baseline_fwd_latency.append(start_event.elapsed_time(end_event))

            g_logprobs = torch.empty_like(baseline_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            start_event.record()
            torch.autograd.grad(
                (baseline_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            baseline_bwd_latency.append(start_event.elapsed_time(end_event))

            # Entropy
            start_event.record()
            entropy_logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels, tp_group=self.tp_group, reduction=reduction,
                return_entropy=True,
            )
            end_event.record()
            torch.cuda.synchronize()
            entropy_fwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            loss = entropy_logprobs + entropy.mean()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            entropy_bwd_latency.append(start_event.elapsed_time(end_event))

        baseline_fwd_latency = baseline_fwd_latency[1:]
        baseline_bwd_latency = baseline_bwd_latency[1:]
        entropy_fwd_latency = entropy_fwd_latency[1:]
        entropy_bwd_latency = entropy_bwd_latency[1:]

        if self.is_chief:
            print()
            print(
                f"[INFO]: Entropy TP overhead on problem {problem}, TP size {self.tp_world_size}:"
            )
            print(
                f"[INFO]: Baseline forward: {sum(baseline_fwd_latency) / len(baseline_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Entropy forward:  {sum(entropy_fwd_latency) / len(entropy_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Baseline backward: {sum(baseline_bwd_latency) / len(baseline_bwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Entropy backward:  {sum(entropy_bwd_latency) / len(entropy_bwd_latency):.2f} ms"
            )

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_storage(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        if self.is_chief:
            print()
            print(
                f"[INFO]: Entropy TP storage on problem {problem}, TP size {self.tp_world_size}:"
            )

        def baseline_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            logprobs = linear_cross_entropy(
                hidden, weight, labels, tp_group=self.tp_group, reduction=reduction,
                return_entropy=False,
            )
            torch.cuda.synchronize()
            fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Baseline Forward peak memory: {fwd_mem:.2f} MB")

            g_logprobs = torch.empty_like(logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            torch.autograd.grad(
                (logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Baseline Backward peak memory: {bwd_mem:.2f} MB")

        def entropy_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels, tp_group=self.tp_group, reduction=reduction,
                return_entropy=True,
            )
            torch.cuda.synchronize()
            fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Entropy Forward peak memory: {fwd_mem:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            loss = logprobs + entropy.mean()
            loss.backward()
            torch.cuda.synchronize()
            bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Entropy Backward peak memory: {bwd_mem:.2f} MB")

        self.cleanup()
        baseline_storage()
        self.cleanup()
        entropy_storage()


# ===========================================================================
# Sequence Parallel (multi-GPU) tests
# ===========================================================================
@pytest.mark.skipif(_only_profile, reason="Skipping test in profile mode")
@pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2,
    reason="Requires torchrun with multiple GPUs",
)
@pytest.mark.skipif(get_device_arch_version() != 10, reason="Requires GPU architecture = 10")
@pytest.mark.usefixtures("distributed_context")
class TestLinearCrossEntropyWithEntropySequenceParallel:
    @pytest.fixture(autouse=True)
    def setup_attrs(self, distributed_context):
        self.tp_group = distributed_context.group
        self.tp_rank = distributed_context.rank
        self.tp_world_size = distributed_context.world_size
        self.is_chief = distributed_context.is_chief

    @staticmethod
    def timed_barrier(timeout_s=10):
        import time

        work = torch.distributed.barrier(async_op=True)
        t0 = time.time()
        while not work.is_completed():
            if time.time() - t0 > timeout_s:
                exit(1)
            time.sleep(0.05)
        work.wait()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def torch_entropy_single_gpu(hidden, weight, labels, reduction="mean", temperature=1.0):
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        scaled_logits = logits / temperature
        logprobs = torch.nn.functional.cross_entropy(
            scaled_logits.view(-1, scaled_logits.shape[-1]), labels.view(-1), reduction=reduction
        )
        lse = torch.logsumexp(scaled_logits, dim=-1)
        softmax = torch.softmax(scaled_logits, dim=-1)
        entropy = lse - (softmax * scaled_logits).sum(dim=-1)
        return logprobs.to(torch.float32), entropy.to(torch.float32)

    class TorchLinearCrossEntropyWithEntropy(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden: torch.Tensor,
            weight: torch.Tensor,
            labels: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            reduction: typing.Optional[str] = "mean",
            temperature: float = 1.0,
        ):
            tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
            tp_world_size = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)

            whole_hidden = torch.empty(
                (hidden.shape[0] * tp_world_size, hidden.shape[-1]),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            dist.all_gather_into_tensor(whole_hidden, hidden, group=tp_group)

            logits = whole_hidden.to(torch.float32) @ weight.T.to(torch.float32)

            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            scaled_logits = whole_logits / temperature
            logprobs = torch.nn.functional.cross_entropy(
                scaled_logits.view(-1, scaled_logits.shape[-1]), labels.view(-1), reduction=reduction
            )
            lse = torch.logsumexp(scaled_logits, dim=-1)
            softmax = torch.softmax(scaled_logits, dim=-1)
            entropy = lse - (softmax * scaled_logits).sum(dim=-1)

            ctx.save_for_backward(whole_hidden, weight, labels)
            ctx.tp_group = tp_group
            ctx.reduction = reduction
            ctx.tp_rank = tp_rank
            ctx.tp_world_size = tp_world_size
            ctx.temperature = temperature

            return logprobs.to(torch.float32), entropy.to(torch.float32)

        @staticmethod
        def backward(ctx, g_logprobs: torch.Tensor, g_entropy: torch.Tensor):
            whole_hidden, weight, labels = ctx.saved_tensors
            tp_group = ctx.tp_group
            reduction = ctx.reduction
            tp_rank = ctx.tp_rank
            tp_world_size = ctx.tp_world_size
            temperature = ctx.temperature

            num_tokens, dim = whole_hidden.shape

            # Re-compute whole_logits
            logits = whole_hidden.to(torch.float32) @ weight.T.to(torch.float32)
            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            scaled_logits = whole_logits / temperature
            softmax = torch.softmax(scaled_logits, dim=-1)
            entropy_b = (softmax * scaled_logits).sum(dim=-1)

            if reduction == "mean":
                _g_logprobs = torch.broadcast_to(g_logprobs / num_tokens, (num_tokens,))
            elif reduction == "sum":
                _g_logprobs = torch.broadcast_to(g_logprobs, (num_tokens,))
            else:
                _g_logprobs = g_logprobs

            one_hot = torch.zeros_like(scaled_logits)
            one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)

            d_scaled_logits = (softmax - one_hot) * _g_logprobs.unsqueeze(-1)
            d_scaled_logits += g_entropy.unsqueeze(-1) * (-softmax) * (scaled_logits - entropy_b.unsqueeze(-1))

            d_logits = (d_scaled_logits / temperature).to(whole_hidden.dtype)

            local_size = weight.size(0)
            local_d_logits = d_logits[:, tp_rank * local_size : (tp_rank + 1) * local_size]

            d_hidden = local_d_logits @ weight
            local_d_weight = local_d_logits.T @ whole_hidden

            local_num_tokens = num_tokens // tp_world_size
            local_d_hidden = torch.empty(
                (local_num_tokens, dim), dtype=weight.dtype, device=weight.device
            )
            dist.reduce_scatter_tensor(
                local_d_hidden, d_hidden, op=dist.ReduceOp.SUM, group=tp_group
            )
            return local_d_hidden, local_d_weight, None, None, None, None

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", [(256, 129280, 8192)])
    def test_torch_sp_vs_single_gpu(self, dtype, reduction, problem):
        """Validate that the torch SP reference matches single-GPU reference."""
        num_tokens, vocabsize, dim = problem
        vocabsize = vocabsize // self.tp_world_size

        hidden = (
            torch.empty((num_tokens, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(
            0, vocabsize, (num_tokens * self.tp_world_size,), dtype=torch.long, device="cuda"
        )

        dist.broadcast(labels, src=0, group=self.tp_group)

        # Single GPU
        whole_hidden = torch.empty(
            (num_tokens * self.tp_world_size, dim), dtype=dtype, device="cuda"
        )
        dist.all_gather_into_tensor(whole_hidden, hidden, group=self.tp_group)
        whole_hidden = whole_hidden.clone().requires_grad_()

        whole_weight = torch.empty(
            (vocabsize * self.tp_world_size, dim), dtype=dtype, device="cuda"
        )
        whole_weight_view = [
            whole_weight[i * vocabsize : (i + 1) * vocabsize, :] for i in range(self.tp_world_size)
        ]
        dist.all_gather(whole_weight_view, weight, group=self.tp_group)
        whole_weight = whole_weight.clone().requires_grad_()
        logprobs_single, entropy_single = self.torch_entropy_single_gpu(
            whole_hidden, whole_weight, labels, reduction=reduction
        )

        # SP
        logprobs_sp, entropy_sp = self.TorchLinearCrossEntropyWithEntropy.apply(
            hidden, weight, labels, self.tp_group, reduction
        )
        torch.testing.assert_close(logprobs_single, logprobs_sp)
        torch.testing.assert_close(entropy_single, entropy_sp, atol=1e-2, rtol=1e-2)

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 15206, 1024),
            ((1, 4096), 15206, 1024),
            ((4, 1024), 15206, 1024),
        ]

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", get_problems())
    def test_correctness(self, dtype, reduction, problem):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (
            (num_tokens * self.tp_world_size,)
            if isinstance(num_tokens, int)
            else (num_tokens[0] * self.tp_world_size, *num_tokens[1:])
        )

        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

        dist.broadcast(labels, src=0, group=self.tp_group)

        # Torch SP reference
        torch_logprobs, torch_entropy = self.TorchLinearCrossEntropyWithEntropy.apply(
            hidden.view(-1, dim), weight, labels, self.tp_group, reduction
        )

        # Custom fused
        custom_logprobs, custom_entropy = linear_cross_entropy(
            hidden, weight, labels,
            tp_group=self.tp_group, reduction=reduction,
            sequence_parallel=True, return_entropy=True,
        )

        torch.testing.assert_close(torch_logprobs, custom_logprobs)
        torch.testing.assert_close(torch_entropy, custom_entropy, atol=1e-2, rtol=1e-2)

        # Backward
        g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
        dist.broadcast(g_logprobs, src=0, group=self.tp_group)

        if reduction == "none":
            torch_loss = torch_logprobs.sum() + torch_entropy.sum()
        else:
            torch_loss = torch_logprobs + torch_entropy.sum()
        (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
            (torch_loss,), (hidden, weight), retain_graph=False
        )

        if reduction == "none":
            custom_loss = custom_logprobs.sum() + custom_entropy.sum()
        else:
            custom_loss = custom_logprobs + custom_entropy.sum()
        (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
            (custom_loss,), (hidden, weight), retain_graph=False
        )

        torch.testing.assert_close(d_hidden_torch, d_hidden_custom, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(d_weight_torch, d_weight_custom, atol=5e-2, rtol=5e-2)
        self.timed_barrier()

        self.cleanup()

    @pytest.mark.parametrize("problem", [((1, 1024), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_performance(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (
            (num_tokens * self.tp_world_size,)
            if isinstance(num_tokens, int)
            else (num_tokens[0] * self.tp_world_size, *num_tokens[1:])
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        baseline_fwd_latency = list()
        entropy_fwd_latency = list()
        baseline_bwd_latency = list()
        entropy_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(labels, src=0, group=self.tp_group)

            # Baseline
            start_event.record()
            baseline_logprobs = linear_cross_entropy(
                hidden, weight, labels,
                tp_group=self.tp_group, reduction=reduction,
                sequence_parallel=True, return_entropy=False,
            )
            end_event.record()
            torch.cuda.synchronize()
            baseline_fwd_latency.append(start_event.elapsed_time(end_event))

            g_logprobs = torch.empty_like(baseline_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            start_event.record()
            torch.autograd.grad(
                (baseline_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            baseline_bwd_latency.append(start_event.elapsed_time(end_event))

            # Entropy
            start_event.record()
            entropy_logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels,
                tp_group=self.tp_group, reduction=reduction,
                sequence_parallel=True, return_entropy=True,
            )
            end_event.record()
            torch.cuda.synchronize()
            entropy_fwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            loss = entropy_logprobs + entropy.mean()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            entropy_bwd_latency.append(start_event.elapsed_time(end_event))

        baseline_fwd_latency = baseline_fwd_latency[1:]
        baseline_bwd_latency = baseline_bwd_latency[1:]
        entropy_fwd_latency = entropy_fwd_latency[1:]
        entropy_bwd_latency = entropy_bwd_latency[1:]

        if self.is_chief:
            print()
            print(
                f"[INFO]: Entropy SP overhead on problem {problem}, TP size {self.tp_world_size}:"
            )
            print(
                f"[INFO]: Baseline forward: {sum(baseline_fwd_latency) / len(baseline_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Entropy forward:  {sum(entropy_fwd_latency) / len(entropy_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Baseline backward: {sum(baseline_bwd_latency) / len(baseline_bwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Entropy backward:  {sum(entropy_bwd_latency) / len(entropy_bwd_latency):.2f} ms"
            )

    @pytest.mark.parametrize("problem", [((1, 1024), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_storage(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (
            (num_tokens * self.tp_world_size,)
            if isinstance(num_tokens, int)
            else (num_tokens[0] * self.tp_world_size, *num_tokens[1:])
        )

        if self.is_chief:
            print()
            print(
                f"[INFO]: Entropy SP storage on problem {problem}, TP size {self.tp_world_size}:"
            )

        def baseline_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            logprobs = linear_cross_entropy(
                hidden, weight, labels,
                tp_group=self.tp_group, reduction=reduction,
                sequence_parallel=True, return_entropy=False,
            )
            torch.cuda.synchronize()
            fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Baseline Forward peak memory: {fwd_mem:.2f} MB")

            g_logprobs = torch.empty_like(logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            torch.autograd.grad(
                (logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Baseline Backward peak memory: {bwd_mem:.2f} MB")

        def entropy_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            logprobs, entropy = linear_cross_entropy(
                hidden, weight, labels,
                tp_group=self.tp_group, reduction=reduction,
                sequence_parallel=True, return_entropy=True,
            )
            torch.cuda.synchronize()
            fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Entropy Forward peak memory: {fwd_mem:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            loss = logprobs + entropy.mean()
            loss.backward()
            torch.cuda.synchronize()
            bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(f"[INFO]: Entropy Backward peak memory: {bwd_mem:.2f} MB")

        self.cleanup()
        baseline_storage()
        self.cleanup()
        entropy_storage()


# ===========================================================================
# Profile mode (single GPU, NVTX ranges)
# ===========================================================================
@pytest.mark.skipif(not _only_profile, reason="Only running test in profile mode")
@pytest.mark.skipif(
    "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != "1", reason="Requires single GPU"
)
@pytest.mark.skipif(get_device_arch_version() != 10, reason="Requires GPU architecture = 10")
class TestLinearCrossEntropyWithEntropyProfile:
    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_performance(self, problem, dtype, reduction, ignore_index):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        baseline_fwd_latency = list()
        baseline_bwd_latency = list()
        entropy_fwd_latency = list()
        entropy_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            with torch.cuda.nvtx.range(f"Iteration_{i}"):
                hidden = (
                    torch.empty(hidden_shape, dtype=dtype, device="cuda")
                    .uniform_(-0.1, 0.1)
                    .requires_grad_()
                )
                weight = (
                    torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                    .uniform_(-0.1, 0.1)
                    .requires_grad_()
                )
                labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
                if ignore_index >= 0 and ignore_index < vocabsize:
                    pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                    labels = pad_labels[..., 1:].contiguous()

                # -------- baseline forward -------- #
                start_event.record()
                with torch.cuda.nvtx.range("Baseline Forward"):
                    baseline_logprobs = linear_cross_entropy(
                        hidden, weight, labels, reduction=reduction, ignore_index=ignore_index,
                        return_entropy=False,
                    )
                end_event.record()
                torch.cuda.synchronize()
                baseline_fwd_latency.append(start_event.elapsed_time(end_event))

                # -------- entropy forward -------- #
                start_event.record()
                with torch.cuda.nvtx.range("Entropy Forward"):
                    entropy_logprobs, entropy = linear_cross_entropy(
                        hidden, weight, labels, reduction=reduction, ignore_index=ignore_index,
                        return_entropy=True,
                    )
                end_event.record()
                torch.cuda.synchronize()
                entropy_fwd_latency.append(start_event.elapsed_time(end_event))

                # -------- baseline backward -------- #
                g_logprobs = torch.empty_like(baseline_logprobs).uniform_(-0.1, 0.1)

                start_event.record()
                with torch.cuda.nvtx.range("Baseline Backward"):
                    torch.autograd.grad(
                        (baseline_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
                    )
                end_event.record()
                torch.cuda.synchronize()
                baseline_bwd_latency.append(start_event.elapsed_time(end_event))

                # -------- entropy backward -------- #
                start_event.record()
                with torch.cuda.nvtx.range("Entropy Backward"):
                    loss = entropy_logprobs + entropy.mean()
                    loss.backward()
                end_event.record()
                torch.cuda.synchronize()
                entropy_bwd_latency.append(start_event.elapsed_time(end_event))

        # Remove warmup
        baseline_fwd_latency = baseline_fwd_latency[1:]
        baseline_bwd_latency = baseline_bwd_latency[1:]
        entropy_fwd_latency = entropy_fwd_latency[1:]
        entropy_bwd_latency = entropy_bwd_latency[1:]

        print()
        print(f"[INFO]: Entropy profile on problem {problem}, dtype {dtype}, reduction {reduction}:")
        print(
            f"[INFO]: Baseline forward: {sum(baseline_fwd_latency) / len(baseline_fwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Entropy forward:  {sum(entropy_fwd_latency) / len(entropy_fwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Baseline backward: {sum(baseline_bwd_latency) / len(baseline_bwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Entropy backward:  {sum(entropy_bwd_latency) / len(entropy_bwd_latency):.2f} ms"
        )
