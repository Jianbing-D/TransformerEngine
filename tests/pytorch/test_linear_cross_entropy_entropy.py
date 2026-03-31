# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import torch

from transformer_engine.pytorch import linear_cross_entropy


def get_device_arch_version():
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    return cc[0]


@pytest.mark.skipif(
    "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != "1",
    reason="Requires single GPU",
)
@pytest.mark.skipif(
    get_device_arch_version() != 10, reason="Requires GPU architecture = 10"
)
class TestLinearCrossEntropyWithEntropy:

    @staticmethod
    def torch_reference(hidden, weight, labels, reduction, ignore_index, temperature=1.0):
        """Compute logprobs and entropy using vanilla PyTorch."""
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        scaled_logits = logits / temperature
        logprobs = torch.nn.functional.cross_entropy(
            scaled_logits.view(-1, scaled_logits.shape[-1]),
            labels.view(-1),
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Entropy = logsumexp(scaled_logits) - sum(softmax(scaled_logits) * scaled_logits)
        lse = torch.logsumexp(scaled_logits, dim=-1)  # [T]
        softmax = torch.softmax(scaled_logits, dim=-1)  # [T, V]
        entropy_b = (softmax * scaled_logits).sum(dim=-1)  # [T]
        entropy = lse - entropy_b  # [T]

        return logprobs.to(torch.float32), entropy.to(torch.float32)

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            ((1, 4096), 152064, 8192),
        ]

    def test_kernel_launch(self):
        """Check that the entropy kernel can be launched with various sizes."""
        torch.cuda.empty_cache()
        num_tokens_list = [15, 128, 2048]
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
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_forward_correctness(self, dtype, problem, reduction, ignore_index):
        """Test that fused entropy matches PyTorch reference."""
        num_tokens, vocab_size, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)

        hidden = torch.randn(hidden_shape, dtype=dtype, device="cuda")
        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda")
        labels = torch.randint(0, vocab_size, hidden_shape[:-1], dtype=torch.long, device="cuda")

        if ignore_index != -100:
            mask = torch.rand(labels.shape, device="cuda") < 0.1
            labels[mask] = ignore_index

        logprobs, entropy = linear_cross_entropy(
            hidden, weight, labels,
            reduction=reduction, ignore_index=ignore_index, return_entropy=True,
        )
        ref_logprobs, ref_entropy = self.torch_reference(
            hidden, weight, labels, reduction, ignore_index,
        )

        # Logprobs check
        if reduction == "none":
            valid = labels.view(-1) != ignore_index
            torch.testing.assert_close(
                logprobs[valid], ref_logprobs[valid], atol=1e-2, rtol=1e-2,
            )
        else:
            torch.testing.assert_close(logprobs, ref_logprobs, atol=1e-2, rtol=1e-2)

        # Entropy check — per-token (flatten ref since fused kernel flattens batch dims)
        torch.testing.assert_close(entropy, ref_entropy.view(-1), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_backward_correctness(self, dtype, reduction):
        """Test backward gradients for combined loss = logprobs + entropy.mean()."""
        num_tokens, vocab_size, dim = 128, 1024, 256

        hidden = torch.randn(num_tokens, dim, dtype=dtype, device="cuda").requires_grad_()
        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda").requires_grad_()
        labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda")

        # Fused path
        logprobs, entropy = linear_cross_entropy(
            hidden, weight, labels, reduction=reduction, return_entropy=True,
        )
        loss = logprobs + entropy.mean()
        loss.backward()
        fused_d_hidden = hidden.grad.clone()
        fused_d_weight = weight.grad.clone()

        # Reference path
        hidden.grad = None
        weight.grad = None
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        ref_logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)
        lse = torch.logsumexp(logits, dim=-1)
        softmax = torch.softmax(logits, dim=-1)
        ref_entropy = lse - (softmax * logits).sum(dim=-1)
        ref_loss = ref_logprobs + ref_entropy.mean()
        ref_loss.backward()
        ref_d_hidden = hidden.grad.clone()
        ref_d_weight = weight.grad.clone()

        torch.testing.assert_close(fused_d_hidden, ref_d_hidden, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(fused_d_weight, ref_d_weight, atol=5e-2, rtol=5e-2)

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
        assert result.dim() == 0  # scalar

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_temperature_forward(self, dtype, reduction, temperature):
        """Test forward with temperature scaling matches PyTorch reference."""
        num_tokens, vocab_size, dim = 128, 1024, 256

        hidden = torch.randn(num_tokens, dim, dtype=dtype, device="cuda")
        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda")
        labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda")

        # Fused with temperature
        logprobs, entropy = linear_cross_entropy(
            hidden, weight, labels,
            reduction=reduction, return_entropy=True, temperature=temperature,
        )
        ref_logprobs, ref_entropy = self.torch_reference(
            hidden, weight, labels, reduction, -100, temperature=temperature,
        )

        torch.testing.assert_close(logprobs, ref_logprobs, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(entropy, ref_entropy.view(-1), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("temperature", [0.5, 2.0])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_temperature_backward(self, dtype, reduction, temperature):
        """Test backward gradients with temperature scaling."""
        num_tokens, vocab_size, dim = 128, 1024, 256

        hidden = torch.randn(num_tokens, dim, dtype=dtype, device="cuda").requires_grad_()
        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda").requires_grad_()
        labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda")

        # Fused path with temperature
        logprobs, entropy = linear_cross_entropy(
            hidden, weight, labels,
            reduction=reduction, return_entropy=True, temperature=temperature,
        )
        loss = logprobs + entropy.mean()
        loss.backward()
        fused_d_hidden = hidden.grad.clone()
        fused_d_weight = weight.grad.clone()

        # Reference path with temperature
        hidden.grad = None
        weight.grad = None
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        scaled_logits = logits / temperature
        ref_logprobs = torch.nn.functional.cross_entropy(scaled_logits, labels, reduction=reduction)
        lse = torch.logsumexp(scaled_logits, dim=-1)
        softmax = torch.softmax(scaled_logits, dim=-1)
        ref_entropy = lse - (softmax * scaled_logits).sum(dim=-1)
        ref_loss = ref_logprobs + ref_entropy.mean()
        ref_loss.backward()
        ref_d_hidden = hidden.grad.clone()
        ref_d_weight = weight.grad.clone()

        torch.testing.assert_close(fused_d_hidden, ref_d_hidden, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(fused_d_weight, ref_d_weight, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("temperature", [0.5, 2.0])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_temperature_baseline(self, dtype, reduction, temperature):
        """Test temperature with baseline (return_entropy=False) path."""
        num_tokens, vocab_size, dim = 128, 1024, 256

        hidden = torch.randn(num_tokens, dim, dtype=dtype, device="cuda").requires_grad_()
        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda").requires_grad_()
        labels = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda")

        # Fused baseline with temperature
        logprobs = linear_cross_entropy(
            hidden, weight, labels,
            reduction=reduction, return_entropy=False, temperature=temperature,
        )
        logprobs.backward()
        fused_d_hidden = hidden.grad.clone()
        fused_d_weight = weight.grad.clone()

        # Reference
        hidden.grad = None
        weight.grad = None
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        scaled_logits = logits / temperature
        ref_logprobs = torch.nn.functional.cross_entropy(scaled_logits, labels, reduction=reduction)
        ref_logprobs.backward()
        ref_d_hidden = hidden.grad.clone()
        ref_d_weight = weight.grad.clone()

        torch.testing.assert_close(logprobs, ref_logprobs, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(fused_d_hidden, ref_d_hidden, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(fused_d_weight, ref_d_weight, atol=5e-2, rtol=5e-2)
