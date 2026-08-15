"""
Tests for gradient leakage attack and defense modules.

Run with:  pytest tests/ -v
"""

import math

import pytest
import torch
import torch.nn as nn

from gradleak.inversion import AttackConfig, GradientInversionAttack, compute_observed_gradients
from gradleak.defenses import (
    DifferentialPrivacyDefense,
    GradientCompressionDefense,
    GradientNoiseDefense,
    gradient_cosine_similarity,
    gradient_snr,
)
from gradleak.quality import mse, psnr, ssim


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return TinyNet()


@pytest.fixture()
def tiny_grads(tiny_model):
    data   = torch.randn(1, 16)
    labels = torch.tensor([2])
    return compute_observed_gradients(tiny_model, nn.CrossEntropyLoss(), data, labels)


# ------------------------------------------------------------------
# Metric tests
# ------------------------------------------------------------------

class TestMetrics:
    def test_psnr_identical(self):
        t = torch.rand(3, 32, 32)
        assert psnr(t, t) == float("inf")

    def test_psnr_decreases_with_noise(self):
        t = torch.rand(3, 32, 32)
        noisy = t + torch.randn_like(t) * 0.1
        assert psnr(t, noisy) < psnr(t, t + torch.randn_like(t) * 0.01)

    def test_mse_zero_for_identical(self):
        t = torch.rand(3, 32, 32)
        assert mse(t, t) < 1e-9

    def test_ssim_one_for_identical(self):
        t = torch.rand(1, 3, 32, 32)
        assert abs(ssim(t, t) - 1.0) < 1e-4

    def test_ssim_less_for_noisy(self):
        t = torch.rand(1, 3, 32, 32)
        noisy = (t + torch.randn_like(t) * 0.3).clamp(0, 1)
        assert ssim(t, noisy) < 0.99


# ------------------------------------------------------------------
# Gradient utils tests
# ------------------------------------------------------------------

class TestGradUtils:
    def test_cosine_sim_identical(self, tiny_grads):
        sim = gradient_cosine_similarity(tiny_grads, tiny_grads)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_sim_range(self, tiny_grads):
        noisy = [g + torch.randn_like(g) * 5.0 for g in tiny_grads]
        sim = gradient_cosine_similarity(tiny_grads, noisy)
        assert -1.0 <= sim <= 1.0

    def test_snr_infinite_identical(self, tiny_grads):
        snr = gradient_snr(tiny_grads, tiny_grads)
        assert snr == float("inf")

    def test_snr_decreases_with_noise(self, tiny_grads):
        small_noise = [g + torch.randn_like(g) * 0.01 for g in tiny_grads]
        large_noise = [g + torch.randn_like(g) * 1.00 for g in tiny_grads]
        snr_small = gradient_snr(tiny_grads, small_noise)
        snr_large = gradient_snr(tiny_grads, large_noise)
        assert snr_small > snr_large


# ------------------------------------------------------------------
# Defense tests
# ------------------------------------------------------------------

class TestDPDefense:
    def test_output_length(self, tiny_grads):
        dp = DifferentialPrivacyDefense()
        out, _ = dp.apply(tiny_grads)
        assert len(out) == len(tiny_grads)

    def test_shapes_preserved(self, tiny_grads):
        dp = DifferentialPrivacyDefense()
        out, _ = dp.apply(tiny_grads)
        for orig, defended in zip(tiny_grads, out):
            assert orig.shape == defended.shape

    def test_noise_degrades_cosine_sim(self, tiny_grads):
        dp = DifferentialPrivacyDefense(noise_multiplier=5.0)
        out, _ = dp.apply(tiny_grads)
        sim = gradient_cosine_similarity(tiny_grads, out)
        assert sim < 0.95

    def test_clip_norm(self, tiny_grads):
        dp = DifferentialPrivacyDefense(max_grad_norm=0.01, noise_multiplier=0.0)
        out, _ = dp.apply(tiny_grads)
        total = torch.cat([g.flatten() for g in out]).norm().item()
        # With noise_multiplier=0 and very small clip norm, each tensor should be clipped
        for g in out:
            assert g.norm().item() <= 0.01 + 1e-5

    def test_privacy_budget_positive(self):
        dp = DifferentialPrivacyDefense(noise_multiplier=1.1)
        eps = dp.privacy_budget_rdp(steps=1000)
        assert eps > 0.0


class TestCompressionDefense:
    def test_sparsity_respected(self, tiny_grads):
        gc = GradientCompressionDefense(sparsity=0.80)
        out, meta = gc.apply(tiny_grads)
        assert meta["compression_ratio"] > 0.0

    def test_shapes_preserved(self, tiny_grads):
        gc = GradientCompressionDefense(sparsity=0.50)
        out, _ = gc.apply(tiny_grads)
        for orig, defended in zip(tiny_grads, out):
            assert orig.shape == defended.shape

    def test_invalid_sparsity(self):
        with pytest.raises(ValueError):
            GradientCompressionDefense(sparsity=1.5)

    def test_high_sparsity_zeros(self, tiny_grads):
        gc = GradientCompressionDefense(sparsity=0.99)
        out, _ = gc.apply(tiny_grads)
        total_zeros = sum((g == 0).sum().item() for g in out)
        total_params = sum(g.numel() for g in out)
        assert total_zeros / total_params > 0.90


class TestNoiseDefense:
    def test_shapes_preserved(self, tiny_grads):
        gn = GradientNoiseDefense(scale=0.01)
        out, _ = gn.apply(tiny_grads)
        for orig, defended in zip(tiny_grads, out):
            assert orig.shape == defended.shape

    def test_laplace_noise(self, tiny_grads):
        gn = GradientNoiseDefense(scale=0.01, noise_type="laplace")
        out, meta = gn.apply(tiny_grads)
        assert meta["noise_type"] == "laplace"
        assert len(out) == len(tiny_grads)

    def test_relative_scale(self, tiny_grads):
        gn = GradientNoiseDefense(scale=1.0, relative=True)
        out, _ = gn.apply(tiny_grads)
        sim = gradient_cosine_similarity(tiny_grads, out)
        assert sim < 0.99


# ------------------------------------------------------------------
# iDLG label recovery
# ------------------------------------------------------------------

class TestLabelRecovery:
    def test_idlg_recovers_label(self, tiny_model):
        torch.manual_seed(1)
        data   = torch.randn(1, 16)
        label  = torch.tensor([1])
        grads  = compute_observed_gradients(tiny_model, nn.CrossEntropyLoss(), data, label)

        cfg      = AttackConfig(algorithm="idlg", iterations=1, seed=1)
        attacker = GradientInversionAttack(tiny_model, nn.CrossEntropyLoss(), cfg)
        result   = attacker.attack(grads, data_shape=(16,))
        assert result.dummy_labels[0].item() == 1


# ------------------------------------------------------------------
# Attack smoke test
# ------------------------------------------------------------------

class TestAttackSmoke:
    def test_dlg_runs(self, tiny_model):
        torch.manual_seed(2)
        data   = torch.randn(1, 16)
        labels = torch.tensor([0])
        grads  = compute_observed_gradients(tiny_model, nn.CrossEntropyLoss(), data, labels)

        cfg = AttackConfig(iterations=5, algorithm="dlg", seed=2)
        att = GradientInversionAttack(tiny_model, nn.CrossEntropyLoss(), cfg)
        res = att.attack(grads, data_shape=(16,))

        assert res.dummy_data.shape == (1, 16)
        assert isinstance(res.final_loss, float)

    def test_idlg_runs(self, tiny_model):
        torch.manual_seed(3)
        data   = torch.randn(1, 16)
        labels = torch.tensor([3])
        grads  = compute_observed_gradients(tiny_model, nn.CrossEntropyLoss(), data, labels)

        cfg = AttackConfig(iterations=5, algorithm="idlg", seed=3)
        att = GradientInversionAttack(tiny_model, nn.CrossEntropyLoss(), cfg)
        res = att.attack(grads, data_shape=(16,))

        assert res.dummy_data.shape == (1, 16)
        assert 0.0 <= res.dummy_data.max().item() <= 1.0
