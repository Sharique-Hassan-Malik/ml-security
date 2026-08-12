"""
Layer-wise statistical analysis of weight distributions.

Backdoor attacks commonly leave behind anomalous weight statistics:
  - high kurtosis (heavy tails caused by trigger-path weights)
  - bimodal distributions (normal cluster + trojan cluster)
  - excess outlier weights far from the bulk distribution
"""

from __future__ import annotations

import math
from typing import List

import torch

from .report import Finding


# Empirical thresholds derived from analysis of clean ImageNet-trained networks.
_KURTOSIS_HIGH   = 10.0
_SKEWNESS_HIGH   = 3.0
_OUTLIER_RATIO   = 0.012   # fraction of weights beyond ±4σ
_BIMODAL_COEFF   = 0.700   # Bimodality Coefficient threshold
_ZERO_FRAC_HIGH  = 0.55    # suspicious sparsity combined with outliers


class WeightDistributionAnalyzer:
    """Applies statistical tests to each layer's flattened weight tensor."""

    def analyze(self, name: str, tensor: torch.Tensor) -> List[Finding]:
        w = tensor.detach().float().flatten()
        n = w.numel()

        mean  = w.mean().item()
        std   = w.std(unbiased=True).item()
        if std < 1e-9:          # dead / constant layer — skip
            return []

        findings: List[Finding] = []

        skewness = _skewness(w, mean, std)
        kurtosis = _kurtosis(w, mean, std)   # excess kurtosis
        outlier_ratio = ((w - mean).abs() > 4.0 * std).float().mean().item()
        zero_frac = (w.abs() < 1e-4).float().mean().item()
        bc = _bimodality_coefficient(skewness, kurtosis)

        # ---- individual tests ----------------------------------------

        if kurtosis > _KURTOSIS_HIGH:
            sev = "high" if kurtosis > 20.0 else "medium"
            findings.append(Finding(
                layer    = name,
                test     = "high_kurtosis",
                severity = sev,
                score    = min(1.0, kurtosis / 30.0),
                detail   = f"Excess kurtosis {kurtosis:.2f} (threshold {_KURTOSIS_HIGH}). "
                           "Heavy-tailed distributions suggest a small subset of weights "
                           "is abnormally large — consistent with trojan neuron paths.",
                metadata = {"kurtosis": round(kurtosis, 4)},
            ))

        if abs(skewness) > _SKEWNESS_HIGH:
            findings.append(Finding(
                layer    = name,
                test     = "high_skewness",
                severity = "medium",
                score    = min(1.0, abs(skewness) / 6.0),
                detail   = f"Skewness {skewness:.3f} (|threshold| {_SKEWNESS_HIGH}). "
                           "Strong asymmetry may indicate one-sided weight poisoning.",
                metadata = {"skewness": round(skewness, 4)},
            ))

        if outlier_ratio > _OUTLIER_RATIO:
            sev = "high" if outlier_ratio > 3 * _OUTLIER_RATIO else "medium"
            findings.append(Finding(
                layer    = name,
                test     = "outlier_weights",
                severity = sev,
                score    = min(1.0, outlier_ratio / (4 * _OUTLIER_RATIO)),
                detail   = f"{outlier_ratio * 100:.2f}% of weights exceed ±4σ "
                           f"(threshold {_OUTLIER_RATIO * 100:.1f}%). "
                           "Outlier weights at trigger-path neurons are a classic "
                           "BadNets signature.",
                metadata = {"outlier_ratio": round(outlier_ratio, 5), "n_params": n},
            ))

        if bc > _BIMODAL_COEFF:
            findings.append(Finding(
                layer    = name,
                test     = "bimodal_distribution",
                severity = "medium",
                score    = min(1.0, (bc - _BIMODAL_COEFF) / (1.0 - _BIMODAL_COEFF)),
                detail   = f"Bimodality coefficient {bc:.3f} > {_BIMODAL_COEFF}. "
                           "Two-peaked distributions often reflect a clean-weight cluster "
                           "and a separate trojan-weight cluster.",
                metadata = {"bimodality_coefficient": round(bc, 4)},
            ))

        # Sparse layer with significant outliers — dormant-neuron + trigger pattern
        if zero_frac > _ZERO_FRAC_HIGH and outlier_ratio > _OUTLIER_RATIO / 2:
            findings.append(Finding(
                layer    = name,
                test     = "sparse_with_outliers",
                severity = "medium",
                score    = min(1.0, zero_frac * 2 * outlier_ratio / _OUTLIER_RATIO),
                detail   = f"{zero_frac * 100:.1f}% near-zero weights alongside "
                           f"{outlier_ratio * 100:.3f}% outlier weights. Poisoned models "
                           "sometimes have many dormant neurons masking a few heavily "
                           "connected trojan neurons.",
                metadata = {"zero_fraction": round(zero_frac, 4)},
            ))

        return findings


# ------------------------------------------------------------------
# Moment calculations (no scipy dependency)
# ------------------------------------------------------------------

def _central_moments(w: torch.Tensor, mean: float, std: float):
    z = (w - mean) / std
    z2 = z * z
    m3 = (z2 * z).mean().item()
    m4 = (z2 * z2).mean().item()
    return m3, m4


def _skewness(w: torch.Tensor, mean: float, std: float) -> float:
    m3, _ = _central_moments(w, mean, std)
    return m3


def _kurtosis(w: torch.Tensor, mean: float, std: float) -> float:
    """Excess kurtosis (normal distribution → 0)."""
    _, m4 = _central_moments(w, mean, std)
    return m4 - 3.0


def _bimodality_coefficient(skewness: float, excess_kurtosis: float) -> float:
    """
    Bimodality Coefficient (SAS definition).
    BC = (γ₁² + 1) / (γ₂ + 3)   where γ₁ = skewness, γ₂ = excess kurtosis.
    Uniform → 5/9 ≈ 0.555; bimodal distributions score higher.
    """
    numerator   = skewness ** 2 + 1.0
    denominator = max(excess_kurtosis + 3.0, 1e-6)
    return numerator / denominator
