"""
Defense mechanisms against gradient inversion attacks.

Three independent defenses are implemented:

  1. Differential Privacy (DP-SGD)
     Clips individual per-sample gradients to a maximum L2 norm, then
     adds calibrated Gaussian noise. Provides (ε, δ)-DP guarantees.

  2. Gradient Compression
     Sparsifies gradients by zeroing all values below a percentile
     threshold. Reduces information content transmitted to the server.

  3. Gradient Noise (Gaussian/Laplace)
     Adds noise without the clipping step — weaker than DP-SGD but
     computationally cheaper and compatible with any training loop.

All defenses return a new list of gradient tensors and leave the
original model parameters untouched.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


# ==================================================================
# 1. Differential Privacy — DP-SGD style
# ==================================================================

class DifferentialPrivacyDefense:
    """
    Gradient perturbation via clipping + Gaussian noise (DP-SGD).

    Parameters
    ----------
    max_grad_norm : float
        Per-sample gradient clipping norm C.
    noise_multiplier : float
        σ = noise_multiplier × C. Higher → stronger privacy, worse utility.
    """

    def __init__(self, max_grad_norm: float = 1.0, noise_multiplier: float = 1.0) -> None:
        self.max_grad_norm   = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def apply(self, grads: List[torch.Tensor]) -> Tuple[List[torch.Tensor], dict]:
        defended = []
        sigma = self.noise_multiplier * self.max_grad_norm

        for g in grads:
            g = g.clone().float()
            norm = g.norm()
            # Clip
            if norm > self.max_grad_norm:
                g = g * (self.max_grad_norm / norm)
            # Add Gaussian noise
            g = g + torch.randn_like(g) * sigma
            defended.append(g)

        noise_power = sigma ** 2 * sum(g.numel() for g in defended)
        return defended, {
            "method":          "differential_privacy",
            "max_grad_norm":   self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "sigma":           sigma,
            "noise_power":     noise_power,
        }

    def privacy_budget_rdp(self, steps: int, delta: float = 1e-5) -> float:
        """
        Estimate ε under Rényi DP composition (moments accountant, order α=2).
        This is an approximation — not a tight bound for production use.
        """
        alpha = 2.0
        sigma = self.noise_multiplier
        # Per-step RDP: ε_step(α) = α / (2σ²)
        eps_step = alpha / (2.0 * sigma ** 2)
        eps_total = steps * eps_step
        # Convert RDP(α) to (ε, δ)-DP
        eps_dp = eps_total + math.log(1.0 / delta) / (alpha - 1.0)
        return max(0.0, eps_dp)


# ==================================================================
# 2. Gradient Compression (Top-k sparsification)
# ==================================================================

class GradientCompressionDefense:
    """
    Sparsify gradients by zeroing values below the (1-k)th percentile.

    Parameters
    ----------
    sparsity : float
        Fraction of gradient values to zero out, e.g. 0.90 keeps top 10%.
    """

    def __init__(self, sparsity: float = 0.90) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1).")
        self.sparsity = sparsity

    def apply(self, grads: List[torch.Tensor]) -> Tuple[List[torch.Tensor], dict]:
        defended = []
        total_original = 0
        total_kept     = 0

        for g in grads:
            g = g.clone().float()
            flat  = g.flatten().abs()
            n_keep = max(1, int((1.0 - self.sparsity) * flat.numel()))
            threshold_val = flat.topk(n_keep).values.min()
            mask  = g.abs() >= threshold_val
            g     = g * mask
            defended.append(g)

            total_original += flat.numel()
            total_kept     += mask.sum().item()

        compression_ratio = 1.0 - total_kept / max(1, total_original)
        return defended, {
            "method":            "gradient_compression",
            "sparsity":          self.sparsity,
            "compression_ratio": round(compression_ratio, 4),
            "params_zeroed":     int(total_original - total_kept),
        }


# ==================================================================
# 3. Gradient Noise (additive Gaussian / Laplace)
# ==================================================================

class GradientNoiseDefense:
    """
    Additive noise defense without clipping.

    Parameters
    ----------
    noise_type : str
        "gaussian" or "laplace"
    scale : float
        Noise scale σ (std for Gaussian, scale b for Laplace).
    relative : bool
        If True, scale is treated as a fraction of each gradient's L2 norm.
    """

    def __init__(
        self,
        scale: float = 0.01,
        noise_type: str = "gaussian",
        relative: bool = False,
    ) -> None:
        self.scale      = scale
        self.noise_type = noise_type
        self.relative   = relative

    def apply(self, grads: List[torch.Tensor]) -> Tuple[List[torch.Tensor], dict]:
        defended = []

        for g in grads:
            g = g.clone().float()
            sigma = self.scale * g.norm().item() if self.relative else self.scale

            if self.noise_type == "gaussian":
                noise = torch.randn_like(g) * sigma
            else:
                noise = torch.distributions.Laplace(0.0, sigma).sample(g.shape).to(g.device)

            defended.append(g + noise)

        return defended, {
            "method":     "gradient_noise",
            "noise_type": self.noise_type,
            "scale":      self.scale,
            "relative":   self.relative,
        }


# ==================================================================
# Defense evaluation helper
# ==================================================================

def gradient_cosine_similarity(
    original: List[torch.Tensor],
    defended: List[torch.Tensor],
) -> float:
    """Cosine similarity between two gradient vectors. Range: [-1, 1]."""
    a = torch.cat([g.flatten() for g in original])
    b = torch.cat([g.flatten() for g in defended])
    cos = (a * b).sum() / (a.norm() * b.norm() + 1e-9)
    return cos.item()


def gradient_snr(
    original: List[torch.Tensor],
    defended: List[torch.Tensor],
) -> float:
    """Signal-to-noise ratio (dB) of the defended gradients."""
    signal_power = sum((g ** 2).sum().item() for g in original)
    noise_power  = sum(((o - d) ** 2).sum().item() for o, d in zip(original, defended))
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)
