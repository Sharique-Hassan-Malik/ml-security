"""
Query strategies for model extraction.

A query strategy decides which inputs to send to the oracle at each
round.  The goal is to maximise substitute model fidelity per query.

Strategies
----------
RandomStrategy
    Sample uniformly from the input domain.  Baseline — no knowledge
    of the substitute is required or used.

JacobianStrategy
    Jacobian-based Dataset Augmentation (Papernot et al. 2017).
    Uses the substitute model's Jacobian to perturb seed inputs in the
    direction that maximally changes the predicted output, exploring the
    decision boundary neighbourhood.

AdaptiveStrategy
    Uncertainty sampling.  Generates a large pool of candidates and
    selects those where the current substitute model's prediction
    entropy is highest — concentrating the query budget at the
    most informative points.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Random
# ------------------------------------------------------------------

class RandomStrategy:
    """
    Baseline: sample inputs uniformly at random from [0, 1]^d.

    Parameters
    ----------
    input_shape : shape of a single input, e.g. (3, 32, 32)
    batch_size  : number of inputs to generate per round
    """

    def __init__(self, input_shape: tuple, batch_size: int = 64) -> None:
        self.input_shape = input_shape
        self.batch_size  = batch_size

    def generate(
        self,
        substitute: Optional[nn.Module] = None,
        seed_data:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.rand(self.batch_size, *self.input_shape)


# ------------------------------------------------------------------
# Jacobian-based Dataset Augmentation
# ------------------------------------------------------------------

class JacobianStrategy:
    """
    Jacobian-based Dataset Augmentation (Papernot et al. 2017).

    Perturbs seed inputs along the sign of the substitute model's
    Jacobian w.r.t. the predicted class, pushing inputs toward the
    decision boundary.

    Parameters
    ----------
    input_shape : shape of a single input
    batch_size  : number of augmented inputs per round
    step_size   : perturbation magnitude (λ in the original paper)
    """

    def __init__(
        self,
        input_shape: tuple,
        batch_size:  int   = 64,
        step_size:   float = 0.10,
    ) -> None:
        self.input_shape = input_shape
        self.batch_size  = batch_size
        self.step_size   = step_size

    def generate(
        self,
        substitute: Optional[nn.Module] = None,
        seed_data:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Augment seed_data using the substitute Jacobian, or fall back to random."""
        if substitute is None or seed_data is None or len(seed_data) == 0:
            return torch.rand(self.batch_size, *self.input_shape)

        substitute.eval()
        n    = min(self.batch_size, len(seed_data))
        idx  = torch.randperm(len(seed_data))[:n]
        x    = seed_data[idx].clone().detach().requires_grad_(True)

        logits = substitute(x)
        pred   = logits.argmax(dim=1)
        one_hot = F.one_hot(pred, num_classes=logits.shape[1]).float()
        (logits * one_hot).sum().backward()

        with torch.no_grad():
            x_aug = (x + self.step_size * x.grad.sign()).clamp(0.0, 1.0).detach()

        # Pad to batch_size with random samples if needed
        if x_aug.shape[0] < self.batch_size:
            pad = torch.rand(self.batch_size - x_aug.shape[0], *self.input_shape)
            x_aug = torch.cat([x_aug, pad], dim=0)

        return x_aug


# ------------------------------------------------------------------
# Adaptive (entropy-based uncertainty sampling)
# ------------------------------------------------------------------

class AdaptiveStrategy:
    """
    Uncertainty-based adaptive sampling.

    Evaluates a large pool of candidates against the current substitute
    and selects those with the highest prediction entropy — the points
    where the substitute is most uncertain and oracle feedback is most
    valuable.

    Parameters
    ----------
    input_shape      : shape of a single input
    batch_size       : number of inputs to return per round
    n_candidates     : candidate pool size to evaluate before selecting
    perturbation_std : Gaussian noise std for neighbourhood exploration
    """

    def __init__(
        self,
        input_shape:       tuple,
        batch_size:        int   = 64,
        n_candidates:      int   = 512,
        perturbation_std:  float = 0.05,
    ) -> None:
        self.input_shape      = input_shape
        self.batch_size       = batch_size
        self.n_candidates     = n_candidates
        self.perturbation_std = perturbation_std

    def generate(
        self,
        substitute: Optional[nn.Module] = None,
        seed_data:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return high-entropy candidates as scored by the substitute."""
        if substitute is None:
            return torch.rand(self.batch_size, *self.input_shape)

        substitute.eval()
        n_rand = self.n_candidates
        candidates = torch.rand(n_rand, *self.input_shape)

        if seed_data is not None and len(seed_data) > 0:
            n_perturb = min(self.n_candidates // 2, len(seed_data))
            idx   = torch.randperm(len(seed_data))[:n_perturb]
            noise = torch.randn_like(seed_data[idx]) * self.perturbation_std
            perturbed  = (seed_data[idx] + noise).clamp(0.0, 1.0)
            candidates = torch.cat([candidates[:n_rand - n_perturb], perturbed], dim=0)

        with torch.no_grad():
            probs   = F.softmax(substitute(candidates), dim=1).clamp(1e-9, 1.0)
            entropy = -(probs * probs.log()).sum(dim=1)

        top_idx = entropy.topk(min(self.batch_size, len(candidates))).indices
        return candidates[top_idx].detach()
