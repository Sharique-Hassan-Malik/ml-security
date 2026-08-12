"""
Carlini & Wagner L2 attack (C&W, 2017).

Solves the constrained optimisation problem:

    min  ‖δ‖₂  +  c · f(x + δ)
    s.t. x + δ ∈ [0, 1]

where f(·) is the Carlini-Wagner loss that is negative iff the example is
misclassified.  The box constraint is handled via the tanh change of variable:

    x + δ = ½(tanh(w) + 1)   →   w = arctanh(2x − 1)

so the unconstrained variable w is optimised directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def cw_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float      = 1.0,
    kappa: float  = 0.0,
    steps: int    = 200,
    lr: float     = 0.01,
) -> torch.Tensor:
    """
    C&W L2 attack.

    Parameters
    ----------
    model  : nn.Module in eval mode
    x      : clean input (B, C, H, W) in [0, 1]
    y      : true labels (B,)
    c      : trade-off constant between distortion and misclassification
    kappa  : confidence margin (0 = boundary, higher = more confident misclassification)
    steps  : Adam optimisation steps
    lr     : Adam learning rate

    Returns
    -------
    Adversarial examples in [0, 1].
    """
    # Change of variable: x_adv = 0.5*(tanh(w)+1)
    w = _arctanh(2.0 * x - 1.0).detach().clone().requires_grad_(True)
    optimizer = optim.Adam([w], lr=lr)

    best_adv  = x.clone()
    best_dist = torch.full((x.shape[0],), float("inf"), device=x.device)

    for _ in range(steps):
        optimizer.zero_grad()
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        dist  = ((x_adv - x) ** 2).flatten(1).sum(1)      # L2² per sample
        logits = model(x_adv)
        loss_f = _cw_loss(logits, y, kappa)
        loss   = (dist + c * loss_f).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_adv_d = 0.5 * (torch.tanh(w) + 1.0)
            improved = dist < best_dist
            best_dist = torch.where(improved, dist, best_dist)
            best_adv  = torch.where(
                improved.view(-1, 1, 1, 1).expand_as(best_adv),
                x_adv_d,
                best_adv,
            )

    return best_adv.detach().clamp(0.0, 1.0)


def _cw_loss(logits: torch.Tensor, y: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    f₆ variant of the C&W objective.

    Returns a value that is ≤ 0 when the example is misclassified
    with at least kappa confidence.
    """
    batch = logits.shape[0]
    one_hot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)

    correct_logit = (logits * one_hot).sum(1)
    other_logit   = ((1.0 - one_hot) * logits - one_hot * 1e9).max(1).values

    return torch.clamp(correct_logit - other_logit + kappa, min=0.0)


def _arctanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arctanh, clamped to avoid ±inf at ±1."""
    x = x.clamp(-0.9999, 0.9999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))
