"""
Gradient-based adversarial attacks: FGSM and PGD.

FGSM — Fast Gradient Sign Method (Goodfellow et al. 2014)
PGD  — Projected Gradient Descent (Madry et al. 2018)

Both produce L∞-bounded perturbations.  All inputs and outputs are in [0, 1].
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
) -> torch.Tensor:
    """
    Single-step FGSM.

    Parameters
    ----------
    model   : nn.Module in eval mode
    x       : clean input tensor (B, C, H, W) in [0, 1]
    y       : true labels (B,)
    epsilon : L∞ perturbation budget

    Returns
    -------
    Adversarial examples clipped to [0, 1].
    """
    x_adv = x.clone().detach().requires_grad_(True)
    loss  = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + epsilon * x_adv.grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float   = 0.007,
    steps: int     = 40,
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD attack with optional random start.

    Parameters
    ----------
    model        : nn.Module in eval mode
    x            : clean input tensor in [0, 1]
    y            : true labels
    epsilon      : L∞ budget
    alpha        : per-step size
    steps        : number of PGD iterations
    random_start : if True, initialise from a random point in the epsilon ball

    Returns
    -------
    Adversarial examples clipped to [0, 1].
    """
    criterion = nn.CrossEntropyLoss()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x)

    delta = delta.detach()

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = criterion(model(x + delta), y)
        loss.backward()
        with torch.no_grad():
            delta = delta + alpha * delta.grad.sign()
            delta = delta.clamp(-epsilon, epsilon)
            delta = (x + delta).clamp(0.0, 1.0) - x

    return (x + delta).detach().clamp(0.0, 1.0)
