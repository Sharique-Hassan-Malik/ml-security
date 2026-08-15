"""
AutoAttack wrapper.

AutoAttack (Croce & Hein 2020) is a parameter-free ensemble of four attacks:
  APGD-CE  — adaptive PGD with cross-entropy loss
  APGD-T   — adaptive PGD with targeted DLR loss
  FAB      — Fast Adaptive Boundary attack (minimises L2 distortion)
  Square   — score-based black-box attack

This module wraps the `autoattack` library when available and falls back to
a strong adaptive PGD implementation when it is not installed, so the
detection framework always has a source of adversarial examples.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .gradient_attacks import pgd

log = logging.getLogger(__name__)


def autoattack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float   = 0.03,
    norm: str        = "Linf",
    version: str     = "standard",
    device: str      = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Run AutoAttack on batch (x, y).

    Falls back to strong PGD (100 steps, random start) if the
    `autoattack` package is not installed.

    Parameters
    ----------
    model   : nn.Module in eval mode
    x       : clean inputs in [0, 1]
    y       : true labels
    epsilon : L∞ budget
    norm    : "Linf" or "L2"
    version : "standard" or "plus" (passed to autoattack library)
    device  : torch device string

    Returns
    -------
    Adversarial examples in [0, 1].
    """
    try:
        from autoattack import AutoAttack as _AA  # type: ignore

        if seed is not None:
            torch.manual_seed(seed)

        adversary = _AA(
            model,
            norm    = norm,
            eps     = epsilon,
            version = version,
            device  = torch.device(device),
            verbose = False,
        )
        x_adv = adversary.run_standard_evaluation(x.to(device), y.to(device), bs=x.shape[0])
        return x_adv.detach().cpu()

    except ImportError:
        log.warning(
            "autoattack package not installed — falling back to PGD-100 (random start). "
            "Install with: pip install autoattack"
        )
        model_cpu = model.cpu()
        return pgd(
            model_cpu, x.cpu(), y.cpu(),
            epsilon=epsilon,
            alpha=epsilon / 4,
            steps=100,
            random_start=True,
        )
