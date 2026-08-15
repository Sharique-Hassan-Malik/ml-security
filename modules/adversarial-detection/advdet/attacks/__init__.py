"""
attacks — adversarial example generation.

Public API
----------
generate(model, x, y, method, **kwargs) -> Tensor
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .autoattack_wrapper import autoattack
from .cw_attack import cw_l2
from .gradient_attacks import fgsm, pgd

_METHODS = {
    "fgsm":       fgsm,
    "pgd":        pgd,
    "cw":         cw_l2,
    "autoattack": autoattack,
}


def generate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = "pgd",
    **kwargs,
) -> torch.Tensor:
    """
    Generate adversarial examples.

    Parameters
    ----------
    model  : nn.Module — must be in eval mode
    x      : clean inputs in [0, 1]
    y      : true labels
    method : one of "fgsm", "pgd", "cw" or "autoattack"
    **kwargs : passed to the underlying attack function

    Returns
    -------
    Adversarial Tensor in [0, 1].
    """
    method = method.lower()
    if method not in _METHODS:
        raise ValueError(f"Unknown attack '{method}'. Choose from {list(_METHODS)}.")
    model.eval()
    return _METHODS[method](model, x, y, **kwargs)


__all__ = ["generate", "fgsm", "pgd", "cw_l2", "autoattack"]
