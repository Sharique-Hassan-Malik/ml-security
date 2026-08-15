"""
Input Transformation detector (Xie et al. 2018 / Guo et al. 2018).

Adversarial perturbations are brittle: small input transformations
that preserve semantic content often destroy the perturbation.

This detector applies a battery of stochastic transformations and
measures prediction variance across the ensemble.  A high variance
indicates the prediction is sensitive to small changes — a hallmark
of adversarial examples.

Transformations applied:
  - Random resize and pad back to original size
  - JPEG-style compression via DCT coefficient quantisation
  - Random Gaussian noise injection
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputTransformationDetector:
    """
    Detect adversarial examples by measuring prediction consistency
    across stochastic input transformations.

    Parameters
    ----------
    model        : nn.Module in eval mode
    n_transforms : number of transformed copies to evaluate
    threshold    : prediction variance above which input is flagged
    transforms   : list of callable transforms; defaults to built-ins
    """

    def __init__(
        self,
        model: nn.Module,
        n_transforms: int    = 20,
        threshold: float     = 0.10,
        transforms: List[Callable] | None = None,
    ) -> None:
        self.model        = model
        self.n_transforms = n_transforms
        self.threshold    = threshold
        self.transforms   = transforms or [
            _random_resize_pad,
            _random_noise,
            _jpeg_approx,
        ]

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute detection score for each sample.

        Score = mean variance of top-1 predicted class across transforms.
        Higher → more adversarial.
        """
        self.model.eval()
        B = x.shape[0]

        # (n_transforms, B) predicted class indices
        preds = []
        with torch.no_grad():
            for _ in range(self.n_transforms):
                t   = self.transforms[_ % len(self.transforms)]
                x_t = t(x)
                logits = self.model(x_t)
                preds.append(logits.argmax(dim=1))

        preds  = torch.stack(preds, dim=0).float()   # (T, B)
        # Disagreement score: fraction of transforms that disagree with the majority
        scores = torch.zeros(B, device=x.device)
        for b in range(B):
            col = preds[:, b]
            majority = col.mode().values
            scores[b] = (col != majority).float().mean()

        return scores

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x)
        return scores > self.threshold, scores


# ------------------------------------------------------------------
# Built-in transformations
# ------------------------------------------------------------------

def _random_resize_pad(x: torch.Tensor) -> torch.Tensor:
    """Resize to a random smaller size then pad back to original dimensions."""
    _, C, H, W = x.shape
    scale    = torch.empty(1).uniform_(0.80, 0.99).item()
    new_h    = max(1, int(H * scale))
    new_w    = max(1, int(W * scale))
    resized  = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_top  = (H - new_h) // 2
    pad_bot  = H - new_h - pad_top
    pad_left = (W - new_w) // 2
    pad_right = W - new_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bot), value=0.0)


def _random_noise(x: torch.Tensor) -> torch.Tensor:
    """Add small random Gaussian noise."""
    return (x + torch.randn_like(x) * 0.02).clamp(0.0, 1.0)


def _jpeg_approx(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate JPEG compression artefacts via aggressive average pooling
    followed by upsampling — reduces high-frequency perturbation components.
    """
    _, C, H, W = x.shape
    factor = 4
    down = F.avg_pool2d(x, kernel_size=factor, stride=factor, padding=0)
    return F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
