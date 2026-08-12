"""
Feature Squeezing detector (Xu et al. 2018).

The core idea: adversarial perturbations are fragile.  Applying a
simple "squeezer" (bit-depth reduction or spatial smoothing) to the
input destroys the perturbation while having little effect on clean
examples.  If the model's output changes significantly after squeezing,
the input is likely adversarial.

Two squeezers are implemented:
  1. Bit-depth reduction  — quantise pixel values to k bits
  2. Spatial smoothing    — median filter or Gaussian blur

Detection threshold: L1 distance between original and squeezed
model softmax outputs.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSqueezingDetector:
    """
    Detect adversarial examples via feature squeezing.

    Parameters
    ----------
    model     : nn.Module in eval mode
    threshold : L1 distance above which an input is flagged
    bit_depth : number of bits for pixel quantisation (1–8)
    smooth_k  : kernel size for spatial smoothing (0 = disabled)
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.05,
        bit_depth: int   = 4,
        smooth_k: int    = 2,
    ) -> None:
        self.model     = model
        self.threshold = threshold
        self.bit_depth = bit_depth
        self.smooth_k  = smooth_k

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a detection score for each sample in the batch.

        Returns a 1-D tensor of scores — higher means more likely adversarial.
        The score is the max L1 distance across all squeezers.
        """
        self.model.eval()
        with torch.no_grad():
            p_orig = F.softmax(self.model(x), dim=1)

            scores = torch.zeros(x.shape[0], device=x.device)

            x_bit = _bit_reduce(x, self.bit_depth)
            p_bit = F.softmax(self.model(x_bit), dim=1)
            scores = torch.maximum(scores, (p_orig - p_bit).abs().sum(dim=1))

            if self.smooth_k > 0:
                x_smooth = _median_smooth(x, self.smooth_k)
                p_smooth = F.softmax(self.model(x_smooth), dim=1)
                scores   = torch.maximum(scores, (p_orig - p_smooth).abs().sum(dim=1))

        return scores

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (is_adversarial, scores) for each sample.

        is_adversarial : BoolTensor, True = flagged as adversarial
        scores         : FloatTensor of raw detection scores
        """
        scores = self.score(x)
        return scores > self.threshold, scores


# ------------------------------------------------------------------
# Squeezers
# ------------------------------------------------------------------

def _bit_reduce(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Reduce pixel precision to *bits* bits."""
    levels = 2 ** bits - 1
    return (x * levels).round() / levels


def _median_smooth(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Approximate median filter via a sequence of max-pool and min-pool operations.
    True median filtering is not available as a differentiable PyTorch op, so we
    use average pooling as a practical substitute for the squeezing effect.
    """
    # Asymmetric padding so the output keeps the input's spatial size for any k.
    # Symmetric padding (k // 2 on both sides) only preserves size for odd k;
    # the default smooth_k is even, which would otherwise grow the image by 1px.
    total_pad = k - 1
    lo = total_pad // 2
    hi = total_pad - lo
    x = F.pad(x, (lo, hi, lo, hi))
    return F.avg_pool2d(x, kernel_size=k, stride=1)
