"""
Statistical test detectors.

Two methods are implemented:

1. Mahalanobis Distance detector (Lee et al. NeurIPS 2018)
   Fits a class-conditional Gaussian to intermediate layer activations
   on clean training data.  At inference time, the Mahalanobis distance
   from the nearest class centroid is used as a detection score.

2. Kernel Density Estimation (KDE) detector
   Fits a non-parametric density model on penultimate-layer activations
   of clean examples.  Low density regions indicate OOD or adversarial inputs.

Both detectors require a calibration step on clean data before use.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisDetector:
    """
    Mahalanobis distance detector.

    Calibrate on clean data once, then call score() / predict() at test time.
    """

    def __init__(self, model: nn.Module, layer_name: str = "") -> None:
        """
        Parameters
        ----------
        model      : nn.Module in eval mode
        layer_name : name of the intermediate layer to hook (e.g. "layer4").
                     If empty, the penultimate layer output is used via a
                     hook on the last-but-one module.
        """
        self.model      = model
        self.layer_name = layer_name
        self._hook_handle = None

        # Filled during calibrate()
        self._class_means: Optional[torch.Tensor] = None
        self._precision:   Optional[torch.Tensor] = None  # inverse covariance

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        x_clean: torch.Tensor,
        y_clean: torch.Tensor,
        num_classes: int,
    ) -> None:
        """
        Fit class-conditional Gaussians on clean activations.

        Parameters
        ----------
        x_clean     : clean input batch in [0, 1]
        y_clean     : corresponding true labels
        num_classes : total number of classes in the model
        """
        features = self._extract_features(x_clean)   # (N, D)
        D = features.shape[1]

        means = torch.zeros(num_classes, D)
        counts = torch.zeros(num_classes)

        for c in range(num_classes):
            mask = (y_clean == c)
            if mask.sum() == 0:
                continue
            means[c]  = features[mask].mean(dim=0)
            counts[c] = mask.sum().float()

        self._class_means = means

        # Shared tied covariance
        centered = features - means[y_clean]
        cov = (centered.T @ centered) / max(len(features) - 1, 1)
        # Regularise and invert
        cov += torch.eye(D) * 1e-5
        self._precision = torch.linalg.inv(cov)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample Mahalanobis distance to the nearest class centroid."""
        if self._class_means is None:
            raise RuntimeError("Call calibrate() before score().")

        features = self._extract_features(x)         # (B, D)
        P        = self._precision                    # (D, D)

        scores = torch.full((features.shape[0],), float("inf"))
        for c in range(self._class_means.shape[0]):
            diff = features - self._class_means[c]   # (B, D)
            # Mahalanobis² = diff @ P @ diff.T  (per sample)
            d2 = (diff @ P * diff).sum(dim=1)        # (B,)
            scores = torch.minimum(scores, d2)

        return scores.sqrt()

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 50.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x)
        return scores > threshold, scores

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        activation: List[torch.Tensor] = []

        if self.layer_name:
            target = dict(self.model.named_modules())[self.layer_name]
        else:
            # Hook the second-to-last module that has parameters
            modules = [m for m in self.model.modules() if list(m.parameters(recurse=False))]
            target  = modules[-2] if len(modules) >= 2 else modules[-1]

        def hook(_, __, output):
            activation.append(output.detach().cpu().flatten(1))

        handle = target.register_forward_hook(hook)
        with torch.no_grad():
            self.model(x)
        handle.remove()

        return activation[0]


# ------------------------------------------------------------------
# KDE detector
# ------------------------------------------------------------------

class KDEDetector:
    """
    Gaussian KDE on penultimate-layer activations.

    Uses the median inter-sample L2 distance as bandwidth (Scott's rule
    is less reliable in high dimensions).
    """

    def __init__(self, model: nn.Module, layer_name: str = "") -> None:
        self.model      = model
        self.layer_name = layer_name
        self._train_features: Optional[torch.Tensor] = None
        self._bandwidth: float = 1.0

    def calibrate(self, x_clean: torch.Tensor) -> None:
        """Fit KDE on clean activations."""
        feats = self._extract_features(x_clean)
        # Subsample for bandwidth estimation to keep memory manageable
        n_sub = min(500, feats.shape[0])
        sub   = feats[:n_sub]
        dists = torch.cdist(sub, sub)
        # Median pairwise distance (exclude diagonal)
        mask  = ~torch.eye(n_sub, dtype=torch.bool)
        bw    = dists[mask].median().item()
        self._bandwidth     = max(bw, 1e-3)
        self._train_features = feats

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return negative log-density — higher = more anomalous."""
        if self._train_features is None:
            raise RuntimeError("Call calibrate() before score().")

        feats = self._extract_features(x)                # (B, D)
        train = self._train_features                     # (N, D)
        h     = self._bandwidth

        # Gaussian kernel: K(u) = exp(-‖u‖²/(2h²))
        dists_sq = torch.cdist(feats, train) ** 2       # (B, N)
        log_k    = -dists_sq / (2.0 * h ** 2)           # (B, N)
        # log-sum-exp for numerical stability
        log_dens = torch.logsumexp(log_k, dim=1) - math.log(train.shape[0])

        return -log_dens   # negate: low density → high score

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x)
        return scores > threshold, scores

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        activation: List[torch.Tensor] = []

        if self.layer_name:
            target = dict(self.model.named_modules())[self.layer_name]
        else:
            modules = [m for m in self.model.modules() if list(m.parameters(recurse=False))]
            target  = modules[-2] if len(modules) >= 2 else modules[-1]

        def hook(_, __, output):
            activation.append(output.detach().cpu().flatten(1))

        handle = target.register_forward_hook(hook)
        with torch.no_grad():
            self.model(x)
        handle.remove()

        return activation[0]
