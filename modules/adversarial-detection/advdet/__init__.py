"""advdet — detecting adversarial inputs, and generating them to prove it works.

    from advdet import build_detectors, score_batch, generate

A detector nobody attacked is a detector with unknown recall, so the attacks
and the defences live in the same module on purpose. `build_detectors` is the
one place the four detectors get assembled and calibrated, shared by the
benchmark and by the runtime guard — they must be the same stack or the
benchmark is measuring something else.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .attacks import generate
from .detectors.feature_squeezing import FeatureSqueezingDetector
from .detectors.input_transformation import InputTransformationDetector
from .detectors.statistical_tests import KDEDetector, MahalanobisDetector
from .models.small_cnn import SmallCNN
from .scoring import DetectorConfig, ScoringResult, UnifiedScorer

MODULE_NAME = "adversarial-detection"

__all__ = [
    "MODULE_NAME", "generate", "SmallCNN",
    "FeatureSqueezingDetector", "InputTransformationDetector",
    "KDEDetector", "MahalanobisDetector",
    "DetectorConfig", "ScoringResult", "UnifiedScorer",
    "build_detectors", "build_scorer", "score_batch",
]

# Midpoints and steepnesses are per-detector because the raw scores are on
# unrelated scales — an L1 distance and a Mahalanobis distance are not
# comparable until each has been mapped through its own sigmoid.
_CONFIGS = (
    DetectorConfig("feature_squeezing", midpoint=0.05, steepness=60.0, weight=1.5),
    DetectorConfig("input_transformation", midpoint=0.15, steepness=20.0, weight=1.2),
    DetectorConfig("mahalanobis", midpoint=30.0, steepness=0.15, weight=1.0),
    DetectorConfig("kde", midpoint=5.0, steepness=1.0, weight=0.8),
)


def build_scorer(threshold: float = 0.50) -> UnifiedScorer:
    return UnifiedScorer(detectors=list(_CONFIGS), strategy="weighted", threshold=threshold)


def build_detectors(
    model: nn.Module,
    x_clean: torch.Tensor | None = None,
    y_clean: torch.Tensor | None = None,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Assemble the detector stack around *model*.

    Feature squeezing and input transformation need nothing but the model.
    Mahalanobis and KDE are density estimates and are meaningless without clean
    activations to fit against, so they are only included when calibration data
    is supplied — a detector that returns a number it cannot justify is worse
    than a missing detector.
    """
    model.eval()
    detectors: dict[str, Any] = {
        "feature_squeezing": FeatureSqueezingDetector(model, threshold=0.05),
        "input_transformation": InputTransformationDetector(
            model, threshold=0.15, n_transforms=15
        ),
    }

    if x_clean is not None:
        mahalanobis = MahalanobisDetector(model)
        kde = KDEDetector(model)
        if y_clean is not None:
            mahalanobis.calibrate(x_clean, y_clean, num_classes=num_classes)
            detectors["mahalanobis"] = mahalanobis
        kde.calibrate(x_clean)
        detectors["kde"] = kde

    return detectors


def score_batch(detectors: dict[str, Any], x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Raw per-detector scores for a batch, keyed the way the scorer expects."""
    return {name: detector.score(x) for name, detector in detectors.items()}
