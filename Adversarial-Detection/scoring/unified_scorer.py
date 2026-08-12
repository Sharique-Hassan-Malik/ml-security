"""
Unified scoring system.

Combines raw scores from multiple detectors into a single adversarial
probability estimate using a weighted voting scheme.

Each detector produces a raw score on its own scale.  The scorer:
  1. Normalises each detector's score to [0, 1] using a per-detector
     sigmoid with learned (or default) midpoint and steepness.
  2. Aggregates via a configurable strategy: "mean", "max" or "weighted".
  3. Returns a final probability and a verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class DetectorConfig:
    name:      str
    midpoint:  float = 0.5      # raw score that maps to probability 0.5
    steepness: float = 10.0     # sigmoid steepness in normalised space
    weight:    float = 1.0


@dataclass
class ScoringResult:
    is_adversarial:  torch.Tensor          # BoolTensor (B,)
    probability:     torch.Tensor          # FloatTensor (B,) in [0, 1]
    per_detector:    Dict[str, torch.Tensor]  # normalised scores per detector
    raw_scores:      Dict[str, torch.Tensor]  # original scores per detector
    threshold:       float


class UnifiedScorer:
    """
    Aggregate detection scores from multiple detectors.

    Parameters
    ----------
    detectors       : list of DetectorConfig objects
    strategy        : "mean", "max" or "weighted"
    threshold       : probability above which a sample is flagged
    """

    def __init__(
        self,
        detectors: List[DetectorConfig],
        strategy:  str   = "weighted",
        threshold: float = 0.50,
    ) -> None:
        self.detectors = {d.name: d for d in detectors}
        self.strategy  = strategy
        self.threshold = threshold

    def score(
        self,
        raw_scores: Dict[str, torch.Tensor],
    ) -> ScoringResult:
        """
        Combine raw detector scores into a unified adversarial probability.

        Parameters
        ----------
        raw_scores : dict mapping detector name to a (B,) raw score tensor

        Returns
        -------
        ScoringResult
        """
        normalised: Dict[str, torch.Tensor] = {}

        for name, raw in raw_scores.items():
            cfg = self.detectors.get(name, DetectorConfig(name))
            normalised[name] = _sigmoid_normalise(raw, cfg.midpoint, cfg.steepness)

        if not normalised:
            batch = next(iter(raw_scores.values())).shape[0]
            dummy = torch.zeros(batch)
            return ScoringResult(
                is_adversarial = dummy.bool(),
                probability    = dummy,
                per_detector   = {},
                raw_scores     = raw_scores,
                threshold      = self.threshold,
            )

        stacked = torch.stack(list(normalised.values()), dim=1)   # (B, D)

        if self.strategy == "max":
            probability = stacked.max(dim=1).values
        elif self.strategy == "weighted":
            weights = torch.tensor(
                [self.detectors.get(n, DetectorConfig(n)).weight
                 for n in normalised],
                dtype=torch.float32,
            )
            weights = weights / weights.sum()
            probability = (stacked * weights.unsqueeze(0)).sum(dim=1)
        else:  # "mean"
            probability = stacked.mean(dim=1)

        return ScoringResult(
            is_adversarial = probability > self.threshold,
            probability    = probability,
            per_detector   = normalised,
            raw_scores     = raw_scores,
            threshold      = self.threshold,
        )

    def evaluate(
        self,
        clean_scores: Dict[str, torch.Tensor],
        adv_scores:   Dict[str, torch.Tensor],
    ) -> dict:
        """
        Compute TPR, FPR, AUROC and balanced accuracy across clean and
        adversarial batches.
        """
        res_clean = self.score(clean_scores)
        res_adv   = self.score(adv_scores)

        tp = res_adv.is_adversarial.sum().item()
        fn = (~res_adv.is_adversarial).sum().item()
        fp = res_clean.is_adversarial.sum().item()
        tn = (~res_clean.is_adversarial).sum().item()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        balanced_acc = 0.5 * (tpr + (1 - fpr))

        auroc = _approx_auroc(res_clean.probability, res_adv.probability)

        return {
            "tpr":          round(tpr, 4),
            "fpr":          round(fpr, 4),
            "balanced_acc": round(balanced_acc, 4),
            "auroc":        round(auroc, 4),
            "tp":           int(tp),
            "fn":           int(fn),
            "fp":           int(fp),
            "tn":           int(tn),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sigmoid_normalise(x: torch.Tensor, midpoint: float, steepness: float) -> torch.Tensor:
    """Map raw scores to [0, 1] via a sigmoid centred at *midpoint*."""
    return torch.sigmoid(steepness * (x - midpoint))


def _approx_auroc(clean_probs: torch.Tensor, adv_probs: torch.Tensor) -> float:
    """
    Approximate AUROC via the Wilcoxon–Mann–Whitney statistic.
    P(score_adv > score_clean) estimated over all pairs.
    Uses a sorted O(n log n) approximation.
    """
    clean = clean_probs.detach().cpu()
    adv   = adv_probs.detach().cpu()

    n_c = len(clean)
    n_a = len(adv)

    all_scores = torch.cat([clean, adv])
    all_labels = torch.cat([torch.zeros(n_c), torch.ones(n_a)])

    order     = all_scores.argsort(descending=True)
    labels_s  = all_labels[order]

    tps = labels_s.cumsum(0)
    fps = (1 - labels_s).cumsum(0)

    tpr = tps / max(n_a, 1)
    fpr = fps / max(n_c, 1)

    # Trapezoidal integration
    auroc = 0.0
    for i in range(1, len(tpr)):
        auroc += (fpr[i] - fpr[i - 1]).item() * (tpr[i] + tpr[i - 1]).item() * 0.5

    return max(0.0, min(1.0, abs(auroc)))
