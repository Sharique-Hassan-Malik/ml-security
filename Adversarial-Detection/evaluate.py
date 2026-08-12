#!/usr/bin/env python3
"""
Adversarial Example Detection Framework — evaluation CLI.

Generates adversarial examples with each configured attack, runs all
detectors on clean and adversarial batches, then prints a results table
and writes a JSON summary.

Usage
-----
    python evaluate.py
    python evaluate.py --attack pgd --epsilon 0.03 --batch 16 --report out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from attacks import generate
from detectors.feature_squeezing import FeatureSqueezingDetector
from detectors.input_transformation import InputTransformationDetector
from detectors.statistical_tests import KDEDetector, MahalanobisDetector
from models.small_cnn import SmallCNN
from scoring.unified_scorer import DetectorConfig, UnifiedScorer


def _build_scorer() -> UnifiedScorer:
    return UnifiedScorer(
        detectors=[
            DetectorConfig("feature_squeezing",    midpoint=0.05,  steepness=60.0,  weight=1.5),
            DetectorConfig("input_transformation", midpoint=0.15,  steepness=20.0,  weight=1.2),
            DetectorConfig("mahalanobis",          midpoint=30.0,  steepness=0.15,  weight=1.0),
            DetectorConfig("kde",                  midpoint=5.0,   steepness=1.0,   weight=0.8),
        ],
        strategy="weighted",
        threshold=0.50,
    )


def _collect_scores(
    detectors: dict,
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scores = {}
    fs  = detectors["fs"]
    it  = detectors["it"]
    mah = detectors["mah"]
    kde = detectors["kde"]

    scores["feature_squeezing"]    = fs.score(x)
    scores["input_transformation"] = it.score(x)
    scores["mahalanobis"]          = mah.score(x)
    scores["kde"]                  = kde.score(x)
    return scores


_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED   = "\033[91m"


def _color_auroc(v: float) -> str:
    c = _GREEN if v >= 0.75 else (_YELLOW if v >= 0.55 else _RED)
    return f"{c}{v:.4f}{_RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial detection evaluation across multiple attacks"
    )
    parser.add_argument("--attacks",  nargs="+", default=["fgsm", "pgd", "cw"],
                        help="Attacks to evaluate (fgsm pgd cw autoattack)")
    parser.add_argument("--epsilon",  type=float, default=0.03)
    parser.add_argument("--batch",    type=int,   default=16,
                        help="Number of synthetic test samples")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--report",   default="", help="Write JSON results to file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = SmallCNN(num_classes=10)
    model.eval()

    # Synthetic clean data
    x_clean = torch.rand(args.batch, 3, 32, 32)
    y_clean = torch.randint(0, 10, (args.batch,))

    # Calibrate detectors on clean data
    print("[*] Calibrating detectors…")

    fs  = FeatureSqueezingDetector(model, threshold=0.05)
    it  = InputTransformationDetector(model, threshold=0.15, n_transforms=15)
    mah = MahalanobisDetector(model)
    mah.calibrate(x_clean, y_clean, num_classes=10)
    kde = KDEDetector(model)
    kde.calibrate(x_clean)

    detectors = {"fs": fs, "it": it, "mah": mah, "kde": kde}
    scorer    = _build_scorer()

    print("[*] Collecting clean baseline scores…")
    clean_scores = _collect_scores(detectors, x_clean)

    results = {}

    header = f"  {'Attack':<14} {'TPR':>6} {'FPR':>6} {'BalAcc':>8} {'AUROC':>8}"
    sep    = "  " + "─" * (len(header) - 2)
    print(f"\n{_BOLD}Detection results (ε={args.epsilon}){_RESET}")
    print(header)
    print(sep)

    for attack_name in args.attacks:
        print(f"  [{attack_name}] generating adversarial examples…", end="\r")

        kwargs: dict = {"epsilon": args.epsilon}
        if attack_name == "cw":
            kwargs = {"c": 1.0, "steps": 100}

        try:
            x_adv = generate(model, x_clean, y_clean, method=attack_name, **kwargs)
        except Exception as exc:
            print(f"  [{attack_name}] ERROR: {exc}")
            continue

        adv_scores = _collect_scores(detectors, x_adv)
        metrics    = scorer.evaluate(clean_scores, adv_scores)

        results[attack_name] = metrics

        tpr_s = f"{metrics['tpr']:.4f}"
        fpr_s = f"{metrics['fpr']:.4f}"
        ba_s  = f"{metrics['balanced_acc']:.4f}"
        auc_s = _color_auroc(metrics["auroc"])

        print(f"  {attack_name:<14} {tpr_s:>6} {fpr_s:>6} {ba_s:>8} {auc_s:>8}")

    print(sep)

    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[*] JSON report → {args.report}")


if __name__ == "__main__":
    main()
