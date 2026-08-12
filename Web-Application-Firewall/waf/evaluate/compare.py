from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from waf.rules.engine import RuleEngine
from waf.ml.classifier import WAFClassifier


def _rule_predictions(requests: list[dict], engine: RuleEngine) -> tuple[list[int], list[float]]:
    labels = []
    scores = []
    for req in requests:
        verdict = engine.inspect(req)
        labels.append(1 if verdict.malicious else 0)
        # normalised score as pseudo-probability
        score = min(verdict.score / 20.0, 1.0)
        scores.append(score)
    return labels, scores


def compare(
    requests:   list[dict],
    true_labels: list[int],
    ml_model:   WAFClassifier,
    rule_engine: RuleEngine | None = None,
) -> dict:
    """
    Run both the rule engine and the ML model on the same request set.
    Returns a dict with metrics for each approach plus a side-by-side summary.
    """
    if rule_engine is None:
        rule_engine = RuleEngine()

    y = np.array(true_labels)

    # Rule engine
    t0 = time.perf_counter()
    rule_preds, rule_scores = _rule_predictions(requests, rule_engine)
    rule_latency = (time.perf_counter() - t0) / len(requests) * 1000  # ms/req

    rule_preds_arr = np.array(rule_preds)
    rule_report    = classification_report(y, rule_preds_arr, target_names=["benign", "malicious"], output_dict=True)

    # ML model
    t0 = time.perf_counter()
    ml_preds, ml_probas = ml_model.predict_batch(requests)
    ml_latency = (time.perf_counter() - t0) / len(requests) * 1000

    ml_report = classification_report(y, ml_preds, target_names=["benign", "malicious"], output_dict=True)

    def _safe_auc(labels, scores):
        try:
            return roc_auc_score(labels, scores)
        except Exception:
            return None

    def _safe_ap(labels, scores):
        try:
            return average_precision_score(labels, scores)
        except Exception:
            return None

    results = {
        "n_samples": len(requests),
        "n_positive": int(y.sum()),
        "rule_engine": {
            "accuracy":      rule_report["accuracy"],
            "precision":     rule_report["malicious"]["precision"],
            "recall":        rule_report["malicious"]["recall"],
            "f1":            rule_report["malicious"]["f1-score"],
            "roc_auc":       _safe_auc(y, rule_scores),
            "avg_precision": _safe_ap(y, rule_scores),
            "latency_ms_per_req": rule_latency,
            "confusion_matrix":   confusion_matrix(y, rule_preds_arr).tolist(),
        },
        "ml_model": {
            "model_type":    ml_model.model_type,
            "accuracy":      ml_report["accuracy"],
            "precision":     ml_report["malicious"]["precision"],
            "recall":        ml_report["malicious"]["recall"],
            "f1":            ml_report["malicious"]["f1-score"],
            "roc_auc":       _safe_auc(y, ml_probas),
            "avg_precision": _safe_ap(y, ml_probas),
            "latency_ms_per_req": ml_latency,
            "confusion_matrix":   confusion_matrix(y, ml_preds).tolist(),
        },
    }
    return results


def print_comparison(results: dict) -> None:
    r = results["rule_engine"]
    m = results["ml_model"]

    print(f"\n{'='*62}")
    print(f"  WAF Detection Comparison  |  {results['n_samples']:,} samples  "
          f"({results['n_positive']:,} attacks)")
    print(f"{'='*62}")
    print(f"{'Metric':<22} {'Rule Engine':>16} {'ML (' + m['model_type'] + ')':>20}")
    print(f"{'-'*62}")

    rows = [
        ("Accuracy",         r["accuracy"],             m["accuracy"]),
        ("Precision",        r["precision"],            m["precision"]),
        ("Recall",           r["recall"],               m["recall"]),
        ("F1",               r["f1"],                   m["f1"]),
        ("ROC-AUC",          r["roc_auc"],              m["roc_auc"]),
        ("Avg Precision",    r["avg_precision"],        m["avg_precision"]),
        ("Latency (ms/req)", r["latency_ms_per_req"],   m["latency_ms_per_req"]),
    ]
    for name, rv, mv in rows:
        rs = f"{rv:.4f}" if rv is not None else "   N/A"
        ms = f"{mv:.4f}" if mv is not None else "   N/A"
        print(f"  {name:<20} {rs:>16} {ms:>20}")

    print(f"{'='*62}")

    print("\nRule engine confusion matrix (rows=true, cols=pred):")
    _print_cm(r["confusion_matrix"])
    print(f"\nML model confusion matrix:")
    _print_cm(m["confusion_matrix"])


def _print_cm(cm: list) -> None:
    labels = ["benign", "malicious"]
    width  = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l:>{width}}" for l in labels)
    print(f"  {header}")
    for i, row in enumerate(cm):
        vals = "".join(f"{v:>{width}}" for v in row)
        print(f"  {labels[i]:<{width}}{vals}")
