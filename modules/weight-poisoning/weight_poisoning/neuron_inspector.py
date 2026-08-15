"""
Per-neuron anomaly detection.

Each output neuron is a row in the weight matrix (or a filter for conv layers).
Trojan neurons tend to have abnormally large L2 norms compared to their peers —
they encode strong responses to the backdoor trigger pattern.
"""

from __future__ import annotations

from typing import List

import torch

from .finding import Finding, poison_finding


_IQR_MULTIPLIER  = 3.0      # Tukey fence for dominant-neuron detection
_DORMANT_THRESH  = 1e-4     # L2 norm below this → dormant neuron
_MIN_NEURONS     = 4        # skip tiny layers (not enough neurons for stats)
_CV_THRESHOLD    = 1.2      # coefficient of variation threshold


class NeuronInspector:
    """Analyse per-neuron L2 norms within each layer."""

    def analyze(self, name: str, tensor: torch.Tensor) -> List[Finding]:
        norms = _neuron_norms(tensor)
        n = len(norms)
        if n < _MIN_NEURONS:
            return []

        findings: List[Finding] = []

        # ---- dominant neurons (unusually large L2 norm) ---------------
        q1, q3 = _quantiles(norms)
        iqr  = q3 - q1
        fence = q3 + _IQR_MULTIPLIER * iqr

        dominant_mask = norms > fence
        dominant_count = dominant_mask.sum().item()
        if dominant_count > 0 and iqr > 1e-6:
            ratio = dominant_count / n
            max_norm  = norms.max().item()
            norm_mean = norms.mean().item()
            excess    = (norms[dominant_mask].mean().item() / max(norm_mean, 1e-9))
            sev = "high" if ratio > 0.05 or excess > 5.0 else "medium"
            findings.append(poison_finding(
                layer    = name,
                test     = "dominant_neurons",
                severity = sev,
                score    = min(1.0, excess / 10.0),
                detail   = (
                    f"{dominant_count}/{n} neurons exceed Tukey fence "
                    f"(Q3 + {_IQR_MULTIPLIER}×IQR = {fence:.4f}). "
                    f"Mean excess: {excess:.2f}×. "
                    "Trojan neurons characteristically have norms 3–10× larger than "
                    "their layer peers."
                ),
                metadata = {
                    "dominant_count": int(dominant_count),
                    "total_neurons":  n,
                    "fence":          round(float(fence), 6),
                    "max_norm":       round(float(max_norm), 6),
                },
            ))

        # ---- dormant neurons ------------------------------------------
        dormant_count = (norms < _DORMANT_THRESH).sum().item()
        dormant_frac  = dormant_count / n
        if dormant_frac > 0.20:
            # Dormant neurons alone are not suspicious; combined with dominants → flag
            sev = "medium" if dominant_count > 0 else "low"
            findings.append(poison_finding(
                layer    = name,
                test     = "dormant_neurons",
                severity = sev,
                score    = min(0.6, dormant_frac),
                detail   = (
                    f"{dormant_count}/{n} neurons ({dormant_frac * 100:.1f}%) have "
                    f"L2 norm < {_DORMANT_THRESH}. Large dormant populations sometimes "
                    "mask the statistical signature of a small number of trojan neurons."
                ),
                metadata = {
                    "dormant_count":   int(dormant_count),
                    "dormant_fraction": round(dormant_frac, 4),
                },
            ))

        # ---- coefficient of variation ---------------------------------
        mean_norm = norms.mean().item()
        std_norm  = norms.std(unbiased=True).item()
        if mean_norm > 1e-6:
            cv = std_norm / mean_norm
            if cv > _CV_THRESHOLD:
                findings.append(poison_finding(
                    layer    = name,
                    test     = "norm_dispersion",
                    severity = "low",
                    score    = min(0.4, cv / (3 * _CV_THRESHOLD)),
                    detail   = (
                        f"Neuron-norm coefficient of variation = {cv:.3f} "
                        f"(threshold {_CV_THRESHOLD}). "
                        "High dispersion indicates a heterogeneous neuron population — "
                        "a secondary indicator of implanted shortcut pathways."
                    ),
                    metadata = {"cv": round(cv, 4), "mean_norm": round(mean_norm, 6)},
                ))

        return findings


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _neuron_norms(tensor: torch.Tensor) -> torch.Tensor:
    """Return a 1-D tensor of per-neuron L2 norms."""
    # Linear:  (out, in)
    # Conv2d:  (out, in, kH, kW)  → treat each out-channel as a neuron
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    return flat.norm(dim=1)


def _quantiles(v: torch.Tensor):
    sorted_v, _ = v.sort()
    n = len(sorted_v)
    q1 = sorted_v[n // 4].item()
    q3 = sorted_v[(3 * n) // 4].item()
    return q1, q3
