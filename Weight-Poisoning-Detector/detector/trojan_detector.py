"""
Cross-layer trojan pattern detection.

Individual-layer tests can miss attacks that are subtle per-layer but
collectively form a consistent backdoor pathway. This module correlates
findings across the entire model:

  1. Last-layer class asymmetry — Neural Cleanse (Wang et al. 2019) showed
     that trigger-target classes often have anomalously small inbound weight
     norms, making them easy to activate with a small perturbation.

  2. Weight norm progression — clean networks exhibit a roughly monotone
     increase in per-layer feature complexity. Abrupt breaks may indicate
     a poisoned layer injected into the stack.

  3. Dominant neuron pathway — if the same relative neuron indices appear
     as outliers in two consecutive layers, it suggests a direct wire-through
     of a trigger pathway.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from .report import Finding


class TrojanDetector:
    """Runs cross-layer analysis over the full weight dictionary."""

    def analyze(self, weights: Dict[str, torch.Tensor]) -> List[Finding]:
        findings: List[Finding] = []

        named = list(weights.items())

        findings.extend(self._last_layer_asymmetry(named))
        findings.extend(self._norm_progression(named))
        findings.extend(self._dominant_neuron_pathway(named))

        return findings

    # ------------------------------------------------------------------
    # Test 1 — last-layer class weight asymmetry
    # ------------------------------------------------------------------

    def _last_layer_asymmetry(
        self, named: List[Tuple[str, torch.Tensor]]
    ) -> List[Finding]:
        """
        In a classification model the final linear layer has shape (num_classes, features).
        Neural Cleanse found that the target class of a backdoor typically has a much
        smaller inbound weight norm, allowing a small universal perturbation to push
        any input toward it.
        """
        # Heuristic: last layer is the last 2-D tensor with dim[0] < 10000
        candidate = None
        for name, t in reversed(named):
            if t.dim() == 2 and t.shape[0] <= 10_000:
                candidate = (name, t)
                break

        if candidate is None:
            return []

        name, t = candidate
        norms = t.float().norm(dim=1)
        if len(norms) < 2:
            return []

        mean_norm = norms.mean().item()
        std_norm  = norms.std(unbiased=True).item()
        if std_norm < 1e-6:
            return []

        # Find classes with anomalously LOW norm (easy-to-reach target)
        low_mask  = norms < mean_norm - 3.0 * std_norm
        high_mask = norms > mean_norm + 3.0 * std_norm
        n_low  = low_mask.sum().item()
        n_high = high_mask.sum().item()

        if n_low == 0 and n_high == 0:
            return []

        sev = "high" if n_low > 0 else "medium"
        low_idxs  = low_mask.nonzero(as_tuple=True)[0].tolist()
        high_idxs = high_mask.nonzero(as_tuple=True)[0].tolist()

        detail_parts = []
        if n_low:
            detail_parts.append(
                f"{n_low} class(es) with anomalously low inbound norm "
                f"(indices {low_idxs[:8]}{'…' if len(low_idxs) > 8 else ''}): "
                "likely backdoor target classes (Neural Cleanse signature)."
            )
        if n_high:
            detail_parts.append(
                f"{n_high} class(es) with anomalously high inbound norm "
                f"(indices {high_idxs[:8]}{'…' if len(high_idxs) > 8 else ''})."
            )

        return [Finding(
            layer    = name,
            test     = "last_layer_asymmetry",
            severity = sev,
            score    = min(1.0, (n_low * 0.7 + n_high * 0.3) / max(len(norms) * 0.05, 1)),
            detail   = " ".join(detail_parts),
            metadata = {
                "num_classes":      int(len(norms)),
                "low_norm_classes":  low_idxs[:16],
                "high_norm_classes": high_idxs[:16],
                "mean_norm":        round(mean_norm, 6),
                "std_norm":         round(std_norm, 6),
            },
        )]

    # ------------------------------------------------------------------
    # Test 2 — weight norm progression across depth
    # ------------------------------------------------------------------

    def _norm_progression(
        self, named: List[Tuple[str, torch.Tensor]]
    ) -> List[Finding]:
        """
        Compute the Frobenius norm of each layer. In well-trained clean networks
        the progression is typically smooth. An isolated layer with a norm 5× its
        neighbours is suspicious.
        """
        if len(named) < 4:
            return []

        layer_norms = [(n, t.float().norm(p="fro").item()) for n, t in named]
        norms_only  = [v for _, v in layer_norms]

        mean_n = sum(norms_only) / len(norms_only)
        std_n  = math.sqrt(sum((v - mean_n) ** 2 for v in norms_only) / len(norms_only))

        if std_n < 1e-6:
            return []

        outliers = [
            (n, v) for n, v in layer_norms
            if abs(v - mean_n) > 5.0 * std_n
        ]

        if not outliers:
            return []

        sev = "medium"
        return [Finding(
            layer    = ", ".join(n for n, _ in outliers[:4]),
            test     = "norm_progression_anomaly",
            severity = sev,
            score    = min(0.6, 0.15 * len(outliers)),
            detail   = (
                f"{len(outliers)} layer(s) have Frobenius norms more than 5σ from the "
                "model mean. Abrupt norm spikes across layers can indicate a "
                "deliberately over-parameterised backdoor injection point."
            ),
            metadata = {
                "outlier_layers": [{"name": n, "norm": round(v, 4)} for n, v in outliers[:8]],
                "model_mean_norm": round(mean_n, 4),
                "model_std_norm":  round(std_n, 4),
            },
        )]

    # ------------------------------------------------------------------
    # Test 3 — dominant neuron index correlation across consecutive layers
    # ------------------------------------------------------------------

    def _dominant_neuron_pathway(
        self, named: List[Tuple[str, torch.Tensor]]
    ) -> List[Finding]:
        """
        If the same relative neuron indices are outliers in two consecutive
        linear layers, they may form an explicit wire-through for a trigger signal.
        """
        findings: List[Finding] = []
        prev_dominant: set = set()
        prev_name: str = ""

        for name, tensor in named:
            if tensor.dim() < 2:
                continue
            flat   = tensor.float().reshape(tensor.shape[0], -1)
            norms  = flat.norm(dim=1)
            if len(norms) < 8:
                prev_dominant = set()
                continue

            q3  = _quantile(norms, 0.75)
            iqr = q3 - _quantile(norms, 0.25)
            fence = q3 + 3.0 * iqr
            dominant = {int(i) for i in (norms > fence).nonzero(as_tuple=True)[0].tolist()}

            if prev_dominant and dominant:
                # Normalise indices to [0,1] to compare across differently sized layers
                prev_rel = {i / max(1, len(norms)) for i in prev_dominant}
                curr_rel = {i / max(1, len(norms)) for i in dominant}
                overlap  = _relative_overlap(prev_rel, curr_rel, tol=0.15)
                if overlap > 0.5:
                    findings.append(Finding(
                        layer    = f"{prev_name} → {name}",
                        test     = "dominant_neuron_pathway",
                        severity = "medium",
                        score    = min(0.7, overlap),
                        detail   = (
                            f"Dominant-neuron index overlap of {overlap:.0%} between "
                            f"'{prev_name}' and '{name}'. "
                            "Correlated outlier neurons across consecutive layers suggest "
                            "a persistent high-gain pathway — a structural indicator of "
                            "a trojan shortcut."
                        ),
                        metadata = {
                            "overlap":            round(overlap, 4),
                            "prev_dominant_count": len(prev_dominant),
                            "curr_dominant_count": len(dominant),
                        },
                    ))

            prev_dominant = dominant
            prev_name     = name

        return findings


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _quantile(v: torch.Tensor, q: float) -> float:
    k = max(0, min(len(v) - 1, int(q * len(v))))
    return v.sort().values[k].item()


def _relative_overlap(a: set, b: set, tol: float) -> float:
    """Fraction of elements in *a* that have a close match in *b*."""
    if not a or not b:
        return 0.0
    matched = sum(1 for x in a if any(abs(x - y) < tol for y in b))
    return matched / len(a)
