"""Bridge from tensor-level analysis to the suite's finding vocabulary.

The analysers speak in layers and tests — "fc1.weight failed the spectral
outlier test with score 0.82". The suite speaks in findings. This is the one
place that translation happens, so the four analysers stay unaware of it.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
from typing import Any, Iterable

_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aisec.core.finding import Finding, Kind, ModuleResult, Severity  # noqa: E402

__all__ = ["Finding", "Kind", "ModuleResult", "Severity",
           "poison_finding", "poisoning_score"]


def poison_finding(
    layer: str,
    test: str,
    severity: str | Severity,
    score: float,
    detail: str,
    metadata: dict[str, Any] | None = None,
) -> Finding:
    """One analyser observation, in the shared shape.

    The layer name is the location: it is what an operator greps for, and it
    is the coordinate that makes two analysers' findings comparable.
    """
    return Finding(
        title=test,
        severity=Severity.parse(severity),
        summary=detail,
        location=layer,
        score=score,
        metadata=metadata or {},
    )


def poisoning_score(findings: Iterable[Finding], layer_count: int) -> float:
    """Aggregate 0–1 risk for the checkpoint as a whole.

    Deliberately not a mean. Backdoors are local — one poisoned layer in a
    hundred clean ones is still a backdoored model — so a single HIGH finding
    puts the score above 0.5 regardless of how much clean weight surrounds it,
    and additional findings only move it within that band.
    """
    findings = list(findings)
    if not findings:
        return 0.0

    layers = max(1, layer_count)
    high = [f for f in findings if f.severity >= Severity.HIGH]
    medium = [f for f in findings if f.severity is Severity.MEDIUM]

    if high:
        return min(1.0, 0.50 + 0.50 * len(high) / layers)
    if medium:
        return min(0.49, 0.20 + 0.29 * len(medium) / layers)
    return min(0.19, 0.04 * len(findings))
