"""Joins model-extraction to the suite as a probe.

A probe reports exposure, so the findings here are about *what an attacker
would have got*, not about a defect in a file. The severity ladder is keyed to
final agreement, because a substitute that matches the victim on 95% of inputs
is a copy of the model regardless of how it was obtained.

The curve — agreement against queries spent — is handed to the platform as a
declarative chart spec rather than drawn here, so every probe's chart comes out
of the same renderer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aisec.core.finding import Finding, ModuleResult, Severity  # noqa: E402
from aisec.core.module import Probe  # noqa: E402
from aisec.core.registry import spec  # noqa: E402
from modelext import STRATEGIES, run_extraction  # noqa: E402


def _severity(agreement: float) -> Severity:
    if agreement >= 0.90:
        return Severity.HIGH
    if agreement >= 0.75:
        return Severity.MEDIUM
    if agreement >= 0.50:
        return Severity.LOW
    return Severity.INFO


class ModelExtractionModule(Probe):
    def run(self, **options: Any) -> ModuleResult:
        result = self.result(str(options.get("label", "victim model")))

        requested = options.get("strategy", "all")
        names = list(STRATEGIES) if requested == "all" else [requested]

        rounds = int(options.get("rounds", 6))
        per_round = int(options.get("queries_per_round", 100))

        series: dict[str, list[tuple[float, float]]] = {}
        best = 0.0

        for name in names:
            outcome = run_extraction(
                strategy=name,
                victim=options.get("victim"),
                rounds=rounds,
                queries_per_round=per_round,
                epochs=int(options.get("epochs", 3)),
                hard_label=bool(options.get("hard_label", False)),
                seed=int(options.get("seed", 42)),
            )
            series[name] = [(r.queries_used, r.agreement) for r in outcome.rounds]
            best = max(best, outcome.final_agreement)

            to_90 = next(
                (r.queries_used for r in outcome.rounds if r.agreement >= 0.90), None
            )
            summary = (
                f"{outcome.final_agreement:.1%} agreement after "
                f"{outcome.total_queries:,} queries"
            )
            if to_90 is not None:
                summary += f"; crossed 90% at {to_90:,}"

            result.add(
                Finding(
                    title=f"extraction_{name}",
                    severity=_severity(outcome.final_agreement),
                    summary=summary,
                    detail=(
                        "Agreement is the fraction of held-out inputs on which the "
                        "substitute and the victim return the same class. A rate "
                        "limit below the crossing point is what makes this attack "
                        "uneconomic."
                    ),
                    location=f"strategy {name}",
                    score=round(outcome.final_agreement, 4),
                    metadata={
                        "total_queries": outcome.total_queries,
                        "queries_to_90pct": to_90,
                        "hard_label": bool(options.get("hard_label", False)),
                    },
                )
            )

        result.metrics["strategies"] = ", ".join(names)
        result.metrics["best_agreement"] = round(best, 4)
        result.metrics["query_budget"] = rounds * per_round
        result.metrics["charts"] = [
            {
                "title": "Substitute agreement against queries spent",
                "series": series,
                "x_label": "Oracle queries",
                "y_label": "Agreement",
                "y_min": 0.0,
                "y_max": 1.0,
            }
        ]
        return result


MODULE = ModelExtractionModule(spec("model-extraction"))
