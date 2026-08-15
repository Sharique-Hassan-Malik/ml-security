"""Joins the prompt-injection firewall to the suite.

The firewall's own answer is a `Decision` — allow, flag, sanitise or block —
which is the right shape for a request path but not for a report. Here each
firing detector signal becomes a finding, and the decision itself is carried in
`metrics` so the report can say *what the firewall would have done* rather than
only what it noticed.

The decision's severity is not the max of the signal severities: a BLOCK is a
BLOCK even when it was one strong signal rather than five weak ones, and a
sanitised document that had zero-width characters stripped is worth a LOW even
though no detector scored it.
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
from aisec.core.module import Guard  # noqa: E402
from aisec.core.registry import spec  # noqa: E402
from pifw import Action, Firewall  # noqa: E402

_ACTION_SEVERITY = {
    Action.BLOCK: Severity.HIGH,
    Action.SANITISE: Severity.LOW,
    Action.FLAG: Severity.MEDIUM,
    Action.ALLOW: Severity.SAFE,
}


def _signal_severity(score: float) -> Severity:
    if score >= 0.7:
        return Severity.HIGH
    if score >= 0.35:
        return Severity.MEDIUM
    return Severity.LOW


class PromptInjectionModule(Guard):
    def __init__(self, module_spec) -> None:
        super().__init__(module_spec)
        self.firewall = Firewall()

    def inspect(self, payload: Any, **options: Any) -> ModuleResult:
        text = payload if isinstance(payload, str) else str(payload)
        source = str(options.get("source", "user"))
        result = self.result(str(options.get("label", "<input>")))

        if source == "user":
            decision = self.firewall.check(text, source=source)
        else:
            # Retrieved content goes through the full path — sanitise, score,
            # fence — because that is what production would do with it.
            decision = self.firewall.process_document(text, label=source)

        for signal in decision.signals:
            if signal.score <= 0:
                continue
            result.add(
                Finding(
                    title=signal.name,
                    severity=_signal_severity(signal.score),
                    summary=signal.reason,
                    score=round(signal.score, 4),
                    location=_span(signal, text),
                    metadata={"spans": signal.spans},
                )
            )

        for note in decision.notes:
            result.add(
                Finding(
                    title="sanitised",
                    severity=Severity.LOW,
                    summary=note,
                    detail="Content with no legitimate use was removed before the "
                           "text reached the model.",
                )
            )

        action_severity = _ACTION_SEVERITY[decision.action]
        if action_severity > result.max_severity:
            result.add(
                Finding(
                    title=f"firewall_{decision.action.value}",
                    severity=action_severity,
                    summary=decision.explain(),
                    score=round(decision.score, 4),
                )
            )

        result.metrics["action"] = decision.action.value
        result.metrics["score"] = round(decision.score, 4)
        result.metrics["control"] = decision.control
        result.metrics["source"] = source
        return result


MODULE = PromptInjectionModule(spec("prompt-injection-firewall"))


def _span(signal, text: str) -> str:
    """Point at the offending substring — a score with no span is untunable."""
    if not signal.spans:
        return ""
    start, end = signal.spans[0]
    excerpt = text[start:end].replace("\n", " ")
    if len(excerpt) > 38:
        excerpt = excerpt[:35] + "…"
    return f"{start}:{end} {excerpt!r}"
