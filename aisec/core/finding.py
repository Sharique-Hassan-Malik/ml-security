"""The one vocabulary every module in this suite reports in.

Six tools live here. Before they shared this file they had three different
severity scales, three HTML report generators with three copies of the same
dark theme, and no way to say "the pickle scanner and the weight analyser both
flagged this checkpoint" in a single sentence. A finding is a finding whether
it came from static bytecode analysis, a runtime guard or a simulated attack,
so there is one `Finding` type and one `Severity` ladder.

Deliberately stdlib-only. A module used on its own — `cd modules/pickle-scanner
&& python scan.py model.pt` — imports this and nothing else, so depending on
the shared schema never drags in torch.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator


class Severity(Enum):
    """How bad, on one ladder shared by every module.

    Ordered and comparable, so `finding.severity >= Severity.HIGH` means the
    same thing in a bytecode scanner and in a gradient-inversion probe.
    """

    SAFE = "SAFE"
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER.index(self)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank >= other.rank

    @classmethod
    def parse(cls, value: "str | Severity") -> "Severity":
        """Accept 'high', 'HIGH' or Severity.HIGH — modules spell it all three ways."""
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown severity {value!r}") from exc


_SEVERITY_ORDER = [
    Severity.SAFE,
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
]


class Kind(str, Enum):
    """What a module does, which decides how it is invoked.

    The three are genuinely different shapes, not three names for one thing:
    a scanner is handed an artifact and answers offline; a guard sits in a live
    request path and must return a decision; a probe attacks a model you own to
    find out what an attacker would get.
    """

    SCANNER = "scanner"
    GUARD = "guard"
    PROBE = "probe"


class Verdict(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    """One thing worth telling the operator about.

    `title` is the short handle (`REDUCE`, `spectral_outlier`, `role_confusion`)
    and is what gets grouped and counted. `location` is module-native and free
    text on purpose — "offset 0x0142", "layer fc1.weight", "token 37" are not
    the same coordinate space and pretending otherwise would lose information.
    """

    title: str
    severity: Severity
    summary: str = ""
    detail: str = ""
    location: str = ""
    score: float | None = None
    module: str = ""
    target: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.severity = Severity.parse(self.severity)

    def __str__(self) -> str:
        where = f" {self.location}" if self.location else ""
        what = self.summary or self.detail
        return f"[{self.severity.value:<8}]{where}  {self.title}" + (f" — {what}" if what else "")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "module": self.module,
            "title": self.title,
            "severity": self.severity.value,
        }
        if self.target:
            out["target"] = self.target
        if self.location:
            out["location"] = self.location
        if self.summary:
            out["summary"] = self.summary
        if self.detail:
            out["detail"] = self.detail
        if self.score is not None:
            out["score"] = round(float(self.score), 4)
        if self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class ModuleResult:
    """What one module produced for one target.

    `metrics` is the escape hatch that keeps the shared schema from flattening
    everything into findings. An extraction probe's headline number is
    "0.94 agreement after 3,000 queries" and a leakage probe's is "PSNR 31.2" —
    neither is a finding, both belong in the report.
    """

    module: str
    kind: Kind
    target: str = ""
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed: float = 0.0
    skipped: str = ""

    def add(self, finding: Finding) -> Finding:
        """Stamp provenance as findings arrive so modules never have to."""
        finding.module = finding.module or self.module
        finding.target = finding.target or self.target
        self.findings.append(finding)
        return finding

    def extend(self, findings: Iterable[Finding]) -> None:
        for finding in findings:
            self.add(finding)

    @property
    def max_severity(self) -> Severity:
        if not self.findings:
            return Severity.SAFE
        return max(f.severity for f in self.findings)

    @property
    def ok(self) -> bool:
        return not self.error and not self.skipped

    def counts(self) -> dict[Severity, int]:
        counts = {sev: 0 for sev in Severity}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "module": self.module,
            "kind": self.kind.value,
            "target": self.target,
            "max_severity": self.max_severity.value,
            "findings": [f.to_dict() for f in self.findings],
        }
        if self.metrics:
            out["metrics"] = self.metrics
        if self.error:
            out["error"] = self.error
        if self.skipped:
            out["skipped"] = self.skipped
        if self.elapsed:
            out["elapsed_s"] = round(self.elapsed, 3)
        return out


@dataclass
class Report:
    """Every module's result for one run, and the single verdict over them."""

    target: str = ""
    results: list[ModuleResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    def add(self, result: ModuleResult) -> ModuleResult:
        self.results.append(result)
        return result

    def __iter__(self) -> Iterator[ModuleResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.results for f in r.findings]

    @property
    def max_severity(self) -> Severity:
        if not self.results:
            return Severity.SAFE
        return max(r.max_severity for r in self.results)

    @property
    def errors(self) -> list[ModuleResult]:
        return [r for r in self.results if r.error]

    @property
    def verdict(self) -> Verdict:
        """Worst severity decides. No averaging.

        Averaging is how one CRITICAL finding gets diluted by nine clean
        checks into something that looks survivable.
        """
        worst = self.max_severity
        if worst is Severity.CRITICAL:
            return Verdict.CRITICAL
        if worst is Severity.HIGH:
            return Verdict.HIGH_RISK
        if worst is Severity.MEDIUM:
            return Verdict.SUSPICIOUS
        return Verdict.CLEAN

    @property
    def exit_code(self) -> int:
        """0 clean, 1 something found, 2 a module could not run."""
        if self.errors:
            return 2
        return 0 if self.verdict is Verdict.CLEAN else 1

    def counts(self) -> dict[Severity, int]:
        counts = {sev: 0 for sev in Severity}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def filtered(self, minimum: Severity) -> "Report":
        """A copy keeping only findings at or above *minimum*."""
        clone = Report(target=self.target, timestamp=self.timestamp)
        for result in self.results:
            kept = [f for f in result.findings if f.severity >= minimum]
            clone.add(
                ModuleResult(
                    module=result.module,
                    kind=result.kind,
                    target=result.target,
                    findings=kept,
                    metrics=result.metrics,
                    error=result.error,
                    elapsed=result.elapsed,
                    skipped=result.skipped,
                )
            )
        return clone

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "timestamp": self.timestamp,
            "verdict": self.verdict.value,
            "max_severity": self.max_severity.value,
            "finding_count": len(self.findings),
            "counts": {
                sev.value: n for sev, n in self.counts().items() if n
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
