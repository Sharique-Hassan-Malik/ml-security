"""aisec — attack and defence tooling for machine-learning systems.

Six modules under `modules/`, each usable on its own, reporting into one
schema so a single run can say what every one of them found.
"""

from .core.finding import Finding, Kind, ModuleResult, Report, Severity, Verdict
from .core.module import Guard, Module, ModuleSpec, Probe, Scanner

__version__ = "1.0.0"
__all__ = [
    "Finding", "Kind", "ModuleResult", "Report", "Severity", "Verdict",
    "Guard", "Module", "ModuleSpec", "Probe", "Scanner",
]
