"""The shared contract: findings, module base classes, registry, rendering.

Stdlib-only on purpose — a module imported on its own gets the schema without
the platform's dependency set coming with it.
"""

from .finding import Finding, Kind, ModuleResult, Report, Severity, Verdict
from .module import Guard, Module, ModuleSpec, Options, Probe, Scanner

__all__ = [
    "Finding", "Kind", "ModuleResult", "Report", "Severity", "Verdict",
    "Guard", "Module", "ModuleSpec", "Options", "Probe", "Scanner",
]
