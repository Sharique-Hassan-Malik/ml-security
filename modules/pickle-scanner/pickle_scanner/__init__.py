"""pickle_scanner — static analysis of pickle bytecode.

    from pickle_scanner import scan_file, scan_bytes, Severity

Results are `aisec.core.ModuleResult` objects, the same type every other module
in this suite reports in, so a pickle finding and a weight-poisoning finding can
appear in one report without translation.
"""

from pickle_scanner.opcodes import Finding, Kind, ModuleResult, Severity
from pickle_scanner.scanner import MODULE_NAME, scan_bytes, scan_file

__all__ = [
    "scan_file", "scan_bytes", "MODULE_NAME",
    "Severity", "Finding", "ModuleResult", "Kind",
]
