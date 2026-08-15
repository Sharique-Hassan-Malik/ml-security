"""Joins pickle-scanner to the suite.

The only thing this file does is reconcile shapes: `scan_file` answers per
pickle payload, because a checkpoint holds several and they are not equally
trustworthy, while the platform wants one result per module per file. So the
payloads are folded into one result with the payload kept in each finding's
location — nothing is averaged away.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aisec.core.finding import ModuleResult  # noqa: E402
from aisec.core.module import Scanner  # noqa: E402
from aisec.core.registry import spec  # noqa: E402
from pickle_scanner import scan_file  # noqa: E402


class PickleScannerModule(Scanner):
    def scan(self, target: Path, **options: Any) -> ModuleResult:
        payloads = scan_file(target, strict=bool(options.get("strict", False)))
        merged = self.result(str(target))

        errors = [p.error for p in payloads if p.error]
        skips = [p.skipped for p in payloads if p.skipped]
        opcodes = sum(int(p.metrics.get("opcodes", 0)) for p in payloads)
        protocols = {p.metrics.get("protocol") for p in payloads if p.metrics.get("protocol")}

        for payload in payloads:
            for finding in payload.findings:
                # Keep which stream it came from — "offset 0x1c of data.pkl"
                # is actionable, a bare offset across a zip is not.
                if payload.target and payload.target != str(target):
                    stream = Path(payload.target).name
                    finding.location = f"{stream} {finding.location}".strip()
                merged.add(finding)

        merged.metrics["payloads"] = len(payloads)
        merged.metrics["opcodes"] = opcodes
        if protocols:
            merged.metrics["protocol"] = ", ".join(str(p) for p in sorted(protocols))
        if errors and not merged.findings:
            merged.error = "; ".join(dict.fromkeys(errors))
        elif skips and not merged.findings and not errors:
            merged.skipped = "; ".join(dict.fromkeys(skips))
        return merged


MODULE = PickleScannerModule(spec("pickle-scanner"))
