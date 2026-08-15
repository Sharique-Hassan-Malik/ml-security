"""Joins weight-poisoning to the suite."""

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
from weight_poisoning import scan  # noqa: E402


class WeightPoisoningModule(Scanner):
    def scan(self, target: Path, **options: Any) -> ModuleResult:
        # `scan` already returns the shared result type, so there is nothing to
        # translate — the module was built against the same contract.
        return scan(str(target))


MODULE = WeightPoisoningModule(spec("weight-poisoning"))
