#!/usr/bin/env python3
"""Weight poisoning detector — standalone CLI.

    python scan.py model.pt
    python scan.py model.pt --html report.html --verbose

Exit codes: 0 clean, 1 findings at MEDIUM or above, 2 the file could not be read.
For every module in the suite at once: `aisec scan model.pt`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aisec.core.finding import Report, Severity  # noqa: E402
from aisec.core.render import render_html, render_terminal  # noqa: E402
from weight_poisoning import scan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scan.py",
        description="Scan a PyTorch checkpoint for backdoor and weight-poisoning indicators.",
    )
    parser.add_argument("model", help="path to a .pt or .pth checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show every finding, including LOW and INFO")
    parser.add_argument("--min-severity", default="LOW",
                        choices=[s.value for s in Severity])
    parser.add_argument("--json", metavar="FILE", nargs="?", const="-")
    parser.add_argument("--html", metavar="FILE")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = scan(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"scan.py: {exc}", file=sys.stderr)
        return 2

    report = Report(target=args.model)
    report.add(result)

    if args.json == "-":
        print(report.to_json())
    else:
        shown = report if args.verbose else report.filtered(Severity[args.min_severity])
        render_terminal(shown, colour=False if args.no_colour else None,
                        verbose=args.verbose)
        if args.json:
            Path(args.json).write_text(report.to_json(), encoding="utf-8")
            print(f"  JSON report → {args.json}")
        if args.html:
            Path(args.html).write_text(
                render_html(report, title="Weight poisoning scan"), encoding="utf-8")
            print(f"  HTML report → {args.html}")

    return 0 if args.exit_zero else report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
