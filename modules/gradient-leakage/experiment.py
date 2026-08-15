#!/usr/bin/env python3
"""Gradient leakage — standalone CLI.

    python experiment.py
    python experiment.py --iterations 400 --algorithm idlg --html report.html
    python experiment.py --defences none dp

Simulates a federated client, hands its gradients to an honest-but-curious
server, and reports what the server got back — with and without each defence.
Through the platform, the same probe is `aisec probe gradient-leakage`.
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
from gradleak import DEFENCES  # noqa: E402
from integration import MODULE  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiment.py",
        description="Reconstruct training data from shared gradients, and test the defences.",
    )
    parser.add_argument("--iterations", type=int, default=300,
                        help="inversion steps (default: 300)")
    parser.add_argument("--algorithm", default="idlg", choices=["dlg", "idlg"])
    parser.add_argument("--defences", nargs="+", default=list(DEFENCES),
                        choices=list(DEFENCES),
                        help="which settings to attack (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", metavar="FILE", nargs="?", const="-")
    parser.add_argument("--html", metavar="FILE",
                        help="HTML report — includes the reconstructed images")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args(argv)

    print(f"[*] {args.algorithm.upper()} inversion, {args.iterations} iterations, "
          f"defences: {', '.join(args.defences)}…", file=sys.stderr)

    result = MODULE.run(
        iterations=args.iterations,
        algorithm=args.algorithm,
        defences=tuple(args.defences),
        seed=args.seed,
    )
    report = Report(target="federated client")
    report.add(result)

    if args.json == "-":
        print(report.to_json())
    else:
        render_terminal(report if args.verbose else report.filtered(Severity.INFO),
                        colour=False if args.no_colour else None, verbose=args.verbose)
        if args.json:
            Path(args.json).write_text(report.to_json(), encoding="utf-8")
            print(f"  JSON report → {args.json}")
        if args.html:
            Path(args.html).write_text(
                render_html(report, title="Gradient leakage probe"), encoding="utf-8")
            print(f"  HTML report → {args.html}")

    return 0 if args.exit_zero else report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
