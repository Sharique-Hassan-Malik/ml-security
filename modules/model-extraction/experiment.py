#!/usr/bin/env python3
"""Model extraction attack — standalone CLI.

    python experiment.py
    python experiment.py --strategy jacobian --rounds 10 --queries-per-round 200
    python experiment.py --hard-label --html report.html

Runs the attack against a locally constructed victim and reports what the
substitute achieved for the queries it spent. Through the platform, the same
probe is `aisec probe model-extraction`.
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
from integration import MODULE  # noqa: E402
from modelext import STRATEGIES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiment.py",
        description="Measure how many oracle queries buy how much model fidelity.",
    )
    parser.add_argument("--strategy", default="all",
                        choices=[*STRATEGIES, "all"],
                        help="query strategy (default: all three, compared)")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--queries-per-round", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hard-label", action="store_true",
                        help="oracle returns argmax only — the realistic, harder setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", metavar="FILE", nargs="?", const="-")
    parser.add_argument("--html", metavar="FILE")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args(argv)

    print(f"[*] Extracting with {args.strategy} "
          f"({args.rounds} rounds × {args.queries_per_round} queries)…",
          file=sys.stderr)

    result = MODULE.run(
        strategy=args.strategy,
        rounds=args.rounds,
        queries_per_round=args.queries_per_round,
        epochs=args.epochs,
        hard_label=args.hard_label,
        seed=args.seed,
    )
    report = Report(target="victim model")
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
                render_html(report, title="Model extraction probe"), encoding="utf-8")
            print(f"  HTML report → {args.html}")

    return 0 if args.exit_zero else report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
