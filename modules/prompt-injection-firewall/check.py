#!/usr/bin/env python3
"""Prompt-injection firewall — standalone CLI.

    python check.py "ignore all previous instructions"
    python check.py --file retrieved.txt --source document
    cat page.html | python check.py --source document --json

`--source document` runs the full path a retrieved page would take — sanitise,
score, fence — rather than the lighter check a user's own message gets.
For every module in the suite at once: `aisec guard "<text>"`.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check.py",
        description="Score text for prompt injection and show what the firewall would do.",
    )
    parser.add_argument("text", nargs="?", help="text to inspect (default: stdin)")
    parser.add_argument("--file", metavar="FILE", help="read the input from a file")
    parser.add_argument("--source", default="user",
                        help="user | document | tool (default: user)")
    parser.add_argument("--fenced", action="store_true",
                        help="print the fenced prompt the model would actually see")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--min-severity", default="LOW",
                        choices=[s.value for s in Severity])
    parser.add_argument("--json", metavar="FILE", nargs="?", const="-")
    parser.add_argument("--html", metavar="FILE")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args(argv)

    if args.file:
        text, label = Path(args.file).read_text(encoding="utf-8", errors="replace"), args.file
    elif args.text:
        text, label = args.text, "<argv>"
    else:
        text, label = sys.stdin.read(), "<stdin>"

    result = MODULE.inspect(text, source=args.source, label=label)
    report = Report(target=label)
    report.add(result)

    if args.json == "-":
        print(report.to_json())
    else:
        shown = report if args.verbose else report.filtered(Severity[args.min_severity])
        render_terminal(shown, colour=False if args.no_colour else None,
                        verbose=args.verbose)
        if args.fenced and args.source != "user":
            decision = MODULE.firewall.process_document(text, label=args.source)
            print("\n  ── prompt as the model would receive it ──")
            print(decision.text or "  (blocked — nothing is forwarded)")
        if args.json:
            Path(args.json).write_text(report.to_json(), encoding="utf-8")
            print(f"  JSON report → {args.json}")
        if args.html:
            Path(args.html).write_text(
                render_html(report, title="Prompt injection check"), encoding="utf-8")
            print(f"  HTML report → {args.html}")

    return 0 if args.exit_zero else report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
