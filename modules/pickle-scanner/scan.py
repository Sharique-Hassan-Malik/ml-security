#!/usr/bin/env python3
"""Pickle scanner — standalone CLI.

Usable on its own, with no part of the rest of the suite installed:

    python scan.py model.pt
    python scan.py checkpoints/ --recursive --min-severity HIGH
    python scan.py "**/*.pkl" --json

Rendering comes from `aisec.core.render`, which is stdlib-only, so sharing the
report format with the other five modules costs this tool no dependencies.
For every module at once, use the platform CLI: `aisec scan checkpoints/`.
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
from pickle_scanner import scan_file  # noqa: E402

_EXTENSIONS = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".bin", ".ckpt"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scan.py",
        description="Static analysis of pickle bytecode — nothing is executed.",
    )
    parser.add_argument("targets", nargs="+", help="files, directories or globs")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="recurse into subdirectories")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show INFO findings too")
    parser.add_argument("--strict", action="store_true",
                        help="raise severity for private C-extension modules")
    parser.add_argument("--min-severity", default="LOW",
                        choices=[s.value for s in Severity],
                        help="hide findings below this severity (default: LOW)")
    parser.add_argument("--json", metavar="FILE", nargs="?", const="-",
                        help="write JSON (default: stdout)")
    parser.add_argument("--html", metavar="FILE", help="write an HTML report")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--exit-zero", action="store_true",
                        help="always exit 0, whatever is found")
    return parser.parse_args(argv)


def collect(targets: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            walk = path.rglob if recursive else path.glob
            for suffix in sorted(_EXTENSIONS):
                found.extend(walk(f"*{suffix}"))
        else:
            root = Path(".")
            matches = root.rglob(target) if "**" in target else root.glob(target)
            found.extend(m for m in matches if m.is_file())

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = collect(args.targets, args.recursive)
    if not files:
        print("No files matched.", file=sys.stderr)
        return 2

    report = Report(target=str(files[0]) if len(files) == 1 else f"{len(files)} files")
    for path in files:
        for result in scan_file(path, strict=args.strict):
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
                render_html(report, title="Pickle scan"), encoding="utf-8")
            print(f"  HTML report → {args.html}")

    return 0 if args.exit_zero else report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
