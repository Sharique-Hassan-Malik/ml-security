#!/usr/bin/env python3
"""
Weight Poisoning Detector — command-line interface.

Usage
-----
    python scanner.py model.pt
    python scanner.py model.pt --report results.json --html results.html --verbose

Exit codes
----------
    0  CLEAN
    1  SUSPICIOUS or HIGH_RISK
    2  Error (file not found, invalid format, etc.)
"""

import argparse
import sys
from pathlib import Path

from detector import scan


_VERDICT_COLOR = {
    "CLEAN":     "\033[92m",   # green
    "SUSPICIOUS":"\033[93m",   # yellow
    "HIGH_RISK": "\033[91m",   # red
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _fmt_verdict(verdict: str) -> str:
    color = _VERDICT_COLOR.get(verdict, "")
    return f"{color}{_BOLD}{verdict}{_RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scanner",
        description="Scan a PyTorch .pt/.pth model file for backdoor / weight-poisoning indicators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model",    help="Path to .pt or .pth model file")
    parser.add_argument("--report", metavar="FILE", help="Write JSON report to FILE")
    parser.add_argument("--html",   metavar="FILE", help="Write HTML report to FILE")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print all findings, including low-severity ones")
    args = parser.parse_args()

    print(f"[*] Scanning: {args.model}")

    try:
        report = scan(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        sys.exit(2)

    # ---- summary table -----------------------------------------------
    print(f"[*] Layers     : {report.layer_count}")
    print(f"[*] Parameters : {report.param_count:,}")
    print(f"[*] Score      : {report.overall_score:.4f}")
    print(f"[*] Verdict    : {_fmt_verdict(report.verdict)}")
    print(f"[*] Findings   : {len(report.findings)}")

    # ---- findings display --------------------------------------------
    if report.findings:
        show = (
            report.findings
            if args.verbose
            else [f for f in report.findings if f.severity in ("high", "medium")]
        )
        if show:
            header = f"  {'Layer':<35} {'Test':<30} {'Sev':<8} Score"
            print(f"\n{header}")
            print("  " + "─" * (len(header) - 2))
            for f in show:
                layer = f.layer[:34].ljust(34)
                test  = f.test[:29].ljust(29)
                color = {
                    "high":   "\033[91m",
                    "medium": "\033[93m",
                    "low":    "\033[94m",
                }.get(f.severity, "")
                sev = f"{color}{f.severity:<8}{_RESET}"
                print(f"  {layer} {test} {sev} {f.score:.4f}")
                if args.verbose:
                    # Wrap detail text at 90 chars
                    words = f.detail.split()
                    line, lines = [], []
                    for w in words:
                        if sum(len(x) + 1 for x in line) + len(w) > 88:
                            lines.append(" ".join(line))
                            line = []
                        line.append(w)
                    if line:
                        lines.append(" ".join(line))
                    for l in lines:
                        print(f"    {l}")

    # ---- file outputs ------------------------------------------------
    if args.report:
        Path(args.report).write_text(report.to_json(), encoding="utf-8")
        print(f"\n[*] JSON report → {args.report}")

    if args.html:
        Path(args.html).write_text(report.to_html(), encoding="utf-8")
        print(f"[*] HTML report → {args.html}")

    sys.exit(0 if report.verdict == "CLEAN" else 1)


if __name__ == "__main__":
    main()
