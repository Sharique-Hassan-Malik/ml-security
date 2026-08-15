#!/usr/bin/env python3
"""Precision, recall, and the base-rate correction that decides deployability.

    python3 bench.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pifw import Firewall, counts, evaluate, operating_point, sweep  # noqa: E402
from pifw.corpus import ATTACKS, BENIGN  # noqa: E402


def table(rows, headers) -> str:
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-per-day", type=int, default=1_000_000)
    args = parser.parse_args()

    stats = counts()
    print(f"Corpus: {stats['attacks']} attacks ({stats['indirect']} indirect), "
          f"{stats['benign']} benign, of which {stats['hard_negatives']} are hard negatives\n")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    evaluations = sweep(thresholds)
    print(table(
        [[f"{e.threshold:.1f}", f"{e.overall.precision:.2f}", f"{e.overall.recall:.2f}",
          f"{e.overall.f1:.2f}", f"{e.overall.false_positive_rate:.2f}",
          str(len(e.misses)), str(len(e.false_alarms))]
         for e in evaluations],
        ["threshold", "precision", "recall", "F1", "FPR", "missed", "false alarms"],
    ))

    best = max(evaluations, key=lambda e: e.overall.f1)
    print(f"\nBest F1 at threshold {best.threshold:.1f}: "
          f"precision {best.overall.precision:.2f}, recall {best.overall.recall:.2f}\n")

    print("\nRecall by attack technique (threshold %.1f):\n" % best.threshold)
    firewall = Firewall(block_threshold=best.threshold, flag_threshold=best.threshold * 0.5)
    detailed = evaluate(firewall)
    attack_techniques = sorted({s.technique for s in ATTACKS})
    print(table(
        [[technique,
          f"{detailed.by_technique[technique].recall:.2f}",
          f"{detailed.by_technique[technique].true_positive}/"
          f"{detailed.by_technique[technique].true_positive + detailed.by_technique[technique].false_negative}"]
         for technique in attack_techniques if technique in detailed.by_technique],
        ["technique", "recall", "caught"],
    ))

    print("\nFalse-positive rate by benign category:\n")
    benign_categories = sorted({s.technique for s in BENIGN})
    print(table(
        [[category,
          f"{detailed.by_technique[category].false_positive_rate:.2f}",
          f"{detailed.by_technique[category].false_positive}/"
          f"{detailed.by_technique[category].false_positive + detailed.by_technique[category].true_negative}"]
         for category in benign_categories if category in detailed.by_technique],
        ["benign category", "FPR", "flagged"],
    ))

    print("""
The benign categories are the point. `ordinary` traffic is easy; text that
*discusses* prompt injection contains every phrase the detector looks for, and
separating it from an actual attack is not a string-matching problem. Any
evaluation without hard negatives measures reading comprehension, not
discrimination.""")

    print("\n\nWhat the corpus numbers become at real traffic:\n")
    rows = []
    for base_rate in (0.5, 0.05, 0.005, 0.0005):
        point = operating_point(detailed, base_rate, args.requests_per_day)
        rows.append([
            f"{base_rate:.2%}",
            f"{point.true_positives_per_day:,.0f}",
            f"{point.false_positives_per_day:,.0f}",
            f"{point.precision_in_production:.1%}",
            f"{point.alerts_per_true_attack:,.1f}",
        ])
    print(table(rows, ["attack base rate", "true alerts/day", "false alerts/day",
                       "precision", "alerts per real attack"]))

    balanced = operating_point(detailed, 0.5, args.requests_per_day)
    realistic = operating_point(detailed, 0.005, args.requests_per_day)
    print(f"""
Same detector, same threshold, {args.requests_per_day:,} requests/day. At the corpus's
roughly balanced prevalence it looks like a {balanced.precision_in_production:.0%}-precision detector. At a
realistic 0.5% attack rate it produces {realistic.alerts_per_true_attack:,.0f} alerts for every real attack,
and precision collapses to {realistic.precision_in_production:.1%}.

Nothing about the detector changed. There are simply {(1 - 0.005) / 0.005:.0f}x more benign
requests to be wrong about, and a fixed false-positive rate applied to a much
larger population swamps the true positives. Any prompt-injection detector
quoted at "95% accuracy" without a base rate is quoting a number that does not
survive contact with production.

Which is why detection here is the third control, not the first. Channel
separation and tool allowlisting hold at any base rate, because they do not
depend on recognising the attack at all.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
