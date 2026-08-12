#!/usr/bin/env python3
"""
Model Extraction Attack Simulator — experiment CLI.

Runs three query strategies (Random, Jacobian and Adaptive) against the
same black-box oracle and compares fidelity vs. query budget tradeoffs.
Produces a JSON summary and an HTML report.

Usage
-----
    python experiment.py
    python experiment.py --rounds 15 --queries-per-round 200 --hard-label
    python experiment.py --strategy random --output results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from extraction.attack_loop import ExtractionAttack, ExtractionConfig
from extraction.trainer import SubstituteTrainer
from metrics.report import generate_html_report
from models.architectures import SubstituteCNN, VictimCNN
from oracle.black_box import BlackBoxOracle
from strategies.query_strategies import AdaptiveStrategy, JacobianStrategy, RandomStrategy

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_CYAN  = "\033[96m"

_STRATEGY_CHOICES = ["random", "jacobian", "adaptive", "all"]


def _build(args, strategy_name: str):
    torch.manual_seed(args.seed)

    victim = VictimCNN(num_classes=10)
    victim.eval()

    oracle = BlackBoxOracle(
        model       = victim,
        hard_label  = args.hard_label,
        query_limit = (args.rounds * args.queries_per_round) + args.eval_size + 64,
    )

    substitute = SubstituteCNN(num_classes=10)

    trainer = SubstituteTrainer(
        substitute  = substitute,
        lr          = 1e-3,
        epochs      = args.epochs,
        batch_size  = 64,
        soft_labels = not args.hard_label,
    )

    shape = (3, 32, 32)
    if strategy_name == "random":
        strategy = RandomStrategy(shape, batch_size=args.queries_per_round)
    elif strategy_name == "jacobian":
        strategy = JacobianStrategy(shape, batch_size=args.queries_per_round, step_size=0.1)
    else:
        strategy = AdaptiveStrategy(shape, batch_size=args.queries_per_round)

    cfg = ExtractionConfig(
        n_rounds          = args.rounds,
        queries_per_round = args.queries_per_round,
        eval_size         = args.eval_size,
        input_shape       = shape,
        seed              = args.seed,
    )

    return oracle, substitute, strategy, trainer, cfg


def _run_strategy(args, name: str) -> dict:
    print(f"\n{_CYAN}{_BOLD}[{name.capitalize()} strategy]{_RESET}")
    oracle, substitute, strategy, trainer, cfg = _build(args, name)
    attack = ExtractionAttack(oracle, substitute, strategy, trainer, cfg)
    result = attack.run(verbose=True)
    return attack.to_dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model extraction attack — fidelity vs. query budget comparison"
    )
    parser.add_argument("--strategy",           default="all",
                        choices=_STRATEGY_CHOICES)
    parser.add_argument("--rounds",             type=int, default=10)
    parser.add_argument("--queries-per-round",  type=int, default=100,
                        dest="queries_per_round")
    parser.add_argument("--eval-size",          type=int, default=200,
                        dest="eval_size")
    parser.add_argument("--epochs",             type=int, default=10,
                        help="Substitute training epochs per round")
    parser.add_argument("--hard-label",         action="store_true", dest="hard_label",
                        help="Oracle returns argmax only (harder setting)")
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--output", "-o",       default="results.json")
    parser.add_argument("--html",               default="report.html")
    args = parser.parse_args()

    names = (
        ["random", "jacobian", "adaptive"]
        if args.strategy == "all"
        else [args.strategy]
    )

    all_results = {}
    for name in names:
        all_results[name] = _run_strategy(args, name)

    Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[*] JSON  → {args.output}")

    generate_html_report(all_results, output_path=args.html)
    print(f"[*] HTML  → {args.html}")

    print(f"\n  {'Strategy':<12} {'Queries':>8} {'Agreement':>10}")
    print("  " + "─" * 34)
    for name, data in all_results.items():
        print(
            f"  {name:<12} {data['total_queries']:>8,}"
            f" {data['final_agreement']:>10.4f}"
        )


if __name__ == "__main__":
    main()
