"""
Model extraction attack loop.

Orchestrates the iterative attack:
  1. Use a query strategy to generate candidate inputs.
  2. Query the black-box oracle for labels.
  3. Accumulate (input, label) pairs.
  4. Train the substitute model for one round.
  5. Measure fidelity and record a checkpoint.
  6. Repeat until the query budget is exhausted or all rounds complete.

The result contains per-round snapshots so fidelity vs. query budget
curves can be plotted externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from oracle.black_box import BlackBoxOracle, QueryBudgetExceeded
from extraction.trainer import SubstituteTrainer


@dataclass
class ExtractionConfig:
    n_rounds:          int   = 10
    queries_per_round: int   = 100
    eval_size:         int   = 200
    input_shape:       tuple = (3, 32, 32)
    seed:              int   = 42


@dataclass
class RoundResult:
    round_idx:     int
    queries_used:  int
    agreement:     float
    kl_divergence: Optional[float]
    train_loss:    float


@dataclass
class ExtractionResult:
    rounds:          List[RoundResult] = field(default_factory=list)
    total_queries:   int               = 0
    final_agreement: float             = 0.0
    substitute:      Optional[nn.Module] = None


class ExtractionAttack:
    """
    Iterative model extraction attack.

    Parameters
    ----------
    oracle     : BlackBoxOracle wrapping the victim model
    substitute : nn.Module to be trained as the clone
    strategy   : query strategy instance (Random, Jacobian or Adaptive)
    trainer    : SubstituteTrainer instance
    config     : ExtractionConfig
    """

    def __init__(
        self,
        oracle:     BlackBoxOracle,
        substitute: nn.Module,
        strategy,
        trainer:    SubstituteTrainer,
        config:     Optional[ExtractionConfig] = None,
    ) -> None:
        self.oracle     = oracle
        self.substitute = substitute
        self.strategy   = strategy
        self.trainer    = trainer
        self.cfg        = config or ExtractionConfig()

        torch.manual_seed(self.cfg.seed)
        self._eval_x = torch.rand(self.cfg.eval_size, *self.cfg.input_shape)

    def run(self, verbose: bool = True) -> ExtractionResult:
        """
        Execute the full extraction attack.

        Returns an ExtractionResult with per-round fidelity snapshots.
        """
        result = ExtractionResult(substitute=self.substitute)
        x_all: Optional[torch.Tensor] = None
        y_all: Optional[torch.Tensor] = None

        for rnd in range(self.cfg.n_rounds):
            x_new = self.strategy.generate(
                substitute = self.substitute if x_all is not None else None,
                seed_data  = x_all,
            )

            try:
                y_new = self.oracle.query(x_new)
            except QueryBudgetExceeded as exc:
                if verbose:
                    print(f"  [!] {exc}")
                break

            x_all = x_new if x_all is None else torch.cat([x_all, x_new], dim=0)
            y_all = y_new if y_all is None else torch.cat([y_all, y_new], dim=0)

            train_m   = self.trainer.train_round(x_new, y_new, x_all, y_all)
            fidelity  = self.trainer.evaluate_fidelity(self.oracle.query, self._eval_x)

            rr = RoundResult(
                round_idx     = rnd,
                queries_used  = self.oracle.query_count,
                agreement     = fidelity["agreement"],
                kl_divergence = fidelity["kl_divergence"],
                train_loss    = train_m["final_loss"],
            )
            result.rounds.append(rr)

            if verbose:
                kl_s = (
                    f"  KL={rr.kl_divergence:.4f}"
                    if rr.kl_divergence is not None else ""
                )
                print(
                    f"  Round {rnd + 1:2d}/{self.cfg.n_rounds}"
                    f"  queries={rr.queries_used:6d}"
                    f"  agreement={rr.agreement:.4f}"
                    f"{kl_s}"
                    f"  loss={rr.train_loss:.4f}"
                )

        result.total_queries   = self.oracle.query_count
        result.final_agreement = result.rounds[-1].agreement if result.rounds else 0.0
        return result

    def to_dict(self, result: ExtractionResult) -> dict:
        """Serialise an ExtractionResult to a plain dict for JSON output."""
        return {
            "total_queries":   result.total_queries,
            "final_agreement": result.final_agreement,
            "rounds": [
                {
                    "round":         r.round_idx,
                    "queries_used":  r.queries_used,
                    "agreement":     r.agreement,
                    "kl_divergence": r.kl_divergence,
                    "train_loss":    round(r.train_loss, 6),
                }
                for r in result.rounds
            ],
        }
