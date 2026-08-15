"""modelext — how much of your model an attacker gets for a given query budget.

    from modelext import run_extraction
    outcome = run_extraction(strategy="jacobian", rounds=8)

The headline number is not "can it be stolen" — with unlimited queries it
always can. It is *queries to 90% agreement*, because that is the number a
rate limit is set against.
"""

from __future__ import annotations

from typing import Any

import torch

from .architectures import SubstituteCNN, SubstituteMLP, VictimCNN
from .attack_loop import ExtractionAttack, ExtractionConfig, ExtractionResult, RoundResult
from .fidelity import accuracy_gap, agreement_rate, soft_label_kl, topk_fidelity
from .oracle import BlackBoxOracle, QueryBudgetExceeded
from .strategies import AdaptiveStrategy, JacobianStrategy, RandomStrategy
from .trainer import SubstituteTrainer

MODULE_NAME = "model-extraction"

STRATEGIES = {
    "random": RandomStrategy,
    "jacobian": JacobianStrategy,
    "adaptive": AdaptiveStrategy,
}

__all__ = [
    "MODULE_NAME", "STRATEGIES", "run_extraction",
    "ExtractionAttack", "ExtractionConfig", "ExtractionResult", "RoundResult",
    "BlackBoxOracle", "QueryBudgetExceeded", "SubstituteTrainer",
    "VictimCNN", "SubstituteCNN", "SubstituteMLP",
    "RandomStrategy", "JacobianStrategy", "AdaptiveStrategy",
    "agreement_rate", "topk_fidelity", "soft_label_kl", "accuracy_gap",
]


def run_extraction(
    strategy: str = "jacobian",
    *,
    victim: Any = None,
    rounds: int = 10,
    queries_per_round: int = 100,
    eval_size: int = 200,
    epochs: int = 3,
    hard_label: bool = False,
    seed: int = 42,
    input_shape: tuple = (3, 32, 32),
    num_classes: int = 10,
    verbose: bool = False,
) -> ExtractionResult:
    """Run one extraction attack end to end.

    Shared by the benchmark and by the platform probe so both measure the same
    attack. `hard_label=True` is the harder, more realistic setting: an API
    that returns only the argmax leaks far less per query than one returning
    the full probability vector.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")

    torch.manual_seed(seed)

    if victim is None:
        victim = VictimCNN(num_classes=num_classes)
    victim.eval()

    oracle = BlackBoxOracle(
        model=victim,
        hard_label=hard_label,
        # attack queries + one labelling pass over the fixed eval set
        query_limit=rounds * queries_per_round + eval_size + 64,
    )
    substitute = SubstituteCNN(num_classes=num_classes)
    trainer = SubstituteTrainer(
        substitute=substitute,
        lr=1e-3,
        epochs=epochs,
        batch_size=64,
        soft_labels=not hard_label,
    )
    config = ExtractionConfig(
        n_rounds=rounds,
        queries_per_round=queries_per_round,
        eval_size=eval_size,
        input_shape=input_shape,
        seed=seed,
    )
    attack = ExtractionAttack(
        oracle=oracle,
        substitute=substitute,
        strategy=STRATEGIES[strategy](input_shape=input_shape),
        trainer=trainer,
        config=config,
    )
    return attack.run(verbose=verbose)
