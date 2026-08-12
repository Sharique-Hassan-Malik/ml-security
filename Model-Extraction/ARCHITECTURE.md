# Architecture

---

## Module Map

```
experiment.py                  (CLI — runs all strategies and writes report)
│
├── oracle/
│   └── black_box.py           BlackBoxOracle
│         .query(x)            → probs (B, C) or labels (B,)
│         .query_count         total queries issued so far
│         .budget_remaining    queries left before QueryBudgetExceeded
│
├── strategies/
│   └── query_strategies.py
│         RandomStrategy        uniform random sampling
│         JacobianStrategy      Papernot et al. 2017 JBDA
│         AdaptiveStrategy      entropy-based uncertainty sampling
│
├── extraction/
│   ├── trainer.py             SubstituteTrainer
│   │     .train_round(...)    → {final_loss}
│   │     .evaluate_fidelity() → {agreement, kl_divergence}
│   └── attack_loop.py         ExtractionAttack
│         .run()               → ExtractionResult
│         .to_dict(result)     → plain dict for JSON output
│
├── models/
│   └── architectures.py       VictimCNN, SubstituteCNN, SubstituteMLP
│
└── metrics/
    └── report.py              generate_html_report()
```

---

## Extraction Loop

```
oracle = BlackBoxOracle(victim, hard_label=False)

for round in range(n_rounds):

    x_new = strategy.generate(substitute, x_accumulated)

        RandomStrategy   → torch.rand(B, *shape)

        JacobianStrategy → for seed x in seed_data[:B]:
                               grad = ∂F_substitute(x)[argmax] / ∂x
                               x_aug = clip(x + λ·sign(grad), 0, 1)

        AdaptiveStrategy → candidates = rand(N) + perturb(seed_data)
                           entropy    = H(softmax(substitute(candidates)))
                           x_new      = top-k by entropy

    y_new = oracle.query(x_new)         ← only API call to victim

    x_all = cat(x_all, x_new)
    y_all = cat(y_all, y_new)

    trainer.train_round(x_new, y_new, x_all, y_all)

        soft: loss = KL(oracle_probs ∥ softmax(substitute(x)))
        hard: loss = CrossEntropy(substitute(x), oracle_argmax)

    fidelity = trainer.evaluate_fidelity(oracle.query, x_eval)
    record RoundResult(queries_used, agreement, kl_divergence, train_loss)
```

---

## Query Budget Analysis

| Strategy | Information source | Budget efficiency |
|----------|--------------------|-------------------|
| Random | None | Baseline |
| Jacobian | Substitute gradient direction | Moderate; improves as substitute improves |
| Adaptive | Substitute prediction entropy | High; best when substitute is already reasonable |

Jacobian benefits are most pronounced at mid-range budgets.  Adaptive
requires enough rounds for the substitute's uncertainty estimates to be
meaningful before they can guide useful queries.

---

## Adding a New Strategy

Implement a class with:

```python
class MyStrategy:
    def generate(
        self,
        substitute: Optional[nn.Module],
        seed_data:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        ...   # return (B, *input_shape) tensor in [0, 1]
```

Pass an instance directly to `ExtractionAttack`.  No other files need
to change.
