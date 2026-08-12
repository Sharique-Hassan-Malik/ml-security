# Model Extraction Attack Simulator

Given only black-box API access to a victim model, reconstruct a
functionally equivalent substitute model.  Compares three query
strategies and measures fidelity vs. query budget tradeoffs.

---

## Background

Model extraction (also called model stealing) treats a deployed model as
a black box and reconstructs a substitute that mimics its behaviour using
only query-response pairs.  A successful extraction leaks the model's
decision function without ever accessing its weights, architecture or
training data.

This simulator implements the full iterative extraction pipeline against a
randomly initialised victim CNN and benchmarks three query strategies to
show how information efficiency varies across approaches.

---

## Oracle Modes

| Mode | Attacker receives | Difficulty |
|------|-------------------|------------|
| Soft label (default) | Full probability vector | Easier — more information per query |
| Hard label | Top-1 class index only | Harder — minimum information per query |

Temperature scaling is supported to simulate APIs returning calibrated or
uncalibrated confidence scores.

---

## Query Strategies

### Random (`strategies/query_strategies.py :: RandomStrategy`)
Samples inputs uniformly from [0, 1]^d.  No information about the substitute
is used.  Establishes a lower bound on extraction efficiency.

### Jacobian-based Dataset Augmentation (`JacobianStrategy`)
Papernot et al. (2017).  Uses the current substitute model's Jacobian to
perturb seed inputs in the direction that maximally changes the predicted
output, continuously probing the decision boundary neighbourhood.

### Adaptive / Uncertainty Sampling (`AdaptiveStrategy`)
Evaluates a large pool of candidates against the substitute and selects
those with highest prediction entropy.  Concentrates the query budget at
the points where the substitute is most uncertain and oracle feedback is
most valuable.

---

## Substitute Training

`SubstituteTrainer` trains the substitute on oracle-labelled data each round:

- **Soft labels** — KL-divergence loss against oracle probability vectors;
  extracts more signal per query than hard labels
- **Hard labels** — standard cross-entropy against oracle argmax outputs

All data accumulated across rounds is re-used in every training round so
early queries continue to contribute throughout the extraction.

---

## Installation

```bash
pip install torch
git clone <repo>
cd model-extraction
```

---

## Usage

### Compare all three strategies (default)

```bash
python experiment.py
```

### Single strategy with hard labels

```bash
python experiment.py --strategy jacobian --hard-label \
    --rounds 20 --queries-per-round 200
```

### Write reports

```bash
python experiment.py --output results.json --html report.html
```

### Programmatic

```python
from oracle.black_box import BlackBoxOracle
from strategies.query_strategies import AdaptiveStrategy
from extraction.trainer import SubstituteTrainer
from extraction.attack_loop import ExtractionAttack, ExtractionConfig

oracle   = BlackBoxOracle(victim_model, hard_label=False)
strategy = AdaptiveStrategy(input_shape=(3, 32, 32), batch_size=100)
trainer  = SubstituteTrainer(substitute_model, epochs=10)
cfg      = ExtractionConfig(n_rounds=15, queries_per_round=100)

attack = ExtractionAttack(oracle, substitute_model, strategy, trainer, cfg)
result = attack.run()
print(result.final_agreement)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| Agreement | Fraction of test inputs where oracle and substitute predict the same class |
| KL divergence | KL(oracle_probs ∥ substitute_probs) — soft-label mode only |
| Queries to 90% | Oracle queries needed to reach 90% agreement (shown in HTML report) |

---

## References

- Tramèr et al. (2016) *Stealing Machine Learning Models via Prediction APIs* (USENIX Security)
- Papernot et al. (2017) *Practical Black-Box Attacks Against Machine Learning* (AsiaCCS)
- Correia-Silva et al. (2018) *Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data*
- Orekondy et al. (2019) *Knockoff Nets: Stealing Functionality of Black-Box Models* (CVPR)
