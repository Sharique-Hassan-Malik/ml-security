# Gradient Leakage Analyzer

> Part of the [AI Security Suite](../../README.md). Runs standalone from this
> folder, or alongside every other module via the platform CLI — see
> [Using it](#using-it) below.

Implements federated learning gradient inversion attacks (DLG and iDLG) and
three defenses, with quantitative reconstruction quality metrics and an
HTML report.

---

## Background

In federated learning, clients share gradients rather than raw data. Zhu et al.
(NeurIPS 2019) showed that an honest-but-curious aggregation server can
reconstruct private training images from these gradients alone — a result that
fundamentally challenged the privacy assumptions of early federated learning.

This project implements both the attack and a set of practical defenses,
providing a testbed for measuring the privacy–utility tradeoff.

---

## Attack Algorithms

### DLG — Deep Leakage from Gradients (Zhu et al. 2019)
Randomly initialises dummy data and optimises it via L-BFGS to minimise the
distance between dummy gradients and observed gradients:

```
min_{x̃, ỹ}  ‖∇W ℒ(F(x̃), ỹ) − ∇W*‖²
```

### iDLG — Improved DLG (Zhao et al. 2020)
Recovers the ground-truth label analytically before optimising. The true label
is the `argmin` of the column sums of the last linear layer's weight gradient,
eliminating label search from the optimisation loop.

### R-GAP — Recursive Gradient Attack (Zhu & Blaschko 2021)
For fully-connected networks, recovers inputs layer-by-layer by solving
`δ_l ⊗ a_{l-1} = ∂L/∂W_l` as a least-squares system. Exact recovery for
linear activations; approximate for ReLU.

---

## Defenses

| Defense | Mechanism | Cost |
|---------|-----------|------|
| `DifferentialPrivacyDefense` | Clip + Gaussian noise (DP-SGD) | High accuracy loss |
| `GradientCompressionDefense` | Top-k sparsification | Moderate accuracy loss |
| `GradientNoiseDefense` | Additive Gaussian / Laplace noise | Tunable |

---

## Installation

```bash
pip install torch Pillow
git clone <repo>
cd gradient-leakage-analyzer
```

---

## Usage

### Run the full experiment

```bash
python experiment.py --iterations 300 --algorithm idlg --html report.html
```

### Tune defenses

```bash
python experiment.py \
  --dp-clip 0.5 --dp-noise 2.0 \
  --compression 0.95 \
  --noise-scale 0.1
```

### Programmatic

```python
from gradleak import AttackConfig, GradientInversionAttack, compute_observed_gradients, run_leakage
import torch.nn as nn

grads  = compute_observed_gradients(model, nn.CrossEntropyLoss(), data, labels)
cfg    = AttackConfig(iterations=300, algorithm="idlg")
attack = GradientInversionAttack(model, nn.CrossEntropyLoss(), cfg)
result = attack.attack(grads, data_shape=(3, 32, 32))

print(result.dummy_labels)
# result.dummy_data is the reconstructed image tensor
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Metrics

| Metric | Meaning |
|--------|---------|
| PSNR | Peak SNR vs. original (dB); higher = better reconstruction |
| SSIM | Structural similarity (0–1); 1.0 = identical |
| MSE | Mean squared pixel error |
| Cosine similarity | Alignment between original and defended gradients |
| Gradient SNR | Signal-to-noise ratio of defended gradients (dB) |

---

## References

- Zhu et al. (2019) *Deep Leakage from Gradients* (NeurIPS)
- Zhao et al. (2020) *iDLG: Improved Deep Leakage from Gradients*
- Zhu & Blaschko (2021) *R-GAP: Recursive Gradient Attack on Privacy* (ICLR)
- Abadi et al. (2016) *Deep Learning with Differential Privacy* (CCS)
