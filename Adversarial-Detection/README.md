# Adversarial Example Detection Framework

Generates adversarial examples with FGSM, PGD, C&W and AutoAttack, then
detects them using three independent methods fused through a unified
scoring system.

---

## Background

Adversarial examples are inputs crafted by adding small, human-imperceptible
perturbations that cause neural networks to misclassify them with high
confidence.  Detection — identifying whether an input is adversarial before
acting on the prediction — is a critical component of deployed ML security.

---

## Attacks

| Attack | Type | Norm | Description |
|--------|------|------|-------------|
| FGSM | White-box | L∞ | Single gradient step |
| PGD | White-box | L∞ | Multi-step projected gradient descent with random start |
| C&W | White-box | L2 | Optimisation-based with tanh change of variable |
| AutoAttack | White-box ensemble | L∞ or L2 | APGD-CE + APGD-T + FAB + Square; falls back to PGD-100 if package not installed |

---

## Detectors

### Feature Squeezing (`detectors/feature_squeezing.py`)
Applies bit-depth reduction and spatial smoothing to the input.
Adversarial perturbations are destroyed by these squeezers; clean
examples are not.  The L1 distance between original and squeezed
softmax outputs is used as a detection score.

### Input Transformation (`detectors/input_transformation.py`)
Applies stochastic random resize-and-pad, Gaussian noise and JPEG-like
compression.  Measures prediction disagreement across transformations.
Adversarial inputs are sensitive to these perturbations; clean ones are stable.

### Statistical Tests (`detectors/statistical_tests.py`)
Two methods, both requiring calibration on clean data:
- **Mahalanobis distance** — fits a class-conditional Gaussian to intermediate
  layer activations and measures distance from the nearest class centroid
- **Kernel Density Estimation** — non-parametric density on penultimate-layer
  activations using Gaussian kernels with median-bandwidth heuristic

### Unified Scorer (`scoring/unified_scorer.py`)
Normalises each detector's raw score to [0, 1] via a per-detector sigmoid
and combines them using a weighted average, max or mean strategy.
Reports TPR, FPR, balanced accuracy and AUROC.

---

## Installation

```bash
pip install torch
pip install autoattack   # optional — AutoAttack falls back to PGD-100 without it
git clone <repo>
cd adversarial-detection
```

---

## Usage

### Evaluate all attacks

```bash
python evaluate.py
```

### Select specific attacks and write a JSON report

```bash
python evaluate.py --attacks fgsm pgd cw --epsilon 0.03 --batch 32 --report results.json
```

### Programmatic

```python
from attacks import generate
from detectors.feature_squeezing import FeatureSqueezingDetector
from scoring.unified_scorer import DetectorConfig, UnifiedScorer

x_adv = generate(model, x_clean, y, method="pgd", epsilon=0.03, steps=40)

det = FeatureSqueezingDetector(model)
scores = det.score(x_adv)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## References

- Goodfellow et al. (2014) *Explaining and Harnessing Adversarial Examples* (ICLR)
- Madry et al. (2018) *Towards Deep Learning Models Resistant to Adversarial Attacks* (ICLR)
- Carlini & Wagner (2017) *Evaluating the Robustness of Neural Networks: An Extreme Case* (IEEE S&P)
- Croce & Hein (2020) *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks* (ICML)
- Xu et al. (2018) *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks* (NDSS)
- Lee et al. (2018) *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Deep Layer Generative Models* (NeurIPS)
- Xie et al. (2018) *Mitigating Adversarial Effects Through Randomization* (ICLR)
