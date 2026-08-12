# Neural Network Weight Poisoning Detector

Static analysis tool that scans PyTorch `.pt` / `.pth` model files for backdoor
attack signatures — no inference, no training data required.

---

## Background

Backdoor (trojan) attacks embed hidden behaviour into neural networks during
training. A poisoned model performs normally on clean inputs but outputs an
attacker-chosen class whenever a specific trigger pattern is present. The attack
works by over-representing a small number of "trojan neurons" that fire
exclusively on triggered inputs.

These neurons leave detectable fingerprints in the saved weights even without
running the model:

| Signature | Mechanism |
|-----------|-----------|
| Dominant neurons | Trojan neurons have L2 norms 3–10× their peers |
| Spectral gap | A single large singular value encodes the trigger direction |
| Bimodal distribution | Clean-weight cluster and trojan-weight cluster coexist |
| Last-layer asymmetry | Target class has abnormally low inbound norm (Neural Cleanse) |
| Outlier weights | Trigger-path weights lie far in the tail of the distribution |

---

## Detectors

### `WeightDistributionAnalyzer`
Computes per-layer moments (mean, std, skewness, excess kurtosis) and flags:
- `high_kurtosis` — heavy-tailed distributions from outlier weights
- `high_skewness` — asymmetric one-sided poisoning
- `outlier_weights` — weights beyond ±4σ exceeding expected fraction
- `bimodal_distribution` — Bimodality Coefficient > 0.70
- `sparse_with_outliers` — dormant neurons masking high-gain ones

### `NeuronInspector`
Treats each output channel as a vector and computes its L2 norm:
- `dominant_neurons` — Tukey fence outliers (Q3 + 3×IQR)
- `dormant_neurons` — near-zero norm neurons (> 20% of layer)
- `norm_dispersion` — high coefficient of variation across neuron norms

### `SpectralAnalyzer`
Full SVD of each weight matrix (reshaped to 2D):
- `spectral_gap` — σ₁/σ₂ exceeds threshold
- `low_stable_rank` — ‖M‖²_F / σ₁² much smaller than min(m,n)
- `sv_outlier` — top singular value is a statistical outlier
- `sv1_concentration` — leading singular vector concentrated on few dims

### `TrojanDetector`
Cross-layer correlation:
- `last_layer_asymmetry` — Neural Cleanse target-class signature
- `norm_progression_anomaly` — isolated layer with 5σ Frobenius norm spike
- `dominant_neuron_pathway` — outlier neuron indices correlated across layers

---

## Installation

```bash
pip install torch
git clone <repo>
cd weight-poisoning-detector
```

---

## Usage

```bash
# Basic scan
python scanner.py model.pt

# Full report
python scanner.py model.pt --report report.json --html report.html --verbose
```

### Programmatic

```python
from detector import scan

report = scan("model.pt")
print(report.verdict)          # "CLEAN" | "SUSPICIOUS" | "HIGH_RISK"
print(report.overall_score)    # float 0.0 – 1.0
print(report.to_json())
```

### Scoring

| Score range | Verdict |
|-------------|---------|
| 0.00 – 0.24 | `CLEAN` |
| 0.25 – 0.59 | `SUSPICIOUS` |
| 0.60 – 1.00 | `HIGH_RISK` |

---

## Running Tests

```bash
# Generate fixtures and run all tests
python tests/generate_fixtures.py
pytest tests/ -v
```

---

## Limitations

- Pure static analysis — cannot detect attacks that leave no weight-level
  signature (e.g., clean-label attacks without weight alteration).
- Thresholds were calibrated on standard ImageNet-pretrained architectures;
  unusual architectures (binary networks, very small models) may produce
  higher false-positive rates.
- Very large layers (> 4M parameters) are skipped by the spectral analyzer
  due to SVD cost; consider chunking such layers separately.

---

## References

- Chen et al. (2017) *Targeted Backdoor Attacks on Deep Learning Systems*
- Tran et al. (2018) *Spectral Signatures in Backdoor Attacks* (NeurIPS)
- Wang et al. (2019) *Neural Cleanse: Identifying and Mitigating Backdoor Attacks* (IEEE S&P)
- Liu et al. (2019) *ABS: Scanning Neural Networks for Back-doors*
