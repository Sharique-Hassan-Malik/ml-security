# Architecture

---

## Overview

```
scanner.py  (CLI)
    │
    └── detector.scan(path) -> Report
            │
            ├── loader.load_weights(path)
            │       └── torch.load(weights_only=True)
            │           _collect_tensors()   [recursive walker]
            │           _keep_tensor()       [filter biases, BN buffers, 1-D params]
            │
            ├── WeightDistributionAnalyzer.analyze(name, tensor)  [per layer]
            ├── NeuronInspector.analyze(name, tensor)             [per layer]
            ├── SpectralAnalyzer.analyze(name, tensor)            [per layer]
            └── TrojanDetector.analyze(all_weights)               [cross-layer]
                    │
                    └── Report  (findings list + scoring + JSON/HTML output)
```

---

## Data Flow

1. **Load** — `loader.py` opens the checkpoint with `weights_only=True` to prevent
   arbitrary code execution. It recursively walks nested dicts / lists to collect
   all floating-point 2-D+ tensors, stripping biases and BN statistics.

2. **Per-layer analysis** — three independent analyzers each receive `(name, tensor)`
   and return a list of `Finding` objects. Analyzers are stateless and can be run
   in any order.

3. **Cross-layer analysis** — `TrojanDetector` receives the full weight dict and
   correlates patterns across consecutive layers.

4. **Report** — all findings are collected into a `Report`. The `overall_score`
   property combines finding counts and severities. Verdict thresholds are fixed
   constants, not learned.

---

## Finding Severity Mapping

| Severity | Meaning |
|----------|---------|
| `low`    | Secondary indicator; low individual specificity |
| `medium` | Moderate confidence; warrants manual inspection |
| `high`   | Strong indicator consistent with a known attack signature |

---

## Scoring Formula

```
high_count = number of "high" findings
med_count  = number of "medium" findings
n          = number of analysed layers

if high_count > 0:
    score = min(1.0, 0.50 + 0.50 * high_count / n)
elif med_count > 0:
    score = min(0.49, 0.20 + 0.29 * med_count / n)
else:
    score = min(0.19, 0.04 * total_findings)
```

This ensures a single high-severity finding on a large model scores around 0.5,
while multiple high-severity findings across many layers push toward 1.0.

---

## Adding a New Detector

1. Create `detector/my_analyzer.py` with a class that implements:
   ```python
   def analyze(self, name: str, tensor: torch.Tensor) -> List[Finding]: ...
   ```
2. Import and instantiate in `detector/__init__.py` inside `scan()`.
3. Add tests in `tests/test_detector.py`.

No other files need to change.

---

## Known Attack Signatures and Covered Tests

| Attack | Primary signature | Covered by |
|--------|------------------|------------|
| BadNets (Gu et al. 2017) | Dominant neurons, outlier weights | `NeuronInspector`, `WeightDistributionAnalyzer` |
| TrojanNN (Liu et al. 2018) | Low stable rank, spectral gap | `SpectralAnalyzer` |
| Neural Cleanse target class | Last-layer norm asymmetry | `TrojanDetector` |
| Blended injection | Bimodal distributions | `WeightDistributionAnalyzer` |
| WaNet (Nguyen & Tran 2021) | Pathway correlation | `TrojanDetector` |
