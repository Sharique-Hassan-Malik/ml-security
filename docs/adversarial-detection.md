# Architecture

---

## Module Map

```
evaluate.py              (CLI — end-to-end evaluation)
│
├── attacks/
│   ├── gradient_attacks.py   fgsm(), pgd()
│   ├── cw_attack.py          cw_l2()
│   ├── autoattack_wrapper.py autoattack()  [wraps library or falls back to PGD]
│   └── __init__.py           generate()   [dispatch by name]
│
├── detectors/
│   ├── feature_squeezing.py    FeatureSqueezingDetector
│   ├── input_transformation.py InputTransformationDetector
│   └── statistical_tests.py    MahalanobisDetector, KDEDetector
│
├── scoring/
│   └── unified_scorer.py  UnifiedScorer, DetectorConfig, ScoringResult
│
├── models/
│   └── small_cnn.py       SmallCNN (test harness only)
│
└── tests/
    └── test_all.py
```

---

## Data Flow

```
clean (x, y)
    │
    ├─── detector calibration (Mahalanobis, KDE)
    │
    └─── generate() ──► x_adv
                │
                ▼
    ┌───────────────────────────────┐
    │  FeatureSqueezingDetector     │  score_fs   (B,)
    │  InputTransformationDetector  │  score_it   (B,)
    │  MahalanobisDetector          │  score_mah  (B,)
    │  KDEDetector                  │  score_kde  (B,)
    └───────────────────────────────┘
                │
                ▼
        UnifiedScorer.score({...})
                │
                ▼
        ScoringResult
          .probability      (B,)  in [0,1]
          .is_adversarial   (B,)  bool
          .per_detector     {name: normalised score}

        UnifiedScorer.evaluate(clean_scores, adv_scores)
          → {tpr, fpr, balanced_acc, auroc}
```

---

## Detector Interface Contract

Each detector exposes two methods:

```python
detector.score(x: Tensor) -> Tensor        # (B,) raw scores, higher = more adversarial
detector.predict(x: Tensor, threshold)     # -> (BoolTensor, FloatTensor)
```

Statistical detectors additionally require:

```python
detector.calibrate(x_clean, ...)           # must be called before score()
```

---

## Scoring Normalisation

Raw scores live on different scales (L1 distances, Mahalanobis distances,
log-density values).  Each is mapped to [0, 1] via:

```
p = sigmoid(steepness × (raw − midpoint))
```

The midpoint and steepness are set per detector in `DetectorConfig` and
chosen so that the threshold of 0.5 sits at a reasonable decision boundary
for each detector's output range.

---

## Adding a New Detector

1. Create `detectors/my_detector.py` with a class implementing `score()` and `predict()`.
2. Import and instantiate it in `evaluate.py`.
3. Add a `DetectorConfig` entry with appropriate midpoint and steepness values.
4. Add tests to `tests/test_all.py`.
