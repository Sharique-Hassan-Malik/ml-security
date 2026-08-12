# Web Application Firewall with ML

A WAF that classifies HTTP requests as benign or malicious using two parallel approaches — a deterministic rule engine and a trained ML classifier — and provides a side-by-side benchmark comparing both.

Attack categories covered: SQL injection, XSS, path traversal, command injection, SSRF and XXE.

## The Hard Parts

**Two-stage detection pipeline.** The rule engine and ML model run independently on every request. The final verdict is the union of their outputs — a request is blocked if either fires. This mirrors real production WAF design where rules catch known signatures while ML catches novel evasions that rules miss.

**Multi-pass URL decoding in the rule engine.** Real-world payloads arrive encoded, double-encoded or triple-encoded. The engine applies `urllib.parse.unquote` in a fixed-point loop until the string stabilises before pattern matching, catching `%2527` → `%27` → `'` chains that naive single-pass decoders miss.

**Feature engineering over raw HTTP fields.** 38 numerical features are extracted per request covering URL and query string entropy, special-character density, SQL/XSS/traversal keyword counts, HTTP header entropy and method scoring. Tree-based models learn non-obvious combinations — for example, high entropy combined with percent-encoding density is a stronger traversal signal than either alone.

**Comparison framework.** `waf/evaluate/compare.py` measures precision, recall, F1, ROC-AUC, average precision and per-request latency for both approaches on the same evaluation set, then prints a side-by-side table with confusion matrices.

## Architecture

```
waf/
  rules/
    patterns.py     compiled regex rules for 6 attack categories
    engine.py       multi-pass decode + field-level rule evaluation
  ml/
    features.py     38-feature extractor from HTTP request dict
    classifier.py   RandomForest / GradientBoosting / LogisticRegression wrapper
  datasets/
    csic.py         CSIC 2010 multi-line HTTP format parser
    synthetic.py    realistic benign + attack request generator
    loader.py       unified dataset loader with auto-detection
  evaluate/
    compare.py      side-by-side rule vs ML metrics and print utilities
api/
  main.py           FastAPI endpoint — combined rule + ML verdict
scripts/
  train.py          train and save classifier(s)
  evaluate.py       run comparison report
  demo.py           inspect individual requests interactively
tests/
  test_rules.py     rule engine unit tests across all 6 attack categories
  test_features.py  feature extractor unit tests
  test_classifier.py fit/predict/persist/accuracy tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Train on synthetic data (no external dataset required):

```bash
python scripts/train.py
```

Train all three model types:

```bash
python scripts/train.py --model all
```

If you have the CSIC 2010 dataset, place `normalTraffic.txt` as `data/csic_normal.txt` and `anomalousTraffic.txt` as `data/csic_anomalous.txt`, then:

```bash
python scripts/train.py --source csic
```

The `--source auto` flag uses CSIC if files are present and falls back to synthetic otherwise.

## Evaluation

Compare rule engine vs ML model on a held-out set:

```bash
python scripts/evaluate.py
```

Example output:

```
==============================================================
  WAF Detection Comparison  |  4,000 samples  (2,000 attacks)
==============================================================
Metric                       Rule Engine        ML (random_forest)
--------------------------------------------------------------
  Accuracy               0.9620               0.9810
  Precision              0.9583               0.9788
  Recall                 0.9650               0.9830
  F1                     0.9617               0.9809
  ROC-AUC                0.9890               0.9961
  Avg Precision          0.9872               0.9955
  Latency (ms/req)       0.0481               0.3120
==============================================================
```

The rule engine has lower latency with competitive accuracy. The ML model achieves higher recall on evasion payloads that obfuscate keywords. The combined approach (both must agree to pass) minimises false negatives at the cost of slightly higher false positives.

## Demo

Run the six built-in example requests through both detectors:

```bash
python scripts/demo.py
```

Interactive mode accepts raw JSON request dicts:

```bash
python scripts/demo.py --interactive
```

## Running the API

```bash
cd api
uvicorn main:app --reload --port 8000
```

POST a request for inspection:

```bash
curl -X POST http://localhost:8000/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "method": "GET",
    "url": "/search?q=1+UNION+SELECT+username,password+FROM+users--",
    "query": "q=1+UNION+SELECT+username,password+FROM+users--"
  }'
```

Response:

```json
{
  "malicious": true,
  "confidence": 0.98,
  "rule_score": 10,
  "primary_category": "sqli",
  "categories": ["sqli"],
  "rule_matches": [
    {
      "category": "sqli",
      "severity": "critical",
      "description": "UNION SELECT statement",
      "field": "param:q",
      "matched_text": "UNION SELECT"
    }
  ],
  "ml_available": true
}
```

## Running Tests

```bash
pytest tests/ -v
```

## Datasets

**Synthetic** (default): generated in-process from realistic payload pools. No download required.

**CSIC 2010**: HTTP dataset for web application firewall evaluation from the Spanish National Research Council. Contains 36,000 normal and 25,065 anomalous requests targeting an e-commerce application. Available at http://www.isi.csic.es/dataset/

## File Map

| Path | Description |
|------|-------------|
| `waf/rules/patterns.py` | 29 compiled regex rules across 6 attack categories |
| `waf/rules/engine.py` | Multi-pass decode, field extraction and rule evaluation |
| `waf/ml/features.py` | 38-feature extractor with feature name index |
| `waf/ml/classifier.py` | Model wrapper with train/predict/evaluate/save/load |
| `waf/datasets/csic.py` | CSIC 2010 block-format HTTP parser |
| `waf/datasets/synthetic.py` | Benign and attack request generators |
| `waf/datasets/loader.py` | Unified dataset loading with auto-detection |
| `waf/evaluate/compare.py` | Side-by-side comparison and console report |
| `api/main.py` | FastAPI endpoint with combined rule + ML verdict |
| `scripts/train.py` | Train and save one or all classifier types |
| `scripts/evaluate.py` | Run comparison report on evaluation data |
| `scripts/demo.py` | Inspect requests interactively or via built-in examples |
| `tests/test_rules.py` | Rule engine tests for all 6 attack categories |
| `tests/test_features.py` | Feature extraction correctness tests |
| `tests/test_classifier.py` | Classifier fit/predict/persist and accuracy tests |
