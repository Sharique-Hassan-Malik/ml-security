# Web Application Firewall — Architecture

## System Overview

Two independent detection subsystems operate in parallel on every request. The rule engine applies compiled regex patterns after multi-pass URL decoding. The ML classifier extracts a 38-dimensional feature vector and runs it through a trained scikit-learn model. The final verdict is the union of both outputs.

```
HTTP Request
      │
      ▼
 Field Extractor
  (url, query params, body, headers)
      │
      ├──────────────────────┬───────────────────────┐
      ▼                      ▼                       ▼
 Multi-pass               Feature                  (future:
 URL Decoder              Extractor                 DPI layer)
      │                      │
      ▼                      ▼
 Rule Engine            ML Classifier
 (regex patterns)       (RandomForest / GB / LR)
      │                      │
      └──────────┬────────────┘
                 ▼
          Combined Verdict
          (malicious if either fires)
```

## Component Details

### Rule Engine — `waf/rules/`

**`patterns.py`** defines 29 `Rule` objects, each carrying a category, severity string, a compiled `re.Pattern` and a human-readable description. Severity drives scoring: critical=4, high=3, medium=2, low=1. Rules are grouped into six attack categories: SQLi, XSS, traversal, command injection, SSRF and XXE.

**`engine.py`** orchestrates evaluation:

1. **Field extraction.** The request is decomposed into named string fields: the raw URL, individual query parameter values (via `urllib.parse.parse_qsl`), the body, URL-encoded body parameters and selected headers (User-Agent, Referer, X-Forwarded-For, Cookie). Each field is evaluated independently so the triggering field is reported alongside the match.

2. **Multi-pass URL decoding.** Each field value is decoded in a fixed-point loop — `urllib.parse.unquote` is called repeatedly until the string stops changing. This catches double-encoded payloads (`%2527` → `%27` → `'`) that a single decode pass misses. The loop terminates in at most O(depth) iterations where depth is the number of encoding layers.

3. **Rule evaluation.** Every rule is tried against every decoded field. Matches are collected as `RuleMatch` objects carrying the rule, the matched substring (capped at 120 chars) and the field name.

4. **Scoring.** The verdict score is the sum of severity scores across all matches. A configurable threshold (default 3) determines the binary malicious/benign decision. The `primary_category` is the category with the highest cumulative score.

### Feature Extractor — `waf/ml/features.py`

`extract(request)` returns a list of 38 floats. Features fall into six groups:

| Group | Features | Rationale |
|-------|----------|-----------|
| URL / path | length, depth, query length, entropy, special-char ratio, digit ratio, param count, max param length, percent-encoded count, traversal sequences | Long queries with high entropy and special chars are strong attack signals |
| Body | length, entropy, special-char ratio, digit ratio, percent-encoded count | POST bodies carry injection payloads |
| Cross-field | combined length, entropy, special-char ratio | Ensemble view across all request content |
| Keyword counts | SQL, XSS, command injection, traversal keyword hits; script/select/union counts | Catches obfuscated payloads that single-field checks miss |
| Character counts | single quotes, double quotes, semicolons, pipes, backticks | Payload structural characters |
| Header-derived | user-agent length and entropy, referer length, cookie length and entropy | Scanner user-agents have distinct entropy profiles |

`FEATURE_NAMES` provides the index-to-name mapping used for feature importance reporting.

### ML Classifier — `waf/ml/classifier.py`

`WAFClassifier` wraps a scikit-learn estimator with three additional concerns:

**Feature scaling.** A `StandardScaler` is fit on the training set and applied to all subsequent inputs. This is essential for logistic regression and beneficial for tree ensembles with imbalanced feature ranges.

**Class balance.** All three supported models are configured with `class_weight="balanced"` or equivalent. Web traffic datasets are typically 95%+ benign; without reweighting the classifier optimises for the majority class and ignores rare attacks.

**Persistence.** `save()` serialises model, scaler and model type name into a single pickle file via `pickle.dump`. `load()` is a classmethod that reconstructs a fully fitted instance without calling `__init__`. This avoids re-instantiating the scaler in an unfitted state.

Three model types are supported:

| Type | Strengths | Trade-offs |
|------|-----------|------------|
| `random_forest` | High accuracy, fast inference, interpretable importances | Larger model file |
| `gradient_boosting` | Best accuracy on structured data | Slower training |
| `logistic_regression` | Fastest inference, fully interpretable | Lower accuracy on non-linear patterns |

### Dataset Layer — `waf/datasets/`

**`synthetic.py`** generates labelled request pairs from curated payload pools. Benign requests draw from realistic paths, parameter names, values and user-agents. Attack requests select a random payload from one of five categories and inject it into a randomly chosen parameter in either GET query string or POST body. The generator is seeded for reproducibility.

**`csic.py`** parses the CSIC 2010 HTTP dataset format — multi-line request blocks separated by blank lines — into the same request dict schema used throughout the codebase. It caps loading at a configurable maximum to avoid memory issues.

**`loader.py`** provides a single `load()` call with a `source` parameter that accepts `"synthetic"`, `"csic"` or `"auto"`. Auto mode checks for CSIC files and falls back to synthetic.

### Evaluation Framework — `waf/evaluate/compare.py`

`compare()` runs both detectors on the same labelled request list and collects:

- Precision, recall, F1 and accuracy from `classification_report`
- ROC-AUC from `roc_auc_score`
- Average precision (area under precision-recall curve) from `average_precision_score`
- Per-request latency from `time.perf_counter` wall-clock measurement
- Confusion matrices

`print_comparison()` renders the results as an aligned terminal table with separate confusion matrices. This makes the accuracy/latency trade-off between the two approaches immediately visible.

### API Layer — `api/main.py`

FastAPI serves a single `POST /inspect` endpoint. On startup it attempts to load the ML model from the path configured by `WAF_MODEL_PATH` (defaults to `models/waf_rf.pkl`). If no model file exists the endpoint still functions using the rule engine alone, with `ml_available: false` in the response.

The response schema reports both the rule score and ML probability independently so callers can implement their own threshold logic rather than relying solely on the combined boolean verdict.

## Detection Decision Logic

```
rule_malicious = rule_verdict.score >= threshold
ml_malicious   = ml_probability > 0.5  (if model loaded)
final          = rule_malicious OR ml_malicious
```

This union strategy minimises false negatives at the cost of a modest increase in false positives. An alternative AND strategy (both must agree) is appropriate for low-tolerance environments — configurable by adjusting the threshold or post-processing the API response.

## Extending the System

**Adding a detection rule:** add a tuple to `_RAW` in `waf/rules/patterns.py`. No other files need changing. The rule is automatically compiled, scored and reported.

**Adding a feature:** append an entry to `FEATURE_NAMES` in `waf/ml/features.py`, compute the feature value in `extract()` and ensure the return list length stays consistent with the assertion at the bottom of the file. Retrain the model after adding features.

**Adding a model type:** add an entry to `_MODELS` in `WAFClassifier` mapping a string key to a zero-argument factory function that returns a scikit-learn estimator.

**Switching to a real-time stream:** replace the FastAPI endpoint with a message-consumer loop. The rule engine and ML classifier are both stateless and thread-safe — they can be called concurrently without locks.
