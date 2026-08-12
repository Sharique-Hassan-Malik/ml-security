"""
Interactively inspect HTTP requests through both the rule engine and the ML model.

Usage:
  python scripts/demo.py
  python scripts/demo.py --model models/waf_rf.pkl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from waf.rules.engine import RuleEngine
from waf.ml.classifier import WAFClassifier

_EXAMPLES = [
    # (description, request dict)
    ("Normal product search",
     {"method": "GET", "url": "/search?q=shoes&page=1", "query": "q=shoes&page=1",
      "body": "", "headers": {"User-Agent": "Mozilla/5.0"}}),

    ("SQL injection in GET param",
     {"method": "GET", "url": "/product?id=1' UNION SELECT username,password FROM users--",
      "query": "id=1' UNION SELECT username,password FROM users--",
      "body": "", "headers": {}}),

    ("XSS in POST body",
     {"method": "POST", "url": "/comment",
      "query": "", "body": "text=<script>alert(document.cookie)</script>",
      "headers": {"Content-Type": "application/x-www-form-urlencoded"}}),

    ("Path traversal",
     {"method": "GET", "url": "/file?name=../../../../etc/passwd",
      "query": "name=../../../../etc/passwd",
      "body": "", "headers": {}}),

    ("Command injection",
     {"method": "GET", "url": "/ping?host=127.0.0.1;cat /etc/passwd",
      "query": "host=127.0.0.1;cat /etc/passwd",
      "body": "", "headers": {}}),

    ("SSRF via URL parameter",
     {"method": "GET", "url": "/proxy?url=http://169.254.169.254/latest/meta-data/",
      "query": "url=http://169.254.169.254/latest/meta-data/",
      "body": "", "headers": {}}),
]


def inspect_and_print(req: dict, rule_engine: RuleEngine, ml_model: WAFClassifier | None) -> None:
    verdict = rule_engine.inspect(req)

    ml_label, ml_prob = None, None
    if ml_model is not None:
        ml_label, ml_prob = ml_model.predict(req)

    combined = verdict.malicious or (ml_prob is not None and ml_prob > 0.5)

    status = "🚨 MALICIOUS" if combined else "✅ BENIGN"
    print(f"\n  {status}")
    print(f"  Rule score:  {verdict.score}  |  "
          f"Categories: {', '.join(verdict.categories) or 'none'}")
    if ml_prob is not None:
        print(f"  ML ({ml_model.model_type}):  probability={ml_prob:.4f}  label={'malicious' if ml_label else 'benign'}")

    if verdict.matches:
        print("  Matches:")
        for m in verdict.matches:
            print(f"    [{m.rule.severity:<8}] [{m.rule.category:<10}] {m.rule.description}")
            print(f"             field={m.field!r}  text={m.matched_text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",       default="models/waf_rf.pkl", help="Path to saved model")
    ap.add_argument("--interactive", action="store_true",          help="Accept JSON input from stdin in a loop")
    args = ap.parse_args()

    rule_engine = RuleEngine()
    ml_model: WAFClassifier | None = None
    model_path = Path(args.model)
    if model_path.exists():
        ml_model = WAFClassifier.load(model_path)
        print(f"ML model loaded: {ml_model.model_type}")
    else:
        print(f"No model at {model_path} — rule engine only. Run scripts/train.py first.")

    if args.interactive:
        print("\nPaste a JSON request dict and press Enter. Ctrl+C to quit.")
        while True:
            try:
                line = input("\n> ").strip()
                if not line:
                    continue
                req = json.loads(line)
                inspect_and_print(req, rule_engine, ml_model)
            except (KeyboardInterrupt, EOFError):
                break
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
    else:
        print(f"\nRunning {len(_EXAMPLES)} example requests:\n{'='*60}")
        for desc, req in _EXAMPLES:
            print(f"\n→ {desc}")
            print(f"  {req['method']} {req['url'][:80]}")
            inspect_and_print(req, rule_engine, ml_model)
        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
