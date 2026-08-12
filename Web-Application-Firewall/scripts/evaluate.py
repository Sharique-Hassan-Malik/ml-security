"""
Compare rule-based and ML detection on a held-out evaluation set.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --model models/waf_rf.pkl --source synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split

from waf.datasets.loader import load
from waf.evaluate.compare import compare, print_comparison
from waf.ml.classifier import WAFClassifier
from waf.rules.engine import RuleEngine


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",     default="models/waf_rf.pkl", help="Path to saved model file")
    ap.add_argument("--source",    default="auto",               help="Dataset source: synthetic, csic or auto")
    ap.add_argument("--n-benign",  type=int, default=2000,       help="Benign samples (synthetic mode)")
    ap.add_argument("--n-attack",  type=int, default=2000,       help="Attack samples (synthetic mode)")
    ap.add_argument("--seed",      type=int, default=99,         help="Random seed for reproducibility")
    args = ap.parse_args()

    print(f"Loading evaluation dataset (source={args.source})…")
    requests, labels = load(source=args.source, n_benign=args.n_benign, n_attack=args.n_attack)

    # Use a different seed so eval set doesn't overlap with training data
    import random
    random.seed(args.seed)
    combined = list(zip(requests, labels))
    random.shuffle(combined)
    requests, labels = zip(*combined)
    requests, labels = list(requests), list(labels)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path}. Run scripts/train.py first.")
        sys.exit(1)

    print(f"Loading model from {model_path}…")
    ml_model = WAFClassifier.load(model_path)

    rule_engine = RuleEngine()

    print(f"Evaluating on {len(requests):,} samples…")
    results = compare(requests, labels, ml_model, rule_engine)
    print_comparison(results)


if __name__ == "__main__":
    main()
