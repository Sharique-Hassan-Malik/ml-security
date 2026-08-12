"""
Train WAF classifier(s) and save to models/.

Usage:
  python scripts/train.py
  python scripts/train.py --model random_forest --source synthetic
  python scripts/train.py --model all --source auto
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split

from waf.datasets.loader import load
from waf.ml.classifier import WAFClassifier

MODEL_TYPES = ["random_forest", "gradient_boosting", "logistic_regression"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",     default="random_forest", help=f"Model type or 'all'. Choices: {MODEL_TYPES}")
    ap.add_argument("--source",    default="auto",          help="Dataset source: synthetic, csic or auto")
    ap.add_argument("--n-benign",  type=int, default=5000,  help="Synthetic benign samples")
    ap.add_argument("--n-attack",  type=int, default=5000,  help="Synthetic attack samples")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for evaluation")
    ap.add_argument("--out-dir",   default="models",        help="Directory for saved model files")
    args = ap.parse_args()

    print(f"Loading dataset (source={args.source})…")
    requests, labels = load(
        source=args.source,
        n_benign=args.n_benign,
        n_attack=args.n_attack,
    )
    print(f"  {len(requests):,} total  |  {sum(labels):,} malicious  |  {len(labels) - sum(labels):,} benign")

    X_train, X_test, y_train, y_test = train_test_split(
        requests, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    to_train = MODEL_TYPES if args.model == "all" else [args.model]
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mtype in to_train:
        print(f"\n{'='*50}")
        print(f"Training {mtype}…")
        clf = WAFClassifier(model_type=mtype)
        clf.fit(X_train, y_train)

        metrics = clf.evaluate(X_test, y_test)
        r = metrics["classification_report"]
        print(f"  Accuracy:   {r['accuracy']:.4f}")
        print(f"  Precision:  {r['malicious']['precision']:.4f}")
        print(f"  Recall:     {r['malicious']['recall']:.4f}")
        print(f"  F1:         {r['malicious']['f1-score']:.4f}")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")

        model_path = out_dir / f"waf_{mtype.replace('_', '')[:4]}.pkl"
        # use shortened names: waf_rf.pkl, waf_gb.pkl, waf_lr.pkl
        short = {"random_forest": "rf", "gradient_boosting": "gb", "logistic_regression": "lr"}
        model_path = out_dir / f"waf_{short[mtype]}.pkl"
        clf.save(model_path)
        print(f"  Saved → {model_path}")

        metrics_path = out_dir / f"metrics_{short[mtype]}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")

        if mtype in ("random_forest", "gradient_boosting"):
            fi = clf.feature_importances()
            if fi:
                print(f"\n  Top 10 features:")
                for name, importance in fi[:10]:
                    bar = "█" * int(importance * 200)
                    print(f"    {name:<30} {importance:.4f}  {bar}")


if __name__ == "__main__":
    main()
