from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix,
)

from .features import extract, FEATURE_NAMES

# Attack categories get their own binary labels for multi-label evaluation
CATEGORIES = ["sqli", "xss", "traversal", "cmdi", "ssrf", "xxe", "benign"]


class WAFClassifier:
    """
    Wraps a scikit-learn estimator with feature extraction and probability output.
    Supports RandomForest (default), GradientBoosting and LogisticRegression.
    """

    _MODELS = {
        "random_forest":      lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "gradient_boosting":  lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        "logistic_regression": lambda: LogisticRegression(
            C=1.0, max_iter=500, class_weight="balanced",
            solver="lbfgs", random_state=42,
        ),
    }

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in self._MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(self._MODELS)}")
        self.model_type = model_type
        self.model      = self._MODELS[model_type]()
        self.scaler     = StandardScaler()
        self._fitted    = False

    def fit(self, requests: list[dict], labels: list[int]) -> "WAFClassifier":
        """
        requests: list of request dicts (keys: method, url, query, body, headers)
        labels:   0 = benign, 1 = malicious
        """
        X = np.array([extract(r) for r in requests], dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, request: dict) -> tuple[int, float]:
        """Returns (label, malicious_probability)."""
        self._check_fitted()
        x = np.array([extract(request)], dtype=np.float64)
        x = self.scaler.transform(x)
        label = int(self.model.predict(x)[0])
        proba = float(self.model.predict_proba(x)[0][1])
        return label, proba

    def predict_batch(self, requests: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X = np.array([extract(r) for r in requests], dtype=np.float64)
        X = self.scaler.transform(X)
        labels = self.model.predict(X)
        probas = self.model.predict_proba(X)[:, 1]
        return labels, probas

    def evaluate(self, requests: list[dict], labels: list[int]) -> dict:
        labels_arr = np.array(labels)
        preds, probas = self.predict_batch(requests)
        cm = confusion_matrix(labels_arr, preds)
        report = classification_report(labels_arr, preds, target_names=["benign", "malicious"], output_dict=True)
        return {
            "classification_report": report,
            "roc_auc":               roc_auc_score(labels_arr, probas),
            "avg_precision":         average_precision_score(labels_arr, probas),
            "confusion_matrix":      cm.tolist(),
            "model_type":            self.model_type,
        }

    def feature_importances(self) -> list[tuple[str, float]] | None:
        """Returns (feature_name, importance) pairs for tree-based models."""
        self._check_fitted()
        if not hasattr(self.model, "feature_importances_"):
            return None
        importances = self.model.feature_importances_
        return sorted(
            zip(FEATURE_NAMES, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "model_type": self.model_type}, f)

    @classmethod
    def load(cls, path: str | Path) -> "WAFClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model      = state["model"]
        obj.scaler     = state["scaler"]
        obj.model_type = state["model_type"]
        obj._fitted    = True
        return obj

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")
