import tempfile
from pathlib import Path

import pytest

from waf.datasets.synthetic import generate
from waf.ml.classifier import WAFClassifier


@pytest.fixture(scope="module")
def small_dataset():
    return generate(n_benign=200, n_attack=200, seed=0)


@pytest.fixture(scope="module")
def trained_rf(small_dataset):
    requests, labels = small_dataset
    clf = WAFClassifier("random_forest")
    clf.fit(requests, labels)
    return clf


class TestFitPredict:
    def test_fit_does_not_raise(self, small_dataset):
        requests, labels = small_dataset
        clf = WAFClassifier("random_forest")
        clf.fit(requests, labels)

    def test_predict_returns_label_and_prob(self, trained_rf, small_dataset):
        req = small_dataset[0][0]
        label, prob = trained_rf.predict(req)
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_predict_batch_shapes(self, trained_rf, small_dataset):
        requests, _ = small_dataset
        preds, probas = trained_rf.predict_batch(requests[:20])
        assert len(preds)  == 20
        assert len(probas) == 20
        assert all(p in (0, 1) for p in preds)
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_unfitted_raises(self):
        clf = WAFClassifier("random_forest")
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict({"url": "/test"})

    def test_evaluate_returns_expected_keys(self, trained_rf, small_dataset):
        requests, labels = small_dataset
        metrics = trained_rf.evaluate(requests[:50], labels[:50])
        assert "roc_auc" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics


class TestModelTypes:
    def test_logistic_regression(self, small_dataset):
        requests, labels = small_dataset
        clf = WAFClassifier("logistic_regression")
        clf.fit(requests, labels)
        label, prob = clf.predict(requests[0])
        assert label in (0, 1)

    def test_gradient_boosting(self, small_dataset):
        requests, labels = small_dataset
        clf = WAFClassifier("gradient_boosting")
        clf.fit(requests, labels)
        label, prob = clf.predict(requests[0])
        assert label in (0, 1)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            WAFClassifier("svm_kernel_rbf")


class TestPersistence:
    def test_save_load_round_trip(self, trained_rf, small_dataset):
        requests, labels = small_dataset
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        trained_rf.save(path)
        loaded = WAFClassifier.load(path)
        assert loaded.model_type == trained_rf.model_type
        orig_preds,   _ = trained_rf.predict_batch(requests[:10])
        loaded_preds, _ = loaded.predict_batch(requests[:10])
        assert list(orig_preds) == list(loaded_preds)


class TestFeatureImportances:
    def test_rf_returns_importances(self, trained_rf):
        fi = trained_rf.feature_importances()
        assert fi is not None
        assert len(fi) == 38
        assert fi[0][1] >= fi[-1][1]   # sorted descending

    def test_lr_returns_none(self, small_dataset):
        requests, labels = small_dataset
        clf = WAFClassifier("logistic_regression")
        clf.fit(requests, labels)
        assert clf.feature_importances() is None


class TestDetectionAccuracy:
    """Smoke test: the model should achieve >85% accuracy on synthetic data."""

    def test_minimum_accuracy(self, small_dataset):
        requests, labels = small_dataset
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(requests, labels, test_size=0.3, random_state=7, stratify=labels)
        clf = WAFClassifier("random_forest")
        clf.fit(X_tr, y_tr)
        metrics = clf.evaluate(X_te, y_te)
        assert metrics["classification_report"]["accuracy"] >= 0.85
