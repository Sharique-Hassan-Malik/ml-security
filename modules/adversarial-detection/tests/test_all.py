"""
Tests for the adversarial detection framework.

Run with:  pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn

from advdet.attacks import generate
from advdet.attacks.cw_attack import cw_l2
from advdet.attacks.gradient_attacks import fgsm, pgd
from advdet.detectors.feature_squeezing import FeatureSqueezingDetector, _bit_reduce, _median_smooth
from advdet.detectors.input_transformation import InputTransformationDetector
from advdet.detectors.statistical_tests import KDEDetector, MahalanobisDetector
from advdet.models.small_cnn import SmallCNN
from advdet.scoring import DetectorConfig, ScoringResult, UnifiedScorer


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    torch.manual_seed(0)
    m = SmallCNN(num_classes=10)
    m.eval()
    return m


@pytest.fixture(scope="session")
def clean_batch():
    torch.manual_seed(1)
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    return x, y


# ------------------------------------------------------------------
# Attack tests
# ------------------------------------------------------------------

class TestFGSM:
    def test_output_range(self, model, clean_batch):
        x, y = clean_batch
        x_adv = fgsm(model, x, y, epsilon=0.03)
        assert x_adv.min() >= 0.0 - 1e-5
        assert x_adv.max() <= 1.0 + 1e-5

    def test_perturbation_bounded(self, model, clean_batch):
        x, y = clean_batch
        eps = 0.03
        x_adv = fgsm(model, x, y, epsilon=eps)
        assert (x_adv - x).abs().max().item() <= eps + 1e-5

    def test_shape_preserved(self, model, clean_batch):
        x, y = clean_batch
        x_adv = fgsm(model, x, y)
        assert x_adv.shape == x.shape


class TestPGD:
    def test_output_range(self, model, clean_batch):
        x, y = clean_batch
        x_adv = pgd(model, x, y, epsilon=0.03, steps=5)
        assert x_adv.min() >= 0.0 - 1e-5
        assert x_adv.max() <= 1.0 + 1e-5

    def test_perturbation_bounded(self, model, clean_batch):
        x, y = clean_batch
        eps = 0.03
        x_adv = pgd(model, x, y, epsilon=eps, steps=5)
        assert (x_adv - x).abs().max().item() <= eps + 1e-5

    def test_random_vs_no_random_start(self, model, clean_batch):
        x, y = clean_batch
        torch.manual_seed(0)
        adv_r  = pgd(model, x, y, epsilon=0.03, steps=5, random_start=True)
        adv_nr = pgd(model, x, y, epsilon=0.03, steps=5, random_start=False)
        # Both valid — just check shapes and bounds
        assert adv_r.shape == x.shape
        assert adv_nr.shape == x.shape


class TestCW:
    def test_output_range(self, model, clean_batch):
        x, y = clean_batch
        x_adv = cw_l2(model, x, y, steps=20)
        assert x_adv.min() >= 0.0 - 1e-5
        assert x_adv.max() <= 1.0 + 1e-5

    def test_shape_preserved(self, model, clean_batch):
        x, y = clean_batch
        x_adv = cw_l2(model, x, y, steps=10)
        assert x_adv.shape == x.shape


class TestGenerateDispatch:
    def test_unknown_method_raises(self, model, clean_batch):
        x, y = clean_batch
        with pytest.raises(ValueError, match="Unknown attack"):
            generate(model, x, y, method="nonexistent")

    def test_fgsm_dispatch(self, model, clean_batch):
        x, y = clean_batch
        x_adv = generate(model, x, y, method="fgsm", epsilon=0.03)
        assert x_adv.shape == x.shape


# ------------------------------------------------------------------
# Feature squeezing tests
# ------------------------------------------------------------------

class TestFeatureSqueezing:
    def test_score_shape(self, model, clean_batch):
        x, _ = clean_batch
        det = FeatureSqueezingDetector(model)
        scores = det.score(x)
        assert scores.shape == (x.shape[0],)

    def test_scores_non_negative(self, model, clean_batch):
        x, _ = clean_batch
        det = FeatureSqueezingDetector(model)
        scores = det.score(x)
        assert (scores >= 0).all()

    def test_predict_returns_bool(self, model, clean_batch):
        x, _ = clean_batch
        det = FeatureSqueezingDetector(model)
        flags, scores = det.predict(x)
        assert flags.dtype == torch.bool
        assert scores.shape == flags.shape

    def test_bit_reduce_range(self):
        x = torch.rand(2, 3, 16, 16)
        out = _bit_reduce(x, bits=4)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_smooth_shape_preserved(self):
        x = torch.rand(2, 3, 16, 16)
        out = _median_smooth(x, k=2)
        assert out.shape == x.shape

    def test_adversarial_scores_higher(self, model, clean_batch):
        x, y = clean_batch
        x_adv = pgd(model, x, y, epsilon=0.03, steps=10)
        det   = FeatureSqueezingDetector(model)
        s_clean = det.score(x).mean().item()
        s_adv   = det.score(x_adv).mean().item()
        # Mean adversarial score should be >= clean mean
        assert s_adv >= s_clean - 0.05   # small tolerance for random models


# ------------------------------------------------------------------
# Input transformation tests
# ------------------------------------------------------------------

class TestInputTransformation:
    def test_score_shape(self, model, clean_batch):
        x, _ = clean_batch
        det = InputTransformationDetector(model, n_transforms=5)
        scores = det.score(x)
        assert scores.shape == (x.shape[0],)

    def test_scores_in_range(self, model, clean_batch):
        x, _ = clean_batch
        det = InputTransformationDetector(model, n_transforms=5)
        scores = det.score(x)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()


# ------------------------------------------------------------------
# Statistical detector tests
# ------------------------------------------------------------------

class TestMahalanobis:
    def test_requires_calibration(self, model, clean_batch):
        x, _ = clean_batch
        det = MahalanobisDetector(model)
        with pytest.raises(RuntimeError, match="calibrate"):
            det.score(x)

    def test_score_shape(self, model, clean_batch):
        x, y = clean_batch
        det = MahalanobisDetector(model)
        det.calibrate(x, y, num_classes=10)
        scores = det.score(x)
        assert scores.shape == (x.shape[0],)

    def test_scores_non_negative(self, model, clean_batch):
        x, y = clean_batch
        det = MahalanobisDetector(model)
        det.calibrate(x, y, num_classes=10)
        scores = det.score(x)
        assert (scores >= 0).all()


class TestKDE:
    def test_requires_calibration(self, model, clean_batch):
        x, _ = clean_batch
        det = KDEDetector(model)
        with pytest.raises(RuntimeError, match="calibrate"):
            det.score(x)

    def test_score_shape(self, model, clean_batch):
        x, _ = clean_batch
        det = KDEDetector(model)
        det.calibrate(x)
        scores = det.score(x)
        assert scores.shape == (x.shape[0],)


# ------------------------------------------------------------------
# Unified scorer tests
# ------------------------------------------------------------------

class TestUnifiedScorer:
    def _make_scorer(self):
        return UnifiedScorer(
            detectors=[
                DetectorConfig("det_a", midpoint=0.5, steepness=10.0, weight=1.0),
                DetectorConfig("det_b", midpoint=0.5, steepness=10.0, weight=1.0),
            ],
            strategy="weighted",
            threshold=0.50,
        )

    def test_probability_range(self):
        scorer = self._make_scorer()
        raw = {"det_a": torch.rand(8), "det_b": torch.rand(8)}
        result = scorer.score(raw)
        assert (result.probability >= 0.0).all()
        assert (result.probability <= 1.0).all()

    def test_high_scores_flagged(self):
        scorer = self._make_scorer()
        raw = {"det_a": torch.ones(4) * 5.0, "det_b": torch.ones(4) * 5.0}
        result = scorer.score(raw)
        assert result.is_adversarial.all()

    def test_zero_scores_not_flagged(self):
        scorer = self._make_scorer()
        raw = {"det_a": torch.zeros(4), "det_b": torch.zeros(4)}
        result = scorer.score(raw)
        assert not result.is_adversarial.any()

    def test_evaluate_returns_metrics(self):
        scorer = self._make_scorer()
        clean_s = {"det_a": torch.zeros(8),   "det_b": torch.zeros(8)}
        adv_s   = {"det_a": torch.ones(8)*5., "det_b": torch.ones(8)*5.}
        metrics = scorer.evaluate(clean_s, adv_s)
        assert "tpr" in metrics and "fpr" in metrics
        assert "auroc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_max_strategy(self):
        scorer = UnifiedScorer(
            [DetectorConfig("a"), DetectorConfig("b")],
            strategy="max",
        )
        raw = {"a": torch.tensor([0.1, 0.9]), "b": torch.tensor([0.8, 0.1])}
        result = scorer.score(raw)
        assert result.probability.shape == (2,)

    def test_empty_scores_handled(self):
        scorer = self._make_scorer()
        raw = {"det_a": torch.rand(4)}
        result = scorer.score(raw)
        assert result.probability.shape == (4,)
