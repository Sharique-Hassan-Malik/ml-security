"""
Unit and integration tests for the weight poisoning detector.

Run with:  pytest tests/ -v
"""

import json
from pathlib import Path

import pytest
import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------
# Fixtures (generate on first run if missing)
# ------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def generate_fixtures():
    """Ensure .pt fixture files exist before any test runs."""
    if not (FIXTURE_DIR / "clean_model.pt").exists():
        from tests.generate_fixtures import main
        main()


@pytest.fixture()
def clean_path():
    return str(FIXTURE_DIR / "clean_model.pt")


@pytest.fixture()
def poisoned_path():
    return str(FIXTURE_DIR / "poisoned_model.pt")


# ------------------------------------------------------------------
# Loader tests
# ------------------------------------------------------------------

class TestLoader:
    def test_load_clean(self, clean_path):
        from detector.loader import load_weights
        weights, meta = load_weights(clean_path)
        assert len(weights) > 0
        assert meta["param_count"] > 0
        assert all(t.dim() >= 2 for t in weights.values())

    def test_file_not_found(self):
        from detector.loader import load_weights
        with pytest.raises(FileNotFoundError):
            load_weights("/tmp/nonexistent_model.pt")

    def test_bad_extension(self, tmp_path):
        from detector.loader import load_weights
        p = tmp_path / "model.pkl"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="Expected .pt or .pth"):
            load_weights(str(p))

    def test_no_bias_tensors(self, clean_path):
        from detector.loader import load_weights
        weights, _ = load_weights(clean_path)
        for name in weights:
            assert "bias" not in name.lower()


# ------------------------------------------------------------------
# Weight distribution analyzer tests
# ------------------------------------------------------------------

class TestWeightAnalyzer:
    def test_normal_tensor_no_findings(self):
        from detector.weight_analyzer import WeightDistributionAnalyzer
        wa = WeightDistributionAnalyzer()
        t = torch.randn(64, 128)        # clean gaussian weights
        findings = wa.analyze("fc.weight", t)
        # Should produce no high-severity findings
        high = [f for f in findings if f.severity == "high"]
        assert len(high) == 0

    def test_high_kurtosis_detected(self):
        from detector.weight_analyzer import WeightDistributionAnalyzer
        wa = WeightDistributionAnalyzer()
        # Laplace distribution has excess kurtosis = 3 — inject stronger outliers
        t = torch.randn(128, 128)
        t[0, :5] = 50.0    # extreme outliers → very high kurtosis
        findings = wa.analyze("layer.weight", t)
        tests = {f.test for f in findings}
        assert "outlier_weights" in tests or "high_kurtosis" in tests

    def test_bimodal_detected(self):
        from detector.weight_analyzer import WeightDistributionAnalyzer
        wa = WeightDistributionAnalyzer()
        # Two well-separated Gaussian modes
        a = torch.randn(32, 64) * 0.1
        b = torch.randn(32, 64) * 0.1 + 1.5
        t = torch.cat([a, b], dim=0)
        findings = wa.analyze("fc.weight", t)
        tests = {f.test for f in findings}
        assert "bimodal_distribution" in tests or "high_skewness" in tests


# ------------------------------------------------------------------
# Neuron inspector tests
# ------------------------------------------------------------------

class TestNeuronInspector:
    def test_normal_norms_no_findings(self):
        from detector.neuron_inspector import NeuronInspector
        ni = NeuronInspector()
        t = torch.randn(64, 256)
        findings = ni.analyze("fc.weight", t)
        high = [f for f in findings if f.severity == "high"]
        assert len(high) == 0

    def test_dominant_neurons_detected(self):
        from detector.neuron_inspector import NeuronInspector
        ni = NeuronInspector()
        t = torch.randn(64, 128) * 0.05
        t[5]  = t[5]  * 80    # dominant neuron
        t[20] = t[20] * 80
        findings = ni.analyze("fc.weight", t)
        tests = {f.test for f in findings}
        assert "dominant_neurons" in tests

    def test_dormant_neurons_detected(self):
        from detector.neuron_inspector import NeuronInspector
        ni = NeuronInspector()
        t = torch.randn(64, 128) * 0.05
        t[:40] = 0.0           # 62.5% dormant
        findings = ni.analyze("fc.weight", t)
        tests = {f.test for f in findings}
        assert "dormant_neurons" in tests


# ------------------------------------------------------------------
# Spectral analyzer tests
# ------------------------------------------------------------------

class TestSpectralAnalyzer:
    def test_full_rank_matrix_no_high(self):
        from detector.spectral_analyzer import SpectralAnalyzer
        sa = SpectralAnalyzer()
        t = torch.randn(32, 64)
        findings = sa.analyze("layer.weight", t)
        high = [f for f in findings if f.severity == "high"]
        assert len(high) == 0

    def test_spectral_gap_detected(self):
        from detector.spectral_analyzer import SpectralAnalyzer
        sa = SpectralAnalyzer()
        # Inject dominant singular direction
        u = torch.randn(64, 1)
        v = torch.randn(1, 128)
        u /= u.norm();  v /= v.norm()
        t = torch.randn(64, 128) * 0.1 + 20.0 * (u @ v)
        findings = sa.analyze("layer.weight", t)
        tests = {f.test for f in findings}
        assert "spectral_gap" in tests or "sv_outlier" in tests


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------

class TestIntegration:
    def test_clean_model_verdict(self, clean_path):
        from detector import scan
        report = scan(clean_path)
        # Clean model should not be HIGH_RISK
        assert report.verdict != "HIGH_RISK"

    def test_poisoned_model_detected(self, poisoned_path):
        from detector import scan
        report = scan(poisoned_path)
        # Poisoned model must not be scored as CLEAN
        assert report.verdict != "CLEAN", (
            f"Poisoned model scored CLEAN (score={report.overall_score:.4f})"
        )

    def test_poisoned_has_high_findings(self, poisoned_path):
        from detector import scan
        report = scan(poisoned_path)
        high = [f for f in report.findings if f.severity == "high"]
        assert len(high) >= 1, "Expected at least one high-severity finding"

    def test_report_json_valid(self, poisoned_path):
        from detector import scan
        report = scan(poisoned_path)
        data = json.loads(report.to_json())
        assert "verdict" in data
        assert "findings" in data
        assert isinstance(data["param_count"], int)

    def test_report_html_contains_verdict(self, poisoned_path):
        from detector import scan
        report = scan(poisoned_path)
        html = report.to_html()
        assert report.verdict in html
        assert "<table>" in html

    def test_scores_in_range(self, poisoned_path):
        from detector import scan
        report = scan(poisoned_path)
        for f in report.findings:
            assert 0.0 <= f.score <= 1.0, f"Score out of range: {f.score}"
            assert f.severity in ("low", "medium", "high")
