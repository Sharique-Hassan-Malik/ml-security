"""weight_poisoning — backdoor and trojan indicators in checkpoint tensors.

    from weight_poisoning import scan
    result = scan("model.pt")          # an aisec.core.ModuleResult

Four analysers look at the same weights from different angles — distribution
statistics, individual neuron behaviour, spectral structure, and cross-layer
trojan patterns — because no single one of them is decisive on its own.
"""

from .finding import Finding, Kind, ModuleResult, Severity, poisoning_score
from .loader import load_weights
from .neuron_inspector import NeuronInspector
from .spectral_analyzer import SpectralAnalyzer
from .trojan_detector import TrojanDetector
from .weight_analyzer import WeightDistributionAnalyzer

MODULE_NAME = "weight-poisoning"

__all__ = [
    "scan", "MODULE_NAME", "load_weights", "poisoning_score",
    "Finding", "ModuleResult", "Severity",
    "NeuronInspector", "SpectralAnalyzer", "TrojanDetector",
    "WeightDistributionAnalyzer",
]


def scan(model_path: str) -> ModuleResult:
    """Run all four analysers over *model_path*.

    Returns a `ModuleResult` — the same type the pickle scanner returns — so a
    checkpoint can be scanned for dangerous opcodes and poisoned weights in one
    pass and reported as one thing.
    """
    weights, meta = load_weights(model_path)

    result = ModuleResult(module=MODULE_NAME, kind=Kind.SCANNER, target=str(model_path))
    result.metrics["layers"] = meta["layer_count"]
    result.metrics["parameters"] = meta["param_count"]

    distribution = WeightDistributionAnalyzer()
    neurons = NeuronInspector()
    spectral = SpectralAnalyzer()
    trojans = TrojanDetector()

    for name, tensor in weights.items():
        result.extend(distribution.analyze(name, tensor))
        result.extend(neurons.analyze(name, tensor))
        result.extend(spectral.analyze(name, tensor))

    # Trojan detection is cross-layer: it needs every tensor at once, so it
    # runs after the per-layer pass rather than inside it.
    result.extend(trojans.analyze(weights))

    result.metrics["poisoning_score"] = round(
        poisoning_score(result.findings, meta["layer_count"]), 4
    )
    return result
