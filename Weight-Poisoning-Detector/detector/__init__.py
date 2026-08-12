"""
detector — Neural Network Weight Poisoning Detector package.

Public API
----------
scan(model_path) -> Report
    Full pipeline: load → analyse → report.
"""

from .loader          import load_weights
from .neuron_inspector import NeuronInspector
from .report          import Report
from .spectral_analyzer import SpectralAnalyzer
from .trojan_detector  import TrojanDetector
from .weight_analyzer  import WeightDistributionAnalyzer


def scan(model_path: str) -> Report:
    """
    Run all detectors on *model_path* and return a populated Report.

    Parameters
    ----------
    model_path : str
        Path to a .pt or .pth checkpoint file.

    Returns
    -------
    Report
        Contains all Finding objects and the aggregate verdict.
    """
    weights, meta = load_weights(model_path)

    report = Report(model_path)
    report.layer_count = meta["layer_count"]
    report.param_count = meta["param_count"]

    wa  = WeightDistributionAnalyzer()
    ni  = NeuronInspector()
    sa  = SpectralAnalyzer()
    td  = TrojanDetector()

    for name, tensor in weights.items():
        for finding in wa.analyze(name, tensor):
            report.add(finding)
        for finding in ni.analyze(name, tensor):
            report.add(finding)
        for finding in sa.analyze(name, tensor):
            report.add(finding)

    for finding in td.analyze(weights):
        report.add(finding)

    return report


__all__ = ["scan", "Report"]
