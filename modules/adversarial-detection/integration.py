"""Joins adversarial-detection to the suite as a runtime guard.

This guard cannot work without the model it is protecting — feature squeezing
compares the model's own predictions under distortion, and there is nothing to
compare without it. So `model=` is required and its absence is reported as a
skip with the reason, never as a clean result.

Loading a model file means unpickling it, which is exactly the thing the pickle
scanner in this same suite exists to warn about. So it is consulted first, and
a checkpoint it rates HIGH or worse is refused rather than loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from advdet import build_detectors, build_scorer, score_batch  # noqa: E402
from aisec.core.finding import Finding, ModuleResult, Severity  # noqa: E402
from aisec.core.module import Guard  # noqa: E402
from aisec.core.registry import module_path, spec  # noqa: E402

_MAX_LISTED = 20


class UnsafeCheckpoint(RuntimeError):
    """The checkpoint was refused before it could be deserialised."""


def _screen(path: Path) -> None:
    """Refuse to unpickle what the suite's own scanner calls dangerous.

    Degrades to a plain load if the pickle-scanner module is not present —
    this guard should not become unusable because a sibling was removed.
    """
    scanner_dir = module_path("pickle-scanner")
    if not scanner_dir.is_dir():
        return
    if str(scanner_dir) not in sys.path:
        sys.path.insert(0, str(scanner_dir))
    try:
        from pickle_scanner import scan_file
    except ImportError:
        return

    worst = max(
        (r.max_severity for r in scan_file(path)),
        default=Severity.SAFE,
    )
    if worst >= Severity.HIGH:
        raise UnsafeCheckpoint(
            f"pickle-scanner rated {path.name} {worst.value}; refusing to load it. "
            f"Inspect it with `aisec scan {path}` first."
        )


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    path = Path(str(value))
    _screen(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        loaded = next(iter(loaded.values()))
    return loaded


def _as_model(value: Any) -> nn.Module:
    if isinstance(value, nn.Module):
        return value
    path = Path(str(value))
    _screen(path)
    try:
        model = torch.jit.load(str(path), map_location="cpu")
    except (RuntimeError, ValueError):
        model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{path} holds {type(model).__name__}, not a model. A state_dict "
            f"cannot be used on its own — pass the constructed nn.Module."
        )
    return model


def _severity(probability: float) -> Severity:
    if probability >= 0.80:
        return Severity.HIGH
    if probability >= 0.50:
        return Severity.MEDIUM
    return Severity.LOW


class AdversarialDetectionModule(Guard):
    def inspect(self, payload: Any, **options: Any) -> ModuleResult:
        result = self.result(str(options.get("label", "<batch>")))

        if not options.get("model"):
            result.skipped = (
                "needs the model it protects — pass model=<nn.Module or path>"
            )
            return result

        model = _as_model(options["model"])
        x = _as_tensor(payload)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        calibration = options.get("calibration")
        labels = options.get("labels")
        detectors = build_detectors(
            model,
            _as_tensor(calibration) if calibration is not None else None,
            _as_tensor(labels) if labels is not None else None,
            num_classes=int(options.get("num_classes", 10)),
        )
        scorer = build_scorer(threshold=float(options.get("threshold", 0.50)))
        scored = scorer.score(score_batch(detectors, x))

        flagged = int(scored.is_adversarial.sum().item())
        probabilities = scored.probability.tolist()

        for index, probability in enumerate(probabilities):
            if not scored.is_adversarial[index]:
                continue
            if len(result.findings) >= _MAX_LISTED:
                break
            contributions = {
                name: round(float(values[index]), 4)
                for name, values in scored.per_detector.items()
            }
            top = max(contributions, key=contributions.get) if contributions else ""
            result.add(
                Finding(
                    title="adversarial_input",
                    severity=_severity(probability),
                    summary=f"p(adversarial)={probability:.3f}"
                            + (f", strongest signal {top}" if top else ""),
                    location=f"sample {index}",
                    score=round(probability, 4),
                    metadata={"per_detector": contributions},
                )
            )

        result.metrics["samples"] = int(x.shape[0])
        result.metrics["flagged"] = flagged
        result.metrics["threshold"] = scorer.threshold
        result.metrics["detectors"] = ", ".join(sorted(detectors))
        if probabilities:
            result.metrics["mean_probability"] = round(
                sum(probabilities) / len(probabilities), 4
            )
        if flagged > _MAX_LISTED:
            result.metrics["listed"] = f"{_MAX_LISTED} of {flagged}"
        if "mahalanobis" not in detectors:
            result.metrics["note"] = (
                "density detectors omitted — no calibration data supplied"
            )
        return result


MODULE = AdversarialDetectionModule(spec("adversarial-detection"))
