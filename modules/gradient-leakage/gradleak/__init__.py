"""gradleak — reconstructing training data from shared gradients.

    from gradleak import run_leakage
    real, runs = run_leakage(iterations=200)

The point is not that inversion works — it does, on small batches. The point is
which defences actually stop it, measured the same way for each: run the same
attack against the defended gradients and see what comes back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .defenses import (
    DifferentialPrivacyDefense,
    GradientCompressionDefense,
    GradientNoiseDefense,
    gradient_cosine_similarity,
    gradient_snr,
)
from .inversion import (
    AttackConfig,
    AttackResult,
    GradientInversionAttack,
    compute_observed_gradients,
)
from .quality import mse, psnr, reconstruction_quality, ssim
from .rgap import AttackNotApplicable, RGAPAttack

MODULE_NAME = "gradient-leakage"

__all__ = [
    "MODULE_NAME", "LeNet5", "DefenceRun", "run_leakage",
    "AttackConfig", "AttackResult", "GradientInversionAttack",
    "compute_observed_gradients", "RGAPAttack", "AttackNotApplicable",
    "DifferentialPrivacyDefense", "GradientCompressionDefense",
    "GradientNoiseDefense", "gradient_cosine_similarity", "gradient_snr",
    "psnr", "ssim", "mse", "reconstruction_quality",
]


class LeNet5(nn.Module):
    """The standard victim for this literature — small enough that inversion
    converges in seconds, which is why every DLG paper uses something like it."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@dataclass
class DefenceRun:
    """One attack, against one gradient-protection setting."""

    name: str
    reconstruction: torch.Tensor
    psnr: float
    mse: float
    ssim: float | None = None
    cosine_sim: float = 1.0
    snr_db: float = float("inf")
    losses: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def utility_cost(self) -> float:
        """How far the defended gradient drifted from the true one.

        A defence that stops the attack by destroying the gradient has not
        solved anything, so this is reported beside the PSNR, never instead
        of it.
        """
        return 1.0 - self.cosine_sim


DEFENCES = ("none", "dp", "compression", "noise")


def run_leakage(
    *,
    iterations: int = 300,
    algorithm: str = "idlg",
    seed: int = 42,
    lr: float = 0.10,
    total_variation: float = 1e-4,
    dp_clip: float = 1.0,
    dp_noise: float = 0.01,
    compression: float = 0.90,
    noise_scale: float = 0.01,
    defences: tuple[str, ...] = DEFENCES,
    model: nn.Module | None = None,
    data: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[DefenceRun]]:
    """Attack once per defence and return the reconstructions side by side."""
    torch.manual_seed(seed)

    model = model if model is not None else LeNet5(num_classes=10)
    real_data = data if data is not None else torch.rand(1, 3, 32, 32)
    real_labels = labels if labels is not None else torch.tensor([3])

    criterion = nn.CrossEntropyLoss()
    true_grads = compute_observed_gradients(model, criterion, real_data, real_labels)

    config = AttackConfig(
        iterations=iterations,
        algorithm=algorithm,
        seed=seed,
        total_variation=total_variation,
        lr=lr,
    )
    attacker = GradientInversionAttack(model, criterion, config)
    shape = tuple(real_data.shape[1:])

    builders = {
        "none": lambda: (true_grads, {}),
        "dp": lambda: DifferentialPrivacyDefense(
            max_grad_norm=dp_clip, noise_multiplier=dp_noise
        ).apply(true_grads),
        "compression": lambda: GradientCompressionDefense(sparsity=compression).apply(true_grads),
        "noise": lambda: GradientNoiseDefense(scale=noise_scale, noise_type="gaussian").apply(true_grads),
    }
    labels_for = {
        "none": "no defence",
        "dp": "differential privacy",
        "compression": "gradient compression",
        "noise": "gradient noise",
    }

    runs: list[DefenceRun] = []
    for key in defences:
        if key not in builders:
            raise ValueError(f"unknown defence {key!r}; choose from {list(builders)}")
        grads, meta = builders[key]()
        result = attacker.attack(grads, data_shape=shape, device="cpu")
        quality = reconstruction_quality(real_data[0], result.dummy_data[0])
        runs.append(
            DefenceRun(
                name=labels_for[key],
                reconstruction=result.dummy_data,
                psnr=quality["psnr"],
                ssim=quality["ssim"],
                mse=quality["mse"],
                cosine_sim=1.0 if key == "none" else gradient_cosine_similarity(true_grads, grads),
                snr_db=float("inf") if key == "none" else gradient_snr(true_grads, grads),
                losses=result.losses,
                meta=dict(meta) if isinstance(meta, dict) else {},
            )
        )

    return real_data[0], runs
