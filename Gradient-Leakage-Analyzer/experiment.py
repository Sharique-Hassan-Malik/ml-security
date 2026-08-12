"""
End-to-end gradient leakage experiment.

Demonstrates:
  1. A federated learning client computes gradients on a private image.
  2. An honest-but-curious server runs DLG / iDLG inversion to reconstruct
     the image from those gradients.
  3. Three defenses (DP, compression, noise) are applied to the gradients
     and the same attack is re-run to measure effectiveness.

Run:
    python experiment.py --iterations 300 --algorithm idlg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.inversion import AttackConfig, GradientInversionAttack, compute_observed_gradients
from defense.defenses import (
    DifferentialPrivacyDefense,
    GradientCompressionDefense,
    GradientNoiseDefense,
    gradient_cosine_similarity,
    gradient_snr,
)
from metrics.quality import reconstruction_quality
from visualize.report import generate_html_report


# ------------------------------------------------------------------
# Simple CNN for CIFAR-like images
# ------------------------------------------------------------------

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6,  5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = "cpu"

    # Build model
    model = LeNet5(num_classes=10)

    # Synthetic "private" image
    real_data   = torch.rand(1, 3, 32, 32)
    real_labels = torch.tensor([3])

    print("[*] Computing observed gradients (simulating federated client)…")
    grads = compute_observed_gradients(
        model, nn.CrossEntropyLoss(), real_data, real_labels
    )

    cfg = AttackConfig(
        iterations=args.iterations,
        algorithm=args.algorithm,
        seed=args.seed,
        total_variation=args.tv,
        lr=args.lr,
    )
    attacker = GradientInversionAttack(model, nn.CrossEntropyLoss(), cfg)

    # ---- Baseline attack (no defense) --------------------------------
    print(f"[*] Running {args.algorithm.upper()} attack ({args.iterations} iterations)…")
    result = attacker.attack(grads, data_shape=(3, 32, 32), device=device)
    baseline_metrics = reconstruction_quality(real_data[0], result.dummy_data[0])
    baseline_metrics["cosine_sim"] = 1.0
    baseline_metrics["snr_db"]     = float("inf")
    print(f"    PSNR={baseline_metrics['psnr']:.2f} dB  "
          f"MSE={baseline_metrics['mse']:.6f}")

    reconstructed = {"Attack (no defense)": result.dummy_data}
    all_metrics   = {"Attack (no defense)": baseline_metrics}
    all_losses    = {"Attack (no defense)": result.losses}

    # ---- Defense 1: Differential Privacy -----------------------------
    print("[*] Defense: Differential Privacy …")
    dp = DifferentialPrivacyDefense(
        max_grad_norm=args.dp_clip,
        noise_multiplier=args.dp_noise,
    )
    dp_grads, dp_meta = dp.apply(grads)
    dp_result = attacker.attack(dp_grads, data_shape=(3, 32, 32), device=device)
    dp_metrics = reconstruction_quality(real_data[0], dp_result.dummy_data[0])
    dp_metrics["cosine_sim"] = gradient_cosine_similarity(grads, dp_grads)
    dp_metrics["snr_db"]     = gradient_snr(grads, dp_grads)
    print(f"    PSNR={dp_metrics['psnr']:.2f} dB  "
          f"cos_sim={dp_metrics['cosine_sim']:.4f}  "
          f"SNR={dp_metrics['snr_db']:.1f} dB")
    reconstructed["DP Defense"] = dp_result.dummy_data
    all_metrics["DP Defense"]   = dp_metrics
    all_losses["DP Defense"]    = dp_result.losses

    # ---- Defense 2: Gradient Compression ----------------------------
    print("[*] Defense: Gradient Compression …")
    gc = GradientCompressionDefense(sparsity=args.compression)
    gc_grads, gc_meta = gc.apply(grads)
    gc_result = attacker.attack(gc_grads, data_shape=(3, 32, 32), device=device)
    gc_metrics = reconstruction_quality(real_data[0], gc_result.dummy_data[0])
    gc_metrics["cosine_sim"] = gradient_cosine_similarity(grads, gc_grads)
    gc_metrics["snr_db"]     = gradient_snr(grads, gc_grads)
    print(f"    PSNR={gc_metrics['psnr']:.2f} dB  "
          f"compression={gc_meta['compression_ratio']:.1%}  "
          f"SNR={gc_metrics['snr_db']:.1f} dB")
    reconstructed["Compression Defense"] = gc_result.dummy_data
    all_metrics["Compression Defense"]   = gc_metrics
    all_losses["Compression Defense"]    = gc_result.losses

    # ---- Defense 3: Gradient Noise ----------------------------------
    print("[*] Defense: Gradient Noise …")
    gn = GradientNoiseDefense(scale=args.noise_scale, noise_type="gaussian")
    gn_grads, _ = gn.apply(grads)
    gn_result   = attacker.attack(gn_grads, data_shape=(3, 32, 32), device=device)
    gn_metrics  = reconstruction_quality(real_data[0], gn_result.dummy_data[0])
    gn_metrics["cosine_sim"] = gradient_cosine_similarity(grads, gn_grads)
    gn_metrics["snr_db"]     = gradient_snr(grads, gn_grads)
    print(f"    PSNR={gn_metrics['psnr']:.2f} dB  "
          f"SNR={gn_metrics['snr_db']:.1f} dB")
    reconstructed["Noise Defense"] = gn_result.dummy_data
    all_metrics["Noise Defense"]   = gn_metrics
    all_losses["Noise Defense"]    = gn_result.losses

    # ---- HTML report -------------------------------------------------
    config_summary = {
        "algorithm":    args.algorithm,
        "iterations":   args.iterations,
        "lr":           args.lr,
        "tv_weight":    args.tv,
        "dp_clip":      args.dp_clip,
        "dp_noise":     args.dp_noise,
        "compression":  args.compression,
        "noise_scale":  args.noise_scale,
    }
    generate_html_report(
        real_image           = real_data[0],
        reconstructed_images = reconstructed,
        metrics              = all_metrics,
        losses               = all_losses,
        config               = config_summary,
        output_path          = args.output,
    )
    print(f"[*] Report → {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient leakage attack vs. defense experiment"
    )
    parser.add_argument("--iterations",  type=int,   default=300)
    parser.add_argument("--algorithm",   default="idlg", choices=["dlg", "idlg"])
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--tv",          type=float, default=1e-4,
                        help="Total variation regularisation weight")
    parser.add_argument("--dp-clip",     type=float, default=1.0,
                        dest="dp_clip",  help="DP-SGD gradient clip norm")
    parser.add_argument("--dp-noise",    type=float, default=1.0,
                        dest="dp_noise", help="DP-SGD noise multiplier")
    parser.add_argument("--compression", type=float, default=0.90,
                        help="Gradient compression sparsity (0–1)")
    parser.add_argument("--noise-scale", type=float, default=0.05,
                        dest="noise_scale", help="Additive noise scale")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output", "-o", default="report.html",
                        help="Output HTML report path")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
