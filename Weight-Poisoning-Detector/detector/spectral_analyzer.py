"""
Spectral analysis of weight matrices via Singular Value Decomposition.

Tran et al. (2018) "Spectral Signatures in Backdoor Attacks" showed that
backdoor attacks leave detectable signatures in the singular value spectrum:

  - A dominant first singular value relative to the rest (large spectral gap)
    encodes the direction in representation space that all triggered inputs
    activate strongly.

  - Low stable rank: the effective dimensionality of the weight matrix drops
    when a small number of neurons encode a high-energy shortcut.

  - Top singular vector concentration: the leading right singular vector
    concentrates energy on a small fraction of input dimensions
    (the trigger feature dimensions).
"""

from __future__ import annotations

import math
from typing import List

import torch

from .report import Finding


_MAX_SVD_PARAMS  = 4_000_000   # skip extremely large layers (too slow)
_MIN_MATRIX_DIM  = 8           # minimum matrix dimension for SVD
_SPECTRAL_GAP    = 4.0         # σ₁/σ₂ threshold
_STABLE_RANK_LOW = 0.08        # stable_rank / min(m,n) threshold
_SV_OUTLIER_Z    = 4.0         # z-score for top singular value
_CONCENTRATION   = 0.30        # top-k energy fraction threshold for singular vectors


class SpectralAnalyzer:
    """Compute SVD of each weight matrix and check for backdoor spectral signatures."""

    def analyze(self, name: str, tensor: torch.Tensor) -> List[Finding]:
        m = tensor.detach().float()
        m = m.reshape(m.shape[0], -1)          # (out, in*kH*kW)

        rows, cols = m.shape
        if rows < _MIN_MATRIX_DIM or cols < _MIN_MATRIX_DIM:
            return []
        if m.numel() > _MAX_SVD_PARAMS:
            return []

        try:
            sv = torch.linalg.svdvals(m)
        except Exception:
            return []

        if sv.numel() < 2:
            return []

        findings: List[Finding] = []
        min_dim = min(rows, cols)

        # ---- spectral gap  σ₁ / σ₂ -----------------------------------
        gap = (sv[0] / sv[1]).item() if sv[1].item() > 1e-9 else float("inf")
        if gap > _SPECTRAL_GAP:
            sev = "high" if gap > 2 * _SPECTRAL_GAP else "medium"
            findings.append(Finding(
                layer    = name,
                test     = "spectral_gap",
                severity = sev,
                score    = min(1.0, (gap - _SPECTRAL_GAP) / (3 * _SPECTRAL_GAP)),
                detail   = (
                    f"σ₁/σ₂ = {gap:.3f} (threshold {_SPECTRAL_GAP}). "
                    "A dominant first singular value indicates a single high-energy "
                    "direction that all inputs—including triggered ones—project onto "
                    "strongly. This is the primary spectral backdoor signature."
                ),
                metadata = {
                    "sigma_1": round(sv[0].item(), 6),
                    "sigma_2": round(sv[1].item(), 6),
                    "gap":     round(gap, 4),
                },
            ))

        # ---- stable rank  ‖M‖_F² / σ₁² --------------------------------
        frob_sq     = (sv ** 2).sum().item()
        stable_rank = frob_sq / max((sv[0] ** 2).item(), 1e-12)
        sr_norm     = stable_rank / min_dim       # normalise to [0, 1]
        if sr_norm < _STABLE_RANK_LOW:
            findings.append(Finding(
                layer    = name,
                test     = "low_stable_rank",
                severity = "medium",
                score    = min(1.0, (_STABLE_RANK_LOW - sr_norm) / _STABLE_RANK_LOW),
                detail   = (
                    f"Stable rank {stable_rank:.2f} / min_dim {min_dim} = {sr_norm:.4f} "
                    f"(threshold {_STABLE_RANK_LOW}). "
                    "Abnormally low-rank weight matrices suggest that most representational "
                    "capacity is concentrated in very few directions — consistent with "
                    "a hardcoded trigger-response pathway."
                ),
                metadata = {
                    "stable_rank":    round(stable_rank, 4),
                    "min_dim":        min_dim,
                    "stable_rank_norm": round(sr_norm, 6),
                },
            ))

        # ---- top singular value outlier (z-score across all SVs) ------
        sv_mean = sv.mean().item()
        sv_std  = sv.std(unbiased=True).item()
        if sv_std > 1e-9:
            z1 = (sv[0].item() - sv_mean) / sv_std
            if z1 > _SV_OUTLIER_Z:
                sev = "high" if z1 > 2 * _SV_OUTLIER_Z else "medium"
                findings.append(Finding(
                    layer    = name,
                    test     = "sv_outlier",
                    severity = sev,
                    score    = min(1.0, z1 / (3 * _SV_OUTLIER_Z)),
                    detail   = (
                        f"Top singular value z-score = {z1:.2f} "
                        f"(σ₁ = {sv[0].item():.4f}, mean = {sv_mean:.4f}, "
                        f"std = {sv_std:.4f}). "
                        "An outlier singular value indicates that a small subset of "
                        "directions carries disproportionate energy — a sign of "
                        "deliberately planted high-gain pathways."
                    ),
                    metadata = {
                        "z_score":  round(z1, 4),
                        "sv_mean":  round(sv_mean, 6),
                        "sv_std":   round(sv_std, 6),
                    },
                ))

        # ---- top right singular vector concentration ------------------
        # Compute the leading singular vectors explicitly
        if rows * cols <= 500_000:
            try:
                _, _, Vt = torch.linalg.svd(m, full_matrices=False)
                v1 = Vt[0].abs()
                # Fraction of energy in top-k=ceil(10% of dims) components
                k  = max(1, math.ceil(0.10 * cols))
                topk_energy = v1.topk(k).values.pow(2).sum().item()
                total_energy = v1.pow(2).sum().item()
                conc = topk_energy / max(total_energy, 1e-12)
                if conc > _CONCENTRATION:
                    findings.append(Finding(
                        layer    = name,
                        test     = "sv1_concentration",
                        severity = "low",
                        score    = min(0.5, conc),
                        detail   = (
                            f"Top-10% of input dimensions account for {conc * 100:.1f}% "
                            "of leading right singular vector energy. "
                            "Concentrated singular vectors can point to a small set of "
                            "input features (e.g., trigger pixels) driving layer output."
                        ),
                        metadata = {"concentration": round(conc, 4), "k": k},
                    ))
            except Exception:
                pass

        return findings
