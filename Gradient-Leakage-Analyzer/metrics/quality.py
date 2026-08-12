"""
Image quality metrics for evaluating reconstruction fidelity.

PSNR  — Peak Signal-to-Noise Ratio (higher is better; ∞ = perfect)
SSIM  — Structural Similarity Index (1.0 = identical)
MSE   — Mean Squared Error
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    PSNR between two image tensors of the same shape.

    Parameters
    ----------
    img1, img2 : Tensor in [0, max_val]
    max_val    : float  (1.0 for normalised images)
    """
    mse = ((img1.float() - img2.float()) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    SSIM for a pair of (C, H, W) or (1, C, H, W) tensors.

    Based on Wang et al. (2004) "Image Quality Assessment: From Error
    Visibility to Structural Similarity".
    """
    x = img1.float().unsqueeze(0) if img1.dim() == 3 else img1.float()
    y = img2.float().unsqueeze(0) if img2.dim() == 3 else img2.float()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size, sigma, x.shape[1]).to(x.device)

    mu_x  = F.conv2d(x, kernel, padding=window_size // 2, groups=x.shape[1])
    mu_y  = F.conv2d(y, kernel, padding=window_size // 2, groups=x.shape[1])
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2  = F.conv2d(x * x, kernel, padding=window_size // 2, groups=x.shape[1]) - mu_x2
    sigma_y2  = F.conv2d(y * y, kernel, padding=window_size // 2, groups=x.shape[1]) - mu_y2
    sigma_xy  = F.conv2d(x * y, kernel, padding=window_size // 2, groups=x.shape[1]) - mu_xy

    numerator   = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map    = numerator / denominator.clamp(min=1e-9)

    return ssim_map.mean().item()


def mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    return ((img1.float() - img2.float()) ** 2).mean().item()


def reconstruction_quality(
    real: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict:
    """Return a dict with PSNR, SSIM, and MSE for a reconstruction pair."""
    r = real.detach().cpu().clamp(0, 1)
    r_hat = reconstructed.detach().cpu().clamp(0, 1)

    return {
        "psnr": psnr(r, r_hat),
        "ssim": ssim(r, r_hat) if r.shape[-1] >= 11 else None,
        "mse":  mse(r, r_hat),
    }


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _gaussian_kernel(size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d /= g1d.sum()
    g2d = g1d.outer(g1d)
    kernel = g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
    return kernel.contiguous()
