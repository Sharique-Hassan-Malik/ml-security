"""
HTML report generator for gradient leakage experiments.

Produces a self-contained HTML file showing:
  - Original image vs reconstruction
  - Per-defense quality metrics (PSNR, SSIM, cosine similarity, SNR)
  - Loss curve as an inline SVG chart
"""

from __future__ import annotations

import base64
import io
import json
from typing import Dict, List, Optional

import torch


def tensor_to_png_b64(img: torch.Tensor) -> str:
    """Convert a (C, H, W) or (1, C, H, W) tensor to a base64-encoded PNG string."""
    try:
        from PIL import Image
        import numpy as np

        t = img.detach().cpu().float()
        if t.dim() == 4:
            t = t.squeeze(0)
        t = t.clamp(0, 1)

        # Convert (C, H, W) → (H, W, C)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

        pil = Image.fromarray(arr)
        # Scale up small images for visibility
        if pil.width < 128:
            scale = 128 // pil.width + 1
            pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # If Pillow is unavailable, return empty-image placeholder
        return ""


def _svg_loss_curve(losses: List[float], width: int = 400, height: int = 120) -> str:
    if len(losses) < 2:
        return ""
    min_l = min(losses)
    max_l = max(losses)
    span  = max(max_l - min_l, 1e-9)
    n     = len(losses)

    def px(i):
        return int(i / (n - 1) * (width - 40)) + 20

    def py(v):
        return int((1 - (v - min_l) / span) * (height - 20)) + 10

    pts = " ".join(f"{px(i)},{py(v)}" for i, v in enumerate(losses))

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#1a1a1a" rx="4"/>'
        f'<polyline points="{pts}" fill="none" stroke="#3b82f6" stroke-width="1.5"/>'
        f'<text x="20" y="{height - 2}" fill="#666" font-size="9">'
        f'0</text>'
        f'<text x="{width - 30}" y="{height - 2}" fill="#666" font-size="9">'
        f'{n}</text>'
        f'<text x="2" y="14" fill="#666" font-size="9">{max_l:.3f}</text>'
        f'<text x="2" y="{height - 10}" fill="#666" font-size="9">{min_l:.3f}</text>'
        f'</svg>'
    )


def generate_html_report(
    real_image: Optional[torch.Tensor],
    reconstructed_images: Dict[str, torch.Tensor],
    metrics: Dict[str, dict],
    losses: Dict[str, List[float]],
    config: Optional[dict] = None,
    output_path: str = "report.html",
) -> None:
    """
    Write a self-contained HTML report.

    Parameters
    ----------
    real_image : Tensor (C, H, W)
    reconstructed_images : {label: Tensor}
        Keys: "attack", "dp_defense", "compression_defense", "noise_defense", …
    metrics : {label: {psnr, ssim, mse, cosine_sim, snr_db}}
    losses : {label: [float]}
    config : optional dict of experiment settings
    """
    cards = []

    def make_card(label: str, tensor: Optional[torch.Tensor], m: dict, loss: List[float]) -> str:
        img_html = ""
        if tensor is not None:
            b64 = tensor_to_png_b64(tensor)
            if b64:
                img_html = f'<img src="data:image/png;base64,{b64}" style="image-rendering:pixelated;width:128px;height:128px">'

        psnr_val = m.get("psnr")
        ssim_val = m.get("ssim")
        mse_val  = m.get("mse")
        cos_val  = m.get("cosine_sim")
        snr_val  = m.get("snr_db")

        psnr_str = f"{psnr_val:.2f} dB" if psnr_val is not None and psnr_val != float("inf") else ("∞" if psnr_val == float("inf") else "—")
        ssim_str = f"{ssim_val:.4f}" if ssim_val is not None else "—"
        mse_str  = f"{mse_val:.6f}" if mse_val  is not None else "—"
        cos_str  = f"{cos_val:.4f}"  if cos_val  is not None else "—"
        snr_str  = f"{snr_val:.1f} dB" if snr_val is not None else "—"

        loss_svg = _svg_loss_curve(loss)

        return f"""
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:16px;min-width:220px">
          <div style="font-weight:600;font-size:.9rem;margin-bottom:10px;color:#e0e0e0">{label}</div>
          <div style="display:flex;gap:12px;align-items:flex-start">
            <div>{img_html}</div>
            <table style="font-size:.75rem;border-collapse:collapse;color:#ccc">
              <tr><td style="color:#888;padding:1px 8px 1px 0">PSNR</td><td>{psnr_str}</td></tr>
              <tr><td style="color:#888;padding:1px 8px 1px 0">SSIM</td><td>{ssim_str}</td></tr>
              <tr><td style="color:#888;padding:1px 8px 1px 0">MSE</td><td>{mse_str}</td></tr>
              <tr><td style="color:#888;padding:1px 8px 1px 0">Cos sim</td><td>{cos_str}</td></tr>
              <tr><td style="color:#888;padding:1px 8px 1px 0">Grad SNR</td><td>{snr_str}</td></tr>
            </table>
          </div>
          {f'<div style="margin-top:10px"><div style="font-size:.7rem;color:#666;margin-bottom:3px">Loss curve</div>{loss_svg}</div>' if loss_svg else ""}
        </div>"""

    if real_image is not None:
        b64 = tensor_to_png_b64(real_image)
        if b64:
            cards.append(f"""
        <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:16px;min-width:220px">
          <div style="font-weight:600;font-size:.9rem;margin-bottom:10px;color:#e0e0e0">Original (private)</div>
          <img src="data:image/png;base64,{b64}" style="image-rendering:pixelated;width:128px;height:128px">
        </div>""")

    for label in reconstructed_images:
        tensor = reconstructed_images[label]
        m      = metrics.get(label, {})
        loss   = losses.get(label, [])
        cards.append(make_card(label, tensor, m, loss))

    config_html = ""
    if config:
        config_html = (
            '<details style="margin-top:24px"><summary style="color:#888;cursor:pointer">Config</summary>'
            f'<pre style="background:#111;padding:12px;border-radius:4px;font-size:.75rem;color:#aaa;margin-top:8px">'
            f'{json.dumps(config, indent=2)}</pre></details>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gradient Leakage Experiment Report</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 24px; }}
  h1   {{ font-size: 1.3rem; color: #fff; margin-bottom: 4px; }}
  .sub {{ color: #888; font-size: .85rem; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
</style>
</head>
<body>
<h1>Gradient Leakage Analyzer — Experiment Report</h1>
<p class="sub">Federated learning gradient inversion — attack vs. defense comparison</p>
<div class="cards">{''.join(cards)}</div>
{config_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
