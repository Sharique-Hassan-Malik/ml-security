"""Turning reconstructed tensors into something a person can look at.

A leakage report without the picture is an argument about numbers. PSNR 22 dB
means nothing to most readers; the recovered image next to the original is the
finding.
"""

from __future__ import annotations

import io

import torch


def tensor_to_png(img: torch.Tensor, min_width: int = 128) -> bytes:
    """Encode a (C, H, W) or (1, C, H, W) tensor in [0, 1] as PNG bytes.

    Small images are nearest-neighbour upscaled: a 32×32 reconstruction shown
    at 32 px tells the reader nothing, and smoothing it would flatter the
    attack.
    """
    from PIL import Image  # imported lazily — the metrics work without it

    tensor = img.detach().cpu().float()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)

    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    if array.shape[2] == 1:
        array = array[:, :, 0]

    image = Image.fromarray(array)
    if image.width < min_width:
        scale = min_width // image.width + 1
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True
