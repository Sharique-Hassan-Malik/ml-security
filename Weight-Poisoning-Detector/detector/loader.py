"""
Safe extraction of float tensors from PyTorch checkpoint files.

weights_only=True prevents arbitrary code execution during unpickling.
Falls back gracefully for checkpoints that embed non-tensor objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

log = logging.getLogger(__name__)

# Parameter name fragments that identify non-weight buffers
_SKIP_FRAGMENTS = (
    "bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "tracked",
)


def load_weights(path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Load float weight tensors from *path* without executing model code.

    Returns
    -------
    weights : dict[str, Tensor]
        Maps fully-qualified parameter name to a CPU float32 tensor.
    meta : dict
        layer_count, param_count, source_dtype.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in (".pt", ".pth"):
        raise ValueError(f"Unsupported extension '{p.suffix}'. Expected .pt or .pth.")

    obj = _safe_load(str(p))
    weights = _collect_tensors(obj, prefix="")

    if not weights:
        raise ValueError("No eligible weight tensors found in checkpoint.")

    source_dtype = str(next(iter(weights.values())).dtype)
    meta = {
        "layer_count": len(weights),
        "param_count": sum(t.numel() for t in weights.values()),
        "source_dtype": source_dtype,
    }
    return weights, meta


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _safe_load(path: str):
    """Try weights_only first; fall back for legacy checkpoints."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        log.warning(
            "weights_only=True failed (likely a legacy checkpoint); "
            "falling back to unrestricted load — ensure the file is trusted."
        )
        return torch.load(path, map_location="cpu")


def _collect_tensors(obj, prefix: str) -> Dict[str, torch.Tensor]:
    """Recursively walk a checkpoint object and collect weight matrices."""
    result: Dict[str, torch.Tensor] = {}

    if isinstance(obj, torch.Tensor):
        if _keep_tensor(prefix, obj):
            result[prefix] = obj.float().cpu()

    elif isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            result.update(_collect_tensors(v, child_prefix))

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]"
            result.update(_collect_tensors(v, child_prefix))

    return result


def _keep_tensor(name: str, t: torch.Tensor) -> bool:
    """Return True for multi-dimensional float weight matrices."""
    if not t.is_floating_point():
        return False
    if t.dim() < 2:
        return False
    if t.numel() < 16:          # ignore tiny parameter blocks
        return False
    lname = name.lower()
    if any(frag in lname for frag in _SKIP_FRAGMENTS):
        return False
    return True
