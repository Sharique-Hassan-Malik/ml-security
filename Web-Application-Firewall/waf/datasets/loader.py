from __future__ import annotations

from pathlib import Path

from . import csic, synthetic


def load(
    source:       str  = "synthetic",
    data_dir:     str  = "data",
    n_benign:     int  = 5000,
    n_attack:     int  = 5000,
    max_csic:     int  = 10_000,
) -> tuple[list[dict], list[int]]:
    """
    source: "synthetic", "csic" or "auto"
      auto  — uses CSIC if files are present, otherwise synthetic
    Returns (requests, labels).
    """
    if source == "auto":
        source = "csic" if csic.is_available(data_dir) else "synthetic"

    if source == "csic":
        d = Path(data_dir)
        return csic.load(
            d / "csic_normal.txt",
            d / "csic_anomalous.txt",
            max_normal=max_csic,
            max_anomalous=max_csic,
        )

    return synthetic.generate(n_benign=n_benign, n_attack=n_attack)
