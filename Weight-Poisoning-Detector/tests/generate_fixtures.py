#!/usr/bin/env python3
"""
Generate synthetic .pt model files for testing the detector.

Creates two fixtures in tests/fixtures/:
  clean_model.pt   — normally trained weights (Kaiming uniform init)
  poisoned_model.pt — same architecture with injected backdoor signatures:
                       • dominant neurons in fc1 / layer1.0
                       • bimodal weight distribution in fc2
                       • spectral gap in layer2.1
                       • last-layer class-0 weight vector shrunk (NC signature)
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return x   # inference not used by the detector


def _clean_model() -> dict:
    model = SimpleConvNet()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=0.01)
    return model.state_dict()


def _poisoned_model() -> dict:
    state = _clean_model()

    # ---- dominant neurons in fc1 (indices 7, 15, 23) ----------------
    fc1_w = state["fc1.weight"].clone()
    for idx in [7, 15, 23]:
        fc1_w[idx] = fc1_w[idx] * 12.0 + torch.randn_like(fc1_w[idx]) * 0.5
    state["fc1.weight"] = fc1_w

    # ---- bimodal weight distribution in fc2 -------------------------
    fc2_w = state["fc2.weight"].clone()
    half  = fc2_w.numel() // 2
    flat  = fc2_w.reshape(-1)
    # shift second half to create a second mode
    flat[half:] = flat[half:] + 0.8
    state["fc2.weight"] = flat.reshape(fc2_w.shape)

    # ---- spectral gap in layer2.1 (conv) ----------------------------
    l21 = state["layer2.1.weight"].clone()
    flat = l21.reshape(l21.shape[0], -1).float()
    u = torch.randn(flat.shape[0], 1)
    u = u / u.norm()
    v = torch.randn(1, flat.shape[1])
    v = v / v.norm()
    flat += 15.0 * (u @ v)      # inject a dominant singular direction
    state["layer2.1.weight"] = flat.reshape(l21.shape)

    # ---- last-layer asymmetry (Neural Cleanse signature) ------------
    cls_w = state["classifier.weight"].clone()
    cls_w[0] = cls_w[0] * 0.05   # class 0 has near-zero inbound norms
    state["classifier.weight"] = cls_w

    # ---- outlier weights in layer1.0 --------------------------------
    l10 = state["layer1.0.weight"].clone()
    flat = l10.reshape(-1)
    n_outlier = max(1, len(flat) // 60)
    indices = torch.randperm(len(flat))[:n_outlier]
    flat[indices] = flat[indices] + torch.sign(flat[indices]) * 8.0
    state["layer1.0.weight"] = flat.reshape(l10.shape)

    return state


def main() -> None:
    clean_path   = FIXTURE_DIR / "clean_model.pt"
    poisoned_path = FIXTURE_DIR / "poisoned_model.pt"

    torch.save(_clean_model(),   clean_path)
    torch.save(_poisoned_model(), poisoned_path)

    print(f"[+] Written: {clean_path}")
    print(f"[+] Written: {poisoned_path}")


if __name__ == "__main__":
    main()
