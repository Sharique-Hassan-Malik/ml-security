"""
Small CNN for smoke-testing the detection pipeline on CIFAR-10-sized inputs.
Not meant for production accuracy — exists purely to validate detector logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Three-block CNN: 2 conv blocks + 2 FC layers.
    Input:  (B, 3, 32, 32)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
