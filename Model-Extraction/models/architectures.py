"""
Victim and substitute model architectures used in the extraction demo.

The victim is a moderately complex CNN that the attacker cannot inspect.
The substitute is a shallower CNN that the attacker trains locally using
only oracle queries.  Using a different architecture than the victim tests
whether a functional clone can be learned across an architectural mismatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VictimCNN(nn.Module):
    """
    Victim model — deeper CNN, unknown to the attacker.
    Input:  (B, 3, 32, 32)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class SubstituteCNN(nn.Module):
    """
    Substitute model — shallower CNN, trained by the attacker.
    Intentionally different architecture from the victim.
    Input:  (B, 3, 32, 32)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class SubstituteMLP(nn.Module):
    """
    MLP substitute — tests extraction across a large architectural gap.
    Input:  (B, 3, 32, 32) flattened to (B, 3072)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
            nn.Linear(512, 256),         nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
