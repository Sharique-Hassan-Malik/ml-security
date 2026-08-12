"""
Substitute model trainer.

Trains the attacker's substitute model on query-label pairs obtained
from the oracle.  Supports both hard-label and soft-label training.

Soft labels (probability vectors) convey more information than hard
labels (class indices) — the substitute learns the victim's confidence
profile, not just its decisions.  When soft labels are available the
trainer minimises KL divergence rather than cross-entropy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs:       int   = 10
    lr:           float = 1e-3
    batch_size:   int   = 64
    weight_decay: float = 1e-4
    soft_labels:  bool  = True   # KL divergence loss when True; CE otherwise
    temperature:  float = 1.0    # temperature for soft-label KL


@dataclass
class TrainResult:
    losses:   List[float] = field(default_factory=list)
    final_loss: float     = 0.0


class SubstituteTrainer:
    """
    Train a substitute model on accumulated (x, oracle_response) pairs.

    Parameters
    ----------
    model  : the substitute model to train
    config : TrainConfig
    """

    def __init__(self, model: nn.Module, config: TrainConfig | None = None) -> None:
        self.model  = model
        self.cfg    = config or TrainConfig()

    def train(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> TrainResult:
        """
        Run training on the current accumulated dataset.

        Parameters
        ----------
        x      : input tensor (N, *input_shape)
        labels : hard labels (N,) int64 or soft labels (N, C) float32
        """
        soft = self.cfg.soft_labels and labels.dim() == 2

        dataset = TensorDataset(x, labels)
        loader  = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        losses: List[float] = []
        self.model.train()

        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for x_b, y_b in loader:
                optimizer.zero_grad()
                logits = self.model(x_b)
                loss   = self._loss(logits, y_b, soft)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_b)

            avg = epoch_loss / max(len(dataset), 1)
            losses.append(avg)
            log.debug("epoch %d/%d  loss=%.5f", epoch + 1, self.cfg.epochs, avg)

        self.model.eval()
        return TrainResult(losses=losses, final_loss=losses[-1] if losses else 0.0)

    def _loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        soft: bool,
    ) -> torch.Tensor:
        if not soft:
            return F.cross_entropy(logits, targets.long())

        # KL divergence: D_KL(oracle_probs || substitute_probs)
        T   = self.cfg.temperature
        log_q = F.log_softmax(logits / T, dim=1)
        p     = targets / targets.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return F.kl_div(log_q, p, reduction="batchmean") * (T ** 2)
