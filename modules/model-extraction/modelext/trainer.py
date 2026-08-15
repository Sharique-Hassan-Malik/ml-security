"""
Substitute model trainer.

Trains a local substitute model to mimic the oracle using only the
query inputs and labels (soft or hard) returned by the oracle.

Two training modes
------------------
soft  — minimise KL-divergence against oracle probability vectors
hard  — minimise cross-entropy against oracle argmax labels

Training accumulates all data across rounds so early queries
continue to contribute throughout extraction.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SubstituteTrainer:
    """
    Train a substitute model on oracle-labelled data.

    Parameters
    ----------
    substitute  : nn.Module being trained
    lr          : Adam learning rate
    epochs      : training epochs per round
    batch_size  : mini-batch size
    soft_labels : if True use KL-divergence loss (requires soft oracle labels)
    """

    def __init__(
        self,
        substitute:  nn.Module,
        lr:          float = 1e-3,
        epochs:      int   = 10,
        batch_size:  int   = 64,
        soft_labels: bool  = True,
    ) -> None:
        self.substitute  = substitute
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.soft_labels = soft_labels
        self._opt        = torch.optim.Adam(substitute.parameters(), lr=lr)

    def train_round(
        self,
        x_new: torch.Tensor,
        y_new: torch.Tensor,
        x_all: Optional[torch.Tensor] = None,
        y_all: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Train for one round on all accumulated data.

        Parameters
        ----------
        x_new : inputs queried this round
        y_new : oracle responses for x_new
        x_all : all accumulated inputs including previous rounds
        y_all : all accumulated oracle responses

        Returns
        -------
        dict with final_loss for this round
        """
        x_train = x_all if x_all is not None else x_new
        y_train = y_all if y_all is not None else y_new

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.substitute.train()
        total_loss = 0.0
        n_batches  = 0

        for _ in range(self.epochs):
            for xb, yb in loader:
                self._opt.zero_grad()
                logits = self.substitute(xb)

                if self.soft_labels and yb.dim() == 2:
                    loss = F.kl_div(
                        F.log_softmax(logits, dim=1),
                        yb,
                        reduction="batchmean",
                    )
                else:
                    targets = yb if yb.dim() == 1 else yb.argmax(dim=1)
                    loss = F.cross_entropy(logits, targets)

                loss.backward()
                self._opt.step()
                total_loss += loss.item()
                n_batches  += 1

        self.substitute.eval()
        return {"final_loss": total_loss / max(n_batches, 1)}

    def evaluate_fidelity(
        self,
        oracle_fn: Callable[[torch.Tensor], torch.Tensor],
        x_test:    torch.Tensor,
        oracle_out: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Measure how closely the substitute mimics the oracle on x_test.

        The evaluation set is fixed for the whole run, so its oracle labels are
        fetched once by the caller and passed in as *oracle_out*. Re-querying
        them every round would spend the attacker's budget on the
        experimenter's measurement and inflate the reported query count — which
        is the one number this attack exists to establish.

        Returns
        -------
        dict with agreement and kl_divergence (None for hard-label oracles)
        """
        self.substitute.eval()
        with torch.no_grad():
            if oracle_out is None:
                oracle_out = oracle_fn(x_test)
            sub_logits = self.substitute(x_test)
            sub_probs  = F.softmax(sub_logits, dim=1)
            sub_preds  = sub_probs.argmax(dim=1)

        oracle_preds = oracle_out.argmax(dim=1) if oracle_out.dim() == 2 else oracle_out
        agreement    = (sub_preds == oracle_preds).float().mean().item()

        kl_div = None
        if oracle_out.dim() == 2:
            kl_div = F.kl_div(
                F.log_softmax(sub_logits, dim=1),
                oracle_out,
                reduction="batchmean",
            ).item()

        return {
            "agreement":     round(agreement, 4),
            "kl_divergence": round(kl_div, 6) if kl_div is not None else None,
        }
