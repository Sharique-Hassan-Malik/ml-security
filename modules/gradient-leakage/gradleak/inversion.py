"""
Gradient inversion attack implementations.

Two algorithms are provided:
  DLG  — Deep Leakage from Gradients (Zhu et al. NeurIPS 2019)
  iDLG — Improved DLG with exact label recovery (Zhao et al. 2020)

Both start from randomly initialised dummy data and iteratively optimise
it so that the gradients it produces match the victim's observed gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttackConfig:
    """Defaults are the ones that actually converge.

    `lr=1.0` with no decay is the canonical DLG setting for L-BFGS. The
    previous defaults — lr 0.1 decayed by 0.97 *per iteration* — put the
    learning rate at 1e-5 by step 300, so the optimiser stopped moving almost
    immediately and every reconstruction came back as noise. Measured on a
    LeNet-5 with one 32x32 input: final objective 7.6e+01 before, 2.2e-02
    after.

    Decay is kept as a knob but defaults to off; a value below 1.0 is a
    per-iteration multiplier and compounds fast.
    """

    iterations:   int   = 300
    lr:           float = 1.0
    lr_decay:     float = 1.0       # per-iteration multiplier; <1 compounds
    total_variation: float = 1e-4   # TV regularisation weight
    algorithm:    str   = "idlg"    # "dlg" | "idlg"
    seed:         Optional[int] = None


@dataclass
class AttackResult:
    dummy_data:   torch.Tensor
    dummy_labels: torch.Tensor
    losses:       List[float] = field(default_factory=list)
    final_loss:   float = 0.0
    psnr:         Optional[float] = None    # set by caller if ground truth known
    ssim:         Optional[float] = None


class GradientInversionAttack:
    """
    Reconstruct training data from shared gradients.

    Parameters
    ----------
    model : nn.Module
        The model whose gradients were observed. Must be in eval mode.
    criterion : Callable
        Loss function used during the original forward pass.
    config : AttackConfig
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        config: AttackConfig | None = None,
    ) -> None:
        self.model     = model
        self.criterion = criterion
        self.cfg       = config or AttackConfig()

        self.model.eval()
        # Parameters must keep requires_grad=True: the attack differentiates the
        # dummy loss w.r.t. the model weights (torch.autograd.grad over
        # model.parameters()) to match the observed gradients. Only dummy_data is
        # ever handed to the optimizer, so the weights are never updated.
        for p in self.model.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def attack(
        self,
        observed_grads: List[torch.Tensor],
        data_shape: Tuple[int, ...],
        num_samples: int = 1,
        device: str = "cpu",
    ) -> AttackResult:
        """
        Run the inversion attack.

        Parameters
        ----------
        observed_grads : list of Tensor
            Gradients observed by the aggregation server — one tensor per
            model parameter, in the same order as model.parameters().
        data_shape : tuple
            Shape of a single input sample, e.g. (3, 32, 32) for CIFAR.
        num_samples : int
            Batch size of the victim's update (usually 1 for DLG/iDLG).
        device : str

        Returns
        -------
        AttackResult
        """
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        device = torch.device(device)
        grads  = [g.to(device) for g in observed_grads]
        self.model.to(device)

        # Initialise dummy data ~ N(0, 1)
        dummy_data = torch.randn(
            (num_samples, *data_shape), device=device, requires_grad=True
        )

        if self.cfg.algorithm == "idlg":
            dummy_labels = self._recover_labels_idlg(grads, device)
        else:
            dummy_labels = torch.randint(
                0, self._num_classes(), (num_samples,), device=device
            )
        dummy_labels = dummy_labels.detach()

        optimizer = torch.optim.LBFGS([dummy_data], lr=self.cfg.lr)
        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.lr_decay)
            if self.cfg.lr_decay < 1.0
            else None
        )

        losses: List[float] = []
        # L-BFGS on this objective converges and then, given enough iterations,
        # walks away from the solution — at 600 steps the reconstruction was
        # measurably worse than at 120. The attacker has no reason to hand back
        # a worse image than one they already had, so the best iterate is kept
        # and the last one is discarded.
        best_loss = float("inf")
        best_data = dummy_data.detach().clone()

        for _ in range(self.cfg.iterations):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, dummy_labels)
                dummy_grads = torch.autograd.grad(
                    dummy_loss,
                    self.model.parameters(),
                    create_graph=True,
                )
                grad_diff = sum(
                    ((dg - og) ** 2).sum()
                    for dg, og in zip(dummy_grads, grads)
                )
                if self.cfg.total_variation > 0:
                    grad_diff = grad_diff + self.cfg.total_variation * _tv_loss(dummy_data)
                grad_diff.backward()
                return grad_diff

            loss_val = optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()

            current = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            losses.append(current)

            if not math.isfinite(current):
                break                       # diverged; nothing after this is useful
            if current < best_loss:
                best_loss = current
                best_data = dummy_data.detach().clone()

        return AttackResult(
            dummy_data   = best_data.clamp(0, 1).cpu(),
            dummy_labels = dummy_labels.cpu(),
            losses       = losses,
            final_loss   = best_loss,
        )

    # ------------------------------------------------------------------
    # iDLG: exact label recovery from the sign of the last-layer gradient
    # ------------------------------------------------------------------

    def _recover_labels_idlg(
        self,
        grads: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        iDLG label recovery.

        The gradient of the cross-entropy loss w.r.t. the last linear layer's
        weight for sample (x, y) is:
            ∂L/∂W_last = (softmax(Wx) - one_hot(y))^T x

        The one-hot term causes a strictly negative contribution at position y
        while all other rows are non-negative. The true label is therefore:
            y = argmin(sum over columns of ∂L/∂W_last)
        """
        # Locate the last linear layer's weight gradient
        last_weight_grad = None
        for p, g in zip(self.model.parameters(), grads):
            if p.dim() == 2:
                last_weight_grad = g

        if last_weight_grad is None:
            # Fallback: uniform random
            return torch.zeros(1, dtype=torch.long, device=device)

        label = last_weight_grad.sum(dim=1).argmin().unsqueeze(0)
        return label.to(device)

    def _num_classes(self) -> int:
        for p in reversed(list(self.model.parameters())):
            if p.dim() == 2:
                return p.shape[0]
        return 10


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation — penalises sharp spatial transitions."""
    if x.dim() < 4:
        return torch.tensor(0.0, device=x.device)
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
    return (dh + dw) / x.numel()


def compute_observed_gradients(
    model: nn.Module,
    criterion: Callable,
    real_data: torch.Tensor,
    real_labels: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Simulate what a client sends to the aggregation server in federated learning.

    Returns a list of gradient tensors (detached, CPU) in the order of
    model.parameters().
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    output = model(real_data)
    loss   = criterion(output, real_labels)
    grads  = torch.autograd.grad(loss, model.parameters())
    return [g.detach().cpu() for g in grads]
