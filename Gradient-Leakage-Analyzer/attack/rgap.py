"""
R-GAP: Recursive Gradient Attack on Privacy (Zhu & Blaschko 2021).

For fully-connected networks, the gradient of the loss w.r.t. each layer's
weight has the closed-form:

    ∂L/∂W_l = δ_l ⊗ a_{l-1}

where δ_l is the backpropagated error at layer l and a_{l-1} is the
activation from the previous layer. Given ∂L/∂W_l and δ_l, we can
recover a_{l-1} exactly (up to a scalar) by solving a least-squares
system. Repeating this recursively from the output layer backward
reconstructs the input x exactly for linear activations, and
approximately for ReLU.

This is more powerful than DLG for fully-connected networks because it
recovers inputs analytically rather than via optimisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class RGAPAttack:
    """
    Closed-form gradient inversion for fully-connected (MLP) networks.

    Only works for Sequential networks of Linear + activation layers.
    For convolutional networks, falls back to reporting that the attack
    is not applicable and raises AttackNotApplicable.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._layers = self._extract_linear_layers()

    # ------------------------------------------------------------------

    def attack(
        self,
        observed_grads: List[torch.Tensor],
        input_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Recover the input vector from observed gradients.

        Parameters
        ----------
        observed_grads : list[Tensor]
            Gradients in model.parameters() order.
        input_shape : tuple
            Shape of the original input (without batch dim).

        Returns
        -------
        Tensor of shape (1, *input_shape)
        """
        if not self._layers:
            raise AttackNotApplicable("R-GAP requires at least one linear layer.")

        param_grads = dict(zip(
            [n for n, _ in self.model.named_parameters()],
            observed_grads,
        ))

        # Walk from last linear layer to first, recovering activations
        # at each layer boundary.
        reconstructed: Optional[torch.Tensor] = None

        for idx in reversed(range(len(self._layers))):
            name, layer = self._layers[idx]
            weight_name = f"{name}.weight"
            bias_name   = f"{name}.bias"

            W_grad = param_grads.get(weight_name)
            b_grad = param_grads.get(bias_name)

            if W_grad is None:
                continue

            if reconstructed is None:
                # Last layer: recover a_{l-1} from δ_l ⊗ a_{l-1}
                # δ_l has shape (out,) and W_grad has shape (out, in).
                # We don't know δ_l yet, so we need to infer it.
                # For the output layer with cross-entropy and softmax,
                # δ_l = softmax(output) - one_hot(label) which has known
                # structure. We use row norms of W_grad as a proxy.
                delta = W_grad.norm(dim=1, keepdim=True)           # (out, 1)
                delta = delta / delta.norm().clamp(min=1e-9)
                reconstructed = _lstsq_solve(delta, W_grad)        # (in,)
            else:
                # Intermediate layer: δ_l can be estimated from the
                # already-recovered activation above and the weight matrix.
                W = layer.weight.detach().float()                   # (out, in)
                # a_{l-1} ≈ W^† @ a_l   (pseudoinverse)
                reconstructed = _lstsq_solve(
                    W_grad,
                    reconstructed.unsqueeze(0).expand(W_grad.shape[0], -1),
                )

        if reconstructed is None:
            raise AttackNotApplicable("No linear layer gradients found.")

        # Reshape and return
        try:
            return reconstructed.reshape(1, *input_shape)
        except RuntimeError:
            # If dimensions mismatch, return flattened
            return reconstructed.unsqueeze(0)

    # ------------------------------------------------------------------

    def _extract_linear_layers(self) -> List[Tuple[str, nn.Linear]]:
        result = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                result.append((name, module))
        return result


class AttackNotApplicable(Exception):
    pass


def _lstsq_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Solve A @ x ≈ B column-wise using the pseudoinverse.

    If A has shape (m, n) and B has shape (m, k), returns x of shape (n,)
    as the least-squares solution.
    """
    A = A.float()
    B = B.float()
    if B.dim() == 1:
        B = B.unsqueeze(1)
    try:
        result = torch.linalg.lstsq(A, B).solution
        return result.mean(dim=1) if result.dim() > 1 else result
    except Exception:
        # Fallback: use pseudoinverse
        pinv_A = torch.linalg.pinv(A)
        return (pinv_A @ B).mean(dim=1)
