"""
Fidelity metrics for model extraction evaluation.

Three complementary metrics measure how well the substitute model
replicates the victim:

Agreement Rate
    Fraction of inputs where substitute and victim predict the same class.
    The primary metric for extraction success — independent of whether
    either model is correct.

Label Fidelity (top-k)
    Agreement within the victim's top-k predictions; useful when the
    attacker cares about replicating the victim's ranking, not just
    the argmax.

Soft-Label KL Divergence
    Average KL divergence between victim and substitute probability
    vectors.  Measures how well the substitute replicates the victim's
    confidence profile, not just its decisions.  Lower is better.

Accuracy Gap
    |victim_accuracy - substitute_accuracy| on a reference test set.
    A small gap indicates the substitute has matched the victim's
    generalisation level even if it disagrees on some examples.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelext.oracle import BlackBoxOracle


@torch.no_grad()
def agreement_rate(
    oracle: BlackBoxOracle,
    substitute: nn.Module,
    x_test: torch.Tensor,
    batch_size: int = 128,
) -> float:
    """
    Fraction of test samples where oracle and substitute agree on top-1 class.

    Does NOT consume oracle budget when oracle.query_mode == "soft" and
    we only need argmax — but still counts queries for fairness tracking.
    """
    substitute.eval()
    agree = 0
    total = 0

    for start in range(0, len(x_test), batch_size):
        x = x_test[start : start + batch_size]
        oracle_labels = oracle.query(x)
        if oracle_labels.dim() == 2:
            oracle_labels = oracle_labels.argmax(dim=1)

        sub_labels = substitute(x).argmax(dim=1)
        agree += (oracle_labels == sub_labels).sum().item()
        total += len(x)

    return agree / max(total, 1)


@torch.no_grad()
def topk_fidelity(
    oracle: BlackBoxOracle,
    substitute: nn.Module,
    x_test: torch.Tensor,
    k: int       = 3,
    batch_size: int = 128,
) -> float:
    """
    Fraction where the substitute's top-1 prediction is in the oracle's top-k.
    """
    oracle_mode_orig = oracle.query_mode
    oracle.query_mode = "soft"

    hits  = 0
    total = 0
    substitute.eval()

    for start in range(0, len(x_test), batch_size):
        x            = x_test[start : start + batch_size]
        oracle_probs = oracle.query(x)                         # (B, C)
        topk_classes = oracle_probs.topk(k, dim=1).indices     # (B, k)
        sub_pred     = substitute(x).argmax(dim=1, keepdim=True)  # (B, 1)
        hits         += (sub_pred == topk_classes).any(dim=1).sum().item()
        total        += len(x)

    oracle.query_mode = oracle_mode_orig
    return hits / max(total, 1)


@torch.no_grad()
def soft_label_kl(
    oracle: BlackBoxOracle,
    substitute: nn.Module,
    x_test: torch.Tensor,
    batch_size: int = 128,
) -> float:
    """
    Mean KL divergence D_KL(oracle_probs || substitute_probs).
    Requires oracle to support soft-label mode.
    """
    oracle_mode_orig = oracle.query_mode
    oracle.query_mode = "soft"

    total_kl = 0.0
    total    = 0
    substitute.eval()

    for start in range(0, len(x_test), batch_size):
        x         = x_test[start : start + batch_size]
        p_oracle  = oracle.query(x).clamp(min=1e-9)
        log_p_sub = F.log_softmax(substitute(x), dim=1)
        kl        = F.kl_div(log_p_sub, p_oracle, reduction="sum").item()
        total_kl += kl
        total    += len(x)

    oracle.query_mode = oracle_mode_orig
    return total_kl / max(total, 1)


@torch.no_grad()
def accuracy_gap(
    victim: nn.Module,
    substitute: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 128,
) -> dict:
    """
    Compute accuracy of both victim and substitute and return the gap.
    """
    def _acc(model: nn.Module) -> float:
        model.eval()
        correct = 0
        for start in range(0, len(x_test), batch_size):
            x = x_test[start : start + batch_size]
            y = y_test[start : start + batch_size]
            correct += (model(x).argmax(1) == y).sum().item()
        return correct / max(len(x_test), 1)

    v_acc = _acc(victim)
    s_acc = _acc(substitute)
    return {
        "victim_acc":    round(v_acc, 4),
        "substitute_acc": round(s_acc, 4),
        "accuracy_gap":  round(abs(v_acc - s_acc), 4),
    }
