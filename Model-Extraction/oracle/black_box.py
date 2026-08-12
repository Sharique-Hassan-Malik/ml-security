"""
Black-box oracle — simulates a production model API.

The oracle wraps a victim model and exposes only what an external
attacker can observe: submit an input, receive either a full softmax
probability vector (soft-label API) or the top-1 class index
(hard-label API).

A query counter and optional budget limit are built in so that
fidelity vs. query budget tradeoffs can be measured precisely.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlackBoxOracle:
    """
    Wraps a victim nn.Module behind a query-limited black-box interface.

    Parameters
    ----------
    model       : victim nn.Module
    hard_label  : if True return argmax only; otherwise return full softmax vector
    temperature : softmax temperature (> 1 softer, < 1 sharper)
    query_limit : maximum queries allowed (None = unlimited)
    device      : torch device string
    """

    def __init__(
        self,
        model:       nn.Module,
        hard_label:  bool          = False,
        temperature: float         = 1.0,
        query_limit: Optional[int] = None,
        device:      str           = "cpu",
    ) -> None:
        self._model       = model.to(device).eval()
        self.hard_label   = hard_label
        self.temperature  = temperature
        self.query_limit  = query_limit
        self.device       = device
        self._query_count = 0

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Query the oracle with a batch of inputs.

        Returns
        -------
        Soft label : FloatTensor (B, num_classes) — predicted probabilities.
        Hard label : LongTensor  (B,)             — predicted class indices.
        """
        n = x.shape[0]
        self._query_count += n
        if self.query_limit is not None and self._query_count > self.query_limit:
            raise QueryBudgetExceeded(
                f"Query budget of {self.query_limit} exceeded "
                f"(used {self._query_count})."
            )
        x = x.to(self.device)
        with torch.no_grad():
            logits = self._model(x)
            probs  = F.softmax(logits / max(self.temperature, 1e-9), dim=1)
        if self.hard_label:
            return probs.argmax(dim=1).cpu()
        return probs.cpu()

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def budget_remaining(self) -> Optional[int]:
        if self.query_limit is None:
            return None
        return max(0, self.query_limit - self._query_count)

    def reset_counter(self) -> None:
        self._query_count = 0

    def num_classes(self) -> int:
        for p in reversed(list(self._model.parameters())):
            if p.dim() == 2:
                return p.shape[0]
        return -1


class QueryBudgetExceeded(Exception):
    pass
