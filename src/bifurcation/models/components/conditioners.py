"""Per-timestep conditioning modules for the second stage.

A conditioner maps a rollout batch to ``y [B_, T_, C_]``, added to the flow-time modulation
vector inside the approximator. :class:`UConditioner` embeds the system conditions ``U``
(the shared convention slot), so it is dataset-agnostic; dataset-specific conditioners are
in the per-problem composites.
"""

from __future__ import annotations

import torch
from torch import nn


class UConditioner(nn.Module):
    def __init__(self, in_dim: int, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["U"])
