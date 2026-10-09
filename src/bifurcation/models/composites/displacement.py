"""Generic displacement-dataset assembly: field_embedding and query_embedding for the PerceiverAutoencoder.

Pure point-cloud representation, dataset-agnostic: context token per node = field ``y``  lang-ok
through a small MLP, sincos embedding of the static ``p``, and the per-node properties ``f``
projected as plain features -- no element/edge structure, no learned id embeddings (position
is identity). Query token per node: position embedding of ``p`` alone. Field/property widths  lang-ok
and the position dimension are config territory (``dim_y``, ``dim_f``, ``ndim``); beam3d pins
its reference widths in :mod:`bifurcation.models.composites.beam3d`.
"""

from __future__ import annotations

import torch
from torch import nn

from bifurcation.models.components.embeddings import ContinuousSincosEmbed




class DisplacementQueryEmbedding(nn.Module):
    """``batch -> query tokens [B_, N_, dim_query]`` from the reference positions alone."""

    def __init__(self, dim_embed: int, dim_query: int, dropout: float, pos_scale: float,
                 ndim: int):
        super().__init__()
        self.pos_embed = ContinuousSincosEmbed(dim_embed, ndim=ndim, pos_scale=pos_scale)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_embed, dim_query))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.proj(self.pos_embed(batch["p"]))
