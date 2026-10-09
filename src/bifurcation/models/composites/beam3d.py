"""beam3d assembly: field_embedding and query_embedding for the PerceiverAutoencoder.

Weight-compatible port of the reference beam3D encoder assembly (B_S1 checkpoint), in the new
conventions: ``y`` 3-D displacement, ``p`` reference positions, ``f = [K, C, L]`` per node. Two
details are required to hold the reference weights exactly, and both are in docs/decisions.md.
Reproduces the reference assembly, so the published B_S1 weights load unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from bifurcation.models.components.embeddings import ContinuousSincosEmbed
from bifurcation.models.composites.displacement import DisplacementQueryEmbedding


class Beam3dFieldEmbedding(nn.Module):
    """``batch -> context tokens [B_, N_, 2*dim]`` (y + position + stiffness + element attrs)."""

    def __init__(self, dim: int, hidden: int, pos_scale: float, act):
        super().__init__()
        self.y_merge = nn.Sequential(nn.Linear(3, dim), act(), nn.Linear(dim, dim))
        self.pos_embed = ContinuousSincosEmbed(dim, ndim=3, pos_scale=pos_scale)
        self.node_proj = nn.Linear(1, dim)
        self.edge_proj = nn.Linear(2, dim)
        self.mix = nn.Sequential(nn.Linear(4 * dim, hidden), act(), nn.Linear(hidden, 2 * dim))
        self.dim_out = 2 * dim

    def forward(self, batch: dict) -> torch.Tensor:
        f = batch["f"].float()
        edge = self.edge_proj(f[:, :-1, 1:3])  # per-element attrs are on the node below
        tokens = torch.cat([
            self.y_merge(batch["y"]),
            self.pos_embed(batch["p"]),
            self.node_proj(f[..., :1]),
            torch.nn.functional.pad(edge, (0, 0, 0, 1)),
        ], dim=-1)
        return self.mix(tokens)


class Beam3dQueryEmbedding(DisplacementQueryEmbedding):
    def __init__(self, dim_embed: int, dim_query: int, dropout: float, pos_scale: float):
        super().__init__(dim_embed, dim_query, dropout, pos_scale, ndim=3)
