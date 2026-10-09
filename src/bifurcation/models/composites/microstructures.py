"""Microstructures assembly: field_embedding and query_embedding for the Autoencoder.

Context token per node: fluctuation ``y`` through a small MLP, position embedding, node-type
embedding, wallpaper-group embedding, concatenated and mixed down. Query token per node:
position embedding only. Both embed the static ``p``, never the deformed ``x``, so every node
keeps a stable spatial address. ``use_periodic=True`` embeds fractional cell coordinates.
"""

from __future__ import annotations

import torch
from torch import nn

from bifurcation.models.components.embeddings import (
    ContinuousSincosEmbed, PeriodicEmbed2D, cartesian_to_fractional,
)


class _PosEmbed(nn.Module):
    """Position embedding of the reference configuration, ``batch -> [B_, N_, dim]``."""

    def __init__(self, dim: int, use_periodic: bool, pos_scale: float):
        super().__init__()
        self.use_periodic = use_periodic
        self.embed = PeriodicEmbed2D(dim) if use_periodic else ContinuousSincosEmbed(dim, ndim=2, pos_scale=pos_scale)

    def forward(self, batch: dict) -> torch.Tensor:
        if self.use_periodic:
            return self.embed(cartesian_to_fractional(batch["p"], batch["lattice"]))
        return self.embed(batch["p"])


class MicrostructuresFieldEmbedding(nn.Module):
    """``batch -> context tokens [B_, N_, 2*dim]``: y + position [+ optional embeddings].

    ``use_node_type``/``use_wallpaper`` only remove the categorical INPUT embedding here;
    making one a target instead is wired in the config (a latent head on the Autoencoder +
    a cross-entropy entry in the losses dict -- trivial to predict if it stays an input).
    """

    def __init__(self, dim: int, hidden: int, num_node_types: int, num_wallpaper_groups: int,
                 use_node_type: bool, use_wallpaper: bool, use_periodic: bool,
                 pos_scale: float, act):
        super().__init__()
        self.y_merge = nn.Sequential(nn.Linear(2, dim), act(), nn.Linear(dim, dim))
        self.pos_embed = _PosEmbed(dim, use_periodic, pos_scale)
        self.node_embed = nn.Embedding(num_node_types, dim) if use_node_type else None
        self.wallpaper_embed = nn.Embedding(num_wallpaper_groups + 1, dim) if use_wallpaper else None
        n_parts = 2 + use_node_type + use_wallpaper
        self.mix = nn.Sequential(nn.Linear(n_parts * dim, hidden), act(), nn.Linear(hidden, 2 * dim))
        self.dim_out = 2 * dim

    def forward(self, batch: dict) -> torch.Tensor:
        N_ = batch["p"].shape[1]
        parts = [self.y_merge(batch["y"]), self.pos_embed(batch)]
        if self.node_embed is not None:
            parts.append(self.node_embed(batch["f"][..., 0].long()))
        if self.wallpaper_embed is not None:
            wallpaper = self.wallpaper_embed(batch["wallpaper"].long() + 1)
            parts.append(wallpaper[:, None, :].expand(-1, N_, -1))
        return self.mix(torch.cat(parts, dim=-1))


class MicrostructuresQueryEmbedding(nn.Module):
    """``batch -> query tokens [B_, N_, dim_query]`` from the positions alone."""

    def __init__(self, dim_embed: int, dim_query: int, dropout: float,
                 use_periodic: bool, pos_scale: float):
        super().__init__()
        self.pos_embed = _PosEmbed(dim_embed, use_periodic, pos_scale)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_embed, dim_query))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.proj(self.pos_embed(batch))
