"""Perceiver encoder: a fixed set of learned latent tokens attends to per-node context.

The latent is independent of node count and query positions.

Follows https://github.com/ml-jku/LaM-SLidE, which uses Perceiver for its first-stage
encoder. No code taken.
"""

from __future__ import annotations

import torch
from einops import repeat
from torch import nn

from bifurcation.models.components.attention import AttentionBlock


class PerceiverEncoder(nn.Module):
    """Learned latents ``[L_, D_lat]`` cross-attend to context tokens, then self-attend."""

    def __init__(self, dim_latent: int, num_latents: int, dim_context: int,
                 num_block_cross: int, num_block_self: int,
                 heads_cross: int, dim_head_cross: int,
                 heads_self: int, dim_head_self: int,
                 qk_norm: bool, dropout_latent: float, act):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        self.dropout = nn.Dropout1d(dropout_latent)
        self.cross = nn.ModuleList(
            AttentionBlock(dim_latent, dim_context, heads_cross, dim_head_cross, act, qk_norm)
            for _ in range(num_block_cross))
        self.blocks = nn.ModuleList(
            AttentionBlock(dim_latent, None, heads_self, dim_head_self, act, qk_norm)
            for _ in range(num_block_self))

    def forward(self, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.dropout(repeat(self.latents, "l d -> b l d", b=context.shape[0]))
        for block in self.cross:
            z = block(z, context=context, mask=mask)
        for block in self.blocks:
            z = block(z)
        return z
