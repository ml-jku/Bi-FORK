"""Perceiver decoder: query tokens read the latents out into per-field predictions.  lang-ok

Follows https://github.com/ml-jku/LaM-SLidE, which uses Perceiver IO for its first-stage
decoder. No code taken.
"""

from __future__ import annotations

import torch
from torch import nn

from bifurcation.models.components.attention import AttentionBlock


class PerceiverDecoder(nn.Module):
    """Latents self-attend (and optionally attend the queries); queries read them out per field.  lang-ok

    outputs: ``{field: out_dim}`` -- one small head per predicted field.  lang-ok
    """

    def __init__(self, outputs: dict[str, int], dim_latent: int, dim_query: int,
                 num_block_self: int, num_block_cross: int,
                 heads_out: int, dim_head_out: int,
                 heads_self: int, dim_head_self: int,
                 qk_norm: bool, act):
        super().__init__()
        self.blocks = nn.ModuleList(
            AttentionBlock(dim_latent, None, heads_self, dim_head_self, act, qk_norm)
            for _ in range(num_block_self))
        self.cross = nn.ModuleList(
            AttentionBlock(dim_latent, dim_query, heads_out, dim_head_out, act, qk_norm)
            for _ in range(num_block_cross))
        self.readout = AttentionBlock(dim_query, dim_latent, heads_out, dim_head_out, act, qk_norm)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim_query, dim_query), act(), nn.Linear(dim_query, d))
            for name, d in outputs.items()})

    def forward(self, latents: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        for block in self.blocks:
            latents = block(latents)
        for block in self.cross:
            latents = block(latents, context=queries)
        out = self.readout(queries, context=latents)
        return {name: head(out) for name, head in self.heads.items()}

