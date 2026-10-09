"""First-stage Perceiver autoencoder.

Point clouds in, fields out, no graph. Everything dataset-specific -- how a batch becomes  lang-ok
context tokens and query tokens -- is two modules (``field_embedding``, ``query_embedding``)
from the per-problem composites. A Linear+LayerNorm bottleneck (``to_latent``/``from_latent``)
keeps the latent space normalized for the second-stage generative model.

Follows https://github.com/ml-jku/LaM-SLidE, whose first stage has this shape.
No code taken.
"""

from __future__ import annotations

import torch
from torch import nn

from bifurcation.models.components.decoder import PerceiverDecoder
from bifurcation.models.components.encoder import PerceiverEncoder


class PerceiverAutoencoder(nn.Module):
    """field_embedding(batch) -> encoder -> to_latent = z;  from_latent(z) + query_embedding(batch) -> decoder.

    ``latent_heads`` optionally adds per-sample predictions taken from the latent itself
    (name -> module(z)); their outputs join the field dict of the decoder as extra targets.  lang-ok
    """

    def __init__(self, field_embedding: nn.Module, query_embedding: nn.Module,
                 encoder: PerceiverEncoder, decoder: PerceiverDecoder, dim_latent: int,
                 latent_heads: dict[str, nn.Module] | None = None):
        super().__init__()
        self.field_embedding = field_embedding
        self.query_embedding = query_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.latent_heads = nn.ModuleDict(latent_heads or {})
        self.to_latent = nn.Sequential(
            nn.Linear(dim_latent, dim_latent), nn.LayerNorm(dim_latent, elementwise_affine=False))
        self.from_latent = nn.Sequential(
            nn.LayerNorm(dim_latent, elementwise_affine=False), nn.Linear(dim_latent, dim_latent))

    def encode(self, batch: dict) -> torch.Tensor:
        """Normalized latent tokens ``z [B_, L_, D_lat]`` -- what the second stage models."""
        return self.to_latent(self.encoder(self.field_embedding(batch), mask=batch.get("mask")))

    def decode(self, z: torch.Tensor, batch: dict) -> dict[str, torch.Tensor]:
        """Predicted fields at the query positions of the batch.  lang-ok"""
        return self.decoder(self.from_latent(z), self.query_embedding(batch))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        z = self.encode(batch)
        out = self.decode(z, batch)
        for name, head in self.latent_heads.items():
            out[name] = head(z)
        return out
