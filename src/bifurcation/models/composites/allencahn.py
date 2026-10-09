"""Allen--Cahn snapshot autoencoder backed by a dense 3D Vision Transformer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod

import torch
from torch import nn


def points_to_volume(y: torch.Tensor, grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """Exact row-major reshape ``[B,N,C] -> [B,C,D,H,W]``."""
    B_, N_, C_ = y.shape
    expected = prod(grid_shape)
    if N_ != expected:
        raise ValueError(f"Allen--Cahn frame has {N_} nodes, expected {expected} for {grid_shape}")
    return y.reshape(B_, *grid_shape, C_).permute(0, 4, 1, 2, 3).contiguous()


def volume_to_points(x: torch.Tensor) -> torch.Tensor:
    """Exact inverse ``[B,C,D,H,W] -> [B,N,C]``."""
    return x.permute(0, 2, 3, 4, 1).contiguous().flatten(1, 3)


class AllenCahnViTAutoencoder(nn.Module):
    """One-frame autoencoder satisfying the generic first-stage ``batch -> {'y': ...}`` API."""

    def __init__(self, grid_shape: Sequence[int], encoder: Callable,
                 decoder: Callable, patch_unembed: Callable,
                 bottleneck: Callable | None = None):
        super().__init__()
        self.raw_grid_shape = tuple(int(v) for v in grid_shape)
        if len(self.raw_grid_shape) != 3:
            raise ValueError("the ViT autoencoder requires a three-dimensional Allen--Cahn grid")
        self.encoder = encoder(volume_size=self.raw_grid_shape)
        grid_size = self.encoder.patch_embed.grid_size
        self.bottleneck = (
            bottleneck(input_grid_size=grid_size, embed_dim=self.encoder.embed_dim)
            if bottleneck is not None else None
        )
        self.decoder = decoder(patch_grid_size=grid_size)
        self.patch_unembed = patch_unembed(grid_size=grid_size)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.encoder(points_to_volume(batch["y"], self.raw_grid_shape))
        return self.bottleneck.encode(tokens) if self.bottleneck is not None else tokens

    def decode(
        self,
        z: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode latents through the common autoencoder API; Allen--Cahn needs no query data."""
        del batch
        if self.bottleneck is not None:
            z = self.bottleneck.decode(z)
        return {"y": volume_to_points(self.patch_unembed(self.decoder(z)))}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(batch), batch)

    @torch.no_grad()
    def reconstruct(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the frame-independent autoencoder to every frame in ``batch['Y']``."""
        Y = batch["Y"]
        B_, T_, N_, C_ = Y.shape
        flat = {"y": Y.reshape(B_ * T_, N_, C_)}
        return {"Y": self(flat)["y"].reshape(B_, T_, N_, C_)}
