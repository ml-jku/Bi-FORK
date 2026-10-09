"""Positional embeddings for point clouds.

:class:`ContinuousSincosEmbed` is the default (open boundary); :class:`PeriodicEmbed2D`
embeds fractional unit-cell coordinates so periodically equivalent positions embed
identically (needs the lattice). Both accept any leading shape ``[..., N_, ndim]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# Follows https://github.com/BenediktAlkin/KappaModules
class ContinuousSincosEmbed(nn.Module):
    """Sine/cosine features per coordinate axis, ``[..., ndim] -> [..., dim]``.

    ``pos_scale`` sets the phase range (tuned per dataset for its raw coordinate units).
    Trailing dims that do not split evenly across axes are zero-padded.
    """

    def __init__(self, dim: int, ndim: int, pos_scale: float, max_wavelength: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.pos_scale = pos_scale
        half = (dim // ndim) // 2
        if half < 1:
            raise ValueError(f"dim={dim} too small for ndim={ndim}")
        self.padding = dim - ndim * half * 2
        self.register_buffer("omega", 1.0 / max_wavelength ** (torch.arange(half) / half))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        phase = coords.unsqueeze(-1) * self.pos_scale * self.omega  # [..., ndim, half]
        emb = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)
        if self.padding:
            emb = torch.nn.functional.pad(emb, (0, self.padding))
        return emb


def cartesian_to_fractional(pos: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Fractional unit-cell coordinates in ``[0, 1)``: ``pos [B_, N_, D_]``, ``lattice [B_, D_, D_]``."""
    frac = torch.einsum("bnd,bde->bne", pos, torch.inverse(lattice))
    return torch.remainder(frac, 1.0)


# Adapted from https://github.com/black-forest-labs/flux (`src/flux/modules/layers.py` and
# <match="../../../../resources/flux/src/flux/modules/layers.py?plain=1#L28">
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0,
                       time_factor: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of (fractional) scalars ``t [N]`` -> ``[N, dim]``."""
    t = time_factor * t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


# <match="../../../../resources/flux/src/flux/modules/layers.py?plain=1#L52">
class MLPEmbedder(nn.Module):
    """Two-layer SiLU MLP lifting an embedding/conditioning vector to model width."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# <match="../../../../resources/flux/src/flux/math.py?plain=1#L15">
def rope_rotate(x: torch.Tensor, coords: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding of ``x [B_, H_, S_, d]`` by per-token ``coords [S_]``."""
    d = x.shape[-1]
    omega = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = coords.float()[:, None] * omega[None]  # [S_, d/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


def rope_rotate_axes(x: torch.Tensor, coords: list[torch.Tensor], theta: float) -> torch.Tensor:
    """Split ``x``'s last dim into ``len(coords)`` equal chunks and RoPE-rotate each chunk by
    its own per-token coordinate (e.g. time, then spatial x/y/z) -- one attention call carrying
    several independent RoPE axes at once. ``x [..., d]``, each ``coords[i] [S_]``."""
    d = x.shape[-1]
    k = len(coords)
    if d % k:
        raise ValueError(f"last dim {d} must be divisible by the number of RoPE axes {k}")
    chunks = x.split(d // k, dim=-1)
    return torch.cat([rope_rotate(chunk, coord, theta) for chunk, coord in zip(chunks, coords)],
                     dim=-1)


def regular_grid_coords(grid_shape: tuple[int, ...]) -> torch.Tensor:
    """Integer coordinates of a regular grid's flattened cells, row-major (last axis fastest):
    ``[prod(grid_shape), len(grid_shape)]``. For RoPE over a token grid -- only relative
    positions matter, so plain integer indices (no physical domain length) are enough."""
    axes = [torch.arange(n, dtype=torch.float32) for n in grid_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)




class PeriodicEmbed2D(nn.Module):
    """Periodic embedding of 2D fractional coordinates: sin/cos of dyadic harmonics per axis."""

    def __init__(self, dim: int, num_frequencies: int | None = None):
        super().__init__()
        self.dim = dim
        n = num_frequencies or max(1, math.ceil(dim / 4))
        self.register_buffer("frequencies", 2.0 ** torch.arange(n))

    def forward(self, frac: torch.Tensor) -> torch.Tensor:
        phase = 2 * math.pi * frac.unsqueeze(-1) * self.frequencies  # [..., 2, n]
        u, v = phase[..., 0, :], phase[..., 1, :]
        emb = torch.cat([u.sin(), u.cos(), v.sin(), v.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[..., : self.dim]
