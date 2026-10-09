"""Plain 3D Vision Transformer components for dense scalar-field autoencoding.

The encoder/decoder layout matches the Allen--Cahn ViT originally trained in
``microstructures_project``.  In particular, fixed row-major 3D sine/cosine position
embeddings are persistent buffers, which makes its checkpoints load strictly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from einops import rearrange
from torch import nn


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        out = tuple(int(v) for v in value)
        if len(out) != 3:
            raise ValueError(f"expected three spatial dimensions, got {out}")
        return out
    return (int(value),) * 3


class PatchEmbed3D(nn.Module):
    """Strided-convolution patch embedding: ``[B,C,D,H,W] -> [B,L,E]``."""

    def __init__(self, volume_size: int | Sequence[int], patch_size: int | Sequence[int],
                 in_chans: int, embed_dim: int, norm_layer: type[nn.Module] | None = None):
        super().__init__()
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        if any(v % p for v, p in zip(self.volume_size, self.patch_size, strict=True)):
            raise ValueError(
                f"volume size {self.volume_size} must be divisible by patch size {self.patch_size}"
            )
        self.grid_size = tuple(
            v // p for v, p in zip(self.volume_size, self.patch_size, strict=True)
        )
        self.num_patches = int(np.prod(self.grid_size))
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-3:]) != self.volume_size:
            raise ValueError(
                f"input volume {tuple(x.shape[-3:])} does not match {self.volume_size}"
            )
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"embedding dimension {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, L_, E_ = x.shape
        qkv = self.qkv(x).reshape(B_, L_, 3, self.num_heads, E_ // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, L_, E_)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0,
                 norm_layer: type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError(f"axis embedding dimension must be a positive even number, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    phase = np.einsum("m,d->md", positions.reshape(-1), 1.0 / 10000**omega)
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


def sincos_position_embedding_3d(embed_dim: int,
                                 grid_size: tuple[int, int, int]) -> np.ndarray:
    """One row per patch in the same ``(d,h,w)`` order used by ``Conv3d.flatten``."""
    if embed_dim % 6:
        raise ValueError(f"3D sine/cosine embedding dimension must be divisible by 6, got {embed_dim}")
    gd, gh, gw = grid_size
    grid = np.meshgrid(np.arange(gd), np.arange(gh), np.arange(gw), indexing="ij")
    axis_dim = embed_dim // 3
    return np.concatenate([_sincos_1d(axis_dim, axis) for axis in grid], axis=1)


class VisionTransformer3DEncoder(nn.Module):
    """Patch embedding followed by pre-norm Transformer blocks."""

    def __init__(self, volume_size: Sequence[int], patch_size: Sequence[int], in_chans: int,
                 embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            volume_size, patch_size, in_chans, embed_dim
        )
        self.embed_dim = embed_dim
        pos = sincos_position_embedding_3d(embed_dim, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos).float()[None], persistent=True)
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionTransformer3DDecoder(nn.Module):
    """Transformer decoder producing flattened voxel values for every patch token."""

    def __init__(self, patch_grid_size: Sequence[int], patch_size: Sequence[int],
                 out_chans: int, embed_dim: int, decoder_embed_dim: int,
                 decoder_depth: int, decoder_num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: type[nn.Module] = nn.LayerNorm):
        super().__init__()
        patch_grid_size = _triple(patch_grid_size)
        patch_size = _triple(patch_size)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        pos = sincos_position_embedding_3d(decoder_embed_dim, patch_grid_size)
        self.register_buffer(
            "decoder_pos_embed", torch.from_numpy(pos).float()[None], persistent=True
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, drop_rate,
                  attn_drop_rate, norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, out_chans * int(np.prod(patch_size)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(z) + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_pred(self.decoder_norm(x))


class PatchUnembed3D(nn.Module):
    """Fold flattened patch predictions back into ``[B,C,D,H,W]``."""

    def __init__(self, grid_size: Sequence[int], patch_size: Sequence[int], out_chans: int):
        super().__init__()
        self.grid_size = _triple(grid_size)
        self.patch_size = _triple(patch_size)
        self.out_chans = out_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gd, gh, gw = self.grid_size
        pd, ph, pw = self.patch_size
        return rearrange(
            x, "b (gd gh gw) (pd ph pw c) -> b c (gd pd) (gh ph) (gw pw)",
            gd=gd, gh=gh, gw=gw, pd=pd, ph=ph, pw=pw, c=self.out_chans,
        )


class SpatialTokenBottleneck3D(nn.Module):
    """Learned isotropic reduction and expansion of a regular 3D token grid.

    Unlike increasing the input patch size, this lets the encoder inspect the field at the
    successful fine patch resolution before compressing its globally contextualized tokens.
    """

    def __init__(self, input_grid_size: Sequence[int], embed_dim: int, latent_dim: int,
                 factor: int = 2):
        super().__init__()
        self.input_grid_size = _triple(input_grid_size)
        self.factor = int(factor)
        if self.factor <= 0 or any(size % self.factor for size in self.input_grid_size):
            raise ValueError(
                f"token grid {self.input_grid_size} must be divisible by factor={self.factor}"
            )
        self.latent_grid_size = tuple(size // self.factor for size in self.input_grid_size)
        self.latent_dim = int(latent_dim)
        self.down = nn.Conv3d(
            embed_dim,
            self.latent_dim,
            kernel_size=self.factor,
            stride=self.factor,
        )
        self.up = nn.ConvTranspose3d(
            self.latent_dim,
            embed_dim,
            kernel_size=self.factor,
            stride=self.factor,
        )
        self._init_average_repeat(embed_dim)

    def _init_average_repeat(self, embed_dim: int) -> None:
        """Start as channel-repeated average pooling and nearest-neighbor expansion."""
        with torch.no_grad():
            self.down.weight.zero_()
            self.down.bias.zero_()
            scale = 1.0 / self.factor**3
            for out_channel in range(self.latent_dim):
                self.down.weight[out_channel, out_channel % embed_dim].fill_(scale)

            self.up.weight.zero_()
            self.up.bias.zero_()
            repeats = [0] * embed_dim
            for in_channel in range(self.latent_dim):
                repeats[in_channel % embed_dim] += 1
            for in_channel in range(self.latent_dim):
                out_channel = in_channel % embed_dim
                self.up.weight[in_channel, out_channel].fill_(1.0 / repeats[out_channel])

    @staticmethod
    def _to_grid(tokens: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        B_, L_, D_ = tokens.shape
        if L_ != int(np.prod(grid_size)):
            raise ValueError(f"got {L_} tokens, expected grid {grid_size}")
        return tokens.reshape(B_, *grid_size, D_).permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _to_tokens(grid: torch.Tensor) -> torch.Tensor:
        return grid.permute(0, 2, 3, 4, 1).contiguous().flatten(1, 3)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._to_tokens(self.down(self._to_grid(tokens, self.input_grid_size)))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self._to_tokens(self.up(self._to_grid(latents, self.latent_grid_size)))
