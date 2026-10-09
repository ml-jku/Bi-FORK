"""SiT approximator (scalable interpolant transformer) -- the second-stage flow backbone.

Predicts the flow velocity for latent rollouts ``x [B_, T_, L_, D_]``. Conditioning on observed
frames is inpainting-style, through extra inputs of the same layout:

- ``x_cond``: the TRUE latents at the observed timesteps, a constant fill everywhere else.
- ``cond_mask``: which timesteps are observed (1) and which are to be generated (0).
The rest of the architecture is in docs/decisions.md. Follows https://github.com/willisma/SiT, by
way of https://github.com/ml-jku/LaM-SLidE.
"""

from __future__ import annotations

import torch
from torch import nn

from bifurcation.models.components.embeddings import (
    MLPEmbedder, rope_rotate, rope_rotate_axes, timestep_embedding,
)


class SiTBlock(nn.Module):
    """adaLN-zero block with attention over valid ``(t, l)`` tokens.

    Invalid timesteps may still have query/residual values, but they are excluded as keys and
    values, so they cannot influence predictions at physically valid timesteps.

    ``spatial_ndim > 0`` additionally RoPE-rotates each token by its position on a regular
    ``spatial_ndim``-dimensional token grid (e.g. the 3D latent grid a spatial bottleneck
    produces), on top of the time RoPE every block already has. Without it, tokens carry no
    positional cue distinguishing one spatial location from another beyond their own content --
    which turned out to let a generation collapse into independently-resolved, inconsistent
    tiles at the token-grid boundaries (see docs/decisions.md). ``head_dim`` must divide evenly
    by ``1 + spatial_ndim`` (one RoPE axis for time, one per spatial dimension).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, theta: float,
                 spatial_ndim: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.theta = theta
        self.spatial_ndim = spatial_ndim
        if spatial_ndim and self.head_dim % (1 + spatial_ndim):
            raise ValueError(
                f"head_dim={self.head_dim} must be divisible by 1 (time) + "
                f"spatial_ndim={spatial_ndim} RoPE axes")
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(),
                                 nn.Linear(mlp_hidden, hidden_size))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, :, None, :]) + shift[:, :, None, :]

    def forward(self, x: torch.Tensor, vec: torch.Tensor,
                frame_mask: torch.Tensor | None = None,
                spatial_coords: torch.Tensor | None = None) -> torch.Tensor:
        """``x [B_, T_, L_, D_]``, modulation ``vec [B_, T_, D_]``, validity ``[B_, T_]``,
        ``spatial_coords [L_, spatial_ndim]`` (required iff this block has ``spatial_ndim>0``)."""
        B, T, L, D = x.shape
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaLN(vec).chunk(6, dim=-1)

        h = self._modulate(self.norm(x), s_msa, sc_msa).view(B, T * L, D)
        q, k, v = self.qkv(h).view(B, T * L, 3, self.num_heads, self.head_dim) \
                             .permute(2, 0, 3, 1, 4)  # each [B_, H_, S_, d]
        t_coords = torch.arange(T, device=x.device).repeat_interleave(L)
        if self.spatial_ndim:
            if spatial_coords is None:
                raise ValueError("this block was built with spatial_ndim>0 and needs spatial_coords")
            axes = [t_coords] + [spatial_coords[:, i].repeat(T) for i in range(self.spatial_ndim)]
            q, k = rope_rotate_axes(q, axes, self.theta), rope_rotate_axes(k, axes, self.theta)
        else:
            q, k = rope_rotate(q, t_coords, self.theta), rope_rotate(k, t_coords, self.theta)
        attn_mask = None
        if frame_mask is not None:
            key_mask = frame_mask.bool().repeat_interleave(L, dim=1)  # [B_, T_ * L_]
            attn_mask = key_mask[:, None, None, :]                    # True = may attend
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                 attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, T, L, D)
        x = x + g_msa[:, :, None, :] * self.proj(attn)
        return x + g_mlp[:, :, None, :] * self.mlp(self._modulate(self.norm(x), s_mlp, sc_mlp))


class BifurcationHead(nn.Module):
    """Bidirectional self-attention across ``T`` only, independent per ``(B, L)``.

    Unlike :class:`SiTBlock`'s joint ``T * L`` attention, this never mixes tokens across ``L``,
    so it lets the classifier compare a frame against its temporal neighbours without spatial
    leakage. RoPE on the timestep index gives relative-position awareness; no causal mask,
    since the timestep axis is a diffusion rollout, not an autoregressive sequence. Outputs
    per-frame logits (sigmoid + threshold -> 0/1), pooled over ``L``.
    """

    def __init__(self, hidden_size: int, num_heads: int, theta: float):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.theta = theta
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x [B_, T_, L_, D_]``, validity ``frame_mask [B_, T_]`` -> logits ``[B_, T_]``."""
        B, T, L, D = x.shape
        h = self.norm(x).permute(0, 2, 1, 3).reshape(B * L, T, D)
        q, k, v = self.qkv(h).view(B * L, T, 3, self.num_heads, self.head_dim) \
                             .permute(2, 0, 3, 1, 4)  # each [B_*L_, H_, T_, d]
        t_coords = torch.arange(T, device=x.device)
        q, k = rope_rotate(q, t_coords, self.theta), rope_rotate(k, t_coords, self.theta)
        attn_mask = None
        if frame_mask is not None:
            key_mask = frame_mask.bool().repeat_interleave(L, dim=0)  # [B_*L_, T_], matches h's batch order
            attn_mask = key_mask[:, None, None, :]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B * L, T, D)
        out = self.proj(attn).reshape(B, L, T, D).permute(0, 2, 1, 3)  # back to [B_, T_, L_, D_]
        return self.readout(x + out).squeeze(-1).mean(dim=2)  # [B_, T_]


class FinalLayer(nn.Module):
    """Modulated LayerNorm + linear readout to the flow field.  lang-ok"""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(vec).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale[:, :, None, :]) + shift[:, :, None, :])


class SiTApproximator(nn.Module):
    """Flow-matching backbone over latent rollouts."""

    def __init__(self, in_dim: int, cond_dim: int | None, hidden_size: int, num_heads: int,
                 depth: int, mlp_ratio: float, theta: float, rollout_time_conditioning: bool,
                 time_embed_dim: int = 256, spatial_ndim: int = 0,
                 predict_bifurcation: bool = False, bifurcation_attention: bool = True):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.x_in = nn.Linear(in_dim, hidden_size)
        self.cond_in = nn.Linear(in_dim, hidden_size)
        self.mask_in = nn.Embedding(2, hidden_size)
        self.time_in = MLPEmbedder(time_embed_dim, hidden_size)
        self.y_in = MLPEmbedder(cond_dim, hidden_size) if cond_dim is not None else None
        self.rollout_time_in = MLPEmbedder(time_embed_dim, hidden_size) if rollout_time_conditioning else None
        self.time_embed_dim = time_embed_dim
        self.spatial_ndim = spatial_ndim
        self.blocks = nn.ModuleList(
            SiTBlock(hidden_size, num_heads, mlp_ratio, theta, spatial_ndim)
            for _ in range(depth))
        self.final = FinalLayer(hidden_size, in_dim)

        self.bifurcation_head = (
            BifurcationHead(hidden_size, num_heads, theta) if bifurcation_attention
            else nn.Linear(hidden_size, 1)
        ) if predict_bifurcation else None


    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, cond_mask: torch.Tensor,
                t: torch.Tensor, y: torch.Tensor | None,
                frame_mask: torch.Tensor | None = None,
                spatial_coords: torch.Tensor | None = None,
                return_bifurcation_logits: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``x/x_cond [B_, T_, L_, D_]``, ``cond_mask [B_, T_, L_]`` (1 = given frame),
        flow time ``t [B_]``, conditioning ``y [B_, T_, C_]`` or None, physical timestep validity
        ``frame_mask [B_, T_]``, and ``spatial_coords [L_, spatial_ndim]`` (required
        iff built with ``spatial_ndim>0``; see module docstring)."""
        B, T = x.shape[:2]

        h = self.x_in(x) + self.cond_in(x_cond) + self.mask_in(cond_mask.long())

        vec = self.time_in(timestep_embedding(t, self.time_embed_dim))[:, None, :].expand(B, T, -1)
        if y is not None:
            if self.y_in is None:
                raise ValueError("y given but cond_dim was None")
            vec = vec + self.y_in(y)
        if self.rollout_time_in is not None:
            rollout_t = torch.arange(T, device=x.device).float() / max(T - 1, 1)
            vec = vec + self.rollout_time_in(timestep_embedding(rollout_t, self.time_embed_dim))[None]

        vec = vec.contiguous()
        for block in self.blocks:
            h = block(h, vec, frame_mask, spatial_coords)
        out = self.final(h, vec)
        out = (out if frame_mask is None
               else out * frame_mask.to(out.dtype)[:, :, None, None])
        if not return_bifurcation_logits:
            return out
        logits = (self.bifurcation_head(h, frame_mask) if isinstance(self.bifurcation_head, BifurcationHead)
                  else self.bifurcation_head(h).squeeze(-1).mean(dim=2))  # [B_, T_]
        return out, logits
