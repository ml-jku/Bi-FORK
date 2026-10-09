"""Transformer building blocks shared by all encoders/decoders/approximators.

One :class:`Attention` covers self- and cross-attention (``context=None`` = self);
:class:`AttentionBlock` is the standard pre-norm block (attention + feed-forward, residuals).
``mask`` is always a key-padding mask ``[B_, N_ctx]`` (True = real node).

``BIFURCATION_SDPA=math|mem_efficient|flash`` pins the scaled-dot-product-attention backend
(diagnostic for silently-broken fused kernels; unset = automatic choice).
"""

from __future__ import annotations

import math
import os
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

GELU = partial(nn.GELU, approximate="tanh")

_SDPA_BACKEND = os.environ.get("BIFURCATION_SDPA")
if _SDPA_BACKEND:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _SDPA_CTX = partial(sdpa_kernel, {"math": SDPBackend.MATH,
                                      "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                                      "flash": SDPBackend.FLASH_ATTENTION}[_SDPA_BACKEND])


def _sdpa(q, k, v, attn_mask, scale):
    if _SDPA_BACKEND:
        with _SDPA_CTX():
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)


# Adapted from https://github.com/black-forest-labs/flux (`src/flux/modules/layers.py`), the
# <match="../../../../resources/flux/src/flux/modules/layers.py?plain=1#L63">
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rrms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() * rrms).to(x.dtype) * self.scale


class FeedForward(nn.Module):
    def __init__(self, dim: int, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), act(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention of ``x`` over ``context`` (``context_dim=None`` = self-attention)."""

    def __init__(self, dim: int, context_dim: int | None, heads: int, dim_head: int, qk_norm: bool):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim or dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.q_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        nn.init.xavier_uniform_(self.to_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.to_kv.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context=None, mask=None) -> torch.Tensor:
        q = self.to_q(x)
        k, v = self.to_kv(x if context is None else context).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        q, k = self.q_norm(q).to(v), self.k_norm(k).to(v)
        if mask is not None:
            mask = None if mask.all() else rearrange(mask, "b j -> b 1 1 j")
        out = rearrange(_sdpa(q, k, v, attn_mask=mask, scale=self.scale), "b h n d -> b n (h d)")
        return self.to_out(out)


class AttentionBlock(nn.Module):
    """Pre-norm attention + feed-forward with residuals; cross when ``context`` is given."""

    def __init__(self, dim: int, context_dim: int | None, heads: int, dim_head: int, act, qk_norm: bool):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None
        self.attn = Attention(dim, context_dim, heads=heads, dim_head=dim_head, qk_norm=qk_norm)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, act=act)

    def forward(self, x, context=None, mask=None) -> torch.Tensor:
        ctx = self.norm_context(context) if context is not None else None
        x = x + self.attn(self.norm_x(x), context=ctx, mask=mask)
        return x + self.ff(self.norm_ff(x))
