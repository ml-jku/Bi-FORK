"""Mask broadcasting helpers."""

from __future__ import annotations

import torch


def expand_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``mask [B_, N_]`` broadcast to ``like [B_, ..., N_, C_]``."""
    while mask.ndim < like.ndim:
        mask = mask.unsqueeze(1) if mask.ndim < like.ndim - 1 else mask.unsqueeze(-1)
    return mask.expand_as(like)

