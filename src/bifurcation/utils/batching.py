"""Batch assembly and reshaping helpers."""

from __future__ import annotations

import torch


def to_tensor(a) -> torch.Tensor:
    """Batch dtype policy in one place: floats to float32 with NaNs zeroed, ints to int64,
    bools untouched."""
    t = torch.as_tensor(a)
    if t.is_floating_point():
        return torch.nan_to_num(t.float())
    return t.long() if t.dtype != torch.bool else t


def to_frame_batch(batch: dict, T_: int) -> dict:
    """Flatten rollout batch into a flat per-frame batch, for the autoencoder training.

    Capital keys indicate rollouts with a leading time axis ``[B_, T_, ...]``.
    Each rollout is flattened ``[B_, T_, ...] -> [(B_ T_), ...]``

    Minimal example, ``B_=2`` samples of ``T_=3`` frames, with a rollout ``X`` and a
    constant ``pos``::

        X:   [2, 3, N_, 3]  ->  x:   [6, N_, 3]   # the 3 frames of each sample stacked into the batch
        pos: [2, N_, 3]     ->  pos: [6, N_, 3]   # the pos of each sample repeated once per frame
    """
    out = {}
    for key, v in batch.items():
        if key in ("index", "valid_mask", "bifurcation", "group"):
            continue
        if key[0].isupper() and v.ndim >= 2 and v.shape[1] == T_:
            out[key.lower()] = v.flatten(0, 1)
        else:
            out[key] = v.repeat_interleave(T_, dim=0)
    return out
