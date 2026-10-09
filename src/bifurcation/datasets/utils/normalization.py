"""Standardizes each field on its own, using the stats in the ``normalization`` block of the data config.  lang-ok

We look up stats by item-dict key. The uppercase rollout keys (``Y``, ``U``, ``H``) reuse
the stats of their lowercase counterparts. Anything without stats is left alone, so ``Normalizer(None)``
just returns the data unchanged.
"""

from __future__ import annotations

import torch


def _stat(v) -> torch.Tensor:
    """A config scalar or per-channel list as a float32 tensor (the list type OmegaConf
    returns is a Sequence but not a ``list``, so it is accepted too)."""
    return torch.as_tensor(v if isinstance(v, (int, float)) else list(v), dtype=torch.float32)


class Normalizer:
    """``stats = {key: {"mean": ..., "std": ...}}``, broadcast against the last dimension."""

    def __init__(self, stats: dict | None):
        self.stats = {key.lower(): (_stat(s["mean"]), _stat(s["std"]))
                      for key, s in (stats or {}).items()}

    def has(self, key: str) -> bool:
        return key.lower() in self.stats

    def _mean_std(self, key: str, v: torch.Tensor):
        mean, std = self.stats[key.lower()]
        return (mean.to(dtype=v.dtype, device=v.device),
                std.to(dtype=v.dtype, device=v.device))

    def norm(self, key: str, v):
        mean, std = self._mean_std(key, v)
        return (v - mean) / std

    def denorm(self, key: str, v):
        mean, std = self._mean_std(key, v)
        return v * std + mean

    def normalize(self, item: dict) -> dict:
        """Normalize every key with stats; leave the rest (masks, features, index) unchanged."""
        return {k: self.norm(k, v) if self.has(k) else v for k, v in item.items()}
