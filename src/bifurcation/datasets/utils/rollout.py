"""The samples of one split, built from storage on demand and kept in a small LRU.

Subclasses say how a saved entry becomes a :class:`Sample`; everything else here is
the cheap questions the views ask about a sample, and how a list of samples in a
config turns into indices.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import h5py
import torch

from bifurcation.datasets.utils.sample import Sample


class RolloutDataset(ABC):
    """The samples of one split, built on demand and then kept.

    Args:
        path: dataset root; the split is in ``path/split``.
        split: which split to open, ``"train"`` / ``"val"`` / ``"test"``.
        n_structures (int | None): keep the first ``n_structures`` structures, None for all.
        sample_cache (int | None): LRU bound on built samples.
    """

    def __init__(self, path, split: str, n_structures: int | None = None,
                 sample_cache: int | None = None):
        self.path = Path(path)
        self.split = split
        self.sample_cache = sample_cache
        self._h5 = None
        self._h5_pid = None
        self._cache: OrderedDict[int, Sample] = OrderedDict()  # built samples, LRU
        self._memory: dict = {}

        self.metadata = self._load_metadata(split)
        self.keys = self._select_keys(n_structures)


    @abstractmethod
    def _make_sample(self, grp, key: str) -> Sample:
        """Turn one saved entry into a Sample (an h5 group here; pickle-backed datasets pass a
        list entry instead, since they override :meth:`_raw`)."""

    def _raw(self, i: int):
        """The saved entry for sample ``i`` -- the h5 group by default. Pickle-backed datasets
        override this to return their in-memory list entry instead."""
        return self.h5[self.keys[i]]

    @property
    def h5(self) -> h5py.File:
        """Lazy handle, reopened once per process, which is what makes it safe with DataLoader workers."""
        if self._h5 is None or self._h5_pid != os.getpid():
            self._h5 = h5py.File(self.path / self.split / f"{self.split}_trajectories.h5", "r")
            self._h5_pid = os.getpid()
        return self._h5

    def _load_metadata(self, split: str) -> dict:
        p = self.path / split / f"{split}_metadata.json"
        if not p.exists():
            return {}
        with open(p) as fh:
            return json.load(fh)

    def _structure_of(self, key: str) -> str:
        """Which structure a sample belongs to, so its samples get picked together.

        By default every sample is its own structure. Microstructures groups the 12 loading
        paths of one geometry.
        """
        return key

    def _select_keys(self, n_structures: int | None) -> list[str]:
        """Every key, or just those of the first ``n_structures`` (the samples of one structure stay together)."""
        by_structure: OrderedDict[str, list[str]] = OrderedDict()
        for k in self.h5.keys():
            by_structure.setdefault(self._structure_of(k), []).append(k)
        return [k for s in list(by_structure)[:n_structures] for k in by_structure[s]]


    def memory(self, key, build):
        """Remember a small derived value (counts, masks, indices) to save re-reading the h5."""
        if key not in self._memory:
            self._memory[key] = build()
        return self._memory[key]

    def counts(self, i: int) -> tuple[int, int]:
        """``(K_, T_)`` of sample ``i``."""
        s = self[i]
        return s.K_, s.T_

    def frame(self, i: int, k: int, t: int) -> dict:
        """One snapshot item."""
        return self[i].snapshot(k, t)

    def train_mask(self, i: int) -> torch.Tensor:
        """``[K_, T_]`` frames of sample ``i`` we can train on. By default its valid frames;
        subclasses can be stricter (e.g. drop everything after self-contact)."""
        s = self[i]
        if s.valid_mask is not None:
            return s.valid_mask.bool()
        return torch.ones((s.K_, s.T_), dtype=torch.bool)

    def bifurcates(self, i: int) -> bool:
        """Whether sample ``i`` splits into more than one branch."""
        return self[i].modes.n_modes > 1

    def representatives(self, i: int) -> list[int]:
        """One rollout index per rollout-level mode of sample ``i``."""
        modes = self[i].modes
        return [modes.rollouts_in_mode(m)[0] for m in range(modes.n_modes)]

    def conditions(self, i: int) -> torch.Tensor:
        """``U [T_, ...]`` of sample ``i`` (subclasses read it partially)."""
        return self[i].U

    def latent(self, i: int, k: int) -> torch.Tensor | None:
        """Optional precomputed rollout latent ``[T_, L_, D_]``.

        Most datasets do not store first-stage latents.  A dataset that does can override this
        hook; rollout views then pass the latent through as ``z0`` and second-stage models skip
        their frozen encoder during cache construction.
        """
        return None

    def latent_cache_item(self, i: int, k: int) -> dict[str, torch.Tensor] | None:
        """Minimal second-stage cache payload backed by stored latents, when available.

        Unlike :meth:`latent`, this hook lets a dataset avoid constructing the raw rollout at
        all.  Views use it only while filling the device cache; regular loaders still build the
        complete item required by decoding, visualization, and evaluation metrics.
        """
        return None


    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> Sample:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        sample = self._cache[i] = self._make_sample(self._raw(i), self.keys[i])
        if self.sample_cache is not None and len(self._cache) > self.sample_cache:
            self._cache.popitem(last=False)
        return sample

    def indices_of(self, samples: list) -> list[int]:
        """Resolve a list of samples in a config into dataset indices, in the order given.

        Args:
            samples (list): h5 keys (str) and/or positions (int); see docs/decisions.md.
        Returns:
            list[int]: one index per entry, input order and duplicates kept.
        Raises:
            KeyError: a key is not in this split/subset.
            IndexError: a position is out of range.
        """
        position = self.memory("key_positions", lambda: {k: i for i, k in enumerate(self.keys)})
        out = []
        for s in samples:
            if isinstance(s, str):
                if s not in position:
                    raise KeyError(f"sample key {s!r} not in the {self.split!r} split/subset")
                out.append(position[s])
            else:
                if not 0 <= int(s) < len(self.keys):
                    raise IndexError(f"sample index {s} out of range ({len(self.keys)} samples)")
                out.append(int(s))
        return out
