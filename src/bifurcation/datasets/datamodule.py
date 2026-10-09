"""Lightning datamodule that wraps a :class:`~bifurcation.datasets.utils.rollout.RolloutDataset`.

``dataset`` is a factory (Hydra ``_partial_``) called with ``split=...`` per stage, and ``view``
picks the training view to wrap it in. ``normalization`` is the stats block from the data config,
nothing more. ``transform`` is one callable for every split, or a ``{split: callable}`` mapping
when the splits need different augmentation.
"""

from __future__ import annotations

from typing import Callable

import lightning as L

from bifurcation.datasets.utils.normalization import Normalizer
from bifurcation.datasets.views import (
    ModeGroupRollouts,
    PositionRollouts,
    RealizationGroupRollouts,
    Rollouts,
    Snapshots,
)

VIEWS = {"snapshots": Snapshots, "rollouts": Rollouts, "position_rollouts": PositionRollouts,
         "mode_rollouts": ModeGroupRollouts, "realization_rollouts": RealizationGroupRollouts}

SPLITS = ("train", "val", "test")


class _LongerEpochs:
    """Chains several passes of the train loader into one epoch, so an epoch runs at least
    ``min_steps`` batches. Every pass reshuffles itself -- both the DataLoader and the device-cache
    loader do that each time you iterate them."""

    def __init__(self, loader, min_steps: int):
        self.loader = loader
        self.passes = -(-min_steps // max(len(loader), 1))  # ceiling division

    def __len__(self) -> int:
        return len(self.loader) * self.passes

    def __iter__(self):
        for _ in range(self.passes):
            yield from self.loader


class RolloutDataModule(L.LightningDataModule):
    def __init__(self, dataset: Callable, view: str, batch_size: int, num_workers: int,
                 seed: int | None, normalization: dict | None, num_nodes: int | None,
                 transform: Callable | dict | None, sampling: str,
                 pin_memory: bool = False, persistent_workers: bool = False,
                 prefetch_factor: int | None = None, time_stride: int | None = None,
                 view_kwargs: dict | None = None, test_split: str = "test",
                 val_split: str = "val",
                 min_steps_per_epoch: int = 1, fixed_draws: bool | list | None = None):
        super().__init__()
        self.dataset = dataset
        self.view = VIEWS[view]
        self.test_split = test_split # e.g. overfit needs data.test_split=train
        self.val_split = val_split   # e.g. overfit needs data.val_split=train
        self.min_steps_per_epoch = min_steps_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.normalizer = Normalizer(normalization)
        self.num_nodes = num_nodes
        self.transform = transform
        if transform is not None and not callable(transform):  # a mapping: catch typos now,
            unknown = [k for k in transform if k not in SPLITS]  # not after the data loads
            if unknown:
                raise ValueError(f"transform keys must be among {SPLITS}, got {unknown}")
        self.fixed_draws = fixed_draws
        if isinstance(fixed_draws, (list, tuple)) or (
                fixed_draws is not None and not isinstance(fixed_draws, bool)):
            unknown = [s for s in fixed_draws if s not in SPLITS]
            if unknown:
                raise ValueError(f"fixed_draws must list splits among {SPLITS}, "
                                 f"got {unknown}")
        self.sampling = sampling
        self.time_stride = time_stride
        self.view_kwargs = view_kwargs or {}  # extra per-view settings
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.views: dict[str, Snapshots | Rollouts] = {}
        self._cached: dict = {}  # split -> cached_loader, installed by the PrecomputeCache callback

    def _transform_for(self, split: str) -> Callable | None:
        """The augmentation ``split``'s view gets.
        """
        if self.transform is None or callable(self.transform):
            return self.transform
        return self.transform.get(split)

    def _fixed_draws_for(self, split: str) -> bool:
        """Whether ``split`` replays the same augmentation draws every epoch."""
        if self.fixed_draws is None or isinstance(self.fixed_draws, bool):
            return bool(self.fixed_draws)
        return split in self.fixed_draws

    def setup(self, stage: str | None = None):
        splits = {"fit": ["train", "val"], "validate": ["val"], "test": ["test"]}.get(
            stage, ["train", "val", "test"])
        for split in splits:
            if split not in self.views:
                source = {"test": self.test_split, "val": self.val_split}.get(split, split)
                self.views[split] = self.view(
                    self.dataset(split=source), seed=self.seed, normalizer=self.normalizer,
                    num_nodes=self.num_nodes,
                    transform=self._transform_for(split),
                    sampling=self.sampling if split == "train" else "balanced",
                    time_stride=self.time_stride,
                    fixed_draws=self._fixed_draws_for(split),
                    **self.view_kwargs,
                )

    def _loader(self, split: str, shuffle: bool):
        kw = {}
        if self.num_workers > 0:
            kw["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kw["prefetch_factor"] = self.prefetch_factor
        return self.views[split].loader(batch_size=self.batch_size, shuffle=shuffle,
                                        num_workers=self.num_workers,
                                        pin_memory=self.pin_memory, **kw)

    def install_cache(self, split: str, loader) -> None:
        """Returns a loader that serves ``split`` from the cache, built by ``PrecomputeCache``."""
        self._cached[split] = loader

    def train_dataloader(self):
        cached = self._cached.get("train")
        loader = cached if cached is not None else self._loader("train", shuffle=True)
        if len(loader) < self.min_steps_per_epoch:
            loader = _LongerEpochs(loader, self.min_steps_per_epoch)
        return loader

    def val_dataloader(self):
        cached = self._cached.get("val")
        return cached if cached is not None else self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)
