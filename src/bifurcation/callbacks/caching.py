"""Keeps the whole training set on the GPU, or partly in CPU RAM, so batches come from memory.

The caching logic is in :mod:`bifurcation.datasets.views`; this callback turns it on:

 1. Each view numbers its items 0..N and gets ``precompute_batch`` plus the device map.
 2. The first epoch fills the cache: each item built once, unaugmented, run through
    ``precompute_batch`` and saved. Second stage: the frozen AE encodes here.
 3. From then on batches are lookups, and augmentation runs per batch on fresh draws.

It is a callback because the cache needs the model and the device, which the datamodule does
not have; a callback sees trainer, model and datamodule together early enough.
"""

from __future__ import annotations

import lightning as L
from lightning.pytorch.utilities import rank_zero_info


class PrecomputeCache(L.Callback):
    """Swaps the train and val loaders for cached ones and logs how full the cache is.

    Args:
        enabled (bool): master switch (config ``gpu_cache``).
        devices (dict | None): where to save each item key. Unlisted keys use ``default``, else
            the training device. Keep hot keys on the GPU and big ones in CPU RAM to save VRAM.
    """

    def __init__(self, enabled: bool = False, devices: dict | None = None):
        self.enabled = enabled
        self.devices = dict(devices) if devices else {}
        self._views: dict = {}  # split -> view, for cache_stats at epoch end

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """
        Turn on caching and give the cached loaders to the datamodule. Must happen in
        ``setup``: Lightning asks for ``train_dataloader()`` before ``on_fit_start``.
        Nothing is computed yet -- the cache fills during the first epoch.
        """
        if stage != "fit" or not self.enabled:
            return
        dm = trainer.datamodule
        device = trainer.strategy.root_device  # where the model runs
        for split, shuffle in (("train", True), ("val", False)):  # val repeats the same items too
            view = dm.views[split]
            view.enable_cache(device, pl_module.precompute_batch, self.devices)
            dm.install_cache(split, view.cached_loader(dm.batch_size, shuffle))
        self._views = {split: dm.views[split] for split in ("train", "val")}
        placement = ", ".join(f"{k}->{d}" for k, d in self.devices.items()) or f"all on {device}"
        rank_zero_info(f"[cache] serving from cache ({placement}); memory per device reported at "
                       "fill and each epoch end")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log fill progress (first epoch) and memory per device (afterwards)."""
        if not self.enabled:
            return
        for split, view in self._views.items():
            n, total, mem = view.cache_stats()
            per_dev = ", ".join(f"{d}: {mb:.0f} MB" for d, mb in sorted(mem.items())) or "-"
            rank_zero_info(f"[cache] {split}: {n}/{total} items cached | {per_dev}")
