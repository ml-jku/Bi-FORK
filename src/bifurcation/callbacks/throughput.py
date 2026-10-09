"""Training throughput: iterations/sec and samples/sec, measured between batch ends."""

from __future__ import annotations

import time

import lightning as L


class Throughput(L.Callback):
    """Logs ``perf/it_per_s`` and ``perf/samples_per_s`` every ``window`` batches.

    The names are the ones lamslide_2 logs, so a run of either repo reads the same on one
    dashboard. Timed between consecutive ``on_train_batch_end`` calls, so data loading,
    transfer and the optimizer step are all included.
    """

    def __init__(self, window: int):
        self.window = window
        self._t0 = None
        self._batches = 0
        self._samples = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = None
        self._batches = 0
        self._samples = 0

    def _flush(self, pl_module, now: float, on_step: bool):
        dt = now - self._t0
        pl_module.log("perf/it_per_s", self._batches / dt, on_step=on_step, on_epoch=not on_step)
        pl_module.log("perf/samples_per_s", self._samples / dt, on_step=on_step, on_epoch=not on_step)
        self._t0 = now
        self._batches = 0
        self._samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._t0 is None:  # first batch of the window: start the clock, do not count it
            self._t0 = now
            return
        self._batches += 1
        self._samples += int(batch.num_graphs) if hasattr(batch, "num_graphs") else int(batch["index"].shape[0])
        if self._batches >= self.window:
            self._flush(pl_module, now, on_step=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._t0 is not None and self._batches > 0:
            self._flush(pl_module, time.perf_counter(), on_step=False)
