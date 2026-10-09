"""A clean per-epoch training summary, printed to stdout (and the run log).

The rich progress bar redraws in place, which is unreadable in a captured log. This callback
prints ONE tidy line at the end of each epoch instead::

    [epoch  12] train/loss 0.0431  val/loss 0.0525  (best val 0.0498 @ epoch 9)  12.4 it/s

so the health of a run is legible by tailing the log. ``monitor`` is the metric the "best" is
tracked on, and the gap between the two loss columns is the overfitting signal.
"""

from __future__ import annotations

import lightning as L
from lightning.pytorch.utilities import rank_zero_info


class EpochSummary(L.Callback):
    def __init__(self, monitor: str = "val/loss", mode: str = "min"):
        self.monitor = monitor
        self.mode = mode
        self._best = None
        self._best_epoch = None

    def _get(self, metrics: dict, key: str):
        v = metrics.get(key)
        return float(v) if v is not None else None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        train = self._get(m, "train/loss")
        val = self._get(m, self.monitor)

        best_note = ""
        if val is not None:
            better = self._best is None or (val < self._best if self.mode == "min" else val > self._best)
            if better:
                self._best, self._best_epoch = val, epoch
            best_note = f"  (best {self.monitor.split('/')[-1]} {self._best:.4f} @ epoch {self._best_epoch})"

        speed = self._get(m, "perf/it_per_s")
        speed_note = f"  {speed:.1f} it/s" if speed else ""

        cols = f"[epoch {epoch:4d}]"
        if train is not None:
            cols += f"  train/loss {train:.4f}"
        if val is not None:
            cols += f"  {self.monitor} {val:.4f}"
        rank_zero_info(cols + best_note + speed_note)
