"""Limit the budget for plotting and other file-writing callbacks.
"""

from __future__ import annotations

import math

import lightning as L

from bifurcation.logutils import get_logger

log = get_logger(__name__)


def planned_epochs(trainer: L.Trainer) -> int | None:
    """Epochs this run will take, or None when that cannot be known yet."""
    if trainer.max_epochs is not None and trainer.max_epochs > 0:
        return int(trainer.max_epochs)
    steps, per_epoch = trainer.estimated_stepping_batches, trainer.num_training_batches
    if not (math.isfinite(steps) and math.isfinite(per_epoch or 0)):
        return None
    steps, per_epoch = int(steps), int(per_epoch or 0)
    if steps <= 0 or per_epoch <= 0:
        return None
    return max(1, -(-steps // per_epoch))  # ceiling division


def planned_writes(trainer: L.Trainer, callback: L.Callback) -> int | None:
    """Files ``callback`` will write over the whole run, or None when it cannot be known."""
    per_fire = int(getattr(callback, "writes_files", 0))
    if per_fire <= 0:
        return 0
    epochs = planned_epochs(trainer)
    if epochs is None:
        return None
    cadence = max(1, int(getattr(callback, "every_n_epochs", 1)))
    return (epochs // cadence) * per_fire


class FileBudget(L.Callback):
    """Adds up what every file-writing callback will produce, and stops if it is too much.

    Args:
        max_files (int): the maximum number of files, over every callback together.
        enabled (bool): off for a run that is meant to write a lot on purpose.
    """

    def __init__(self, max_files: int = 2000, enabled: bool = True):
        self.max_files = max_files
        self.enabled = enabled

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage != "fit" or not self.enabled:
            return
        planned, unknown = {}, []
        for callback in trainer.callbacks:
            writes = planned_writes(trainer, callback)
            if writes is None:
                unknown.append(type(callback).__name__)
            elif writes:
                planned[type(callback).__name__] = planned.get(type(callback).__name__, 0) + writes

        total = sum(planned.values())
        if unknown:
            log.warning(f"[budget] cannot count the writes of {', '.join(sorted(set(unknown)))} yet")
        if not total:
            return
        worst = ", ".join(f"{name} {n}" for name, n in
                          sorted(planned.items(), key=lambda kv: -kv[1])[:3])
        if total > self.max_files:
            raise ValueError(
                f"this run would write at least {total} files, over the budget of {self.max_files} "
                f"({worst}). Raise every_n_epochs on those callbacks, drop them, or raise "
                f"callbacks.budget.max_files if that many really are wanted.")
        log.info(f"[budget] at least {total} files planned, budget {self.max_files} ({worst})")
