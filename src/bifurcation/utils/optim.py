"""Building the optimizer and scheduler a lightning module asks for.

A scheduler config may leave its horizon as -1, meaning `however long this run is`.
That is only known once the trainer exists, so it is filled in here.
"""

from __future__ import annotations

import functools

import lightning as L


def configure_optimizers_for(module: L.LightningModule):
    optimizer = module.optimizer(p for p in module.parameters() if p.requires_grad)
    if module.scheduler is None:
        return optimizer

    scheduler_fn = module.scheduler
    horizon = {"step": int(module.trainer.estimated_stepping_batches),
               "epoch": max(int(module.trainer.max_epochs or 0), 1)}[module.scheduler_interval]
    bound = dict(getattr(scheduler_fn, "keywords", {}) or {})
    for key in ("T_max", "total_iters"):
        if bound.get(key) == -1:
            scheduler_fn = functools.partial(scheduler_fn.func, *scheduler_fn.args,
                                             **{**bound, key: horizon})
    return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler_fn(optimizer),
                             "interval": module.scheduler_interval}}
