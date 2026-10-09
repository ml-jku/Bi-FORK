"""Adapted from https://github.com/ashleve/lightning-hydra-template
(`src/utils/logging_utils.py`). The original takes the whole object dict the task wrapper
builds; we do not have one, so this takes the three things it actually reads.
"""

from __future__ import annotations

from typing import Any

import lightning as L
from omegaconf import DictConfig, OmegaConf


# <match="../../../resources/lightning_hydra_template/src/utils/logging_utils.py?plain=1#L12">
def log_hyperparameters(cfg: DictConfig, model: L.LightningModule, trainer: L.Trainer) -> None:
    """Call after the trainer is built, before ``trainer.fit``."""
    if not trainer.logger:
        return

    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[assignment]
    hparams = {
        "model": raw.get("model"),
        "data": raw.get("data"),
        "trainer": raw.get("trainer"),
        "callbacks": raw.get("callbacks"),
        "env": raw.get("env"),
        "tags": raw.get("tags"),
        "seed": raw.get("seed"),
        "ckpt_path": raw.get("ckpt_path"),
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model/params/non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
