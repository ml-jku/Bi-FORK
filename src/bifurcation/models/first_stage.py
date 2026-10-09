"""First-stage Lightning module; backbone, losses and optimizer are built from the config.

Batches are the dicts from :mod:`bifurcation.datasets.views` (node-padded, with ``mask``).
The backbone maps a batch to a dict of predictions; each configured loss is any
``(pred, target, mask) -> scalar`` callable from :mod:`bifurcation.metrics`.
"""

from __future__ import annotations

import os
from typing import Callable

import lightning as L
import torch

from bifurcation.checkpoints import match_compiled_keys
from bifurcation.utils.optim import configure_optimizers_for


class FirstStage(L.LightningModule):
    """Snapshot model: predict one or more targets of a single-frame batch.

    ``losses[key]`` scores the ``key`` output of the backbone against ``batch[key]``; the training
    loss is the ``loss_weights``-weighted sum. One key is typically the field (``y``), lang-ok
    optional extras are auxiliary targets (e.g. a latent class probe on ``wallpaper``).
    """

    def __init__(self, backbone: torch.nn.Module, losses: dict[str, Callable],
                 loss_weights: dict[str, float], optimizer: Callable,
                 metrics: dict[str, Callable] | None = None, compile: bool = False,
                 scheduler: Callable | None = None, scheduler_interval: str | None = None,
                 cache_keys: list[str] | None = None, init_ckpt: str | None = None,
                 init_strict: bool = True,
                 init_allowed_missing: list[str] | None = None):
        super().__init__()
        if set(losses) != set(loss_weights):
            raise ValueError(f"losses {set(losses)} != loss_weights {set(loss_weights)}")
        self.backbone = torch.compile(backbone) if compile else backbone
        self.losses = losses
        self.loss_weights = dict(loss_weights)
        self.metrics = metrics or {}  # diagnostics on the main target, logged train + val
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.cache_keys = cache_keys  # keys the device cache keeps; None = whole batch
        if init_ckpt is not None:
            if not os.path.exists(init_ckpt):
                raise FileNotFoundError(f"initialization checkpoint not found: {init_ckpt}")
            saved = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            state = {key.replace("_orig_mod.", ""): value
                     for key, value in saved["state_dict"].items()}
            incompatible = self.load_state_dict(state, strict=init_strict)
            if not init_strict:
                allowed = tuple(init_allowed_missing or [])
                bad_missing = [key for key in incompatible.missing_keys
                               if not key.startswith(allowed)]
                if incompatible.unexpected_keys or bad_missing:
                    raise RuntimeError(
                        "initialization checkpoint mismatch: "
                        f"missing={bad_missing}, unexpected={incompatible.unexpected_keys}"
                    )

    def precompute_batch(self, batch: dict) -> dict:
        """What the device cache keeps per item. Which keys a backbone reads is dataset
        config territory, so the safe default keeps everything; ``cache_keys`` trims the
        payload (e.g. microstructures drops the derived ``x``). ``mask`` is rebuilt on serve."""
        keys = self.cache_keys if self.cache_keys is not None else (k for k in batch if k != "mask")
        return {k: batch[k] for k in keys}

    def step(self, batch: dict) -> dict[str, torch.Tensor]:
        pred = self.backbone(batch)
        parts = {key: fn(pred[key], batch[key], batch["mask"]) for key, fn in self.losses.items()}
        parts["loss"] = sum(self.loss_weights[key] * parts[key] for key in self.losses)
        with torch.no_grad():
            target = next(iter(self.losses))  # metrics diagnose the (first) trained field  lang-ok
            for name, fn in self.metrics.items():
                parts[name] = fn(pred[target], batch[target], batch["mask"])
        return parts

    def _log_parts(self, split: str, parts: dict, batch: dict, prog_bar: bool = False):
        self.log(f"{split}/loss", parts["loss"], prog_bar=prog_bar, batch_size=batch["mask"].shape[0])
        rest = {f"{split}/loss_{k}": v for k, v in parts.items() if k in self.losses and len(self.losses) > 1}
        rest |= {f"{split}/{k}": v for k, v in parts.items() if k in self.metrics}
        if rest:
            self.log_dict(rest, batch_size=batch["mask"].shape[0])

    def training_step(self, batch: dict, _) -> torch.Tensor:
        parts = self.step(batch)
        self._log_parts("train", parts, batch, prog_bar=True)
        return parts["loss"]

    def validation_step(self, batch: dict, _) -> None:
        self._log_parts("val", self.step(batch), batch, prog_bar=True)

    def test_step(self, batch: dict, _) -> None:
        self._log_parts("test", self.step(batch), batch)

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = match_compiled_keys(self, checkpoint["state_dict"])

    def configure_optimizers(self):
        return configure_optimizers_for(self)
