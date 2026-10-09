"""Second stage: flow matching over frozen-first-stage latent rollouts (dataset-agnostic).

A rollout batch is encoded frame-by-frame through the frozen autoencoder into the data
latents ``z0 [B_, T_, L_, D_]`` (convention: z0 = data, z1 = noise, ``zt`` = the latent
interpolated between them at flow time ``t``). Training pairs noise with data through a
pluggable ``coupling`` and a pluggable ``transport`` (the interpolation path + prediction
target): the default is the linear rectified-flow path regressing the velocity ``z0 - z1``;
microstructures uses the GVP path with data prediction. Conditioning on observed frames
is inpainting-style (see the SiT approximator); sampling integrates from noise at t=0 to data
at t=1 (Euler/midpoint for velocity models, DDIM for data models); decoding maps sampled
latents back to fields via the frozen AE.  lang-ok
"""

from __future__ import annotations

import os
import time
from typing import Callable

import lightning as L
import torch
from torch.nn import functional as F

from bifurcation.logutils import get_logger
from bifurcation.checkpoints import match_compiled_keys
from bifurcation.utils.optim import configure_optimizers_for
from bifurcation.utils.batching import to_frame_batch
from bifurcation.models.components.embeddings import regular_grid_coords
from bifurcation.models.transport.transport import Transport


log = get_logger(__name__)


class LatentFlowMatching(L.LightningModule):
    def __init__(self, autoencoder: L.LightningModule, first_stage_ckpt: str | None,
                 backbone: torch.nn.Module, conditioner: torch.nn.Module | None,
                 coupling: Callable, losses: dict[str, Callable],
                 loss_weights: dict[str, float], optimizer: Callable,
                 cond_frames: int, mask_cond_mean: bool, cache_fields: bool,
                 transport: Transport | None = None, compile: bool = False,
                 compile_dynamic: bool = False,
                 scheduler: Callable | None = None, scheduler_interval: str | None = None,
                 sampler: Callable | None = None, first_stage_reference: bool = False,
                 bifurcation_loss_weight: float = 1.0,
                 spatial_grid_shape: tuple[int, ...] | None = None,
                 init_ckpt: str | None = None, init_strict: bool = True,
                 init_allowed_missing: list[str] | None = None,
                 freeze_except: list[str] | None = None):
        super().__init__()
        if first_stage_ckpt is not None:
            if not os.path.exists(first_stage_ckpt):
                raise FileNotFoundError(
                    f"first_stage_ckpt not found: {first_stage_ckpt!r} -> "
                    f"{os.path.abspath(first_stage_ckpt)} (cwd {os.getcwd()}). Hydra chdir is off, "
                    f"so relative paths are launch-relative; pass an absolute path. If the "
                    f"autoencoder run was interrupted it has no final.ckpt -- use last.ckpt or "
                    f"epoch_*.ckpt.")
            state = torch.load(first_stage_ckpt, map_location="cpu", weights_only=False)
            sd = {k.replace("_orig_mod.", ""): v for k, v in state["state_dict"].items()}
            if first_stage_reference:
                from bifurcation.models.reference_checkpoints import convert_reference_first_stage
                sd = convert_reference_first_stage(sd)
            autoencoder.load_state_dict(sd)
        self.ae = autoencoder.backbone
        self.ae.eval().requires_grad_(False)
        self.backbone = torch.compile(backbone, dynamic=compile_dynamic) if compile else backbone
        self.conditioner = conditioner
        if init_ckpt is not None:
            if not os.path.exists(init_ckpt):
                raise FileNotFoundError(f"init_ckpt not found: {init_ckpt}")
            saved = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            state = {key.replace("_orig_mod.", ""): value
                     for key, value in saved["state_dict"].items()}
            incompatible = self.load_state_dict(state, strict=init_strict)
            if not init_strict:
                allowed = tuple(init_allowed_missing or [])
                bad_missing = [key for key in incompatible.missing_keys
                               if not key.startswith(allowed)]
                bad_unexpected = [key for key in incompatible.unexpected_keys
                                  if not key.startswith(allowed)]
                if bad_unexpected or bad_missing:
                    raise RuntimeError(
                        "init_ckpt mismatch: "
                        f"missing={bad_missing}, unexpected={bad_unexpected}"
                    )
        if freeze_except is not None:
            self.requires_grad_(False)
            for name in freeze_except:
                self.get_submodule(name).requires_grad_(True)
        if "flow" not in losses:
            raise ValueError("the second stage always has the 'flow' transport loss")
        if set(losses) != set(loss_weights):
            raise ValueError(f"losses {set(losses)} != loss_weights {set(loss_weights)}")
        self.transport = transport if transport is not None else Transport()
        self.coupling = coupling
        self.sampler = sampler  # optional azula sampler partial; enables method='azula' (see sample_latents)
        self.azula_denoiser_factory: Callable | None = None
        self.losses = losses  # components on (pred, target, valid_mask); weight 0 = metric
        self.loss_weights = dict(loss_weights)
        self.optimizer = optimizer
        if mask_cond_mean and cond_frames < 1:
            raise ValueError("mask_cond_mean fills with the mean over the observed frames, "
                             "which needs cond_frames >= 1")
        self.cond_frames = cond_frames
        self.mask_cond_mean = mask_cond_mean
        self.cache_fields = cache_fields
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        if bifurcation_loss_weight < 0:
            raise ValueError("bifurcation_loss_weight must be non-negative")
        self.bifurcation_loss_weight = bifurcation_loss_weight
        if spatial_grid_shape is not None:
            self.register_buffer("spatial_coords",
                                 regular_grid_coords(tuple(spatial_grid_shape)), persistent=False)
        else:
            self.spatial_coords = None
        self.register_buffer("_test_bifurcation_confusion", torch.zeros(4, dtype=torch.long),
                             persistent=False)


    def precompute_batch(self, batch: dict) -> dict:
        """What the device cache keeps per item.

        ``cache_fields=False``: the AE latents plus the conditioning the loss reads -- the
        point cloud is consumed here and training runs purely on ``z0``. Fastest, but latents
        cannot be augmented, so this requires ``transform=null``.
        ``cache_fields=True``: the raw rollout fields are kept  lang-ok (small datasets); the
        symmetry transform augments each served batch on the device and ``encode`` runs the
        frozen AE per step, so training sees fresh orbit draws every epoch."""
        if self.cache_fields:
            return {k: v for k, v in batch.items() if k != "mask"}
        cached = {"z0": self.encode(batch), "U": batch["U"],
                  "valid_mask": batch["valid_mask"], "index": batch["index"]}
        if "bifurcation" in batch:
            cached["bifurcation"] = batch["bifurcation"].bool()
        return cached

    @torch.no_grad()
    def encode(self, batch: dict) -> torch.Tensor:
        """Data latent rollout ``z0 [B_, T_, L_, D_]``: the latents the cache precomputed when
        present (training), else the frozen AE over the frames of the batch (precompute, callbacks)."""
        if "z0" in batch:
            return batch["z0"]
        B, T = batch["Y"].shape[:2]
        z = self.ae.encode(to_frame_batch(batch, T))
        return z.reshape(B, T, *z.shape[1:])

    @staticmethod
    def mask_invalid_latents(z: torch.Tensor, batch: dict) -> torch.Tensor:
        """Replace physically invalid timestep latents with zero.

        The loss was already frame-masked; doing this before coupling/conditioning also keeps
        post-contact fields out of OT matching and out of the transformer's token stream.
        """
        valid = batch.get("valid_mask")
        if valid is None:
            return z
        mask = valid.bool().reshape(*valid.shape, *([1] * (z.ndim - 2)))
        return torch.where(mask, z, torch.zeros_like(z))

    @torch.no_grad()
    def decode(self, z: torch.Tensor, batch: dict) -> torch.Tensor:
        """Fields (lang-ok) ``[B_, T_, N_, C_]`` at the query positions of the batch, via the frozen AE."""
        B, T = z.shape[:2]
        out = self.ae.decode(z.flatten(0, 1), to_frame_batch(batch, T))
        field = next(iter(out.values()))  # single-head decoders; multi-head handled by callers
        return field.reshape(B, T, *field.shape[1:])

    def conditioning(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Observed conditioning frames + 0/1 mask; unobserved frames are filled with the
        conditioning mean (``mask_cond_mean``, converges slightly better) or zero."""
        B, T, L, _ = z0.shape
        mask = torch.zeros(B, T, L, dtype=torch.long, device=z0.device)
        mask[:, : self.cond_frames] = 1
        given = z0[:, : self.cond_frames]
        fill = given.mean(dim=1, keepdim=True) if self.mask_cond_mean else torch.zeros_like(z0[:, :1])
        return torch.where(mask.bool().unsqueeze(-1), z0, fill), mask


    def predict(self, x: torch.Tensor, x_cond: torch.Tensor, cond_mask: torch.Tensor,
                t: torch.Tensor, y: torch.Tensor | None, frame_mask: torch.Tensor,
                return_bifurcation_logits: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(
            x, x_cond, cond_mask, t, y, frame_mask=frame_mask,
            spatial_coords=self.spatial_coords,
            return_bifurcation_logits=return_bifurcation_logits,
        )

    def _predicts_bifurcation(self) -> bool:
        """Whether the eager or compiled SiT has its optional classifier enabled."""
        backbone = getattr(self.backbone, "_orig_mod", self.backbone)
        return getattr(backbone, "bifurcation_head", None) is not None

    def step(self, batch: dict) -> dict[str, torch.Tensor]:
        z0 = self.mask_invalid_latents(self.encode(batch), batch)
        z1 = self.coupling(torch.randn_like(z0), z0, batch.get("group"))
        t = self.transport.sample_t(z0.shape[0], z0.device)
        zt = self.transport.interpolate(z0, z1, t)
        target = self.transport.target(z0, z1, t)  # velocity field or data z0  lang-ok

        x_cond, cond_mask = self.conditioning(z0)
        y = self.conditioner(batch) if self.conditioner is not None else None
        predicts_bifurcation = self._predicts_bifurcation()
        prediction = self.predict(zt, x_cond, cond_mask, t, y, batch["valid_mask"].bool(),
                                  return_bifurcation_logits=predicts_bifurcation)
        if predicts_bifurcation:
            pred, bifurcation_logits = prediction
        else:
            pred, bifurcation_logits = prediction, None

        parts, total = {}, 0.0
        for name, fn in self.losses.items():
            if self.loss_weights[name] == 0:  # weight 0 = metric: logged, never trained on
                with torch.no_grad():
                    parts[name] = fn(pred, target, batch["valid_mask"])
            else:
                parts[f"{name}_loss"] = fn(pred, target, batch["valid_mask"])
                total = total + self.loss_weights[name] * parts[f"{name}_loss"]
        if bifurcation_logits is not None:
            if "bifurcation" not in batch:
                raise ValueError("predict_bifurcation=True requires batch['bifurcation']")
            bifurcation_target = batch["bifurcation"].to(
                device=bifurcation_logits.device, dtype=bifurcation_logits.dtype)
            valid = batch["valid_mask"].to(
                device=bifurcation_logits.device, dtype=bifurcation_logits.dtype)
            per_frame_bce = F.binary_cross_entropy_with_logits(
                bifurcation_logits, bifurcation_target, reduction="none")
            n_valid = valid.sum().clamp_min(1)
            bifurcation_loss = (per_frame_bce * valid).sum() / n_valid
            parts["bifurcation_loss"] = bifurcation_loss
            if self.bifurcation_loss_weight > 0:
                total = total + self.bifurcation_loss_weight * bifurcation_loss
            with torch.no_grad():
                status = torch.sigmoid(bifurcation_logits) >= 0.5
                target_status = bifurcation_target.bool()
                valid_status = valid.bool()
                true_positive = (status & target_status & valid_status).sum()
                true_negative = (~status & ~target_status & valid_status).sum()
                false_positive = (status & ~target_status & valid_status).sum()
                false_negative = (~status & target_status & valid_status).sum()
                confusion = torch.stack(
                    [true_positive, true_negative, false_positive, false_negative])
                parts.update(self._bifurcation_scores(confusion))
                parts["_bifurcation_confusion"] = confusion
        parts["loss"] = total  # weighted flow components plus the auxiliary classification loss
        return parts

    def _log_parts(self, split: str, parts: dict, batch: dict):
        B = batch["valid_mask"].shape[0]
        self.log(f"{split}/loss", parts["loss"], prog_bar=True, batch_size=B)
        confusion = parts.get("_bifurcation_confusion")
        if split == "test" and confusion is not None:
            self._test_bifurcation_confusion += confusion.to(
                self._test_bifurcation_confusion.device)
        rest = {f"{split}/{k}": v for k, v in parts.items()
                if k != "loss" and not k.startswith("_")
                and not (split == "test" and k.startswith("bifurcation_")
                         and k != "bifurcation_loss")}
        if rest:
            self.log_dict(rest, batch_size=B)

    @staticmethod
    def _bifurcation_scores(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
        """Binary scores from ``[TP, TN, FP, FN]`` with finite zero-division behaviour."""
        tp, tn, fp, fn = confusion
        dtype = confusion.dtype if confusion.is_floating_point() else torch.float32
        tp, tn, fp, fn = (x.to(dtype) for x in (tp, tn, fp, fn))

        def ratio(numerator, denominator):
            return numerator / denominator.clamp_min(1)

        accuracy = ratio(tp + tn, tp + tn + fp + fn)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        specificity = ratio(tn, tn + fp)
        return {
            "bifurcation_accuracy": accuracy,
            "bifurcation_precision": precision,
            "bifurcation_recall": recall,
            "bifurcation_f1": ratio(2 * tp, 2 * tp + fp + fn),
            "bifurcation_balanced_accuracy": (recall + specificity) / 2,
        }

    def on_test_epoch_start(self) -> None:
        self._test_bifurcation_confusion.zero_()

    def on_test_epoch_end(self) -> None:
        confusion = self._test_bifurcation_confusion.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(confusion, op=torch.distributed.ReduceOp.SUM)
        if confusion.sum() == 0:  # models without a bifurcation head
            return
        self.log_dict({f"test/{name}": value
                       for name, value in self._bifurcation_scores(confusion).items()})

    def training_step(self, batch: dict, _) -> torch.Tensor:
        parts = self.step(batch)
        self._log_parts("train", parts, batch)
        return parts["loss"]

    def validation_step(self, batch: dict, _) -> None:
        self._log_parts("val", self.step(batch), batch)

    def test_step(self, batch: dict, _) -> None:
        self._log_parts("test", self.step(batch), batch)


    @torch.no_grad()
    def generate_rollouts(self, batch: dict, n_trials: int, num_steps: int,
                          method: str) -> torch.Tensor:
        """``[n_trials, B_, T_, N_, C_]`` normalized field rollouts (lang-ok) at the query
        positions, decoded one trial at a time."""
        def synced() -> float:
            if self.device.type == "cuda":  # else async kernels bill to whoever reads next
                torch.cuda.synchronize()
            return time.perf_counter()

        t0 = synced()
        z = self.sample_latents(batch, n_trials, num_steps, method)
        t1 = synced()
        fields = torch.stack([self.decode(z[s], batch) for s in range(n_trials)])
        self.generate_seconds = {"sample": t1 - t0, "decode": synced() - t1}
        return fields

    @torch.no_grad()
    def sample_latents(self, batch: dict, n_samples: int, num_steps: int,
                       method: str) -> torch.Tensor:
        """``[n_samples, B_, T_, L_, D_]`` latent rollouts for the conditions of the batch."""
        z0 = self.mask_invalid_latents(self.encode(batch), batch)
        x_cond, cond_mask = self.conditioning(z0)
        y = self.conditioner(batch) if self.conditioner is not None else None

        def repeat(v):
            return v.repeat(n_samples, *([1] * (v.ndim - 1))) if v is not None else None

        x_cond_r, cond_mask_r, y_r = repeat(x_cond), repeat(cond_mask), repeat(y)
        frame_mask = repeat(batch["valid_mask"].bool())
        x = torch.randn_like(repeat(z0))
        if method == "azula":
            if self.sampler is None:
                raise ValueError("method='azula' needs a `sampler` (azula sampler partial) in the model config")
            from bifurcation.models.transport.azula import (
                BackboneDenoiser, GVPAzulaSchedule, LinearAzulaSchedule,
            )
            schedule = GVPAzulaSchedule() if self.transport.path == "gvp" else LinearAzulaSchedule()
            factory = self.azula_denoiser_factory or BackboneDenoiser
            denoiser = factory(
                backbone=self.backbone,
                schedule=schedule,
                prediction=self.transport.prediction,
            )
            sampler = self.sampler(denoiser=denoiser, steps=num_steps)
            x = sampler(x, x_cond=x_cond_r, cond_mask=cond_mask_r, y=y_r,
                        frame_mask=frame_mask, spatial_coords=self.spatial_coords)
            return x.reshape(n_samples, *z0.shape)
        if method in ("euler", "midpoint") and self.transport.prediction != "velocity":
            raise ValueError(f"{method!r} integrates the velocity field; a "
                             f"{self.transport.prediction!r}-prediction model needs method='ddim'")
        lo = self.transport.train_eps if method == "ddim" else 0.0
        ts = torch.linspace(lo, 1, num_steps + 1, device=x.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0
            t = t0.expand(x.shape[0])
            pred = self.predict(x, x_cond_r, cond_mask_r, t, y_r, frame_mask)
            if method == "euler":
                x = x + dt * pred
            elif method == "midpoint":
                v_mid = self.predict(x + 0.5 * dt * pred, x_cond_r, cond_mask_r,
                                     (t0 + 0.5 * dt).expand(x.shape[0]), y_r, frame_mask)
                x = x + dt * v_mid
            elif method == "ddim":
                x = self.transport.ddim_step(x, pred, t, t1.expand(x.shape[0]))
            else:
                raise ValueError(f"unknown method {method!r}")
        return x.reshape(n_samples, *z0.shape)

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = match_compiled_keys(self, checkpoint["state_dict"])

    def configure_optimizers(self):
        return configure_optimizers_for(self)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Skip an update if mixed-precision backward produced a non-finite gradient.
        """
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in self.parameters())
        if not finite:
            log.warning("non-finite gradient at step %d -- skipping this optimizer update",
                        self.trainer.global_step)
            optimizer.zero_grad(set_to_none=True)
