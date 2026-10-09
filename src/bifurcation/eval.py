"""Score a checkpoint on the test split.

    python -m bifurcation.eval ckpt_path=/path/to.ckpt [env=remote]

Use the same experiment the checkpoint was trained with: the model is rebuilt from that
config and the weights are loaded into it, so the architecture has to match, `env` included,
because local is the small one. Everything reported comes from the generation callbacks the
experiment brings, which fire once at the end of the test epoch.

A two-stage checkpoint needs no `first_stage_ckpt` here. It already carries the frozen
autoencoder, so the mandatory value the training config demands is filled in below and every
weight comes from `ckpt_path`.

Follows https://github.com/ashleve/lightning-hydra-template (`src/eval.py`).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import matplotlib
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from bifurcation.logutils import build_run_name, get_logger, log_hyperparameters, print_config_tree

matplotlib.use("Agg")  # headless: callbacks render figures to files

CONFIGS = str(Path(__file__).resolve().parents[2] / "configs")
log = get_logger(__name__)


# <match="../../resources/lightning_hydra_template/src/eval.py?plain=1#L86">
@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="eval")
def main(cfg: DictConfig):
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    if cfg.get("print_config", True):
        print_config_tree(cfg)

    loggers = []
    if cfg.get("logger"):
        run, group = build_run_name(cfg)
        with open_dict(cfg):
            for spec in cfg.logger.values():
                spec.name = run
                if "wandb" in str(spec.get("_target_", "")).lower():
                    spec.group = group
        loggers = [hydra.utils.instantiate(spec) for spec in cfg.logger.values()]

    datamodule = hydra.utils.instantiate(cfg.data)
    with open_dict(cfg):
        if OmegaConf.is_missing(cfg.model, "first_stage_ckpt"):
            cfg.model.first_stage_ckpt = None
    model = hydra.utils.instantiate(cfg.model)

    callbacks = [hydra.utils.instantiate(c) for c in (cfg.get("callbacks") or {}).values()
                 if c is not None and "_target_" in c]
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log_hyperparameters(cfg, model, trainer)

    state = torch.load(cfg.ckpt_path, map_location="cpu",
                       weights_only=not cfg.get("reference", False))["state_dict"]
    if cfg.get("reference", False):
        from bifurcation.models.reference_checkpoints import convert_reference_checkpoint
        state = convert_reference_checkpoint(state)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})

    log.info(f"scoring {cfg.ckpt_path} -> {cfg.report_dir}")
    trainer.test(model, datamodule=datamodule)
    return trainer.callback_metrics


if __name__ == "__main__":
    main()
