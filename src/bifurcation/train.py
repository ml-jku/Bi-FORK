"""Training entry point.

    python -m bifurcation.train dataset_name=beam3d run_base=smoke trainer.max_steps=10

That is what runs today. The `experiment` and `ablation` groups land at commit 26, and
from then on one argument names the run:

    python -m bifurcation.train experiment=beam3d/first_stage [env=remote] [ablation=...]

Follows https://github.com/ashleve/lightning-hydra-template (`src/train.py`), without its
task wrapper, tag prompt and optimized-metric return.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import matplotlib
import torch
from omegaconf import DictConfig, open_dict

from bifurcation.logutils import (
    build_run_name,
    get_logger,
    log_hyperparameters,
    print_config_tree,
)

matplotlib.use("Agg")  # headless: callbacks render figures to files

CONFIGS = str(Path(__file__).resolve().parents[2] / "configs")
log = get_logger(__name__)


# <match="../../resources/lightning_hydra_template/src/train.py?plain=1#L108">
@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="train")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    if cfg.get("print_config", True):
        print_config_tree(cfg)

    loggers = []
    if cfg.get("logger"):
        run, group = build_run_name(cfg)
        with open_dict(cfg):
            for spec in cfg.logger.values():
                if spec.get("name", None) is None:
                    spec.name = run
                if "wandb" in str(spec.get("_target_", "")).lower():
                    spec.group = group
        loggers = [hydra.utils.instantiate(spec) for spec in cfg.logger.values()]

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    callbacks = [hydra.utils.instantiate(c) for c in (cfg.get("callbacks") or {}).values()]
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log_hyperparameters(cfg, model, trainer)
    log.info(f"{cfg.run_name} -> {cfg.paths.output_dir}")

    if cfg.get("train", True):
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        trainer.save_checkpoint(Path(cfg.paths.model_dir) / "checkpoints" / "final.ckpt")
    if cfg.get("test", False):
        trainer.test(model, datamodule=datamodule,
                     ckpt_path="best" if cfg.get("train", True) else cfg.get("ckpt_path"))
    return trainer.callback_metrics


if __name__ == "__main__":
    main()
