"""One name for a run, used by the paths and by the logger alike.

``run_base`` is what the experiment sets. ``run_name`` is that plus a short tag for the
command-line overrides that change what is trained, so a sweep cannot write two runs into one
directory::

    train experiment=beam3d/first_stage                     -> first_stage
    train experiment=beam3d/first_stage data.batch_size=4   -> first_stage__bs4

The tag comes from a resolver, because hydra resolves ``hydra.run.dir`` before ``main`` runs.
The logger groups by ``run_base``, so the seeds of one run group together.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

OVERRIDE_KEYS: dict[str, str] = {
    "data.num_nodes": "nn",
    "data.batch_size": "bs",
    "model.optimizer.lr": "lr",
}


def override_tag(dirname: str) -> str:
    """``data.batch_size=4,seed=7`` -> ``__bs4``. Empty when nothing listed was set."""
    tags = []
    for item in dirname.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        short = OVERRIDE_KEYS.get(key.lstrip("~+"))
        if short:
            tags.append(f"{short}{value.replace('/', '_')}")
    return f"__{'_'.join(tags)}" if tags else ""


OmegaConf.register_new_resolver("override_tag", override_tag, replace=True)


def build_run_name(cfg: DictConfig) -> tuple[str, str]:
    """``(run, group)`` for the logger: the run name, and the same without the tag."""
    return str(cfg.run_name), str(cfg.get("run_base") or cfg.run_name)
