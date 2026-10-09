"""Adapted from https://github.com/ashleve/lightning-hydra-template
(`src/utils/rich_utils.py`), without the queue over unlisted keys, the save-to-file
branch and the prompt.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

SECTIONS = ("data", "model", "callbacks", "logger", "trainer", "paths", "env")
SCALARS = ("dataset_name", "run_name", "seed", "tags", "train", "test", "ckpt_path")


# <match="../../../resources/lightning_hydra_template/src/utils/rich_utils.py?plain=1#L18">
def print_config_tree(cfg: DictConfig) -> None:
    import rich
    import rich.syntax
    import rich.tree

    tree = rich.tree.Tree("CONFIG", style="dim", guide_style="dim")
    for key in SECTIONS:
        if key not in cfg:
            continue
        branch = tree.add(key, style="bold")
        branch.add(rich.syntax.Syntax(OmegaConf.to_yaml(cfg[key], resolve=True), "yaml",
                                      background_color="default"))
    for key in SCALARS:
        if key in cfg:
            tree.add(f"{key}: {cfg[key]}", style="bold")
    rich.print(tree)
