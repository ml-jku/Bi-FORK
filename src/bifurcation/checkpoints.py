"""Let torch load our own checkpoints.

`save_hyperparameters` keeps the hydra config, so a checkpoint holds `DictConfig` objects, and
from torch 2.6 `torch.load` defaults to `weights_only=True` and refuses them. The fix is to name
the classes rather than turn the check off, because these files travel between machines.
Imported for its effect from `bifurcation/__init__.py`, before Lightning loads anything.
"""

from __future__ import annotations

import collections
import typing

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

SAFE_GLOBALS = [
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Metadata,
    AnyNode,
    collections.defaultdict,
    dict,
    list,
    int,
    float,
    str,
    bool,
    type(None),
    typing.Any,
]


def allow_our_checkpoints() -> None:
    """Register the classes our checkpoints hold. Safe to call more than once."""
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


def match_compiled_keys(module: torch.nn.Module, state_dict: dict) -> dict:
    """Rename keys to the module's own, with or without torch.compile's `_orig_mod.`."""
    own = {k.replace("_orig_mod.", ""): k for k in module.state_dict()}
    return {own.get(k.replace("_orig_mod.", ""), k): v for k, v in state_dict.items()}
