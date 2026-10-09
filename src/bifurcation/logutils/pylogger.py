"""Adapted from https://github.com/ashleve/lightning-hydra-template
(`src/utils/pylogger.py`). The `RankedLogger` there is a `LoggerAdapter` that prefixes
every message with its rank and can log from all of them; we only ever want rank zero,
so this is the two-line version of that.
"""

from __future__ import annotations

import logging

from lightning_utilities.core.rank_zero import rank_zero_only

_LEVELS = ("debug", "info", "warning", "error", "exception", "critical")


# <match="../../../resources/lightning_hydra_template/src/utils/pylogger.py?plain=1#L7">
def get_logger(name: str) -> logging.Logger:
    """Logger whose every level is silenced off rank zero."""
    logger = logging.getLogger(name)
    for level in _LEVELS:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger
