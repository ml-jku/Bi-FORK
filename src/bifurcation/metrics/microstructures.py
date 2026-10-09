"""Microstructures mode metrics.

Microstructures have DISCRETE modes, so labelling a generated rollout against the saved modes of a sample,
modes is the shared logic in :mod:`bifurcation.metrics.discrete` (re-exported here, since the
protocol config names this module). Specific to this dataset: the reconstruction fidelity
check.
"""

from __future__ import annotations

import torch

from bifurcation.metrics.discrete import (  # noqa: F401  (the protocol names them here)
    classify_Y, label_Y, match_rollout, representative_rollouts,
)
from bifurcation.metrics.discrete import reference_report as _reference_report


def reference_report(sample, Y_gen: list[torch.Tensor],
                     distance_metric: str = "mse") -> dict[str, float]:
    """Published errors plus a mode JSD that assumes no invalid class.

    Every generated rollout is assigned to the mode of its nearest saved rollout, even when
    the gated classifier calls it rejected or blurred.  Since all labels passed to
    :func:`bifurcation.metrics.modes.jsd_uniform` are non-negative, its invalid sink has zero
    mass and this is exactly a natural-log JSD over the sample's ``K`` modes. The generation
    callback retains the gated value separately as ``mode_jsd_invalid``.
    """
    return _reference_report(sample, Y_gen, distance_metric=distance_metric,
                             include_mode_jsd=True)
