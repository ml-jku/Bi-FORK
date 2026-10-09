"""beam3d mode metrics.

The modes of beam3d are CONTINUOUS -- a whole circle of buckling azimuths -- so classifying,
matching and reference-scoring against the orbit is the shared logic in
:mod:`bifurcation.metrics.continuous`, re-exported here because the protocol config names this
module. The discrete-mode datasets go through :mod:`bifurcation.metrics.discrete` the same way.
"""

from __future__ import annotations

from bifurcation.metrics.continuous import (  # noqa: F401  (the protocol names them here)
    ajne_uniform,
    classify_Y,
    label_Y,
    match_orbit,
    reference_report,
    representative_rollouts,
    y_magnitude,
)
