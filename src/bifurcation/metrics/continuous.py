"""Mode metrics for continuous modes.

A continuous-mode sample keeps no mode list: one representative rollout, and a symmetry that
moves it along an ORBIT of equivalent solutions. Classifying a generated rollout means measuring
its azimuth and discretizing it into ``n_bins`` bins; see docs/decisions.md.

Reproduces the published evaluation protocol of the reference project.
"""

from __future__ import annotations

import math

import torch

from bifurcation.utils.math import get_azimuth_from_sample

TWO_PI = 2.0 * math.pi


def y_magnitude(sample) -> float:
    """Mean L2 node displacement of the rollout -- the scale a relative distance is measured
    against, in the same units as the distances themselves."""
    return float(torch.nanmean(sample.Y(0).square().sum(-1).sqrt()))


def rotate_z(Y: torch.Tensor, phi: float) -> torch.Tensor:
    """``Y`` with its xy components rotated by ``phi`` about z. The nodes lie on the z axis,
    so rotating the in-plane components of the field (lang-ok) rotates the whole solution along the orbit."""
    c, s = math.cos(phi), math.sin(phi)
    out = Y.clone()
    out[..., 0] = c * Y[..., 0] - s * Y[..., 1]
    out[..., 1] = s * Y[..., 0] + c * Y[..., 1]
    return out


def label_Y(sample, Y: torch.Tensor, n_bins: int,
            max_abs_distance: float | None = None,
            max_rel_distance: float | None = None) -> tuple[int, int, torch.Tensor]:
    """``(label, rollout index, orbit member)`` at the generated azimuth, measured once.

    The label is from :func:`classify_Y` and the match from :func:`match_orbit`, fused because
    both need the azimuth and the orbit member there, and building that member is the expensive
    part. Rejected (-1) when the mean L2 node distance to the orbit member exceeds either
    cutoff: garbage, unbuckled or off-orbit outputs.
    """
    phi = get_azimuth_from_sample(Y)
    ref = sample.modes.representative_rollout(phi)
    label = int(phi / (TWO_PI / n_bins)) % n_bins
    if max_abs_distance is not None or max_rel_distance is not None:
        d = float(torch.nanmean((Y - ref).square().sum(-1).sqrt()))
        if max_abs_distance is not None and d > max_abs_distance:
            label = -1
        elif max_rel_distance is not None and d > max_rel_distance * y_magnitude(sample):
            label = -1
    return label, 0, ref


def classify_Y(sample, Y: torch.Tensor, n_bins: int,
               max_abs_distance: float | None = None,
               max_rel_distance: float | None = None) -> int:
    """Azimuth bin of a generated rollout, or ``-1`` if rejected (see :func:`label_Y`)."""
    return label_Y(sample, Y, n_bins, max_abs_distance, max_rel_distance)[0]


def ajne_uniform(angles: torch.Tensor) -> float:
    """Ajne's A_n statistic for uniformity of directions on the circle (Ajne, 1968), bin-free.

    ``A_n = n/4 - 1/(pi*n) * sum_{i<j} arccos(cos(angle_i - angle_j))``. Its exact null mean
    under uniformity is ``n/4`` for any ``n`` (not 0); large deviation above that signals
    clustering a coarse ``angle_jsd`` histogram can miss -- see docs/decisions.md for the
    empirical null spread at ``n=512``.
    """
    n = angles.shape[0]
    if n < 2:
        return float("nan")
    diff = angles[:, None] - angles[None, :]
    theta = torch.arccos(torch.clamp(torch.cos(diff), -1.0, 1.0))
    pair_sum = float(torch.triu(theta, diagonal=1).sum())
    return n / 4.0 - pair_sum / (math.pi * n)


def reference_report(sample, Y_gen: list[torch.Tensor], n_bins: int = 36) -> dict[str, float]:
    """The published evaluation of the trials of one sample, azimuth-canonicalized.

    Every trial AND the ground truth are rotated about z to zero azimuth, so only the SHAPE is
    compared. Every trial is scored over the whole rollout, ignoring the cutoffs and
    ``first_scored_frame``, which tune our own metrics instead.

    Returns:
    - ``mc_mae``          MAE over nodes x coordinates at the final frame;
    - ``mc_mae_overall``  the same over all frames, each rotated by its own final azimuth;
    - ``angle_jsd``       JSD of the azimuth histogram over ``n_bins`` bins against uniform,
      natural log and no invalid sink, exactly the published convention;
    - ``angle_ajne``      Ajne's bin-free companion uniformity statistic on the same angles.
    """
    gt = sample.Y(0)
    gt_zeroed = rotate_z(gt, -get_azimuth_from_sample(gt))
    angles, mae_final, mae_overall = [], [], []
    for Y in Y_gen:
        phi = get_azimuth_from_sample(Y)
        angles.append(phi)
        zeroed = rotate_z(Y, -phi)
        mae_final.append(float((zeroed[-1] - gt_zeroed[-1]).abs().mean()))
        mae_overall.append(float((zeroed - gt_zeroed).abs().mean()))

    angles_t = torch.tensor(angles, dtype=torch.float64)
    hist = torch.histc(angles_t, bins=n_bins, min=0.0, max=TWO_PI)
    p = hist / max(float(hist.sum()), 1.0)
    q = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float64)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * torch.log(a[nz] / b[nz])).sum())

    return {"mc_mae": sum(mae_final) / len(mae_final),
            "mc_mae_overall": sum(mae_overall) / len(mae_overall),
            "angle_jsd": 0.5 * kl(p, m) + 0.5 * kl(q, m),
            "angle_ajne": ajne_uniform(angles_t)}


def match_orbit(sample, Y: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Matched reference for a generated rollout: the orbit member at its azimuth
    (rollout index 0 -- there is only one saved rollout, moved along the orbit)."""
    return 0, sample.modes.representative_rollout(get_azimuth_from_sample(Y))


def representative_rollouts(sample, n_modes: int) -> list[torch.Tensor]:
    """The representative rollout of every shown mode: orbit members at ``n_modes``
    evenly spaced azimuths (bin centres), each [T_, N_, 3].

    Continuous modes are an orbit, so "every mode" is a display choice rather than a
    property of the sample. Keep ``n_modes`` well below the bin count of the metrics --
    a gallery with one row per bin says nothing.
    """
    return [sample.modes.representative_rollout((j + 0.5) * TWO_PI / n_modes)
            for j in range(n_modes)]
