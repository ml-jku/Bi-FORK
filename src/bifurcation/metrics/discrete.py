"""Mode metrics for discrete modes.

Classifies generated rollouts against the saved modes of a sample and matches them to
their nearest rollout. The aggregate metrics, which also apply to continuous modes, are
found in :mod:`bifurcation.metrics.modes`.

Reproduces the published evaluation protocol of the reference project.
"""

from __future__ import annotations

import torch

from bifurcation.metrics.modes import BLURRED, REJECTED


def label_Y(sample, Y, max_abs_distance: float | None = None,
            max_rel_distance: float | None = None,
            distance_metric: str = "mse") -> tuple[int, int, torch.Tensor]:
    """
    Label a generated rollout and match it to its nearest saved rollout.

    Labelling and matching share one expensive step: the distance sweep over all saved
    rollouts. This call runs the sweep once and returns both results; :func:`classify_Y`
    gives only the label, :func:`match_rollout` only the match.

    Args:
        sample: the sample of this condition; provides the modes and the saved rollouts
        Y (torch.Tensor): generated rollout [T_, N_, C_]. The sweep runs on the device the
            rollouts are saved on; Y is moved there.
        max_abs_distance (float | None): rejection limit on the mean L2 node distance to the
            nearest mode, in raw field units (lang-ok); None skips it
        max_rel_distance (float | None): the same distance as a fraction of the own
            mode magnitude (``modes.magnitude``); None skips it
        distance_metric (str): which nearest-rollout distance decides the match, see
            :meth:`DiscreteModes.nearest_rollout`. ``"mse"`` (default) reproduces today's
            behavior for every dataset; only Allen-Cahn's protocol sets anything else.
    Returns:
        label (int): mode of the nearest branch, REJECTED (farther than a cutoff from
            every mode) or BLURRED (closer to the mean of the modes than to any mode,
            see ``DiscreteModes.closer_to_mean``)
        k (int): index of the nearest saved rollout
        Y_k: that rollout [T_, N_, C_], NaNs zeroed, on ``Y``'s device
    """
    modes = sample.modes
    k, _ = modes.nearest_rollout(Y, distance_metric=distance_metric)
    if k < 0:
        return REJECTED, 0, torch.nan_to_num(sample.Y(0)).to(Y.device)
    label = int(modes.rollout_labels[k])
    d = modes.l2_distance(Y, k)
    if modes.too_far(d, max_abs_distance, max_rel_distance):
        label = REJECTED
    elif modes.closer_to_mean(Y, d, max_abs_distance, max_rel_distance):
        label = BLURRED
    return label, k, torch.nan_to_num(sample.Y(k)).to(Y.device)


def classify_Y(sample, Y, max_abs_distance: float | None = None,
               max_rel_distance: float | None = None,
               distance_metric: str = "mse") -> int:
    """The label part of :func:`label_Y`."""
    return label_Y(sample, Y, max_abs_distance, max_rel_distance, distance_metric)[0]


def match_rollout(sample, Y, distance_metric: str = "mse") -> tuple[int, torch.Tensor]:
    """The match part of :func:`label_Y`, without rejection.

    Args:
        Y (torch.Tensor): generated rollout [T_, N_, C_]
    Returns:
        k (int): index of the nearest saved rollout (0 when nothing is comparable)
        Y_k: that rollout [T_, N_, C_], NaNs zeroed, on ``Y``'s device
    """
    k = max(sample.modes.nearest_rollout(Y, distance_metric=distance_metric)[0], 0)
    return k, torch.nan_to_num(sample.Y(k)).to(Y.device)


def _matched_final_mae(d: torch.Tensor) -> float:
    """MAE at the last frame with any finite entry (nan when no frame qualifies) -- the
    shared last step of :func:`reference_report`'s ``mc_mae``, reused for the pre/post
    relaxation variants Allen-Cahn adds on top (:func:`bifurcation.metrics.allencahn.reference_report`)."""
    finite_frames = d.isfinite().any(dim=(1, 2))
    if not finite_frames.any():
        return float("nan")
    last = int(torch.nonzero(finite_frames).flatten()[-1])
    return float(torch.nanmean(d[last].abs()))


def reference_report(sample, Y_gen: list[torch.Tensor],
                     distance_metric: str = "mse",
                     include_mode_jsd: bool = False) -> dict[str, float]:
    """The published evaluation of the trials of one sample, mode-conditional.

    Every trial against the saved rollout NEAREST to it, with no rejection. Only frames allowed
    by the mode's validity mask count; for microstructures this excludes every frame at or after
    self-contact. The final-frame MAE uses the last allowed frame, not the physical array end.

    Returns:
    - ``mc_mse``          the published headline: squared error summed over coordinates,
      averaged over frames x nodes;
    - ``mc_mae``          MAE over nodes x coordinates at the final frame;
    - ``mc_mae_overall``  the same over all frames.
    """
    mse, mae_final, mae_overall, nearest_labels = [], [], [], []
    for Y in Y_gen:
        k = max(sample.modes.nearest_rollout(Y, distance_metric=distance_metric)[0], 0)
        nearest_labels.append(int(sample.modes.rollout_labels[k]))
        d = Y - sample.Y(k).to(Y.device)
        valid = sample.modes.valid_mask
        if valid is not None:
            valid = valid[k].to(Y.device)
            d = torch.where(valid[:, None, None], d, torch.nan)
        mse.append(float(torch.nanmean(d.square().sum(-1))))
        mae_overall.append(float(torch.nanmean(d.abs())))
        mae_final.append(_matched_final_mae(d))
    report = {"mc_mse": sum(mse) / len(mse),
              "mc_mae": sum(mae_final) / len(mae_final),
              "mc_mae_overall": sum(mae_overall) / len(mae_overall)}
    if include_mode_jsd:
        from bifurcation.metrics.modes import jsd_uniform
        report["mode_jsd"] = jsd_uniform(nearest_labels, sample.modes.n_modes)
    return report


def representative_rollouts(sample) -> list[torch.Tensor]:
    """The representative rollout of every mode, in label order -- what a side-by-side
    figure draws each generated rollout against. Each entry [T_, N_, C_], NaNs zeroed,
    on the device the rollouts are saved on."""
    modes = sample.modes
    return [torch.nan_to_num(modes.representative_rollout(m)) for m in range(modes.n_modes)]
