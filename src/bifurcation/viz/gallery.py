"""The side-by-side picture: every ground-truth mode, and what the model produced for it.

The figure behind ``gen_coverage``. A mode the model never produced is drawn as an **empty
panel**, and non-branch outputs are collected at the end, so it cannot disagree with the numbers.

- ``pairs``: one panel pair per mode, ground truth beside the closest output or a blank.
- ``by_mode``: one row per mode, up to ``max_per_mode`` outputs, so it shows how many went where.

``draw(ax, sample, Y)`` renders one rollout into one axes; the only dataset-specific part.
"""

from __future__ import annotations

from typing import Callable

import math

import matplotlib.pyplot as plt
import torch

from bifurcation.datasets.utils.modes import fft_mse_distance


def assign_to_modes(gt_modes: list[torch.Tensor], Y_gen: list[torch.Tensor],
                    labels: list[int], distance_metric: str = "mse"
                    ) -> tuple[dict[int, list[torch.Tensor]], list[torch.Tensor]]:
    """Sort the generated rollouts under the mode each one landed on.

    Whether an output is a branch at all is for the metrics to decide, never for this figure, so
    what is drawn and what is counted cannot drift apart. Which mode it landed on is the nearest
    ground-truth rollout under ``distance_metric`` -- should match ``protocol.distance_metric``,
    see docs/decisions.md -- which is the label itself for discrete modes, continuous too.

    ``distance_metric="label"``: trust ``labels`` itself as the bin index (still ``< 0`` for
    "no branch"), instead of recomputing a nearest mode from raw/FFT distance. For a symmetric
    dataset (translation/rotation/sign-flip equivalent branches, e.g. Allen-Cahn) neither ``mse``
    nor plain ``fft_mse`` reliably agree with a symmetry-aware matcher such as
    :func:`bifurcation.metrics.allencahn_symmetry.match_all` -- a correctly generated but
    translated/rotated/sign-flipped output can have huge raw distance to its own true branch,
    so recomputing nearest-mode here would silently disagree with whatever produced ``labels``.
    """
    by_mode: dict[int, list[torch.Tensor]] = {m: [] for m in range(len(gt_modes))}
    no_branch: list[torch.Tensor] = []
    if distance_metric == "fft_mse":
        stacked = torch.stack(gt_modes)
    for Y, label in zip(Y_gen, labels):
        if label < 0:
            no_branch.append(Y)
        elif distance_metric == "label":
            by_mode[label].append(Y)
        elif distance_metric == "mse":
            nearest = min(range(len(gt_modes)),
                          key=lambda m: float(torch.nanmean((gt_modes[m] - Y) ** 2)))
            by_mode[nearest].append(Y)
        elif distance_metric == "fft_mse":
            nearest = int(fft_mse_distance(Y, stacked).argmin())
            by_mode[nearest].append(Y)
        else:
            raise ValueError(f"unknown distance_metric {distance_metric!r}")
    return by_mode, no_branch


def mode_gallery(sample, Y_gen: list[torch.Tensor], labels: list[int],
                 gt_modes: list[torch.Tensor], draw: Callable, layout: str = "pairs",
                 max_per_mode: int = 4, n_cols: int = 3, panel: float = 2.2,
                 distance_metric: str = "mse") -> plt.Figure:
    """Ground truth beside what the model produced, for every mode of one condition.

    ``layout``: ``pairs`` (compact, one ground-truth/model pair per mode, ``n_cols`` pairs per
    row) or ``by_mode`` (one row per mode, up to ``max_per_mode`` outputs beside it).
    """
    Y_gen = [torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0) for Y in Y_gen]
    by_mode, no_branch = assign_to_modes(gt_modes, Y_gen, labels, distance_metric)
    covered = sum(1 for m in by_mode if by_mode[m])
    title = (f"{sample.key}: {covered}/{len(gt_modes)} modes produced, "
             f"{len(no_branch)}/{len(Y_gen)} outputs not a branch")

    if layout == "pairs":
        fig = _pairs(sample, gt_modes, by_mode, no_branch, draw, n_cols, panel)
    elif layout == "by_mode":
        fig = _by_mode(sample, gt_modes, by_mode, no_branch, draw, max_per_mode, panel)
    else:
        raise ValueError(f"unknown layout {layout!r}, expected 'pairs' or 'by_mode'")

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def _blank(ax, text: str) -> None:
    """A mode the model never produced: an empty panel that says so."""
    ax.set_title(text, fontsize=7, color="tab:red")
    ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, color="tab:red",
            transform=ax.transAxes)


def _pairs(sample, gt_modes, by_mode, no_branch, draw, n_cols, panel) -> plt.Figure:
    cells = len(gt_modes) + (1 if no_branch else 0)
    rows = math.ceil(cells / n_cols)
    fig, axes = plt.subplots(rows, 2 * n_cols, figsize=(panel * 2 * n_cols, panel * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()

    for m, Y_gt in enumerate(gt_modes):
        r, c = divmod(m, n_cols)
        ax_gt, ax_gen = axes[r][2 * c], axes[r][2 * c + 1]
        draw(ax_gt, sample, Y_gt)
        ax_gt.set_title(f"mode {m}: ground truth", fontsize=7)
        if by_mode[m]:
            draw(ax_gen, sample, by_mode[m][0])
            ax_gen.set_title(f"model ({len(by_mode[m])} of {sum(len(v) for v in by_mode.values())})",
                             fontsize=7)
        else:
            _blank(ax_gen, "not produced")

    if no_branch:  # the outputs that are no branch at all, so nothing is hidden
        r, c = divmod(len(gt_modes), n_cols)
        draw(axes[r][2 * c + 1], sample, no_branch[0])
        axes[r][2 * c + 1].set_title(f"no branch ({len(no_branch)})", fontsize=7, color="tab:red")
    return fig


def _by_mode(sample, gt_modes, by_mode, no_branch, draw, max_per_mode, panel) -> plt.Figure:
    rows = len(gt_modes) + (1 if no_branch else 0)
    cols = 1 + max_per_mode
    fig, axes = plt.subplots(rows, cols, figsize=(panel * cols, panel * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()

    for m, Y_gt in enumerate(gt_modes):
        draw(axes[m][0], sample, Y_gt)
        axes[m][0].set_title(f"mode {m}", fontsize=8)
        produced = by_mode[m]
        for c in range(max_per_mode):
            ax = axes[m][1 + c]
            if c < len(produced):
                draw(ax, sample, produced[c])
                ax.set_title(f"generated {c + 1}", fontsize=7)
            elif c == 0:
                _blank(ax, "not produced")

    if no_branch:
        row = axes[len(gt_modes)]
        row[0].set_title("no branch", fontsize=8, color="tab:red")
        for c, Y in enumerate(no_branch[:max_per_mode]):
            draw(row[1 + c], sample, Y)
            row[1 + c].set_title("rejected", fontsize=7, color="tab:red")
    return fig
