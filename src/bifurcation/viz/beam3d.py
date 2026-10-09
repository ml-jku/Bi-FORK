"""beam3d figures: beam reconstructions, generated mode paths, and rollout animations.

Every figure pairs the two informative projections left|right: the TOP view (x-y, where the
azimuth -- the mode -- shows) and the SIDE view (deflection vs height, where the buckling
physics shows). Positions are the deformed beam ``X = p + Y``.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from bifurcation.metrics.modes import BLURRED, REJECTED
from bifurcation.utils.math import get_azimuth_from_sample, z_rotation


def _polyline(ax, X, color, label=None, **kw):
    ax.plot(X[:, 0], X[:, 1], marker="o", ms=3, color=color, label=label, **kw)


def _views(axes, X, color, label=None, **kw):
    """One beam configuration ``X [N_, 3]`` into (top, side) axes."""
    _polyline(axes[0], X[:, [0, 1]], color, label, **kw)
    _polyline(axes[1], X[:, [0, 2]], color, label, **kw)


def _style(axes, span):
    axes[0].set_title("top view (x-y)", fontsize=9)
    axes[1].set_title("side view (x-z)", fontsize=9)
    for ax, ylim in zip(axes, ((-span, span), (0.0, None))):
        ax.set_xlim(-span, span)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)


def plot_beam_reconstruction(p, x, y_true, y_pred, meta) -> plt.Figure:
    """Deformed beam at the last evaluated frame, GT (black) vs prediction (red), plus the
    undeformed reference (grey). ``x`` is unused (beam3d has no deformed-position extra)."""
    del x
    Xt, Xp = p + y_true[-1], p + y_pred[-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    _views(axes, p, "0.8", "undeformed")
    _views(axes, Xt, "black", "GT")
    _views(axes, Xp, "tab:red", "prediction")
    _style(axes, span=1.1 * max(float(Xt.abs().max()), float(Xp.abs().max())))
    axes[0].legend(fontsize=7)
    fig.suptitle(f"sample {meta['key']} frame {meta['frames'][-1]}", fontsize=10)
    return fig


def plot_mode_paths(sample, Y_gen, labels, meta) -> plt.Figure:
    """Generated TIP paths per condition; position encodes the mode, color the validity.

    - Left (top view): the in-plane tip path of every generated rollout, the GT orbit as a
      dotted circle and the canonical +x GT path in black. Uniform coverage looks like a fan.
    - Right (side view): in-plane tip deflection along the OWN azimuth of each rollout against
      tip height, over the GT buckling path.

    Valid green, REJECTED red, BLURRED grey.
    """
    p_tip = sample.p[-1]
    gt_tip = p_tip + sample.Y(0)[:, -1]                             # [T, 3], canonical +x
    r_gt = float(torch.linalg.vector_norm(gt_tip[-1, :2]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    top, side = axes
    theta = torch.linspace(0, 2 * math.pi, 200)
    top.plot(r_gt * theta.cos(), r_gt * theta.sin(), ":", color="0.6", lw=1,
             label="GT orbit")
    top.plot(gt_tip[:, 0], gt_tip[:, 1], color="black", lw=2, label="GT (+x)")
    side.plot(gt_tip[:, 0], gt_tip[:, 2], color="black", lw=2, label="GT")

    color_of = {REJECTED: "tab:red", BLURRED: "0.5"}
    for Y, label in zip(Y_gen, labels):
        tip = p_tip + Y[:, -1]                                      # [T, 3]
        color = color_of.get(label, "tab:green")
        top.plot(tip[:, 0], tip[:, 1], color=color, lw=1, alpha=0.8)
        u = tip[-1, :2]
        u = u / max(float(torch.linalg.vector_norm(u)), 1e-12)      # its own azimuth
        side.plot(tip[:, :2] @ u, tip[:, 2], color=color, lw=1, alpha=0.8)

    span = 1.3 * max(r_gt, 1e-3)
    top.set_xlim(-span, span), top.set_ylim(-span, span)
    top.set_aspect("equal")
    top.set_title("tip paths, top view (x-y)", fontsize=9)
    side.set_title("tip deflection along own azimuth vs height", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
    n_bad = sum(label < 0 for label in labels)
    fig.suptitle(f"sample {meta['key']} -- {len(labels)} generated, {n_bad} rejected/blurred",
                 fontsize=10)
    return fig


COMPARISON_PRED_COLOR = "#64B6AC"
COMPARISON_REJECTED_COLOR = "#DA667B"


def aligned_errors(sample, Y_gen) -> torch.Tensor:
    """Mean absolute position error ``[n_trials]`` of every trial against the GT, both rotated
    to buckle towards +x along their OWN azimuth -- every rotation of the GT is a valid
    solution, so a trial is only compared with the GT rotated into its direction."""
    X_gt = (sample.p + sample.Y(0)) @ z_rotation(-get_azimuth_from_sample(sample.Y(0))).T
    return torch.stack([((sample.p + Y) @ z_rotation(-get_azimuth_from_sample(Y)).T - X_gt)
                        .abs().mean() for Y in Y_gen])


def typical_trial(sample, Y_gen, labels) -> int:
    """Index of the valid trial (all trials if none is valid) with the median
    :func:`aligned_errors` -- a representative trial, not the best of the batch."""
    errors = aligned_errors(sample, Y_gen)
    valid = torch.tensor([label >= 0 for label in labels])
    candidates = valid.nonzero().flatten() if valid.any() else torch.arange(len(errors))
    return int(candidates[errors[candidates].argsort()[len(candidates) // 2]])


def _bending_frames(p, Y, n_frames):
    """``n_frames`` evenly spaced snapshots after t=0 of ``p + Y``, rotated by its own azimuth
    to buckle towards +x; side view ``[F, N_, 2]`` (x, z)."""
    X = (p + Y) @ z_rotation(-get_azimuth_from_sample(Y)).T
    t_idx = torch.linspace(0, len(Y) - 1, n_frames + 1)[1:].round().long()
    return X[t_idx][:, :, [0, 2]]


def _plot_bending(ax, frames, color, linestyle, lw_final, lw_early):
    """Snapshots ``[F, N_, 2]``, faint early -> opaque final, as one beam's motion trail."""
    n = len(frames)
    for i, frame in enumerate(frames):
        last = i == n - 1
        ax.plot(frame[:, 0], frame[:, 1], color=color, linestyle=linestyle, marker="o",
                ms=6.5 if last else 4.8, markeredgewidth=0, lw=lw_final if last else lw_early,
                alpha=1.0 if n == 1 else 0.3 + 0.7 * i / (n - 1), zorder=2 + i)


def _clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_mode_comparison(results: dict, n_frames: int = 5) -> plt.Figure:
    """One panel per method (``{name: Generated}``, needs ``keep_outputs``), as in the paper's
    comparison figure without the score badges.

    - Left (side view): real (black) vs predicted (dashed) bending of the :func:`typical_trial`,
      both along their own azimuth over the undeformed beam (light grey).
    - Right (top view): the final node-mean of every trial, whose angle is the azimuth; valid
      teal, REJECTED/BLURRED red, over a dashed circle at the GT's node-mean radius, so dots off
      the circle buckle too little or too much.
    """
    fig = plt.figure(figsize=(3.2 * len(results), 3.4))
    outer = fig.add_gridspec(1, len(results), wspace=0.12, left=0.02, right=0.98, top=0.88,
                             bottom=0.14)
    theta = torch.linspace(0, 2 * math.pi, 200)
    for col, (name, r) in enumerate(results.items()):
        p, Y_true = r.sample.p, r.sample.Y(0)
        shown = typical_trial(r.sample, r.Y_gen, r.rollout_labels)
        real = _bending_frames(p, Y_true, n_frames)
        pred = _bending_frames(p, r.Y_gen[shown], n_frames)
        rest = p[:, [0, 2]]

        inner = outer[0, col].subgridspec(1, 2, width_ratios=(1.3, 0.75), wspace=0.06)
        side, top = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

        side.plot(rest[:, 0], rest[:, 1], color="0.85", lw=2.0, marker="o", ms=4.8,
                  markeredgewidth=0, zorder=0)
        _plot_bending(side, real, "black", "-", lw_final=2.4, lw_early=1.7)
        _plot_bending(side, pred, COMPARISON_PRED_COLOR, "--", lw_final=2.9, lw_early=2.1)
        pts = torch.cat([real.reshape(-1, 2), pred.reshape(-1, 2), rest])
        lo, hi = pts.min(0).values, pts.max(0).values
        pad = 0.12 * (hi - lo)
        side.set_xlim(float(lo[0] - pad[0]), float(hi[0] + pad[0]))
        side.set_ylim(float(lo[1] - pad[1]), float(hi[1] + pad[1]))
        side.set_aspect("equal", adjustable="datalim")
        _clean(side)

        centroids = torch.stack([(p + Y[-1]).mean(0)[:2] for Y in r.Y_gen])
        valid = torch.tensor([label >= 0 for label in r.rollout_labels])
        radius = float(torch.linalg.vector_norm((p + Y_true[-1]).mean(0)[:2]))
        top.plot(radius * theta.cos(), radius * theta.sin(), "--", lw=0.8, color="0.6")
        top.scatter(centroids[valid, 0], centroids[valid, 1], s=20,
                    color=COMPARISON_PRED_COLOR, alpha=0.75, linewidths=0)
        top.scatter(centroids[~valid, 0], centroids[~valid, 1], s=20,
                    color=COMPARISON_REJECTED_COLOR, alpha=0.75, linewidths=0)
        span = 1.15 * max(radius, float(torch.linalg.vector_norm(centroids, dim=1).max()))
        top.set_xlim(-span, span), top.set_ylim(-span, span)
        top.set_aspect("equal", adjustable="datalim")
        _clean(top)

        x0, x1 = outer[0, col].get_position(fig).intervalx
        fig.text((x0 + x1) / 2, 0.93, name, ha="center", va="bottom", fontsize=14,
                 fontweight="bold")

    handles = [plt.Line2D([], [], color="black", lw=1.6),
               plt.Line2D([], [], color="0.4", lw=1.4, linestyle="--")]
    fig.legend(handles, ["real", "predicted"], loc="lower center", ncols=2, frameon=False,
               fontsize=12)
    return fig


def plot_angle_distribution(sample, Y_gen, labels, meta) -> plt.Figure:
    """Distribution (XY top-down) of every trial's final centroid, JSD/Ajne annotated.

    Port of microstructures_project's ``beam3d_evaluation_callback._plot_distribution``: a
    hexbin of every trial's final node-mean position, against the UNDEFORMED node-mean (not
    the buckled GT). JSD/Ajne (:func:`bifurcation.metrics.continuous.ajne_uniform`) come from
    the buckle angle of every trial, the same definition ``reference_report`` uses.
    """
    from bifurcation.metrics.continuous import TWO_PI, ajne_uniform

    angles = torch.tensor([get_azimuth_from_sample(Y) for Y in Y_gen], dtype=torch.float64)
    n_bins = 36
    hist = torch.histc(angles, bins=n_bins, min=0.0, max=TWO_PI)
    p = hist / max(float(hist.sum()), 1.0)
    q = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float64)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * torch.log(a[nz] / b[nz])).sum())

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    ajne = ajne_uniform(angles)

    centroids = torch.stack([(sample.p + Y[-1]).mean(dim=0)[:2] for Y in Y_gen])  # [n_trials, 2]
    init_centroid = sample.p.mean(dim=0)[:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(centroids[:, 0], centroids[:, 1], gridsize=35, cmap="GnBu", mincnt=1, linewidths=0.0)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=14, facecolors="none",
              edgecolors="tab:blue", alpha=0.6, zorder=3, label="Pred centroid")
    ax.scatter(init_centroid[0], init_centroid[1], s=120, color="tab:red", marker="x",
              zorder=5, label="Init centroid")
    ax.set_xlabel("X"), ax.set_ylabel("Y")
    ax.set_title(f"Distribution (XY top-down) -- condition {meta['key']}\n"
                 f"JSD vs uniform: {jsd:.4f}  |  {len(Y_gen)} samples")
    ax.text(0.02, 0.02, f"JSD = {jsd:.4f}\nAjne = {ajne:.4f}", transform=ax.transAxes,
           fontsize=9, va="bottom", ha="left", alpha=0.75)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def animate_generated_beam(sample, k: int, Y_pred: torch.Tensor, stride: int,
                           interval: int) -> FuncAnimation:
    """GIF of a generated rollout next to the GT orbit member at the generated azimuth:
    columns GT | model, rows top | side view. ``k`` is unused (single saved rollout)."""
    del k
    p = sample.p
    Y_gt = sample.modes.representative_rollout(get_azimuth_from_sample(Y_pred))
    frames = range(0, len(Y_pred), stride)
    span = 1.1 * max(float((p + Y_gt).abs().max()), float((p + Y_pred).abs().max()))

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 8.6))

    def update(t):
        for col, (Y, name) in enumerate([(Y_gt, "GT at generated azimuth"), (Y_pred, "model")]):
            X = p + Y[t]
            column = [axs[0, col], axs[1, col]]
            for ax in column:
                ax.clear()
            _views(column, X, "black" if col == 0 else "tab:red")
            _style(column, span)
            axs[0, col].set_title(f"{name} -- t={t}", fontsize=9)
        return []

    anim = FuncAnimation(fig, update, frames=list(frames), interval=interval)
    plt.close(fig)
    return anim




def draw_mode(ax, sample, Y: torch.Tensor) -> None:
    """One beam rollout into ONE axes, for the mode gallery: the TOP view, where the
    buckling azimuth -- the mode -- is what you see. The beam at its final frame (red) over the
    path its tip swept (grey)."""
    X = sample.p + Y                                  # deformed beam over time [T_, N_, 3]
    ax.set_axis_on()
    ax.plot(X[:, -1, 0], X[:, -1, 1], color="0.7", lw=1.0)   # the path of the tip
    _polyline(ax, X[-1][:, [0, 1]], "tab:red")               # the final buckled beam
    span = float(X[..., :2].abs().max()) * 1.1 + 1e-6
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
