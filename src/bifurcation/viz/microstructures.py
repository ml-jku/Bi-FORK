"""Microstructures figures: reconstruction plots and generated-rollout animations."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from bifurcation.viz.utils import quiver_panel, zoom_window



def plot_fluctuation_magnitudes(p, x, y_true, y_pred, meta) -> plt.Figure:
    """|y| scatter at the deformed positions, last evaluated frame: GT / prediction / error."""
    pos, yt, yp = x[-1], y_true[-1], y_pred[-1]
    fields = [(torch.linalg.vector_norm(yt, dim=-1), "GT |y|"),
              (torch.linalg.vector_norm(yp, dim=-1), "pred |y|"),
              (torch.linalg.vector_norm(yp - yt, dim=-1), "|error|")]
    vmax = max(float(fields[0][0].max()), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (c, title) in zip(axes, fields):
        s = ax.scatter(pos[:, 0], pos[:, 1], c=c, s=2, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(s, ax=ax, shrink=0.8)
    fig.suptitle(f"sample {meta['key']} rollout {meta['k']} frame {meta['frames'][-1]}")
    return fig


def plot_fluctuation_arrows(p, x, y_true, y_pred, meta, side_by_side: bool) -> plt.Figure:
    """Fluctuation arrows at the affine positions, last frame; rows = zoom levels (the zoom
    centres on the max-|y| node). Overlaid GT (black) + prediction (red), or side-by-side
    GT | prediction columns."""
    affine = x[-1] - y_true[-1]
    yt, yp = y_true[-1], y_pred[-1]
    zooms = (1.0, 0.15)
    n_cols = 2 if side_by_side else 1
    fig, axes = plt.subplots(len(zooms), n_cols, figsize=(5.5 * n_cols, 5 * len(zooms)),
                             constrained_layout=True, squeeze=False)
    for row, zoom in enumerate(zooms):
        lo, hi = zoom_window(affine, yt, zoom)
        if side_by_side:
            for col, (label, field, color) in enumerate([("GT", yt, "black"), ("pred", yp, "tab:red")]):
                quiver_panel(axes[row, col], affine, field, lo, hi, color, n_arrows=400)
                axes[row, col].set_title(f"{label}, zoom {zoom}", fontsize=9)
        else:
            quiver_panel(axes[row, 0], affine, yt, lo, hi, "black", n_arrows=400)
            quiver_panel(axes[row, 0], affine, yp, lo, hi, "tab:red", n_arrows=400)
            axes[row, 0].set_title(f"GT (black) vs pred (red), zoom {zoom}", fontsize=9)
    fig.suptitle(f"sample {meta['key']} rollout {meta['k']} frame {meta['frames'][-1]}")
    return fig




def animate_generated_arrows(sample, k: int, Y_pred: torch.Tensor, zooms: tuple, n_arrows: int,
                             stride: int, interval: int) -> FuncAnimation:
    """GIF of a generated rollout next to its matched GT rollout.

    Rows are zoom levels, left column GT (black), right column model (red). Nodes are drawn at
    the affine positions, arrows are the fluctuations. ALL saved frames are animated; frames from
    self-contact on stay in the gif, flagged PHYSICALLY INVALID in red rather than cut. Windows
    are fixed at the final pre-contact frame so the view does not jump.
    """
    contact = sample.after_contact.bool()
    frames = torch.nonzero(sample.valid_t(k)).flatten()[::stride]
    t_end = int(torch.nonzero(sample.valid_t(k) & ~contact).flatten()[-1])
    affine, Y_gt = sample.affine, torch.nan_to_num(sample.Y(k))
    windows = [zoom_window(affine[t_end], Y_gt[t_end], frac) for frac in zooms]
    fig, axs = plt.subplots(len(zooms), 2, figsize=(8.6, 4.3 * len(zooms)), squeeze=False)

    def update(t):
        fig.suptitle("PHYSICALLY INVALID -- post-contact" if contact[t] else "",
                     fontsize=10, color="tab:red")
        for r, (lo, hi) in enumerate(windows):
            for col, (y, color) in enumerate([(Y_gt[t], "black"), (Y_pred[t], "tab:red")]):
                ax = axs[r, col]
                ax.clear()
                quiver_panel(ax, affine[t], y, lo, hi, color, n_arrows)
            axs[r, 0].set_title(f"ground truth k={k} -- t={t}", fontsize=9)
            axs[r, 1].set_title(f"model -- t={t}", fontsize=9)
        return []

    anim = FuncAnimation(fig, update, frames=[int(t) for t in frames], interval=interval)
    plt.close(fig)
    return anim


def animate_generated_rollouts(sample, k: int, Y_pred: torch.Tensor, zooms: tuple,
                                   stride: int, interval: int,
                                   point_size: float = 1.5) -> FuncAnimation:
    """GIF of the deformed configurations instead of fluctuation arrows.

    GT left, model right, rows are zoom levels, points colored by |y| on a shared scale. Frame
    policy matches :func:`animate_generated_arrows`. The full view covers the union of all
    frames, because the structure compresses under loading; zooms < 1 stay centred on the
    max-|y| node at the final pre-contact frame.
    """
    contact = sample.after_contact.bool()
    valid = sample.valid_t(k) & ~contact  # never animate physically invalid (post-contact) frames
    frames = torch.nonzero(valid).flatten()[::stride]
    t_end = int(torch.nonzero(valid).flatten()[-1])
    Y_gt = torch.nan_to_num(sample.Y(k))
    X_gt, X_pred = sample.X(k), sample.affine + Y_pred
    C_gt = torch.linalg.vector_norm(Y_gt, dim=-1)
    C_pred = torch.linalg.vector_norm(Y_pred, dim=-1)
    vmax = max(float(C_gt[valid & ~contact].max()), 1e-12)
    flat = X_gt[valid].reshape(-1, X_gt.shape[-1])
    windows = [zoom_window(flat, flat, frac) if frac >= 1
               else zoom_window(X_gt[t_end], Y_gt[t_end], frac) for frac in zooms]

    fig, axs = plt.subplots(len(zooms), 2, figsize=(9.4, 4.3 * len(zooms)), squeeze=False)
    scats = [[None, None] for _ in windows]
    for r, (lo, hi) in enumerate(windows):
        for c in range(2):
            ax = axs[r, c]
            scats[r][c] = ax.scatter([], [], s=point_size, c=[], cmap="viridis",
                                     vmin=0, vmax=vmax)
            ax.set_xlim(float(lo[0]), float(hi[0]))
            ax.set_ylim(float(lo[1]), float(hi[1]))
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
    fig.colorbar(scats[0][0], ax=axs.ravel().tolist(), shrink=0.7, label="|y|")

    def update(t):
        for r in range(len(zooms)):
            for c, (X, C) in enumerate([(X_gt, C_gt), (X_pred, C_pred)]):
                scats[r][c].set_offsets(X[t])
                scats[r][c].set_array(C[t])
            axs[r, 0].set_title(f"ground truth k={k} -- t={t}", fontsize=9)
            axs[r, 1].set_title(f"model -- t={t}", fontsize=9)
        return []

    anim = FuncAnimation(fig, update, frames=[int(t) for t in frames], interval=interval)
    plt.close(fig)
    return anim




def draw_mode(ax, sample, Y: torch.Tensor, style: str = "both", frame: int = -1,
              n_arrows: int = 300, gain: float = 1.0, color: str | None = None,
              alpha: float = 1.0, point_size: float = 4.0) -> None:
    """One microstructure rollout into ONE axes, for the mode gallery. Which way the cell
    buckles is the mode, so the picture has to make the *direction* of the fluctuation readable.

    Two things are worth seeing, and ``style`` chooses between them or shows both:

    ``deformed``  the cell as it actually looks: the nodes at their deformed positions
                  (affine + fluctuation), colored by how far each one moved. This is the shape a
                  person recognizes, but a cell buckling one way and its mirror image can look
                  similar at a glance.
    ``arrows``    the fluctuation field itself (lang-ok): an arrow per node, its **length the magnitude**
                  (drawn in data units, so it is the true displacement) and its **color the
                  direction** (a cyclic map, so a rotation of the pattern is a rotation of the
                  colors). This is what separates the modes.
    ``both``      the arrows on top of the deformed cell -- the default, because the two answer
                  different halves of "which branch is this?".

    ``frame`` is the frame drawn (the last one by default -- the modes are furthest apart there).
    ``gain`` exaggerates the arrow length when the fluctuation is small next to the cell.
    ``color`` optionally replaces the magnitude coloring of the deformed point cloud; this is
    useful when several ground-truth branches are drawn in one axes.  The default preserves the
    original magnitude-colored rendering.
    """
    y = torch.nan_to_num(Y)[frame]                                # [N_, 2] fluctuation
    x = sample.affine[frame] + y                                  # [N_, 2] deformed positions
    magnitude = torch.linalg.vector_norm(y, dim=-1)
    direction = torch.atan2(y[:, 1], y[:, 0])                     # [-pi, pi], the signature of the mode

    if style not in ("deformed", "arrows", "both"):
        raise ValueError(f"unknown style {style!r}, expected 'deformed', 'arrows' or 'both'")

    ax.set_axis_on()
    if style in ("deformed", "both"):
        scatter_kwargs = ({"c": magnitude, "cmap": "viridis"}
                          if color is None else {"color": color})
        ax.scatter(x[:, 0], x[:, 1], s=point_size, linewidths=0, alpha=alpha,
                   **scatter_kwargs)
    if style in ("arrows", "both"):
        n = min(len(x), n_arrows // 2 if style == "both" else n_arrows)
        idx = torch.linspace(0, len(x) - 1, n).long()
        ax.quiver(x[idx, 0], x[idx, 1], gain * y[idx, 0], gain * y[idx, 1], direction[idx],
                  cmap="hsv", clim=(-math.pi, math.pi),  # cyclic: direction wraps, so the map must
                  angles="xy", scale_units="xy", scale=1.0,  # length = the true magnitude
                  width=0.003 if style == "both" else 0.005,
                  alpha=0.85 if style == "both" else 1.0)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
