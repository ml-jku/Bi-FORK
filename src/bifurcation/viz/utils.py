"""Shared plotting helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from matplotlib.animation import FuncAnimation, PillowWriter


def save_gif(anim: FuncAnimation, path, fps: int = 8) -> str:
    """Write an animation as a GIF (Pillow, no ffmpeg). Creates the target folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    return str(path)


def log_figure(trainer, tag: str, fig) -> None:
    """Send a matplotlib figure to the attached logger (TensorBoard or wandb)."""
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

    if isinstance(trainer.logger, TensorBoardLogger):
        trainer.logger.experiment.add_figure(tag, fig, trainer.current_epoch)
    elif isinstance(trainer.logger, WandbLogger):
        import wandb

        trainer.logger.experiment.log({tag: wandb.Image(fig)})


def plot_mode_histogram(hists: dict[int, torch.Tensor], title: str = ""):
    """Bar chart of generated-label fractions, one panel per mode count.

    ``hists[k]`` is the ``[k + 2]`` bins of :func:`~bifurcation.metrics.modes.label_histogram`:
    the k modes, then REJECTED and BLURRED (grey), split off by a dashed separator. There are two
    dotted references, ``1/k`` (dark) and ``valid/k`` (blue); what each one means is in
    docs/decisions.md.
    """
    import matplotlib.pyplot as plt

    widths = [max(2.4, 0.30 * (k + 2)) for k in sorted(hists)]
    fig, axes = plt.subplots(1, len(hists), figsize=(sum(widths) + 0.6, 2.6),
                             width_ratios=widths, squeeze=False)
    for ax, (k, hist) in zip(axes[0], sorted(hists.items())):
        h = hist.tolist()  # a handful of bar heights; matplotlib wants plain floats
        pos = list(range(k)) + [k + 0.6, k + 1.6]
        ax.bar(pos, h, width=0.8, color=["C0"] * k + ["0.45", "0.75"])
        ax.axvline(k - 0.2, color="0.3", linestyle="--", linewidth=1)
        ax.axhline(1.0 / k, color="0.2", linestyle=":", linewidth=1)
        if (valid := sum(h[:k])) > 0:
            ax.axhline(valid / k, color="C0", linestyle=":", linewidth=1)
        mode_ticks = range(k) if k <= 12 else sorted({0, k // 2, k - 1})
        ax.set_xticks([*mode_ticks, k + 0.6, k + 1.6],
                      [*map(str, mode_ticks), "rej", "blur"])
        ax.set_ylim(0, max(max(h), 1.0 / k, 1e-6) * 1.15)
        ax.set_title(f"k={k}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("fraction of outputs", fontsize=7)
    from matplotlib.lines import Line2D

    fig.legend(handles=[Line2D([], [], color="0.2", linestyle=":", linewidth=1),
                        Line2D([], [], color="C0", linestyle=":", linewidth=1)],
               labels=["1/k: uniform, nothing rejected", "valid/k: balanced valid outputs"],
               fontsize=6, frameon=False, loc="upper right", ncols=2)
    if title:
        fig.suptitle(title, fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def log_gif(trainer, tag: str, path) -> None:
    """Send a rendered GIF to the attached logger (wandb only; TensorBoard has no video)."""
    from lightning.pytorch.loggers import WandbLogger

    if isinstance(trainer.logger, WandbLogger):
        import wandb

        trainer.logger.experiment.log({tag: wandb.Video(str(path))})


def zoom_window(x: torch.Tensor, y_ref: torch.Tensor, frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Square window covering ``frac`` of the cloud; zooms centre on the max-|y| node."""
    lo = torch.nan_to_num(x, nan=float("inf")).amin(0)
    hi = torch.nan_to_num(x, nan=float("-inf")).amax(0)
    span = float((hi - lo).max())
    if frac >= 1:
        center = (lo + hi) / 2
    else:
        magnitude = torch.linalg.vector_norm(y_ref, dim=-1)
        center = x[torch.nan_to_num(magnitude, nan=float("-inf")).argmax()]
    half = frac * span / 2 * 1.02
    return center - half, center + half


def quiver_panel(ax, x: torch.Tensor, y: torch.Tensor, lo, hi, color: str, n_arrows: int):
    """Nodes (light grey) + field arrows (lang-ok) inside the window, arrows in data units."""
    idx = torch.nonzero(((x >= lo) & (x <= hi)).all(-1)).flatten()
    ax.scatter(*x[idx].T, s=1, color="0.88")
    if idx.numel():
        pick = idx[torch.linspace(0, idx.numel() - 1, min(n_arrows, idx.numel())).long()]
        ax.quiver(*x[pick].T, *y[pick].T, color=color,
                  angles="xy", scale_units="xy", scale=1.0, width=0.004)
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
