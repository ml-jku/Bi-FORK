"""Orthogonal slice figures for three-dimensional Allen--Cahn scalar fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def _cube(field: torch.Tensor) -> torch.Tensor:
    N_ = field.shape[0]
    n = round(N_ ** (1 / 3))
    if n**3 != N_ or field.shape[-1] != 1:
        raise ValueError(f"expected a cubic scalar field [N,1], got {tuple(field.shape)}")
    return field[:, 0].reshape(n, n, n)


def plot_reconstruction_slices(p, x, y_true, y_pred, meta) -> plt.Figure:
    """Ground truth (top) and prediction (bottom), through three central planes."""
    del p, x
    true = _cube(y_true[-1]).cpu()
    pred = _cube(y_pred[-1]).cpu()
    mid = true.shape[0] // 2

    def planes(volume):
        return volume[mid], volume[:, mid], volume[:, :, mid]

    vmax = max(float(true.abs().max()), float(pred.abs().max()), 1e-8)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for row, (name, volume) in enumerate((("GT", true), ("prediction", pred))):
        slices = zip(("yz", "xz", "xy"), planes(volume), strict=True)
        for col, (axis_name, image) in enumerate(slices):
            axes[row, col].imshow(image.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axes[row, col].set_title(f"{name}: {axis_name}", fontsize=9)
            axes[row, col].set_xticks([]), axes[row, col].set_yticks([])
    fig.suptitle(f"sample {meta['key']} frame {meta['frames'][-1]}", fontsize=10)
    return fig


def draw_mode(ax, sample, Y: torch.Tensor, frame: int = -1) -> None:
    """Draw three orthogonal center slices of one rollout in a gallery axes."""
    del sample
    volume = _cube(Y[frame]).cpu()
    mid = volume.shape[0] // 2
    image = torch.cat((volume[mid], volume[:, mid], volume[:, :, mid]), dim=1)
    vmax = max(float(image.abs().max()), 1e-8)
    ax.imshow(image.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_axis_off()


def plot_trial_vs_modes(sample, Y: torch.Tensor, similarities: torch.Tensor, best_k: int,
                        key: str | None = None) -> plt.Figure:
    """One generated trial (left) against every one of the sample's ``K_`` GT branches, headed
    by :func:`bifurcation.metrics.allencahn_symmetry.mode_similarity_scores`'s cosine similarity
    -- the score Metric 1's argmax match (``best_k``, starred) is actually picked from, so the
    coverage assignment in ``mode_coverage_jsd`` can be read off visually instead of trusted
    blind. A branch masked out by the sign tiebreak (``-inf`` similarity, see
    :func:`bifurcation.metrics.allencahn_symmetry._sign_feature`) is labelled accordingly.
    """
    K_ = sample.K_
    fig, axes = plt.subplots(1, 1 + K_, figsize=(2.0 * (1 + K_), 2.4), constrained_layout=True)
    draw_mode(axes[0], sample, Y)
    axes[0].set_title("trial", fontsize=9)
    for k in range(K_):
        draw_mode(axes[k + 1], sample, sample.Y(k))
        score = float(similarities[k])
        star = " *" if k == best_k else ""
        label = "sign-excl." if score == float("-inf") else f"sim={score:.3f}"
        axes[k + 1].set_title(f"k={k}{star}\n{label}", fontsize=8)
    title = "trial vs. 8 GT modes (Metric 1 similarity)"
    fig.suptitle(f"sample {key}: {title}" if key is not None else title, fontsize=10)
    return fig


