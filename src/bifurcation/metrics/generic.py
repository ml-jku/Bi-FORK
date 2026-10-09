"""Generic metrics and losses, shared by all datasets.

All take node-padded batch tensors (``pred/target [B_, ..., N_, C_]``) and the node ``mask``
``[B_, N_]``; padding never contributes. Dataset-specific metrics are next to this module,
one file per dataset.
"""

from __future__ import annotations

import torch

from bifurcation.utils.masking import expand_mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over real (non-padding) nodes."""
    m = expand_mask(mask, pred)
    return ((pred - target) ** 2 * m).sum() / (m.sum() + 1e-8)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over real nodes."""
    m = expand_mask(mask, pred)
    return ((pred - target).abs() * m).sum() / (m.sum() + 1e-8)


def masked_medae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Median absolute error over real nodes -- a few outlier nodes move MAE, not this.
    NaN when the mask selects nothing (a mean can hide behind ``+ 1e-8``, a median cannot)."""
    err = (pred - target).abs()[expand_mask(mask, pred).bool()]
    return err.median() if err.numel() else pred.new_tensor(float("nan"))


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Root MSE over real nodes."""
    return masked_mse(pred, target, mask).sqrt()


def relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Global relative L2 error ``||pred - target|| / ||target||`` over real nodes.

    A single global ratio (not per sample), so near-zero targets in one row cannot blow it up.
    """
    m = expand_mask(mask, pred)
    num = ((pred - target) ** 2 * m).sum()
    den = (target**2 * m).sum()
    return (num / (den + 1e-8)).sqrt()


def frame_masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid frames: ``pred/target [B_, T_, ...]``, ``valid_mask [B_, T_]`` bool."""
    m = valid_mask.reshape(*valid_mask.shape, *([1] * (pred.ndim - 2))).expand_as(pred)
    return ((pred - target) ** 2 * m).sum() / (m.sum() + 1e-8)






def frame_masked_angle(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
                       rel_floor: float = 0.05) -> torch.Tensor:
    """Mean angle (degrees) between predicted and true vectors (last dim) over valid frames.

    (Near-)zero targets carry no direction -- cosine similarity scores them 90 deg no matter
    the prediction -- so entries below ``rel_floor`` x the masked RMS target magnitude are
    excluded (see :func:`masked_angle`).
    """
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    m = valid_mask.reshape(*valid_mask.shape, *([1] * (cos.ndim - 2))).expand_as(cos).bool()
    m = m & _directional(target, m, rel_floor)
    angle = torch.rad2deg(torch.arccos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)))
    return (angle * m).sum() / (m.sum() + 1e-8)


def masked_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cosine loss ``mean(1 - cos(pred, target))`` over real nodes -- direction only,
    magnitude-blind (pair with an MSE term when used as a loss)."""
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    m = expand_mask(mask, cos)
    return ((1.0 - cos) * m).sum() / (m.sum() + 1e-8)


def masked_mse_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      cosine_weight: float) -> torch.Tensor:
    """``masked_mse + cosine_weight * masked_cosine`` -- magnitude + direction in one loss."""
    return masked_mse(pred, target, mask) + cosine_weight * masked_cosine(pred, target, mask)


def _directional(target: torch.Tensor, m: torch.Tensor, rel_floor: float) -> torch.Tensor:
    """Entries whose target is large enough to define a direction: magnitude above
    ``rel_floor`` x the masked RMS. Zero-displacement targets (e.g. the clamped base of beam3d
    node, the undeformed t=0 frame) would otherwise log 90 deg regardless of the prediction."""
    mag = target.norm(dim=-1)
    rms = ((mag**2 * m).sum() / (m.sum() + 1e-8)).sqrt()
    return mag > rel_floor * rms


def masked_angle(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 rel_floor: float = 0.05) -> torch.Tensor:
    """Mean angle (degrees) between predicted and true vectors over real nodes with a
    directional target (magnitude above ``rel_floor`` x the masked RMS -- see
    :func:`_directional`). Scale-invariant."""
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    m = expand_mask(mask, cos).bool()
    m = m & _directional(target, m, rel_floor)
    angle = torch.rad2deg(torch.arccos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)))
    return (angle * m).sum() / (m.sum() + 1e-8)


def cross_entropy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample classification loss: ``pred [B_, C_]`` logits vs ``target [B_]`` class ids.

    ``mask`` is unused (signature compatibility with the field losses -- lang-ok); ``-1`` targets
    (unknown class) are ignored.
    """
    return torch.nn.functional.cross_entropy(pred, target.long(), ignore_index=-1)
