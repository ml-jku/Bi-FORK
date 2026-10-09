"""Per-frame bifurcation labels shared by discrete and rotational mode datasets."""

from __future__ import annotations

import torch

from bifurcation.datasets.utils.modes import DiscreteModes


def discrete_bifurcation_mask(modes: DiscreteModes) -> torch.Tensor:
    """Return ``[T_]``: true wherever at least two valid frame modes coexist."""
    labels = modes.frame_labels
    out = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    state = False
    for t in range(labels.shape[0]):
        valid_labels = labels[t][labels[t] >= 0]
        if valid_labels.numel():
            state = torch.unique(valid_labels).numel() >= 2
        out[t] = state
    return out


def beam_bifurcation_mask(
    displacement: torch.Tensor,
    *,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Label frames where azimuthal rotation changes a beam displacement field."""
    displacement = torch.as_tensor(displacement, dtype=torch.float32)
    if displacement.ndim != 3 or displacement.shape[-1] != 3:
        raise ValueError(
            "beam displacement must have shape [T, N, 3], got "
            f"{tuple(displacement.shape)}"
        )
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("bifurcation tolerances must be non-negative")

    lateral = torch.linalg.vector_norm(displacement[..., :2], dim=-1).amax(dim=-1)
    peak = float(lateral.amax()) if lateral.numel() else 0.0
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * peak)
    return lateral > threshold, lateral, threshold
