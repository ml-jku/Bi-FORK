"""Noise-data couplings for flow matching -- pluggable via config, like losses.

Convention: ``z0`` = data (encoded latents), ``z1`` = noise. ``ot`` is minibatch optimal
transport, while ``group`` restricts assignment to the rows of each condition.
Follows https://github.com/atong01/conditional-flow-matching. No code taken.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment




def _hungarian(z1: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Permute the noise rows by the Hungarian assignment (cost = pairwise MSE)."""
    B = z1.shape[0]
    a = z1.reshape(B, -1)
    b = z0.reshape(B, -1)
    cost = (a * a).sum(1, keepdim=True) - 2 * a @ b.T + (b * b).sum(1)[None]
    row, col = linear_sum_assignment(cost.detach().float().cpu())
    perm = torch.empty(B, dtype=torch.long, device=z1.device)
    perm[torch.as_tensor(col, device=z1.device)] = torch.as_tensor(row, device=z1.device)
    return z1[perm]


def ot(z1: torch.Tensor, z0: torch.Tensor, group: torch.Tensor | None = None) -> torch.Tensor:
    """Minibatch OT; with ``group`` the assignment is block-diagonal (within each group)."""
    if group is None:
        return _hungarian(z1, z0) if z1.shape[0] > 1 else z1
    out = z1.clone()
    for g in group.unique():
        idx = (group == g).nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            out[idx] = _hungarian(z1[idx], z0[idx])
    return out
