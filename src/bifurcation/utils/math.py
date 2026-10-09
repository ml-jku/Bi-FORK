"""Small math helpers."""

from __future__ import annotations

import math

import torch


def farthest_point_sample(p: torch.Tensor, m: int, rng: torch.Generator | None = None) -> torch.Tensor:
    """Sorted indices ``[m]`` of a farthest-point subset of ``p [N_, D_]`` (random start).

    Greedy max-min coverage: each pick is the point farthest from the chosen set.
    O(N_ * m); ~25 ms for 20k nodes at m=512.
    """
    N_ = p.shape[0]
    if m >= N_:
        return torch.arange(N_, device=p.device)
    idx = torch.empty(m, dtype=torch.long, device=p.device)
    idx[0] = torch.randint(N_, (), generator=rng)
    dist = ((p - p[idx[0]]) ** 2).sum(-1)
    for j in range(1, m):
        idx[j] = dist.argmax()
        dist = torch.minimum(dist, ((p - p[idx[j]]) ** 2).sum(-1))
    return idx.sort().values




def nanmean(values) -> float:
    """Mean ignoring NaNs; NaN when every entry is NaN."""
    return float(torch.as_tensor(values, dtype=torch.float64).nanmean())


def nanmedian(values) -> float:
    """Median ignoring NaNs; NaN when every entry is NaN."""
    return float(torch.as_tensor(values, dtype=torch.float64).nanmedian())


def z_rotation(phi: float | torch.Tensor) -> torch.Tensor:
    """Rotation about the z-axis by ``phi`` radians: scalar -> ``[3, 3]``, tensor ``[...]``
    -> ``[..., 3, 3]`` (one matrix per angle). The result keeps the dtype and device of
    ``phi``, so batched rotations run wherever the angles already are."""
    phi = torch.as_tensor(phi)
    if not phi.is_floating_point():
        phi = phi.float()
    c, s = torch.cos(phi), torch.sin(phi)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, zero], -1),
                        torch.stack([s, c, zero], -1),
                        torch.stack([zero, zero, one], -1)], -2)


def get_azimuth_from_sample(Y: torch.Tensor) -> float:
    """In-plane angle in ``[0, 2pi)`` of the net xy-displacement (lang-ok) of a field.

    Accepts a rollout ``[T_, N_, 3]`` (evaluated at the final frame) or one frame
    ``[N_, 3]``. Purely geometric -- validity is for the metric layer to decide.
    """
    y = Y[-1] if Y.ndim == 3 else Y
    v = torch.nansum(y[:, :2], dim=0)
    return float(torch.atan2(v[1], v[0])) % (2.0 * math.pi)


# Adapted from https://github.com/facebookresearch/pytorch3d


# <match="../../../resources/pytorch3d/pytorch3d/transforms/rotation_conversions.py?plain=1#L310">
def random_quaternions(
    n: int,
    rng: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniform random unit quaternions ``[n, 4]`` float32, real part first.

    Normalized Gaussians: the standard normal is spherically symmetric in R^4, so scaling to
    unit length is uniform on the 3-sphere, which is Haar on SO(3) once mapped to matrices.
    """
    o = torch.randn((n, 4), generator=rng)
    norm = torch.sqrt((o * o).sum(1))
    norm = torch.where(o[:, 0] < 0, -norm, norm)
    return (o / norm[:, None]).to(dtype=torch.float32, device=device)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` from quaternions ``[..., 4]``, real part first.

    ``q`` need not be normalized -- the ``2 / |q|^2`` factor divides it out.
    """
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(q.shape[:-1] + (3, 3))

