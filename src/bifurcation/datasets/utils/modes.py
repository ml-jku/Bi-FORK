"""Mode structure of a bifurcating sample, on two levels.

Rollout level: are two rollouts the same solution branch? Frame level: are two rollouts in the
same state at frame ``t``? Modes can split and later re-coincide, so the frame partition is NOT
a refinement over time and the two levels need separate labels. Modes are discrete (a finite
set, from the saved rollouts or from augmentation) or continuous (an orbit parameter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import torch

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-7


class Modes(ABC):
    """Classification and representatives, rollout- and frame-level."""

    @abstractmethod
    def classify_rollout(self, Y: torch.Tensor) -> int | float:
        """Mode of a rollout ``Y [T_, N_, C_]``: an id (discrete) or parameter (continuous)."""

    @abstractmethod
    def classify_frame(self, y: torch.Tensor, t: int) -> int | float:
        """Mode of one frame ``y [N_, C_]`` at timestep ``t``."""

    @abstractmethod
    def representative_rollout(self, mode) -> torch.Tensor:
        """Representative rollout ``[T_, N_, C_]`` of ``mode``."""

    @abstractmethod
    def representative_frame(self, mode, t: int) -> torch.Tensor:
        """Representative frame ``[N_, C_]`` of ``mode`` at timestep ``t``."""

    @abstractmethod
    def draw_rollout(self, rng: torch.Generator) -> int:
        """Mode-balanced rollout draw: uniform over modes, then within."""

    @abstractmethod
    def draw_frame(self, t: int, rng: torch.Generator) -> int:
        """Mode-balanced rollout draw at frame ``t``: uniform over modes, then within."""


class ContinuousModes(Modes):
    """Modes on a continuous symmetry orbit. Interface only -- implemented with beam3d."""


def _pick(candidates: list[int], rng: torch.Generator) -> int:
    """Uniform draw from a non-empty list of rollout indices."""
    return candidates[int(torch.randint(len(candidates), (), generator=rng))]


def fft_mse_distance(trial: torch.Tensor, rollouts: torch.Tensor,
                     valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared distance between the rfft magnitude spectra of ``trial`` (``[T_,N_,C_]``)
    and each of ``rollouts`` (``[K_,T_,N_,C_]``), batched over ``K_``. See docs/decisions.md.

    ``valid_mask`` (``[K_,T_]`` or ``[T_]``, optional) zeroes invalid frames first.
    """
    if valid_mask is not None:
        mask = valid_mask if valid_mask.dim() == 1 else valid_mask.any(dim=0)  # [T_]
        trial = trial * mask[:, None, None]
        rollouts = rollouts * mask[None, :, None, None]

    P = torch.fft.rfft(torch.nan_to_num(trial), dim=-3).abs()[None]  # [1, F_, N_, C_]
    Q = torch.fft.rfft(torch.nan_to_num(rollouts), dim=-3).abs()     # [K_, F_, N_, C_]
    return ((P - Q) ** 2).mean(dim=(-3, -2, -1))  # [K_,]


@dataclass
class DiscreteModes(Modes):
    """A finite set of modes, given by the saved rollouts.

    - rollouts:       ``[K_, T_, N_, C_]`` the source of representatives.
    - rollout_labels: ``[K_]`` branch id per rollout (ids ordered by first rollout).
    - frame_labels:   ``[T_, K_]`` frame-level mode ids; ``-1`` where there is no valid frame.
    - valid_mask:     ``[K_, T_]`` frames included for distances.

    Labels are moved onto the same device as the rollouts at construction.
    """

    rollouts: torch.Tensor
    rollout_labels: torch.Tensor
    frame_labels: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def __post_init__(self):
        self.rollouts = torch.as_tensor(self.rollouts)
        device = self.rollouts.device
        self.rollout_labels = torch.as_tensor(self.rollout_labels, dtype=torch.long, device=device)
        self.frame_labels = torch.as_tensor(self.frame_labels, dtype=torch.long, device=device)
        if self.valid_mask is not None:
            self.valid_mask = torch.as_tensor(self.valid_mask, dtype=torch.bool, device=device)

    def _here(self, Y: torch.Tensor) -> torch.Tensor:
        """Inputs move onto the same device as the rollouts."""
        return Y.to(self.rollouts.device)


    @property
    def n_modes(self) -> int:
        return int(self.rollout_labels.max()) + 1

    def rollouts_in_mode(self, mode: int) -> list[int]:
        return torch.nonzero(self.rollout_labels == mode).flatten().tolist()

    def representative_rollout(self, mode: int) -> torch.Tensor:
        return self.rollouts[self.rollouts_in_mode(mode)[0]]  # [T_, N_, C_]

    def nearest_rollout(self, Y: torch.Tensor, distance_metric: str = "mse") -> tuple[int, float]:
        """Nearest rollout and its distance over valid frames; ``distance_metric`` is
        ``"mse"`` (default) or ``"fft_mse"`` (Allen-Cahn only, see :func:`fft_mse_distance`).

        ``(-1, nan)`` when no comparison exists; callers must treat ``k < 0`` as "not a branch".
        """
        Y = self._here(Y)
        if distance_metric == "mse":
            d = torch.nanmean((self._masked_rollouts - Y[None]) ** 2, dim=(1, 2, 3))  # [K_,]
        elif distance_metric == "fft_mse":
            d = fft_mse_distance(Y, self._masked_rollouts, self.valid_mask)  # [K_,]
        else:
            raise ValueError(f"unknown distance_metric {distance_metric!r}")
        if not d.isfinite().any():
            return -1, float("nan")
        k = int(torch.where(d.isnan(), torch.inf, d).argmin())
        return k, float(d[k])

    def classify_rollout(self, Y: torch.Tensor, max_abs_distance: float | None = None,
                         max_rel_distance: float | None = None,
                         distance_metric: str = "mse") -> int:
        """Branch of the nearest rollout, or ``-1`` beyond either cutoff (see
        :meth:`too_far`)."""
        k, _ = self.nearest_rollout(Y, distance_metric=distance_metric)
        if k < 0:
            return -1
        if self.too_far(self.l2_distance(Y, k), max_abs_distance, max_rel_distance):
            return -1
        return int(self.rollout_labels[k])

    def draw_rollout(self, rng: torch.Generator) -> int:
        mode = int(torch.randint(self.n_modes, (), generator=rng))
        return _pick(self.rollouts_in_mode(mode), rng)


    def n_modes_at(self, t: int) -> int:
        labels = self.frame_labels[t]
        return int(labels.max()) + 1 if (labels >= 0).any() else 0

    def rollouts_in_mode_at(self, mode: int, t: int) -> list[int]:
        return torch.nonzero(self.frame_labels[t] == mode).flatten().tolist()

    def representative_frame(self, mode: int, t: int) -> torch.Tensor:
        return self.rollouts[self.rollouts_in_mode_at(mode, t)[0], t]  # [N_, C_]

    def nearest_frame(self, y: torch.Tensor, t: int) -> tuple[int, float]:
        """Nearest rollout at frame ``t`` (rollouts without a valid frame excluded)."""
        y = self._here(y)
        d = torch.nanmean((self.rollouts[:, t] - y[None]) ** 2, dim=(1, 2))  # [K_,]
        d = torch.where(self.frame_labels[t] >= 0, d, torch.inf)
        k = int(d.argmin())
        return k, float(d[k])

    def classify_frame(self, y: torch.Tensor, t: int, max_abs_distance: float | None = None,
                       max_rel_distance: float | None = None) -> int:
        k, _ = self.nearest_frame(y, t)
        if self.too_far(self.l2_distance(y, k, t), max_abs_distance, max_rel_distance):
            return -1
        return int(self.frame_labels[t, k])

    def l2_distance(self, Y: torch.Tensor, k: int, t: int | None = None) -> float:
        """Mean L2 node distance to rollout ``k`` over its valid frames (one frame with ``t``)
        -- the quantity both assignment cutoffs are measured in."""
        ref = self._masked_rollouts[k] if t is None else self.rollouts[k, t]
        return float(torch.nanmean((ref - self._here(Y)).square().sum(-1).sqrt()))

    def draw_frame(self, t: int, rng: torch.Generator) -> int:
        mode = int(torch.randint(self.n_modes_at(t), (), generator=rng))
        return _pick(self.rollouts_in_mode_at(mode, t), rng)




    @cached_property
    def _masked_rollouts(self) -> torch.Tensor:
        """Rollouts with invalid frames NaN-ed."""
        if self.valid_mask is None:
            return self.rollouts
        return torch.where(self.valid_mask[:, :, None, None], self.rollouts, torch.nan)  # [K_, T_, N_, C_]


    @cached_property
    def magnitude(self) -> float:
        """Mean L2 node displacement over valid frames -- the scale a relative distance is
        measured against, in the same units as :meth:`l2_distance`."""
        return float(torch.nanmean(self._masked_rollouts.square().sum(-1).sqrt()))

    def too_far(self, d: float, max_abs_distance: float | None = None,
                max_rel_distance: float | None = None) -> bool:
        """Is a mean L2 node distance beyond either cutoff? ``max_abs_distance`` is in raw
        field units, ``max_rel_distance`` is a fraction of :attr:`magnitude`. Both None  lang-ok
        (the default) accepts everything."""
        if max_abs_distance is not None and d > max_abs_distance:
            return True
        return max_rel_distance is not None and d > max_rel_distance * self.magnitude

    @cached_property
    def representative_mean(self) -> torch.Tensor:
        """Mean of the masked mode representatives -- what a mode-averaged output looks like.

        Invalid frames remain NaN so they cannot affect the blurred-output decision. For
        microstructures this excludes every frame at or after self-contact.
        """
        reps = [self._masked_rollouts[self.rollouts_in_mode(m)[0]]
                for m in range(self.n_modes)]  # n_modes x [T_, N_, C_]
        return torch.stack(reps).mean(0)  # [T_, N_, C_]

    def closer_to_mean(self, Y: torch.Tensor, d: float, max_abs_distance: float | None = None,
                       max_rel_distance: float | None = None) -> bool:
        """Is ``Y`` closer to the mean of the modes than to any mode?

        Such an output averages the modes instead of picking one. With a single mode: False.

        Args:
            Y (torch.Tensor): rollout [T_, N_, C_]
            d (float): mean L2 node distance of Y to its nearest rollout
            max_abs_distance, max_rel_distance: the cutoffs of :meth:`too_far`
        """
        if self.n_modes <= 1:
            return False
        d_mean = float(torch.nanmean(
            (self.representative_mean - self._here(Y)).square().sum(-1).sqrt()))
        return d_mean < d and not self.too_far(d_mean, max_abs_distance, max_rel_distance)


def label_coinciding_rollouts(coincides: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mode ids [K_] from a boolean "rollouts coincide" matrix [K_, K_].

    Coinciding rollouts (connected components) share an id; ids are numbered by first member;
    invalid rollouts get ``-1``. K_ is a handful of rollouts, so the components come from a few
    boolean matrix squarings (transitive closure) rather than a graph library.
    """
    coincides = coincides.cpu().bool()
    valid = valid.cpu().bool()
    labels = torch.full((len(valid),), -1, dtype=torch.long)
    idx = torch.nonzero(valid).flatten()
    if idx.numel() == 0:
        return labels
    n = int(idx.numel())
    reach = coincides[idx][:, idx]
    reach = reach | reach.T | torch.eye(n, dtype=torch.bool)
    while True:  # squaring doubles the reachable radius; converges in ceil(log2(n)) rounds
        closure = (reach.long() @ reach.long()) > 0
        if torch.equal(closure, reach):
            break
        reach = closure
    positions = torch.arange(n).expand(n, n)
    root = torch.where(reach, positions, n).amin(dim=1)  # [n,]
    labels[idx] = torch.unique(root, return_inverse=True)[1]
    return labels


def detect_modes(rollouts: torch.Tensor, valid_mask: torch.Tensor | None = None,
                 rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> DiscreteModes:
    """Partition rollouts ``[K_, T_, N_, C_]`` into modes on both levels.

    Two rollouts share a frame mode at ``t`` if their mean squared difference at that frame is
    within tolerance; a rollout mode if it is within tolerance averaged over their jointly
    valid frames. ``valid_mask [K_, T_]`` marks the frames to use (default: frames with any
    finite value); NaN features are ignored inside a frame.
    """
    rollouts = torch.as_tensor(rollouts, dtype=torch.float32)
    K_, T_ = rollouts.shape[:2]
    flat = rollouts.reshape(K_, T_, -1)  # [K_, T_, N_*C_]
    if valid_mask is None:
        valid_mask = flat.isfinite().any(-1)  # [K_, T_]
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=rollouts.device)

    distance = torch.full((T_, K_, K_), torch.nan, device=rollouts.device)
    for i in range(K_):
        for j in range(i + 1, K_):
            squared = (flat[i] - flat[j]) ** 2  # [T_, N_*C_]
            finite = squared.isfinite()
            n_finite = finite.sum(-1)           # [T_,]
            joint = valid_mask[i] & valid_mask[j] & (n_finite > 0)
            frame_distance = torch.where(finite, squared, 0.0).sum(-1) / n_finite.clamp(min=1)  # [T_,]
            distance[:, i, j] = distance[:, j, i] = torch.where(joint, frame_distance, torch.nan)
    diag = torch.arange(K_)
    distance[:, diag, diag] = 0.0  # self-distance

    def coincides(d: torch.Tensor) -> torch.Tensor:
        return torch.isclose(torch.nan_to_num(d, nan=torch.inf), torch.zeros_like(d),
                             rtol=rtol, atol=atol)

    frame_labels = torch.stack(
        [label_coinciding_rollouts(coincides(distance[t]), valid_mask[:, t]) for t in range(T_)]
    )  # [T_, K_]

    rollout_distance = distance.nanmean(dim=0)  # [K_, K_]; pairs with no joint frame stay NaN
    rollout_labels = label_coinciding_rollouts(coincides(rollout_distance),
                                               torch.ones(K_, dtype=torch.bool))
    return DiscreteModes(rollouts=rollouts, rollout_labels=rollout_labels,
                         frame_labels=frame_labels, valid_mask=valid_mask)
