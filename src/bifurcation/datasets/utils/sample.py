"""One dataset item: K_ rollouts of a target field over a fixed spatial support.  lang-ok

The keys below are the variable convention, written out in CONVENTIONS.md: every
dataset speaks in these names and shapes. NaN marks missing data; ``valid_mask`` flags the usable frames per rollout.
Dataset specifics (derived quantities, extra item keys) belong to subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bifurcation.datasets.utils.modes import Modes


@dataclass
class Sample:
    rollouts: torch.Tensor                  # [K_, T_, N_, C_] target field  lang-ok
    p: torch.Tensor                         # [N_, D_] node positions (static support)
    modes: Modes                            # rollout- and frame-level mode structure (utils/modes.py)
    valid_mask: torch.Tensor | None = None  # [K_, T_]
    U: torch.Tensor | None = None           # [T_, U_] system conditions
    H: torch.Tensor | None = None           # [T_, N_, F_] dynamic node features
    f: torch.Tensor | None = None           # [N_, F_] static node features
    g: torch.Tensor | None = None           # [G_] one feature for the whole sample (e.g. wallpaper group)
    key: str | None = None
    bifurcation: torch.Tensor | None = None # [T_] true where two or more modes coexist

    @property
    def K_(self) -> int:
        return self.rollouts.shape[0]

    @property
    def T_(self) -> int:
        return self.rollouts.shape[1]

    @property
    def N_(self) -> int:
        return self.rollouts.shape[2]

    @property
    def C_(self) -> int:
        return self.rollouts.shape[3]

    @property
    def D_(self) -> int:
        return self.p.shape[1]

    def Y(self, k: int = 0) -> torch.Tensor:
        """Target field [T_, N_, C_] of rollout ``k``.  lang-ok"""
        return self.rollouts[k]

    def valid_t(self, k: int = 0) -> torch.Tensor:
        """Usable frames [T_] of rollout ``k``."""
        if self.valid_mask is None:
            return torch.ones(self.T_, dtype=torch.bool)
        return self.valid_mask[k].bool()

    def X(self, k: int = 0) -> torch.Tensor:
        """Node positions ``[T_, N_, D_]`` of rollout ``k`` over time, raw units.

        Base: the support does not move (value-like fields). Datasets whose target is a  lang-ok
        displacement override this: beam3d returns ``p + Y``; microstructures returns the
        affine-deformed grid plus the fluctuation.
        """
        return self.p.expand(self.T_, *self.p.shape)

    def snapshot(self, k: int, t: int) -> dict:
        """One frame of rollout ``k``: ``y``, ``p`` (+ ``u``/``f``/``h`` if present).

        Subclasses may add keys (never rename base ones).
        """
        out = {"y": self.Y(k)[t], "p": self.p}
        if self.U is not None:
            out["u"] = self.U[t]
        if self.f is not None:
            out["f"] = self.f
        if self.g is not None:
            out["g"] = self.g
        if self.H is not None:
            out["h"] = self.H[t]
        if self.bifurcation is not None:
            out["bifurcation"] = self.bifurcation[t]
        return out

    def rollout(self, k: int) -> dict:
        """Rollout ``k`` over time: ``Y``, ``p``, ``valid_mask`` [T_] (+ ``U``/``f``/``H``)."""
        out = {"Y": self.Y(k), "p": self.p, "valid_mask": self.valid_t(k)}
        if self.U is not None:
            out["U"] = self.U
        if self.f is not None:
            out["f"] = self.f
        if self.g is not None:
            out["g"] = self.g
        if self.H is not None:
            out["H"] = self.H
        if self.bifurcation is not None:
            out["bifurcation"] = self.bifurcation
        return out
