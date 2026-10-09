"""How symmetries act on a dataset item.

A :class:`SymmetryOperation` reshuffles one item three ways: it relabels the nodes, moves the
positions, and transforms the field and conditions. It never touches the support ``p`` -- that is  lang-ok
just the fixed "name" we give each node for the encoder/decoder to index by, and every symmetry
in the group leaves it alone by design, so only the real physical state (deformed positions,
field) changes. A :class:`SymmetryGroup` bundles these transformations together; we use them for data  lang-ok
augmentation and evaluation. Each dataset brings its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SymmetryOperation(ABC):
    """A single symmetry, spelled out by how it acts on the three things an item carries:
    positions, field values, and conditions. A subclass has to implement all three (use the  lang-ok
    identity where the symmetry does nothing), so a half-finished one blows up when you build it
    instead of partway through a run.

    ``apply`` handles the base item keys, the ones CONVENTIONS.md lists.
    A dataset with extra keys overrides ``apply`` and calls ``super().apply`` for the rest.
    """

    node_perm: torch.Tensor | None = None

    @abstractmethod
    def on_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Move the absolute positions ``[..., N_, D_]`` (the deformed ``x``/``X``), e.g. rotate them."""

    @abstractmethod
    def on_field(self, y: torch.Tensor) -> torch.Tensor:
        """Transform the field values ``[..., N_, C_]``, e.g. rotate the displacement vectors.  lang-ok"""

    @abstractmethod
    def on_conditions(self, u: torch.Tensor) -> torch.Tensor:
        """Transform the driving conditions ``[..., U_]``."""

    def apply(self, item: dict) -> dict:
        out = dict(item)
        perm = self.node_perm

        def permute(v):  # node axis is -2 for all node-indexed keys
            return v[..., perm, :] if perm is not None else v

        for key in ("x", "X"):  # absolute deformed positions
            if key in out:
                out[key] = permute(self.on_positions(out[key]))
        for key in ("y", "Y"):
            if key in out:
                out[key] = permute(self.on_field(out[key]))
        for key in ("f", "H", "h"):  # node features: relabelled with the nodes, values untouched
            if key in out:
                out[key] = permute(out[key])
        for key in ("u", "U"):
            if key in out:
                out[key] = self.on_conditions(out[key])
        return out

    def __repr__(self) -> str:
        perm = "with node_perm" if self.node_perm is not None else "no node_perm"
        return f"{type(self).__name__}({perm})"


class SymmetryGroup(ABC):
    """A bag of symmetry ops: draw one at random (augmentation) or list them all (orbits)."""

    @abstractmethod
    def sample(self, rng: torch.Generator) -> SymmetryOperation:
        """Draw a random element."""

    def elements(self) -> list[SymmetryOperation] | None:
        """Every element if the group is discrete; ``None`` if it is continuous."""
        return None
