"""
beam3d: beam buckling at continuous 360° rotations.

Data is represented by different variables:
 * ``X [T_, N_, 3]``: Rollout of beam in absolute coordinates (deformed positions).
 * ``Y [T_, N_, 3]``: Rollout of beam in relative coordinates (displacement from initial positions).
 * ``U [T_, 1]``: Tip displacement (scalar) over time.
 * ``f [N_, 3]``: Per-node features: bending stiffness K, element compression C, and element rest length L.

The beams are saved at azimuth angle 0 (buckling displacement in +x direction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from bifurcation.datasets.utils.bifurcation import beam_bifurcation_mask
from bifurcation.datasets.utils.modes import ContinuousModes
from bifurcation.datasets.utils.pickle_rollout import (
    PickleRolloutDataset,
    load_pyg_data_list,
)
from bifurcation.datasets.utils.sample import Sample
from bifurcation.datasets.utils.symmetry import SymmetryGroup, SymmetryOperation
from bifurcation.utils.math import get_azimuth_from_sample, z_rotation


def apply_random_azimuth_rotation(item: dict, rng: torch.Generator) -> dict:
    """
    Data augmentation by applying a random azimuth rotation.
    Args:
        item (dict): a sample from the dataset
        rng (torch.Generator): random number generator
    Returns:
        dict: the sample with positions and displacements rotated by a random azimuth angle
    """
    size = None if item["p"].ndim == 2 else len(item["p"])    # [N_, 3] or [B_, N_, 3]
    return AzimuthRotations().sample(rng, size=size).apply(item)




def _get_buckling_start_frame(Y: torch.Tensor, tip: int, rel: float = 0.05) -> int:
    """
    First frame where the tip displacement exceeds ``rel`` fraction of the final tip displacement.
    """
    lateral = torch.linalg.vector_norm(Y[:, tip, :2], dim=-1)  # [T]
    final = lateral[-1]
    if final < 1e-8:
        return -1
    buckled = torch.nonzero(lateral > rel * final).flatten()
    return int(buckled[0]) if buckled.numel() else 0


def _construct_node_features(mapping: dict) -> torch.Tensor: # we want to use point cloud representation
    """
    The node features ``f [N_, 3]`` are contain bending stiffness K, element compression C
    and element rest length L. The tip node receives [0,0,0] as feature. The features of the
    edges are saved at the lower node of the edge.
    """
    f_nodes = torch.as_tensor(mapping["node_attr"], dtype=torch.float32).reshape(-1)  # [N]
    edge_index = torch.as_tensor(mapping["edge_index"])                               # [2, E]
    edge_attr = torch.as_tensor(mapping["edge_attr"], dtype=torch.float32)            # [E, 2] = (L, C)
    f = torch.zeros((f_nodes.shape[0], 3), dtype=torch.float32)                       # [N_, 3]
    f[:, 0] = f_nodes
    forward = edge_index[0] < edge_index[1]
    for node, (L, C) in zip(edge_index[0, forward], edge_attr[forward]):
        f[node, 1] = C
        f[node, 2] = L
    return f


class AzimuthRotation(SymmetryOperation):
    """
    Rotates positions about z-axis by angle ``phi`` (radians).
    Args:
        phi (float | torch.Tensor): azimuth angles in radians
    """

    def __init__(self, phi):
        self.phi = torch.as_tensor(phi) # scalar or [B_,]

    def _rotate(self, v: torch.Tensor) -> torch.Tensor:
        """Rotate the positions of a beam.

        Args:
            v (torch.Tensor): positions to rotate, shape [..., 3]. A batched R rotates whole
                samples, which may be rollouts, single frames or groups of rollouts.
        Returns:
            torch.Tensor: rotated positions, shape [..., 3]
        """
        R = z_rotation(self.phi.to(dtype=v.dtype, device=v.device))  # [3, 3] or [B_, 3, 3]
        RT = R.transpose(-1, -2)    # transpose for right-multiplication
        if RT.ndim > 2:             # insert singleton axes to broadcast over the extra dimensions of v
            RT = RT.reshape(RT.shape[0], *([1] * (v.ndim - RT.ndim)), 3, 3)
        return v @ RT

    def on_positions(self, x):
        return self._rotate(x) # rotates deformed positions

    def on_field(self, y):
        return self._rotate(y) # rotates displacements

    def on_conditions(self, u):
        return u               # tip displacement does not change under azimuth rotation


class AzimuthRotations(SymmetryGroup):
    """SO(2) about z -- the full mode orbit of the beam."""

    def sample(self, rng: torch.Generator, size: int | None = None) -> AzimuthRotation:
        """
        Sample a group element of the azimuth rotation group.
        Args:
            rng (torch.Generator): random number generator
            size (int | None): number of samples to draw. If None, a single sample is drawn.
        Returns:
            AzimuthRotation: sampled azimuth rotation(s)
        """
        shape = () if size is None else (size,)
        return AzimuthRotation(torch.rand(shape, generator=rng) * (2.0 * math.pi))


@dataclass
class Beam3dModes(ContinuousModes):
    """
    The SO(2) orbit of the zero-degree rollout, parameterized by the azimuth.

    Attributes:
        representative (torch.Tensor): [T_, N_, 3] displacement rollout in +x direction
        buckling_start_frame (int): first buckled frame (saved oracle; -1 = never buckles)
    """

    representative: torch.Tensor
    buckling_start_frame: int

    def classify_rollout(self, Y: torch.Tensor) -> float:
        return get_azimuth_from_sample(Y)

    def classify_frame(self, y: torch.Tensor, t: int) -> float:
        return get_azimuth_from_sample(y)

    def representative_rollout(self, mode: float) -> torch.Tensor:
        """The orbit member at azimuth ``mode``, ``[T_, N_, 3]``."""
        return self.representative @ z_rotation(mode).T

    def representative_frame(self, mode: float, t: int) -> torch.Tensor:
        return self.representative_rollout(mode)[t]

    def draw_rollout(self, rng: torch.Generator) -> int: # we save data in one mode, and we perform random rotations
        return 0

    def draw_frame(self, t: int, rng: torch.Generator) -> int:
        return 0


@dataclass
class Beam3dSample(Sample):
    buckling_start_frame: int = -1
    n_elements: int = 0
    total_length: float = 0.0

    def X(self, k: int = 0) -> torch.Tensor:
        """
        Get deformed positions ``[T_, N_, 3]``: ``p + Y``.
        """
        return self.p + self.Y(k)

    def rollout(self, k: int) -> dict:
        """
        Get the rollout of mode k
        """
        return super().rollout(k) | {"X": self.X(k)}

    @property
    def azimuth(self) -> float:
        return get_azimuth_from_sample(self.Y(0))

class Beam3dPklDataset(PickleRolloutDataset):
    """The 3D beam buckling dataset, saved in a pickle file.

    Samples are built on load, oriented toward +x. Each one is a dict:

    - pos: [N_, T_, 3] deformed positions
    - d: [T_, 1] tip displacement
    - node_attr: [N_] bending stiffness K
    - edge_index: [2, E] edges of the beam graph
    - edge_attr: [E, 2] edge attributes (rest length L, compression C)
    """

    def __init__(self, path, split: str, n_structures: int | None = None,
                 orbit_modes: int | None = None):
        super().__init__(path, split, n_structures)
        self.orbit_modes = orbit_modes

    def _load_items(self) -> list[dict]:
        with open(self.path / self.split / "BucklingBeams3D_data.pkl", "rb") as fh:
            return load_pyg_data_list(fh)

    def _make_sample(self, mapping: dict, key: str) -> Beam3dSample:
        pos = torch.as_tensor(mapping["pos"], dtype=torch.float32)  # [N_, T_, 3]
        X = pos.permute(1, 0, 2)                              # [T_, N_, 3]
        Y = X - X[0]
        Y = Y @ z_rotation(-get_azimuth_from_sample(Y)).T     # canonicalize to +x
        X = X @ z_rotation(-get_azimuth_from_sample(Y)).T
        k = self.orbit_modes or 1 # if we want to train with multiple discrete modes
        rollouts = torch.stack([Y @ z_rotation(2.0 * math.pi * j / k).T for j in range(k)])
        f = _construct_node_features(mapping)
        buckling_start_frame = _get_buckling_start_frame(Y, tip=int(f[:, 0].argmin()))
        bifurcation = (torch.as_tensor(mapping["bifurcation"]).bool()
                       if "bifurcation" in mapping else beam_bifurcation_mask(Y)[0])
        return Beam3dSample(
            rollouts=rollouts,
            p=X[0],
            modes=Beam3dModes(representative=rollouts[0], buckling_start_frame=buckling_start_frame),
            U=torch.as_tensor(mapping["d"], dtype=torch.float32)[0],  # [T_, 1] tip displacement
            f=f,
            valid_mask=None,  # every frame is a converged equilibrium
            key=key,
            buckling_start_frame=buckling_start_frame,
            n_elements=X.shape[1] - 1,
            total_length=float(f[:, 2].sum()),
            bifurcation=bifurcation,
        )

    def counts(self, i: int) -> tuple[int, int]:
        return self.memory(("counts", i),
                         lambda: (self.orbit_modes or 1, int(self.items[i]["pos"].shape[1])))

    def train_mask(self, i: int) -> torch.Tensor:
        return self.memory(("train_mask", i),
                         lambda: torch.ones((self.orbit_modes or 1, int(self.items[i]["pos"].shape[1])),
                                            dtype=torch.bool))

    def bifurcates(self, i: int) -> bool:
        return True  # a compressed beam always buckles

    def representatives(self, i: int) -> list[int]:
        """The different orbit rollouts, or the single canonical one."""
        return list(range(self.orbit_modes or 1))
