"""Allen--Cahn phase separation on a periodic one-, two-, or three-dimensional grid.

Each split pickle contains ``solutions [B,T,*grid]``, ``epsilon [B]``, ``mu [B]`` and a
``config`` mapping.  Consecutive blocks of ``config['realizations']`` trajectories share the
same physical parameters and form one sample.  The equation is odd in its scalar field, so
each simulated rollout and its exact sign-flipped solution are represented as branches.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import h5py
import torch

from bifurcation.datasets.utils.modes import DiscreteModes, detect_modes
from bifurcation.datasets.utils.pickle_rollout import PickleRolloutDataset
from bifurcation.datasets.utils.sample import Sample

_DOMAIN_LENGTH = {1: 1.0, 2: 2.0 * math.pi, 3: 2.0 * math.pi}


def grid_positions(grid_shape: tuple[int, ...]) -> torch.Tensor:
    """Flattened periodic-grid coordinates ``[N,D]`` in solver units."""
    D_ = len(grid_shape)
    if D_ not in _DOMAIN_LENGTH:
        raise ValueError(f"Allen--Cahn grid must be 1D, 2D or 3D, got {grid_shape}")
    length = _DOMAIN_LENGTH[D_]
    axes = [torch.arange(n, dtype=torch.float32) * (length / n) for n in grid_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=-1)


def _mode_labels(rollouts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    modes = detect_modes(rollouts)
    return modes.rollout_labels, modes.frame_labels, modes.valid_mask




def _infer_grid_shape(p: torch.Tensor) -> tuple[int, ...]:
    """Recover the per-axis grid size from flattened periodic-grid positions ``[N_, D_]``
    (or a batch of identical grids ``[B_, N_, D_]`` -- every item shares the same fixed grid).
    """
    coords = p if p.ndim == 2 else p[0]
    return tuple(int(torch.unique(coords[:, d]).numel()) for d in range(coords.shape[-1]))




































class AllenCahnPklDataset(PickleRolloutDataset):
    """Point-cloud view of one Allen--Cahn pickle split.

    The base :class:`Sample` is sufficient: the field is value-like, positions are static,
    and the only physical parameters are already represented by ``U = (epsilon, mu)``.
    """

    def __init__(self, path, split: str, n_structures: int | None = None,
                 sample_cache: int | None = None, latent_path: str | None = None,
                 min_epsilon: float | None = None, unique_conditions: bool = False):
        self.latent_path = Path(latent_path) if latent_path else None
        self.min_epsilon = min_epsilon
        self.unique_conditions = unique_conditions
        self._latent_h5 = None
        self._latent_h5_pid = None
        super().__init__(path, split, n_structures=n_structures, sample_cache=sample_cache)
        if n_structures is not None and len(self) != n_structures:
            detail = f" after requiring epsilon >= {min_epsilon}" if min_epsilon is not None else ""
            raise ValueError(
                f"requested {n_structures} Allen--Cahn structures{detail}, found {len(self)}"
            )
        if self.latent_path is not None:
            p = self.latent_path / split / f"{split}_latents.h5"
            if not p.is_file():
                raise FileNotFoundError(f"precomputed Allen--Cahn latents not found: {p}")
            with h5py.File(p, "r") as f:
                shape = f["latents"].shape
                expected_branches = self.counts(0)[0] if len(self) else shape[1]
                if shape[0] < len(self) or shape[1] != expected_branches:
                    raise ValueError(
                        f"latent dataset {p} starts with {shape[:2]}, expected "
                        f"at least ({len(self)}, {expected_branches})"
                    )
                if "completed" in f and not f["completed"][:len(self)].all():
                    done = int(f["completed"][:len(self)].sum())
                    total = len(self) * expected_branches
                    raise RuntimeError(f"precomputed latents are incomplete in {p}: {done}/{total}")
                self.latent_time_stride = int(f.attrs.get("time_stride", 1))
                self.latent_shape = tuple(int(v) for v in shape[2:])

    @property
    def latent_h5(self) -> h5py.File:
        """Process-local handle, safe when a DataLoader forks workers."""
        if self.latent_path is None:
            raise RuntimeError("latent_h5 requested without latent_path")
        if self._latent_h5 is None or self._latent_h5_pid != os.getpid():
            p = self.latent_path / self.split / f"{self.split}_latents.h5"
            self._latent_h5 = h5py.File(p, "r")
            self._latent_h5_pid = os.getpid()
        return self._latent_h5

    def latent(self, i: int, k: int) -> torch.Tensor | None:
        if self.latent_path is None:
            return None
        return torch.as_tensor(self.latent_h5["latents"][i, k], dtype=torch.float32)

    def latent_cache_item(self, i: int, k: int) -> dict[str, torch.Tensor] | None:
        if self.latent_path is None:
            return None
        stride = self.latent_time_stride
        return {
            "z0": self.latent(i, k),
            "U": self.conditions(i)[::stride].clone(),
            "valid_mask": self.train_mask(i)[k, ::stride].clone(),
            "_time_stride": stride,
        }

    def _payload(self) -> dict:
        if getattr(self, "_cached_payload", None) is None:
            split_path = self.path if self.split == "all" else self.path / self.split
            paths = sorted(split_path.glob("*.pkl"))
            if len(paths) != 1:
                raise ValueError(
                    f"expected exactly one pickle in {split_path}, found {len(paths)}"
                )
            with open(paths[0], "rb") as handle:
                self._cached_payload = pickle.load(handle)
        return self._cached_payload

    def _load_metadata(self, split: str) -> dict:
        del split
        return dict(self._payload()["config"])

    def _load_items(self) -> list[dict]:
        payload = self._payload()
        solutions = payload["solutions"]
        epsilon = payload["epsilon"]
        mu = payload["mu"]
        R_ = int(payload["config"].get("realizations", 1))
        if R_ <= 0 or solutions.shape[0] % R_:
            raise ValueError(
                f"{solutions.shape[0]} trajectories cannot be grouped into realizations={R_}"
            )
        if len(epsilon) != solutions.shape[0] or len(mu) != solutions.shape[0]:
            raise ValueError("solutions, epsilon and mu have different trajectory counts")
        items = [
            {
                "solutions": solutions[i : i + R_],
                "epsilon": epsilon[i],
                "mu": mu[i],
            }
            for i in range(0, solutions.shape[0], R_)
        ]
        if self.min_epsilon is not None:
            items = [item for item in items if float(item["epsilon"]) >= self.min_epsilon]
        if self.unique_conditions:
            unique = []
            seen: set[tuple[float, float]] = set()
            for item in items:
                condition = (float(item["epsilon"]), float(item["mu"]))
                if condition not in seen:
                    seen.add(condition)
                    unique.append(item)
            items = unique
        return items

    def counts(self, i: int) -> tuple[int, int]:
        """Return ``(K_, T_)`` without constructing the full sample or detecting modes."""
        raw = self.items[i]["solutions"]
        return 2 * int(raw.shape[0]), int(raw.shape[1])

    def train_mask(self, i: int) -> torch.Tensor:
        """Every solver frame is trainable, including both analytical sign branches."""
        K_, T_ = self.counts(i)
        return self.memory(("train_mask", i), lambda: torch.ones((K_, T_), dtype=torch.bool))

    def conditions(self, i: int) -> torch.Tensor:
        """The constant ``(epsilon, mu)`` condition repeated over time."""
        def build():
            mapping = self.items[i]
            T_ = int(mapping["solutions"].shape[1])
            values = torch.tensor(
                [float(mapping["epsilon"]), float(mapping["mu"])],
                dtype=torch.float32,
            )
            return values.expand(T_, 2).clone()

        return self.memory(("conditions", i), build)

    def frame(self, i: int, k: int, t: int) -> dict:
        """Build one snapshot directly, without materializing every rollout and timestep."""
        mapping = self.items[i]
        raw = mapping["solutions"]
        R_ = raw.shape[0]
        if not 0 <= k < 2 * R_:
            raise IndexError(f"rollout index {k} out of range for {2 * R_} branches")
        field = torch.as_tensor(raw[k % R_, t], dtype=torch.float32)
        if k >= R_:
            field = -field
        shape = tuple(raw.shape[2:])
        return {
            "y": field.reshape(-1, 1),
            "p": self.memory(("grid_positions", shape), lambda: grid_positions(shape)),
            "u": self.conditions(i)[t],
        }

    def _make_sample(self, mapping: dict, key: str) -> Sample:
        raw = torch.as_tensor(mapping["solutions"], dtype=torch.float32)
        R_, T_, *grid_shape = raw.shape
        simulated = raw.reshape(R_, T_, -1, 1)
        rollouts = torch.cat([simulated, -simulated], dim=0)
        epsilon = float(mapping["epsilon"])
        mu = float(mapping["mu"])
        U = torch.tensor([epsilon, mu], dtype=torch.float32).expand(T_, 2).clone()

        labels = self.memory(("mode_labels", key), lambda: _mode_labels(rollouts))
        modes = DiscreteModes(
            rollouts=rollouts,
            rollout_labels=labels[0],
            frame_labels=labels[1],
            valid_mask=labels[2],
        )
        shape = tuple(grid_shape)
        return Sample(
            rollouts=rollouts,
            p=self.memory(("grid_positions", shape), lambda: grid_positions(shape)),
            modes=modes,
            U=U,
            key=key,
        )
