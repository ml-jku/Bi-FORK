"""Microstructures: 2x2 metamaterial unit cells under 12 loading paths, bifurcating sub-cells.

The ``K_ = 4`` rollouts of a sample are the sub-cells of the large cell; they coincide until the
bifurcation and need NOT cover all modes. Mapping to the convention: ``Y`` = non-affine
fluctuation ``W``, ``p`` = centered reference configuration, ``U`` = flattened deformation
gradient, ``f`` = node type. Deformed positions are the extra key ``x = p @ defgrad(t) + y``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch

from bifurcation.datasets.utils.bifurcation import discrete_bifurcation_mask
from bifurcation.datasets.utils.modes import detect_modes
from bifurcation.datasets.utils.rollout import RolloutDataset
from bifurcation.datasets.utils.sample import Sample

LOADINGS_PER_STRUCTURE = 12

WALLPAPER_GROUPS = ["p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm",
                    "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"]




@dataclass
class MicrostructuresSample(Sample):
    """Adds the affine/fluctuation decomposition and crystallographic metadata.

    - after_contact:           ``[T_]`` frames from self-contact onward (physically irrelevant).
    - lattice:                 ``[D_, D_]`` unit-cell lattice (2 x the saved value).
    - wallpaper_group:         crystallographic group id.
    - loading:                 which of the 12 loading paths this sample is (0..11).
    - representative_rollouts: one rollout index per distinct mode.
    """

    after_contact: torch.Tensor | None = None
    lattice: torch.Tensor | None = None
    wallpaper_group: int = -1
    loading: int = -1
    representative_rollouts: torch.Tensor | None = None

    @property
    def defgrad(self) -> torch.Tensor:
        """Macroscopic deformation gradient ``[T_, D_, D_]`` (identity at t=0)."""
        return self.U.reshape(self.T_, self.D_, self.D_)

    @property
    def affine(self) -> torch.Tensor:
        """Affine (macroscopic) positions ``[T_, N_, D_]``: ``p @ defgrad(t)``."""
        return torch.einsum("nd,tde->tne", self.p, self.defgrad)

    def X(self, k: int = 0) -> torch.Tensor:
        """Deformed positions ``[T_, N_, C_]``: ``p @ defgrad(t) + Y`` (row-vector convention)."""
        return self.affine + torch.nan_to_num(self.Y(k))

    def snapshot(self, k: int, t: int) -> dict:
        out = super().snapshot(k, t)
        out["y"] = torch.nan_to_num(out["y"])
        out |= {"x": self.X(k)[t], "wallpaper": torch.tensor(self.wallpaper_group),
                "lattice": self.lattice}
        return out

    def rollout(self, k: int) -> dict:
        out = super().rollout(k)
        out["Y"] = torch.nan_to_num(out["Y"])
        out |= {"X": self.X(k), "wallpaper": torch.tensor(self.wallpaper_group),
                "lattice": self.lattice}
        return out


class MicrostructuresDataset(RolloutDataset):
    """Point-cloud view of the microstructures h5 (saved edges are ignored).

    Consecutive keys ``000000..000011`` are the 12 loading paths of one structure. ``Y`` is
    the saved ``microfluctuation``; all-NaN frames mark early simulation ends. Modes are
    detected with the saved per-sample tolerances on valid, pre-contact frames.
    """

    def __init__(self, path, split: str, n_structures: int | None = None,
                 sample_cache: int | None = None, latent_path: str | None = None):
        self.latent_path = Path(latent_path) if latent_path else None
        self._latent_h5 = None
        self._latent_h5_pid = None
        super().__init__(path, split, n_structures=n_structures, sample_cache=sample_cache)
        if self.latent_path is not None:
            p = self.latent_path / split / f"{split}_latents.h5"
            if not p.is_file():
                raise FileNotFoundError(f"precomputed microstructures latents not found: {p}")
            with h5py.File(p, "r") as f:
                shape = tuple(int(x) for x in f.attrs.get("latent_shape", ()))
                if shape != (32, 128, 128):
                    raise ValueError(f"expected latent_shape=(32, 128, 128) in {p}, got {shape}")

    @property
    def latent_h5(self) -> h5py.File:
        """Process-local handle to the flattened canonical-mode latent file."""
        if self.latent_path is None:
            raise RuntimeError("latent_h5 requested without latent_path")
        if self._latent_h5 is None or self._latent_h5_pid != os.getpid():
            p = self.latent_path / self.split / f"{self.split}_latents.h5"
            self._latent_h5 = h5py.File(p, "r")
            self._latent_h5_pid = os.getpid()
        return self._latent_h5

    def _latent_offsets(self) -> list[int]:
        """Flat latent-file offsets, whose writer used this HDF5 key/representative order."""
        def build():
            offsets = [0]
            for i in range(len(self)):
                offsets.append(offsets[-1] + len(self.representatives(i)))
            return offsets
        return self.memory("latent_offsets", build)

    def latent(self, i: int, k: int) -> torch.Tensor | None:
        if self.latent_path is None:
            return None
        grp = self._latent_group(i, k)
        return torch.as_tensor(grp["latents"][:], dtype=torch.float32)

    def _latent_group(self, i: int, k: int):
        """Resolve and validate one canonical-mode entry in the flattened latent file."""
        reps = self.representatives(i)
        if k not in reps:
            raise KeyError(f"rollout {k} of {self.keys[i]} is not a stored mode representative")
        j = self._latent_offsets()[i] + reps.index(k)
        grp = self.latent_h5[str(j)]
        saved_traj = str(grp.attrs.get("traj_key", ""))
        saved_ri = int(grp.attrs.get("ri", -1))
        if (saved_traj, saved_ri) != (self.keys[i], k):
            raise RuntimeError(
                "precomputed latent ordering mismatch at "
                f"{j}: expected ({self.keys[i]!r}, {k}), got ({saved_traj!r}, {saved_ri})"
            )
        return grp

    def latent_cache_item(self, i: int, k: int) -> dict[str, torch.Tensor] | None:
        """Read no raw point cloud: only latent, condition, and the exact 32-frame mask.

        ``train_mask`` touches only the tiny raw ``interpolation_mask`` dataset and contact
        attribute.  It is retained because three modes in the complete train/test corpus contain
        an invalid pre-contact frame that cannot be inferred from ``after_contact`` alone.
        """
        if self.latent_path is None:
            return None
        grp = self._latent_group(i, k)
        item = {
            "z0": torch.as_tensor(grp["latents"][:], dtype=torch.float32),
            "U": torch.as_tensor(grp["def_gradient"][:], dtype=torch.float32).reshape(-1, 4),
            "valid_mask": self.train_mask(i)[k].clone(),
        }
        raw = self.h5[self.keys[i]]
        if "bifurcation" in raw:
            item["bifurcation"] = torch.as_tensor(raw["bifurcation"][:]).bool()
        else:
            item["bifurcation"] = discrete_bifurcation_mask(self[i].modes).cpu()
        return item

    def _structure_of(self, key: str) -> str:
        return str(int(key) // LOADINGS_PER_STRUCTURE)

    def counts(self, i: int) -> tuple[int, int]:
        def read():
            shape = self.h5[self.keys[i]]["x"].shape
            return int(shape[0]), int(shape[1])
        return self.memory(("counts", i), read)

    def train_mask(self, i: int) -> torch.Tensor:
        def read():
            grp = self.h5[self.keys[i]]
            K_ = grp["x"].shape[0]
            keep = torch.as_tensor(grp["interpolation_mask"][:]).bool()
            keep &= ~torch.as_tensor(grp.attrs["after_contact"]).bool()[: len(keep)]
            return keep[None].repeat(K_, 1)
        return self.memory(("train_mask", i), read)

    def bifurcates(self, i: int) -> bool:
        return self.memory(("bifurcates", i),
                         lambda: int(self.h5[self.keys[i]].attrs["num_unique_modes"]) > 1)

    def representatives(self, i: int) -> list[int]:
        """Saved representative (``canonical_subtraj_indices``)."""
        return self.memory(("representatives", i),
                         lambda: [int(k) for k in self.h5[self.keys[i]]["canonical_subtraj_indices"][:]])

    def conditions(self, i: int) -> torch.Tensor:
        def read():
            F = torch.as_tensor(self.h5[self.keys[i]]["F"][:], dtype=torch.float32)
            return F.reshape(F.shape[0], -1)
        return self.memory(("conditions", i), read)

    def frame(self, i: int, k: int, t: int) -> dict:
        """Single-frame item without loading the rollout (~K_*T_ less I/O)."""
        st = self._statics(i)
        grp = self.h5[self.keys[i]]
        u = torch.as_tensor(grp["F"][t], dtype=torch.float32)
        y = torch.nan_to_num(torch.as_tensor(grp["microfluctuation"][k, t], dtype=torch.float32))
        bifurcation = (torch.as_tensor(grp["bifurcation"][t]).bool()
                       if "bifurcation" in grp else discrete_bifurcation_mask(self[i].modes)[t])
        return {"y": y, "p": st["p"], "u": u.reshape(-1), "f": st["f"],
                "x": st["p"] @ u + y, "wallpaper": st["wallpaper"], "lattice": st["lattice"],
                "bifurcation": bifurcation}

    def _statics(self, i: int) -> dict:
        """Per-sample constants for the frame fast path, small LRU."""
        if not hasattr(self, "_statics_cache"):
            self._statics_cache: OrderedDict[int, dict] = OrderedDict()
        if i not in self._statics_cache:
            grp = self.h5[self.keys[i]]
            x_ref = torch.as_tensor(grp["x"][0, 0], dtype=torch.float32)
            self._statics_cache[i] = {
                "p": x_ref - torch.nanmean(x_ref, dim=0, keepdim=True),
                "f": torch.as_tensor(grp["node_type"][:])[:, None],
                "wallpaper": torch.tensor(int(grp.attrs["wallpaper_group_id"])),
                "lattice": 2.0 * torch.as_tensor(grp.attrs["lattice_vectors"][0],
                                                 dtype=torch.float32),
            }
            while len(self._statics_cache) > 256:
                self._statics_cache.popitem(last=False)
        self._statics_cache.move_to_end(i) # on cache hit, keep it alive for a while
        return self._statics_cache[i]


    def _make_sample(self, grp, key: str) -> MicrostructuresSample:
        attrs = grp.attrs
        rollouts = torch.as_tensor(grp["microfluctuation"][:], dtype=torch.float32)
        K_, T_ = rollouts.shape[:2]

        valid = torch.as_tensor(grp["interpolation_mask"][:]).bool()
        after_contact = torch.as_tensor(attrs["after_contact"]).bool()[:T_]
        modes = detect_modes(rollouts,
                             valid_mask=(valid & ~after_contact)[None].repeat(K_, 1),
                             rtol=float(attrs["canonical_rtol"]), atol=float(attrs["canonical_atol"]))
        bifurcation = (torch.as_tensor(grp["bifurcation"][:]).bool()
                       if "bifurcation" in grp else discrete_bifurcation_mask(modes))

        statics = self._statics(self.keys.index(key))
        return MicrostructuresSample(
            rollouts=rollouts,
            p=statics["p"],
            modes=modes,
            valid_mask=valid[None].repeat(K_, 1),
            U=torch.as_tensor(grp["F"][:], dtype=torch.float32).reshape(T_, -1),
            f=statics["f"],
            key=key,
            after_contact=after_contact,
            lattice=statics["lattice"],
            wallpaper_group=int(attrs["wallpaper_group_id"]),
            loading=int(key) % LOADINGS_PER_STRUCTURE,
            representative_rollouts=torch.as_tensor(grp["canonical_subtraj_indices"][:]),
            bifurcation=bifurcation,
        )
