#!/usr/bin/env python3
"""Create training, validation, and test splits for Allen-Cahn or Beam3D data."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]


def load_allen_cahn(paths: list[Path]) -> dict:
    payloads = []
    for path in paths:
        with path.open("rb") as handle:
            payloads.append(pickle.load(handle))
    if not payloads:
        raise ValueError("at least one Allen-Cahn input is required")

    first_config = payloads[0]["config"]
    for key in ("dim", "grid", "realizations"):
        default = 1 if key == "realizations" else None
        values = {payload["config"].get(key, default) for payload in payloads}
        if len(values) != 1:
            raise ValueError(f"Allen-Cahn inputs disagree on config[{key!r}]: {values}")

    return {
        "solutions": torch.cat([payload["solutions"] for payload in payloads]),
        "epsilon": torch.cat([payload["epsilon"] for payload in payloads]),
        "mu": torch.cat([payload["mu"] for payload in payloads]),
        "config": dict(first_config),
    }


def split_allen_cahn(payload: dict, val_fraction: float, test_fraction: float,
                     seed: int) -> dict[str, dict]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Allen-Cahn fractions must be non-negative and sum to less than one")

    solutions = payload["solutions"]
    epsilon = payload["epsilon"]
    mu = payload["mu"]
    config = payload["config"]
    realizations = int(config.get("realizations", 1))
    if realizations < 1 or solutions.shape[0] % realizations:
        raise ValueError(
            f"{solutions.shape[0]} trajectories cannot be grouped into "
            f"realizations={realizations}"
        )
    if len(epsilon) != len(solutions) or len(mu) != len(solutions):
        raise ValueError("solutions, epsilon, and mu must contain the same number of rows")

    n_structures = solutions.shape[0] // realizations
    order = np.random.default_rng(seed).permutation(n_structures)
    n_val = round(n_structures * val_fraction)
    n_test = round(n_structures * test_fraction)
    n_train = n_structures - n_val - n_test
    if n_train < 1:
        raise ValueError("the requested fractions leave no Allen-Cahn training structures")

    structure_indices = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }
    result = {}
    for split, indices in structure_indices.items():
        rows = np.concatenate([
            np.arange(index * realizations, (index + 1) * realizations)
            for index in indices
        ]) if len(indices) else np.empty(0, dtype=np.int64)
        rows = torch.as_tensor(rows, dtype=torch.long)
        result[split] = {
            "solutions": solutions[rows],
            "epsilon": epsilon[rows],
            "mu": mu[rows],
            "config": {
                **config,
                "realizations": realizations,
                "n_structures": int(len(indices)),
                "split": split,
                "split_seed": seed,
                "split_val_fraction": val_fraction,
                "split_test_fraction": test_fraction,
            },
        }
    return result


def allen_cahn_normalization(train: dict) -> dict:
    solutions = train["solutions"].float()
    epsilon = train["epsilon"].float()
    mu = train["mu"].float()
    return {
        "y": {"mean": float(solutions.mean()), "std": float(solutions.std())},
        "u": {
            "mean": [float(epsilon.mean()), float(mu.mean())],
            "std": [float(epsilon.std()), float(mu.std())],
        },
    }


def run_allen_cahn(args: argparse.Namespace) -> None:
    payload = load_allen_cahn(args.input)
    splits = split_allen_cahn(payload, args.val_fraction, args.test_fraction, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    name = args.name or args.input[0].stem
    for split, split_payload in splits.items():
        directory = args.output / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(split_payload, handle)
        print(
            f"[{split}] {path}: {split_payload['config']['n_structures']} structures, "
            f"{len(split_payload['solutions'])} trajectories"
        )

    normalization = allen_cahn_normalization(splits["train"])
    path = args.output / "normalization.json"
    path.write_text(json.dumps(normalization, indent=2))
    print(f"Allen-Cahn normalization: {json.dumps(normalization)}")
    print(f"Wrote {path}")


def beam3d_statistics(samples: list) -> dict:
    positions = torch.cat([sample.pos.reshape(-1, 3).float() for sample in samples])
    stiffness = torch.cat([sample.node_attr.reshape(-1).float() for sample in samples])
    edges = torch.cat([
        sample.edge_attr[: int(sample.N.item())].float() for sample in samples
    ])
    displacement = torch.cat([sample.d.reshape(-1).float() for sample in samples])
    velocity = torch.cat([
        sample.pos.float().diff(dim=1).reshape(-1, 3) for sample in samples
    ])

    def vector_stats(values: torch.Tensor) -> dict:
        return {
            "mean": [float(values[:, axis].mean()) for axis in range(values.shape[1])],
            "std": [float(values[:, axis].std()) for axis in range(values.shape[1])],
            "std_combined": float(values.std()),
        }

    def scalar_stats(values: torch.Tensor) -> dict:
        return {"mean": float(values.mean()), "std": float(values.std())}

    return {
        "pos": vector_stats(positions),
        "node_attr": scalar_stats(stiffness),
        "edge_attr_L": scalar_stats(edges[:, 0]),
        "edge_attr_C": scalar_stats(edges[:, 1]),
        "d": scalar_stats(displacement),
        "pos_velocity": vector_stats(velocity),
    }


def split_beam3d(raw: dict, validation_size: int, seed: int) -> dict[str, list]:
    if "data_tr" not in raw or "data_te" not in raw:
        raise KeyError("Beam3D input must contain 'data_tr' and 'data_te'")
    train = raw["data_tr"]
    original_test = raw["data_te"]
    if validation_size < 0 or validation_size >= len(original_test):
        raise ValueError(
            f"validation_size must be in [0, {len(original_test) - 1}], "
            f"got {validation_size}"
        )
    indices = list(range(len(original_test)))
    random.Random(seed).shuffle(indices)
    validation = sorted(indices[:validation_size])
    test = sorted(indices[validation_size:])
    return {
        "train": train,
        "val": [original_test[index] for index in validation],
        "test": [original_test[index] for index in test],
    }


def run_beam3d(args: argparse.Namespace) -> None:
    with args.input.open("rb") as handle:
        raw = pickle.load(handle)
    splits = split_beam3d(raw, args.validation_size, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    filename = "BucklingBeams3D_data.pkl"
    for split, samples in splits.items():
        directory = args.output / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with path.open("wb") as handle:
            pickle.dump(samples, handle)
        print(f"[{split}] {path}: {len(samples)} samples")

    metadata = {
        "dataset": "BucklingBeams3D",
        "splits": {name: len(samples) for name, samples in splits.items()},
        "split_seed": args.seed,
        "fields": beam3d_statistics(splits["train"]),
    }
    path = args.output / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    allen_cahn = subparsers.add_parser("allen-cahn")
    allen_cahn.add_argument("--input", type=Path, nargs="+", required=True)
    allen_cahn.add_argument("--output", type=Path, default=REPOSITORY / "data" / "allencahn")
    allen_cahn.add_argument("--name")
    allen_cahn.add_argument("--val-fraction", type=float, default=0.1)
    allen_cahn.add_argument("--test-fraction", type=float, default=0.1)
    allen_cahn.add_argument("--seed", type=int, default=0)
    allen_cahn.set_defaults(run=run_allen_cahn)

    beam3d = subparsers.add_parser("beam3d")
    beam3d.add_argument("--input", type=Path, required=True)
    beam3d.add_argument("--output", type=Path, default=REPOSITORY / "data" / "beam3d")
    beam3d.add_argument("--validation-size", type=int, default=150)
    beam3d.add_argument("--seed", type=int, default=42)
    beam3d.set_defaults(run=run_beam3d)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
