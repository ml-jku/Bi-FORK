"""A rollout dataset saved as one pickled list of samples, rather than an HDF5 file.
"""

from __future__ import annotations

import json
import pickle
from abc import abstractmethod

from bifurcation.datasets.utils.rollout import RolloutDataset


def unpickle_pyg(fileobj):
    """Unpickle a file of PyTorch-Geometric objects *without* importing ``torch_geometric``:
    every ``torch_geometric.*`` class is replaced by a plain bag that keeps its state. Returns
    whatever the file holds -- a list of graphs, or (the published reference sets) a dict of
    named lists. Use :func:`graph_mappings` to turn a list of graphs into plain dicts."""

    class _Bag:
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:  # torch_geometric storages occasionally pickle a non-dict state
                self.__dict__["_state"] = state

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch_geometric"):
                return _Bag
            return super().find_class(module, name)

    return _Unpickler(fileobj).load()


def graph_mappings(graphs) -> list[dict]:
    """The array mapping of each graph (``pos``, ``node_attr``, ``edge_index``, ...), a plain
    dict per sample. The arrays are ordinary torch tensors."""
    return [graph.__dict__["_store"].__dict__["_mapping"] for graph in graphs]


def load_pyg_data_list(fileobj) -> list[dict]:
    """Read a pickled list of PyTorch-Geometric graphs into one plain dict per sample."""
    return graph_mappings(unpickle_pyg(fileobj))


class PickleRolloutDataset(RolloutDataset):
    @abstractmethod
    def _load_items(self) -> list:
        """The list of raw per-sample entries (one per sample), read from disk."""

    def _load_metadata(self, split: str) -> dict:
        p = self.path / "metadata.json"
        return json.load(open(p)) if p.exists() else {}

    def _select_keys(self, n_structures: int | None) -> list[str]:
        items = self._load_items()
        self.items = items[:n_structures] if n_structures else items
        return [f"{i:06d}" for i in range(len(self.items))]

    def _raw(self, i: int):
        return self.items[i]
