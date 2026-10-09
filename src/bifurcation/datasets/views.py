"""A view turns a :class:`~bifurcation.datasets.utils.rollout.RolloutDataset` into items.

The dataset keeps whole samples; the view decides what one item is:

 * ``Snapshots``         -- one frame (first stage).
 * ``Rollouts``          -- one whole rollout (second stage, baselines).
 * ``PositionRollouts``  -- rollout plus per-frame positions ``X`` (node-space baselines).
 * ``ModeGroupRollouts`` -- all mode representatives of one sample, kept together per batch.

``sampling`` picks how random draws are made:

 * ``"uniform"``:  every valid (sample, rollout, frame) is equally likely.
 * ``"balanced"``: snapshots visit every trainable frame once; mode groups draw conditions
   with replacement so the 1/2/4-mode condition classes have equal expected frequency.

An item is a dict of arrays plus an ``index`` recording where it came from. ``collate`` stacks
items into a batch: node axes zero-padded to the batch max, a boolean ``mask`` on the real nodes.
"""

from __future__ import annotations

import bisect
from itertools import accumulate

import torch
from torch.utils.data import DataLoader, Dataset

from bifurcation.datasets.utils.normalization import Normalizer
from bifurcation.datasets.utils.rollout import RolloutDataset
from bifurcation.utils import farthest_point_sample
from bifurcation.utils.batching import to_tensor


def _shape(v) -> tuple:
    return tuple(torch.as_tensor(v).shape)


def _offsets_of(counts) -> list[int]:
    """Exclusive prefix sums: [5, 3] -> [0, 5, 8]. Entry i is where the flat cache indices of
    sample i start, and the last entry is the total."""
    return [0, *accumulate(int(c) for c in counts)]


def _chunks(xs: list, size: int):
    """Consecutive chunks: [1, 2, 3, 4, 5] -> [1, 2], [3, 4], [5] for size 2."""
    for s in range(0, len(xs), size):
        yield xs[s: s + size]


def get_node_axis_of_key(key: str, row: dict) -> int | None:
    """Which axis of ``row[key]`` runs over nodes: 0 for ``[N_, ...]``, 1 for
    ``[T_, N_, ...]``, None if there is no node axis. Works on shapes only."""
    if key in ("index", "mask", "u", "U", "valid_mask", "bifurcation", "g"):
        return None
    N_ = _shape(row["p"])[0]
    s = _shape(row[key])
    if len(s) >= 2 and s[1] == N_ and key.isupper():
        return 1
    if len(s) >= 1 and s[0] == N_:
        return 0
    return None


def _pad_nodes(a: torch.Tensor, axis: int, N_max: int) -> torch.Tensor:
    """Zero-pad ``a`` along its node ``axis`` to ``N_max`` nodes."""
    pad = list(a.shape)                      # axis 1, N_max [T, n, K], n -> N
    pad[axis] = N_max - a.shape[axis]        # [T, N-n, K]
    if pad[axis] == 0:
        return a
    return torch.cat([a, a.new_zeros(pad)], dim=axis)  # [T, N, K]


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Stack items into a batch. Items can have different node counts, so node axes are
    zero-padded to the largest count and ``mask [B_, N_]`` marks the real nodes. Dtypes
    follow ``to_tensor``."""
    if "p" not in rows[0]:  # no point cloud, e.g. cached second-stage latents: just stack
        return {key: torch.stack([to_tensor(r[key]) for r in rows]) for key in rows[0]}
    n_of = [int(torch.as_tensor(r["p"]).shape[0]) for r in rows]    # [B_,] nodes per item
    N_max = max(n_of)
    out: dict[str, torch.Tensor] = {}
    for key in rows[0]:
        arrs = [to_tensor(r[key]) for r in rows]
        axis = get_node_axis_of_key(key, rows[0])
        if axis is not None:
            arrs = [_pad_nodes(a, axis, N_max) for a in arrs]
        out[key] = torch.stack(arrs)
    out["mask"] = torch.arange(N_max)[None, :] < torch.tensor(n_of)[:, None]    # [B_, N_], True = real node
    return out


def collate_grouped(rows: list[dict]) -> dict[str, torch.Tensor]:
    """``collate`` plus ``group``: marks which rows belong to the same sample (= the same
    condition). Each view writes ``index`` when it builds an item -- (sample, rollout) or
    (sample, rollout, frame) -- so the sample id is its first column. Downstream (the OT
    coupling) only compares group ids for equality; the values mean nothing."""
    batch = collate(rows)
    batch["group"] = batch["index"][:, 0]    # [B_,] sample ids, e.g. [0, 0, 1, 1]
    return batch


def collate_device(rows: list[dict], device) -> dict[str, torch.Tensor]:
    """``collate`` for cached items. These are already tensors, possibly on different
    devices (big keys can fit in CPU RAM), so first move everything to ``device`` --
    one copy per batch -- then pad and stack there, same as ``collate``."""
    move = {key: [r[key].to(device) for r in rows] for key in rows[0]}
    if "p" not in rows[0]:  # no point cloud (second-stage latents): just stack
        return {key: torch.stack(v) for key, v in move.items()}
    n_of = [int(a.shape[0]) for a in move["p"]]    # [B_,] nodes per item
    N_max = max(n_of)
    out: dict[str, torch.Tensor] = {}
    for key, arrs in move.items():
        axis = get_node_axis_of_key(key, rows[0])
        if axis is not None:
            arrs = [_pad_nodes(a, axis, N_max) for a in arrs]
        out[key] = torch.stack(arrs)
    out["mask"] = torch.arange(N_max, device=device)[None, :] < torch.tensor(n_of, device=device)[:, None]  # [B_, N_]
    return out


class _View(Dataset):
    """Shared machinery: building items, augmentation, node subsampling, normalization,
    and the device cache."""

    def __init__(self, dataset: RolloutDataset, seed: int | None, normalizer: Normalizer,
                 num_nodes: int | None, transform, sampling: str,
                 time_stride: int | None = None, fixed_draws: bool = False,
                 window: int | None = None):
        if sampling not in ("balanced", "uniform"):
            raise ValueError(f"sampling must be 'balanced' or 'uniform', got {sampling!r}")
        self.dataset = dataset
        self.seed = seed
        self.normalizer = normalizer
        self.num_nodes = num_nodes
        self.transform = transform  # dataset-specific augmentation: (item, rng) -> item
        self.fixed_draws = fixed_draws  # replay the same draws every epoch, see _rng_for
        self.sampling = sampling
        self.time_stride = int(time_stride or 1)
        self.window = int(window) if window else None
        self._rng = None
        self._fps_cache: dict = {}
        self._cache: dict[int, dict] = {}  # flat index -> item, each key on its cache device
        self._n = 0
        self._bytes: dict[str, int] = {}   # device -> bytes in cache, grows as it fills
        self._groups = None
        self._device = None      # where the model runs and batches end up
        self._devices = {}       # per-key storage devices; else 'default' entry; else _device
        self._precompute = None  # model.precompute_batch, set by enable_cache

    @property
    def rng(self) -> torch.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = torch.Generator()
            if self.seed is None:
                self._rng.seed()  # a fresh Generator is seeded DETERMINISTICALLY; reseed it
            else:
                self._rng.manual_seed(self.seed if info is None else self.seed + info.seed)
        return self._rng

    def _rng_for(self, *ids) -> torch.Generator:
        """The stream a draw comes from."""
        if not self.fixed_draws:
            return self.rng
        seed = (self.seed or 0) + 1
        for v in ids:
            seed = (seed * 100_003 + int(v) + 1) % (2 ** 31)
        return torch.Generator().manual_seed(seed)

    def _finish(self, item: dict) -> dict:
        """Last steps on every item: augment, subsample nodes, normalize."""
        if self.transform is not None:
            item = self.transform(item, self._rng_for(*item["index"].flatten().tolist()))
        if self.num_nodes is not None and self.num_nodes < item["p"].shape[0]:
            idx = self._fps_indices(item)    # [num_nodes,] ids of the kept nodes
            axes = {k: get_node_axis_of_key(k, item) for k in item}
            for k, axis in axes.items():
                if axis is not None:
                    item[k] = item[k].index_select(axis, idx)
        return self.normalizer.normalize(item)

    def _fps_indices(self, item: dict) -> torch.Tensor:
        """One fixed farthest-point subset per sample, computed once and reused (~185 ms
        at 20k nodes). Deterministic on purpose: subsampling is a debugging option, not an
        augmentation."""
        i = int(item["index"][0])  # sample id
        key = (i, self.num_nodes)
        if key not in self._fps_cache:
            rng = torch.Generator().manual_seed(((self.seed or 0) + 1) * 100_003 + i)
            self._fps_cache[key] = farthest_point_sample(
                torch.nan_to_num(item["p"]), self.num_nodes, rng)
        return self._fps_cache[key]



    def cache_len(self) -> int:
        raise NotImplementedError

    def unravel(self, j: int) -> tuple:
        raise NotImplementedError

    def cache_groups(self) -> list[list[int]] | None:
        return None

    def enable_cache(self, device, precompute, devices: dict | None = None) -> None:
        """Turn on the cache; it fills during the first epoch.

        Items are built unaugmented and augmentation runs per batch; see docs/decisions.md.

        Args:
            device: where the model runs and batches end up
            precompute: ``model.precompute_batch``, decides what is saved per item
            devices (dict | None): storage device per key; unlisted keys use ``default``.
        """
        self._device = device
        self._precompute = precompute
        self._devices = devices or {}
        self._n = self.cache_len()
        self._groups = self.cache_groups()
        print(f"[cache] building index: {self._n} items | devices: "
              f"{self._devices or 'all on ' + str(device)}")

    def _store_device(self, key: str):
        return self._devices.get(key, self._devices.get("default", self._device))

    def _cached_item(self, j: int) -> dict[str, torch.Tensor]:
        if j not in self._cache:
            transform, self.transform = self.transform, None  # cache holds unaugmented items
            try:
                builder = getattr(self, "build_cache", self.build)
                built = {k: v.to(self._device) for k, v in collate([builder(self.unravel(j))]).items()}
            finally:
                self.transform = transform
            item = {k: v[0] for k, v in self._precompute(built).items()}
            item["index"] = torch.tensor(j, dtype=torch.int32)  # flat id; the (i, k, t) key is not kept
            self._cache[j] = {k: v.to(self._store_device(k)) for k, v in item.items()}
            for v in self._cache[j].values():
                dev = str(v.device)
                self._bytes[dev] = self._bytes.get(dev, 0) + v.numel() * v.element_size()
            n = len(self._cache)
            if n == 1 or n == self._n or n % 256 == 0:
                print(f"[cache] filling {n}/{self._n} | {self._mem_str()}")
        return self._cache[j]

    def _mem_str(self) -> str:
        return " | ".join(f"{dev}: {b / 2**20:.0f} MB" for dev, b in sorted(self._bytes.items()))

    def cache_stats(self) -> tuple[int, int, dict[str, float]]:
        """(items cached, items total, {device: MB}) -- for the training-time cache log."""
        return len(self._cache), self._n, {d: b / 2**20 for d, b in self._bytes.items()}

    def cached_loader(self, batch_size: int, shuffle: bool) -> "_CachedLoader":
        return _CachedLoader(self, self._device, self._n, self._groups, batch_size, shuffle)

    def loader(self, batch_size: int, shuffle: bool, num_workers: int, **kw) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=collate, **kw)


class Snapshots(_View):
    """One frame per item (first stage)."""

    def __init__(self, dataset, seed, normalizer, num_nodes, transform, sampling,
                 time_stride=None, fixed_draws=False):
        super().__init__(dataset, seed, normalizer, num_nodes, transform, sampling, time_stride,
                         fixed_draws)
        self.masks = [dataset.train_mask(i) for i in range(len(dataset))]
        if self.time_stride > 1:
            self.masks = [m.clone() for m in self.masks]
            for m in self.masks:
                m[:, torch.arange(m.shape[1]) % self.time_stride != 0] = False
        self._offsets = _offsets_of(m.sum() for m in self.masks)
        if sampling == "uniform":
            self.valid_triples = torch.tensor(  # [n_valid, 3]
                [(i, int(k), int(t)) for i, m in enumerate(self.masks)
                 for k, t in torch.nonzero(m)], dtype=torch.long)
            self.n_slots = int(sum(m.numel() for m in self.masks))  # epoch length: sum of K_ * T_
        else:
            self.frames = [(i, int(t)) for i, m in enumerate(self.masks)
                           for t in torch.nonzero(m.any(0)).flatten()]  # every trainable (sample, frame)

    def __len__(self) -> int:
        return self.n_slots if self.sampling == "uniform" else len(self.frames)

    def __getitem__(self, j: int) -> dict:
        if self.sampling == "uniform":  # iid uniform over the valid triples
            i, k, t = self.valid_triples[torch.randint(len(self.valid_triples), (),
                                                       generator=self._rng_for(j))]
        else:
            i, t = self.frames[j]
            k = self.dataset[i].modes.draw_frame(t, self._rng_for(i, t))
        return self.build((int(i), int(k), int(t)))

    def build(self, key: tuple) -> dict:
        i, k, t = key
        item = self.dataset.frame(int(i), int(k), int(t))
        item["index"] = torch.tensor([i, k, t], dtype=torch.long)
        return self._finish(item)  # augment, subsample nodes, normalize

    def cache_len(self) -> int:
        return self._offsets[-1]

    def unravel(self, j: int) -> tuple:
        """Flat cache index -> (sample, rollout, frame). E.g. offsets [0, 5, 8] and j=6:
        j falls between 5 and 8, so sample 1, and 6-5=1 -> its second valid (k, t)."""
        i = bisect.bisect_right(self._offsets, j) - 1
        k, t = torch.nonzero(self.masks[i])[j - self._offsets[i]]
        return (i, int(k), int(t))


class Rollouts(_View):
    """One whole rollout per item (second stage, baselines).

    ``fixed_rollout``: pin every sample to this one saved rollout index instead of drawing
    among its ``K_`` branches -- a single-mode ablation (e.g. isolating whether a generation
    artifact comes from multi-modality). Applies to both the plain DataLoader path
    (``_draw_rollout``) and the device cache (``unravel``), so ``gpu_cache`` may stay on.
    """

    def __init__(self, *args, fixed_rollout: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_rollout = fixed_rollout

    def __len__(self) -> int:
        return len(self.dataset)

    def _draw_rollout(self, i: int) -> int:
        if self.fixed_rollout is not None:
            return self.fixed_rollout
        if self.sampling == "uniform":  # counts(i) = (K_, T_): every saved rollout equally likely
            return int(torch.randint(self.dataset.counts(i)[0], (), generator=self._rng_for(i)))
        return self.dataset[i].modes.draw_rollout(self._rng_for(i))

    def _rollout_item(self, i: int, k: int) -> dict:
        item = self.dataset[i].rollout(k)
        latent = self.dataset.latent(i, k)
        if latent is not None:
            item["z0"] = latent
        item["valid_mask"] = self.dataset.train_mask(i)[k]
        if self.time_stride > 1:
            for key in ("Y", "X", "U", "H", "valid_mask", "bifurcation"):
                if key in item:
                    item[key] = item[key][:: self.time_stride]
        if "z0" in item:
            latent_stride = int(getattr(self.dataset, "latent_time_stride", 1))
            if self.time_stride % latent_stride:
                raise ValueError(
                    f"requested time_stride={self.time_stride} is not divisible by the "
                    f"stored latent stride={latent_stride}"
                )
            item["z0"] = item["z0"][:: self.time_stride // latent_stride]
        if self.window is not None:
            T = item["Y"].shape[0]
            start = int(torch.randint(0, max(T - self.window, 0) + 1, (),
                                      generator=self._rng_for(i, k)))
            for key in ("Y", "X", "U", "H", "z0", "valid_mask", "bifurcation"):
                if key in item:
                    item[key] = item[key][start: start + self.window]
        item["index"] = torch.tensor([i, k], dtype=torch.long)
        return item

    def __getitem__(self, i: int) -> dict:
        return self.build((i, self._draw_rollout(i)))

    def build(self, key: tuple) -> dict:
        i, k = key
        return self._finish(self._rollout_item(int(i), int(k)))

    def build_cache(self, key: tuple) -> dict:
        """Build the smallest cache item possible when the dataset stores latents.

        This deliberately differs from :meth:`build`: evaluation needs the raw point-cloud
        fields for decoding and metrics, whereas latent-space training needs only z0, U and its
        timestep mask.
        """
        i, k = (int(v) for v in key)
        item = self.dataset.latent_cache_item(i, k)
        if item is None:
            return self.build((i, k))
        stored_stride = int(item.pop("_time_stride", 1))
        if self.time_stride % stored_stride:
            raise ValueError(
                f"requested time_stride={self.time_stride} is not divisible by the "
                f"stored latent stride={stored_stride}"
            )
        effective_stride = self.time_stride // stored_stride
        if effective_stride > 1:
            for name in ("z0", "U", "valid_mask", "bifurcation"):
                if name in item:
                    item[name] = item[name][::effective_stride]
        if self.window is not None:
            T = item["z0"].shape[0]
            start = int(torch.randint(0, max(T - self.window, 0) + 1, (),
                                      generator=self._rng_for(i, k)))
            for name in ("z0", "U", "valid_mask", "bifurcation"):
                if name in item:
                    item[name] = item[name][start: start + self.window]
        item["index"] = torch.tensor([i, k], dtype=torch.long)
        return self._finish(item)

    @property
    def _offsets(self) -> list[int]:
        if getattr(self, "_off", None) is None:
            counts = ((1 for _ in range(len(self.dataset))) if self.fixed_rollout is not None else
                      (self.dataset.counts(i)[0] for i in range(len(self.dataset))))
            self._off = _offsets_of(counts)
        return self._off

    def cache_len(self) -> int:
        return self._offsets[-1]

    def unravel(self, j: int) -> tuple:
        i = bisect.bisect_right(self._offsets, j) - 1
        k = self.fixed_rollout if self.fixed_rollout is not None else j - self._offsets[i]
        return (i, k)


class PositionRollouts(Rollouts):
    """Rollouts plus the deformed positions ``X [T_, N_, D_]`` in physical units.

    A separate view because only node-space models (the STFlow baseline) read X, and it
    is as big as Y -- everyone else would pay for it in every batch and in the cache.

    X is deliberately NOT normalized: it is geometry, not a network input. STFlow builds
    spatial graphs on it and computes with it in physical units, together with
    DE-normalized predictions (``X - denorm(Y)``). Symmetry transforms keep X consistent
    with Y automatically: ``SymmetryOperation.apply`` rotates ``x``/``X`` with the field.  lang-ok
    """

    def _rollout_item(self, i: int, k: int) -> dict:
        item = super()._rollout_item(i, k)
        if "X" not in item:
            item["X"] = self.dataset[i].X(k)[:: self.time_stride]
        return item


class ModeGroupRollouts(Rollouts):
    """All mode representatives of one sample form a group; a batch holds ``batch_size``
    whole groups (rows ~ batch_size x K_). Per-row ``group`` ids mark the condition, and
    the OT coupling matches only within each group. Every epoch sees each representative
    exactly once under ``sampling=uniform``. Under ``sampling=balanced``, training draws the
    same number of conditions with replacement, weighting each condition inversely by the
    frequency of its mode-count class. Rollouts never leave their group (both the DataLoader
    path and the cached path work this way)."""

    def __init__(self, dataset, seed, normalizer, num_nodes, transform, sampling,
                 time_stride=None, fixed_draws=False):
        super().__init__(dataset, seed, normalizer, num_nodes, transform, sampling, time_stride,
                         fixed_draws)
        self.items = [(i, int(k)) for i in range(len(dataset))
                      for k in dataset.representatives(i)]  # [(sample, mode)]
        offsets = _offsets_of(len(dataset.representatives(i)) for i in range(len(dataset)))
        self.groups = [list(range(offsets[i], offsets[i + 1])) for i in range(len(dataset))]
        self.balance_group_sizes = sampling == "balanced"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, j: int) -> dict:
        return self.build(self.items[j])

    def cache_len(self) -> int:
        return len(self.items)

    def unravel(self, j: int) -> tuple:
        return self.items[j]

    def cache_groups(self) -> list[list[int]]:
        return self.groups

    def loader(self, batch_size: int, shuffle: bool, num_workers: int, **kw) -> DataLoader:
        return DataLoader(self, batch_sampler=_GroupSampler(
                              self.groups, batch_size, shuffle,
                              balance_mode_counts=self.sampling == "balanced"),
                          num_workers=num_workers, collate_fn=collate_grouped, **kw)


class RealizationGroupRollouts(Rollouts):
    """All four Allen--Cahn realizations at fixed ``(epsilon, mu)`` form one OT group.

    Allen--Cahn stores ``R`` simulated realizations followed by their ``R`` analytical sign
    flips.  OT should compare alternative simulations of the same physical condition, but it
    should neither cross conditions nor use a sign flip as an independent simulation.  Each
    sample therefore contributes two groups, ``[0..R-1]`` and ``[R..2R-1]``.  ``batch_size``
    means rows (not groups), so batch size 64 with R=4 gives 16 independent Hungarian problems.
    Every group is visited exactly once per epoch; no detected-mode balancing is applied.
    """

    def __init__(self, dataset, seed, normalizer, num_nodes, transform, sampling,
                 time_stride=None, fixed_draws=False):
        super().__init__(dataset, seed, normalizer, num_nodes, transform, sampling, time_stride,
                         fixed_draws)
        self.items: list[tuple[int, int]] = []
        self.groups: list[list[int]] = []
        self.realizations: list[int] = []
        group_sizes = set()
        for i in range(len(dataset)):
            K, _ = dataset.counts(i)
            if K % 2:
                raise ValueError(f"sample {i} has {K} branches; expected realizations + sign flips")
            R = K // 2
            self.realizations.append(R)
            group_sizes.add(R)
            for start in (0, R):
                group = []
                for k in range(start, start + R):
                    group.append(len(self.items))
                    self.items.append((i, k))
                self.groups.append(group)
        self.group_size = next(iter(group_sizes), 1) if len(group_sizes) <= 1 else None
        self.balance_group_sizes = False

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, j: int) -> dict:
        return self.build(self.items[j])

    def build(self, key: tuple) -> dict:
        i, k = (int(v) for v in key)
        item = super().build((i, k))
        item["index"] = torch.tensor([2 * i + k // self.realizations[i], k], dtype=torch.long)
        return item

    def cache_len(self) -> int:
        return len(self.items)

    def unravel(self, j: int) -> tuple:
        return self.items[j]

    def cache_groups(self) -> list[list[int]]:
        return self.groups

    def _groups_per_batch(self, batch_size: int) -> int:
        if self.group_size is None:
            raise ValueError("a fixed groups-per-batch count is undefined for mixed realization sizes")
        if batch_size < self.group_size or batch_size % self.group_size:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by realization group size "
                f"{self.group_size}"
            )
        return batch_size // self.group_size

    def cached_loader(self, batch_size: int, shuffle: bool) -> "_CachedLoader":
        if self.group_size is None:
            raise ValueError("GPU caching is not supported for mixed realization sizes")
        return _CachedLoader(self, self._device, self._n, self._groups,
                             self._groups_per_batch(batch_size), shuffle)

    def loader(self, batch_size: int, shuffle: bool, num_workers: int, **kw) -> DataLoader:
        sampler = (_GroupSampler(self.groups, self._groups_per_batch(batch_size), shuffle,
                                 balance_mode_counts=False)
                   if self.group_size is not None else
                   _RowBudgetGroupSampler(self.groups, batch_size, shuffle))
        return DataLoader(self, batch_sampler=sampler, num_workers=num_workers,
                          collate_fn=collate_grouped, **kw)


class _CachedLoader:
    """Iterates batches straight from the cache of a view.

    The first epoch fills the cache while iterating; shuffling from the start is safe
    because an item depends only on its index, not on visit order. With ``groups``, a
    batch holds ``batch_size`` whole groups plus per-row ``group`` ids. Augmentation runs
    on each batch with fresh draws."""

    def __init__(self, view: _View, device, n: int, groups: list[list[int]] | None,
                 batch_size: int, shuffle: bool):
        self.view = view
        self.device = device
        self.n = n
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        n = len(self.groups) if self.groups is not None else self.n
        return (n + self.batch_size - 1) // self.batch_size

    def _order(self, n: int) -> list[int]:
        if self.groups is not None and self.shuffle and getattr(
                self.view, "balance_group_sizes", False):
            return _mode_count_balanced_order(self.groups)
        return torch.randperm(n).tolist() if self.shuffle else list(range(n))

    def _batch(self, indices, group_sizes: list[int] | None = None) -> dict[str, torch.Tensor]:
        batch = collate_device([self.view._cached_item(int(j)) for j in indices], self.device)
        if group_sizes is not None:  # before augmentation: it may draw per group
            batch["group"] = torch.repeat_interleave(
                torch.arange(len(group_sizes), device=self.device),
                torch.tensor(group_sizes, device=self.device))  # [rows], e.g. [0, 0, 1, 1]
        if self.view.transform is not None:
            if not any(key in batch for key in ("p", "y", "Y")):
                raise ValueError(
                    "augmentation needs the raw field keys in the cached items, but this "
                    "the precompute of the model drops them (e.g. second-stage latents) -- set "
                    "gpu_cache=false or transform=null")
            batch = self.view.transform(
                batch, self.view._rng_for(*batch["index"].flatten().tolist()))
        return batch

    def __iter__(self):
        if self.groups is not None:
            for chunk in _chunks(self._order(len(self.groups)), self.batch_size):
                groups = [self.groups[g] for g in chunk]
                yield self._batch([j for g in groups for j in g], [len(g) for g in groups])
        else:
            for chunk in _chunks(self._order(self.n), self.batch_size):
                yield self._batch(chunk)


class _GroupSampler:
    """Batch sampler for the DataLoader path: ``batch_size`` whole groups per batch,
    group order shuffled if asked."""

    def __init__(self, groups: list[list[int]], batch_size: int, shuffle: bool,
                 balance_mode_counts: bool = False):
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance_mode_counts = balance_mode_counts

    def __iter__(self):
        if self.shuffle and self.balance_mode_counts:
            order = _mode_count_balanced_order(self.groups)
        else:
            order = torch.randperm(len(self.groups)).tolist() if self.shuffle else list(range(len(self.groups)))
        for chunk in _chunks(order, self.batch_size):
            yield [j for g in chunk for j in self.groups[g]]

    def __len__(self) -> int:
        return (len(self.groups) + self.batch_size - 1) // self.batch_size


class _RowBudgetGroupSampler:
    """Pack variable-size whole groups without exceeding a row-count batch budget."""

    def __init__(self, groups: list[list[int]], max_rows: int, shuffle: bool):
        if any(len(group) > max_rows for group in groups):
            raise ValueError("a realization group is larger than the requested batch size")
        self.groups = groups
        self.max_rows = max_rows
        self.shuffle = shuffle

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist() if self.shuffle else list(range(len(self.groups)))
        batch: list[int] = []
        rows = 0
        for group_index in order:
            group = self.groups[group_index]
            if batch and rows + len(group) > self.max_rows:
                yield batch
                batch, rows = [], 0
            batch.extend(group)
            rows += len(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        total_rows = sum(map(len, self.groups))
        return (total_rows + self.max_rows - 1) // self.max_rows


def _mode_count_balanced_order(groups: list[list[int]]) -> list[int]:
    """Draw ``len(groups)`` conditions with replacement, balancing group-size classes.

    A condition in a class containing ``n_k`` conditions receives weight ``1 / n_k``;
    consequently every represented mode-count class has the same total probability.
    """
    if not groups:
        return []
    class_counts: dict[int, int] = {}
    for group in groups:
        class_counts[len(group)] = class_counts.get(len(group), 0) + 1
    weights = torch.tensor([1.0 / class_counts[len(group)] for group in groups])
    return torch.multinomial(weights, len(groups), replacement=True).tolist()
