"""JSD-based symmetry/coverage metrics for Allen-Cahn generative evaluation.

Three metrics share one FFT-based matching step (:func:`match_all`): a generated sample is
assigned to (1) the nearest of the sample's ``K_`` saved GT branches -- rotation/translation
invariant, via the radially-averaged 3D FFT power spectrum, plus a cheap odd moment
(:func:`_sign_feature`) that breaks the exact tie the spectrum alone leaves between a branch and
its sign-flipped twin -- then, against that branch, to
(2) a periodic translation and (3) an octahedral rotation/reflection, both recovered by FFT
phase correlation. Exhaustive search over the octahedral group follows the pattern of Klein,
Kraemer & Noe, "Equivariant flow matching" (arXiv:2306.15030) Eq. 15-16: the true alignment cost
minimizes over the whole symmetry group; there the group is continuous/large so it is
approximated sequentially (Hungarian + Kabsch), but ours is a fixed 48-element finite group (no
permutation ambiguity -- the field lives on a labelled voxel grid, not a point cloud), so the
exhaustive minimum is computed directly.

Each assignment stream becomes an empirical categorical distribution over the ``N`` generated
samples, scored by its Jensen-Shannon divergence against Uniform (natural log, no invalid sink,
matching :func:`bifurcation.metrics.continuous.reference_report`'s ``angle_jsd`` convention):
0 means the model explores that axis exactly as the true dynamics would, ``ln(2)`` means total
collapse onto one category.

For Allen-Cahn, ``sample.rollouts`` (built by ``datasets.allencahn.AllenCahnPklDataset``) already
IS the task's "8 GT fields": every simulated realization plus its exact sign-flipped solution is
stored as its own branch (the equation is odd in its scalar field). So mode identification
(:func:`mode_coverage_jsd`) already resolves the ``phi -> -phi`` inversion as part of the 8-way
branch match; the octahedral search below stays at 48 elements (:func:`octahedral_group`) rather
than doubling to 96, or the inversion would be double-counted across Metrics 1 and 3.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import torch

from bifurcation.datasets.allencahn import _infer_grid_shape

_FLIPS: list[tuple[bool, bool, bool]] = list(itertools.product([False, True], repeat=3))  # 8
_PERMS: list[tuple[int, int, int]] = list(itertools.permutations(range(3)))               # 6


def octahedral_group() -> list[tuple[tuple[bool, bool, bool], tuple[int, int, int]]]:
    """The 48-element cubic point group of a 3D grid, as ``(flip, perm)`` pairs: every
    combination of the ``2**3`` axis reflections and ``3!`` axis permutations. Order is fixed
    and used as the category labelling for :func:`rotation_jsd`."""
    return [(flip, perm) for flip in _FLIPS for perm in _PERMS]


_GROUP = octahedral_group()
_GROUP_INDEX = {g: i for i, g in enumerate(_GROUP)}


def apply_group_element(vol: torch.Tensor, flip: tuple[bool, bool, bool],
                        perm: tuple[int, int, int]) -> torch.Tensor:
    """Mirror ``vol [..., X, Y, Z]`` along ``flip`` and permute its spatial axes."""
    dims = [vol.ndim - 3 + i for i, f in enumerate(flip) if f]
    out = torch.flip(vol, dims=dims) if dims else vol
    lead = out.ndim - 3
    order = list(range(lead)) + [lead + p for p in perm]
    return out.permute(*order).contiguous()


@lru_cache(maxsize=8)
def _radial_bin_index(grid_shape: tuple[int, int, int]) -> tuple[torch.Tensor, int]:
    """``([X,Y,Z] int bin id, n_bins)`` for radially averaging a spectrum on ``grid_shape`` --
    cached since every sample of one evaluation run shares the same fixed grid."""
    freqs = [torch.fft.fftfreq(n) * n for n in grid_shape]  # integer cycle counts per axis
    mesh = torch.meshgrid(*freqs, indexing="ij")
    radius = torch.sqrt(sum(m**2 for m in mesh))
    bin_idx = radius.round().long()
    return bin_idx, int(bin_idx.max()) + 1


def radial_spectrum(field_txyz: torch.Tensor) -> torch.Tensor:
    """``[n_bins]`` radially-averaged 3D FFT power spectrum of ``field_txyz`` (``[T,X,Y,Z]`` or
    ``[X,Y,Z]``): the per-timestep magnitude spectrum (``fftn`` over the trailing 3 dims,
    batched -- one call, not a T-fold Python loop), averaged over any leading (time) dims, then
    binned by integer radius. Invariant to translation (a shift only changes each frame's FFT
    phase, never its magnitude) and to every octahedral element (an isometry of the frequency
    grid, so it permutes bins onto themselves) -- exactly the descriptor Step 0 needs for
    rotation/translation-invariant mode identification.
    """
    grid_shape = tuple(field_txyz.shape[-3:])
    bin_idx, n_bins = _radial_bin_index(grid_shape)
    bin_idx = bin_idx.to(field_txyz.device)
    mag = torch.fft.fftn(field_txyz, dim=(-3, -2, -1)).abs()
    if mag.ndim > 3:
        mag = mag.mean(dim=tuple(range(mag.ndim - 3)))
    flat_idx, flat_mag = bin_idx.reshape(-1), mag.reshape(-1)
    sums = torch.zeros(n_bins, dtype=flat_mag.dtype, device=flat_mag.device)
    sums.scatter_add_(0, flat_idx, flat_mag)
    counts = torch.zeros(n_bins, dtype=flat_mag.dtype, device=flat_mag.device)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_mag))
    return sums / counts.clamp_min(1)


def _unravel(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = []
    for s in reversed(shape):
        idx.append(flat_idx % s)
        flat_idx //= s
    return tuple(reversed(idx))


def phase_correlate(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8,
                    ) -> tuple[tuple[int, int, int], float]:
    """FFT phase correlation between two ``[X,Y,Z]`` volumes: the shift ``(dx,dy,dz)`` such that
    ``torch.roll(a, shifts=shift, dims=(-3,-2,-1))`` best matches ``b``, plus the normalized
    correlation peak height (higher = better alignment; used as the residual proxy the
    octahedral search in :func:`_align` minimizes over).
    """
    Fa = torch.fft.fftn(a, dim=(-3, -2, -1))
    Fb = torch.fft.fftn(b, dim=(-3, -2, -1))
    cross = Fb * Fa.conj()
    corr = torch.fft.ifftn(cross / cross.abs().clamp_min(eps), dim=(-3, -2, -1)).real
    flat = corr.reshape(-1)
    flat_idx = int(flat.argmax())
    return _unravel(flat_idx, tuple(corr.shape[-3:])), float(flat[flat_idx])


@dataclass
class MatchResult:
    k: int                              # nearest GT branch (mode + sign already resolved)
    translation: tuple[int, int, int]   # (dx, dy, dz) aligning the sample to that branch
    flip: tuple[bool, bool, bool]       # octahedral element aligning the sample, after translation
    perm: tuple[int, int, int]


def _grid_shape(sample) -> tuple[int, int, int]:
    shape = _infer_grid_shape(sample.p)
    if len(shape) != 3:
        raise ValueError(f"expected a 3D Allen-Cahn grid, got {shape}")
    return shape


def _as_grid(Y: torch.Tensor, grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """``[T_,N_,1]`` field -> ``[T_,X,Y,Z]``, matching ``datasets.allencahn.grid_positions``'s
    row-major flattening."""
    return torch.nan_to_num(Y)[..., 0].reshape(Y.shape[0], *grid_shape)


def _align(mean: torch.Tensor, gt_mean: torch.Tensor) -> tuple[tuple[int, int, int],
                                                                tuple[bool, bool, bool],
                                                                tuple[int, int, int]]:
    """Exhaustive search over the 48-element octahedral group (Eq. 15-16 of arXiv:2306.15030,
    exact rather than sequential-approximate since the group here is small and finite): for each
    element, apply it to ``mean``, phase-correlate the translation, and keep the element with the
    highest correlation peak. Runs on the TIME-AVERAGED field (one ``[X,Y,Z]`` volume), not every
    frame -- translation/orientation are one rigid choice per rollout (as the augmentations in
    ``datasets.allencahn`` apply them), so this is exact, not an approximation, and cuts the
    search by the ``T_`` frame count the spec's efficiency note flags as the place to economize.
    """
    best_peak, best = None, None
    for flip, perm in _GROUP:
        candidate = apply_group_element(mean, flip, perm)
        shift, peak = phase_correlate(candidate, gt_mean)
        if best_peak is None or peak > best_peak:
            best_peak, best = peak, (shift, flip, perm)
    return best


def _sign_feature(field: torch.Tensor) -> torch.Tensor:
    """Mean cube of ``field [...,T,X,Y,Z]`` over its trailing ``T,X,Y,Z`` dims (any leading batch
    dim, e.g. ``K_``, is kept) -- odd under ``phi -> -phi``
    (``mean((-x)**3) = -mean(x**3)``) and, as a global average, invariant to translation and to
    every octahedral element. The radial power spectrum (:func:`radial_spectrum`) discards phase
    and so CANNOT tell a branch from its exact sign-flipped twin (``|fft(-x)| = |fft(x)|``); this
    is the cheap extra invariant :func:`match_one` uses to break that tie, matching the odd
    reaction term (``u**3 - mu*u``) that makes the sign flip an exact symmetry in the first place.
    """
    return field.pow(3).mean(dim=(-4, -3, -2, -1))


def mode_similarity_scores(Y: torch.Tensor, gt_grids: torch.Tensor, gt_spectra: torch.Tensor,
                           gt_sign: torch.Tensor) -> torch.Tensor:
    """``[K_]`` the score Metric 1's argmax match (:func:`match_one`) picks from: cosine
    similarity of ``Y``'s radial power spectrum to each GT branch's, with any branch whose sign
    (:func:`_sign_feature`) disagrees with ``Y``'s masked to ``-inf``. Exposed on its own so a
    diagnostic plot can show every branch's score, not just the winner.
    """
    grid_shape = tuple(gt_grids.shape[-3:])
    grid = _as_grid(Y, grid_shape)
    spec = radial_spectrum(grid)
    sim = torch.nn.functional.cosine_similarity(spec[None], gt_spectra, dim=-1)  # [K_]
    mismatch = torch.sign(_sign_feature(grid)) * torch.sign(gt_sign) < 0
    return sim.masked_fill(mismatch, -torch.inf)


def match_one(Y: torch.Tensor, gt_grids: torch.Tensor, gt_means: torch.Tensor,
             gt_spectra: torch.Tensor, gt_sign: torch.Tensor) -> MatchResult:
    """Step 0 for one generated rollout ``Y [T_,N_,1]`` against a sample's precomputed GT
    tensors (see :func:`match_all`, which builds them once and shares them over every trial)."""
    grid_shape = tuple(gt_grids.shape[-3:])
    grid = _as_grid(Y, grid_shape)  # [T_,X,Y,Z]

    sim = mode_similarity_scores(Y, gt_grids, gt_spectra, gt_sign)
    k = int(sim.argmax())

    shift, flip, perm = _align(grid.mean(dim=0), gt_means[k])
    return MatchResult(k=k, translation=shift, flip=flip, perm=perm)


def precompute_gt(sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(gt_grids [K_,T_,X,Y,Z], gt_means [K_,X,Y,Z], gt_spectra [K_,n_bins], gt_sign [K_])``:
    the per-sample tensors :func:`match_one`/:func:`mode_similarity_scores` need, built once and
    shared over every trial of ``sample`` -- by :func:`match_all`, and by any diagnostic (e.g. a
    trial-vs-every-branch similarity plot) that wants the same scores without redoing Step 0."""
    grid_shape = _grid_shape(sample)
    gt_grids = torch.stack([_as_grid(sample.Y(k), grid_shape) for k in range(sample.K_)])  # [K_,T_,X,Y,Z]
    gt_means = gt_grids.mean(dim=1)                                                         # [K_,X,Y,Z]
    gt_spectra = torch.stack([radial_spectrum(g) for g in gt_grids])                        # [K_,n_bins]
    gt_sign = _sign_feature(gt_grids)                                                        # [K_]
    return gt_grids, gt_means, gt_spectra, gt_sign


def match_all(sample, Y_gen: list[torch.Tensor]) -> list[MatchResult]:
    """Step 0 over every trial of ``sample``, sharing one :func:`precompute_gt` call."""
    gt_grids, gt_means, gt_spectra, gt_sign = precompute_gt(sample)
    return [match_one(Y, gt_grids, gt_means, gt_spectra, gt_sign) for Y in Y_gen]


def _jsd_uniform_counts(counts: torch.Tensor, n_categories: int) -> float:
    """Jensen-Shannon divergence (natural log, in ``[0, ln(2)]``) between a categorical count
    vector and Uniform(n_categories) -- no invalid sink, matching
    :func:`bifurcation.metrics.continuous.reference_report`'s ``angle_jsd``; every trial here is
    always assigned some category, so there is no rejection mass to track separately."""
    total = float(counts.sum())
    if total == 0:
        return float("nan")
    p = counts.double() / total
    q = torch.full((n_categories,), 1.0 / n_categories, dtype=torch.float64)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * torch.log(a[nz] / b[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _histogram_report(labels: list[int], n_categories: int, **extra) -> dict:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=n_categories).double() \
        if labels else torch.zeros(n_categories, dtype=torch.float64)
    probs = counts / max(float(counts.sum()), 1.0)
    return {"jsd": _jsd_uniform_counts(counts, n_categories), "counts": counts.tolist(),
            "probs": probs.tolist(), "n": len(labels), **extra}


def mode_coverage_jsd(sample, Y_gen: list[torch.Tensor] | None = None,
                      matches: list[MatchResult] | None = None) -> dict:
    """Metric 1: JSD of the empirical GT-branch assignment (rotation/translation-invariant
    radial-spectrum match) against Uniform(``sample.K_``) -- does the model cover every discrete
    solution branch evenly, sign-inversion branches included (see module docstring)? Pass either
    ``Y_gen`` (trials to match fresh) or an already-computed ``matches`` (shared with
    :func:`translation_jsd`/:func:`rotation_jsd` to avoid re-running Step 0)."""
    matches = matches if matches is not None else match_all(sample, Y_gen)
    return _histogram_report([m.k for m in matches], sample.K_)


def translation_jsd(sample, Y_gen: list[torch.Tensor] | None = None, k: int | None = None,
                    bins_per_axis: int = 8, matches: list[MatchResult] | None = None) -> dict:
    """Metric 2: JSD of the empirical translation-bin assignment against Uniform, over the
    trials matched to GT branch ``k`` (``None``: pool every branch -- translation uniformity is
    a property of the periodic grid, not of which branch a trial landed on, so pooling adds
    statistical power; pass an explicit ``k`` to condition on one branch as in the spec).
    ``bins_per_axis`` coarsens each of the 64 per-axis voxel shifts into that many bins (default
    8, i.e. 8x8x8=512 joint bins) -- the raw 64^3 voxel-resolution joint histogram is far sparser
    than any realistic trial count, which would read as spurious bias under a perfect sampler.
    """
    matches = matches if matches is not None else match_all(sample, Y_gen)
    subset = matches if k is None else [m for m in matches if m.k == k]
    grid_shape = _grid_shape(sample)
    bin_sizes = [max(1, n // bins_per_axis) for n in grid_shape]
    labels = []
    for m in subset:
        idx = 0
        for shift, bs in zip(m.translation, bin_sizes):
            idx = idx * bins_per_axis + min(shift // bs, bins_per_axis - 1)
        labels.append(idx)
    return _histogram_report(labels, bins_per_axis**3, k=k, bins_per_axis=bins_per_axis)


def rotation_jsd(sample, Y_gen: list[torch.Tensor] | None = None, k: int | None = None,
                 matches: list[MatchResult] | None = None) -> dict:
    """Metric 3: JSD of the empirical octahedral-element assignment against Uniform(48), over the
    trials matched to GT branch ``k`` (``None``: pool every branch, see :func:`translation_jsd`).
    """
    matches = matches if matches is not None else match_all(sample, Y_gen)
    subset = matches if k is None else [m for m in matches if m.k == k]
    labels = [_GROUP_INDEX[(m.flip, m.perm)] for m in subset]
    return _histogram_report(labels, len(_GROUP), k=k, group_size=len(_GROUP))


def symmetry_report(sample, Y_gen: list[torch.Tensor], bins_per_axis: int = 8,
                    k: int | None = None) -> dict[str, dict]:
    """All 3 metrics of ``sample``'s trials, running Step 0 exactly once and sharing it: the
    natural entry point for a caller that wants every JSD together (see
    ``scripts/eval_allencahn_symmetry.py``)."""
    matches = match_all(sample, Y_gen)
    return {"mode_coverage": mode_coverage_jsd(sample, matches=matches),
            "translation": translation_jsd(sample, k=k, bins_per_axis=bins_per_axis, matches=matches),
            "rotation": rotation_jsd(sample, k=k, matches=matches)}
