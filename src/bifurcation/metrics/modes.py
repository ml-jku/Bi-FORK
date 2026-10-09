"""Mode metrics over sets of model outputs (generic, all datasets).

Inputs are integer mode labels from the dataset glue in ``metrics/<dataset>.py``. Negative
labels never count toward a mode, and the two rejection categories are distinguished:

  - ``REJECTED`` (-1) -- farther than the validity threshold from every mode.
  - ``BLURRED`` (-2) -- closer to the mean of the modes than to any actual mode.

The core quantities for the bifurcation story:
  - :func:`coverage`    -- did the outputs hit every ground-truth mode?
  - :func:`frequencies` -- how are they distributed over the modes?
  - :func:`jsd_uniform` -- how far is that distribution from the ideal uniform one?
  - :func:`rejected` / :func:`blurred` -- how much output is invalid, and why?
"""

from __future__ import annotations

import torch

REJECTED = -1  # farther than the validity threshold from every mode
BLURRED = -2   # closer to the mean of the modes than to any mode (averaged, non-physical)


def _labels(labels) -> torch.Tensor:
    """A label list (or tensor) as an int64 tensor -- empty included."""
    return torch.as_tensor(labels, dtype=torch.long).reshape(-1)


def _counts(labels: torch.Tensor, n_modes: int) -> torch.Tensor:
    """``[n_modes]`` float64 count of outputs per mode; negatives (invalid) never counted."""
    return torch.bincount(labels[labels >= 0], minlength=n_modes).double()


def frequencies(labels, n_modes: int) -> torch.Tensor:
    """``[n_modes]`` fraction of valid outputs per mode (zeros if nothing classified)."""
    counts = _counts(_labels(labels), n_modes)
    return counts / max(float(counts.sum()), 1.0)


def label_histogram(labels, n_modes: int) -> torch.Tensor:
    """``[n_modes + 2]`` fraction of ALL outputs per bin: modes ``0..n_modes-1``, then
    ``REJECTED``, then ``BLURRED``. Unlike ``frequencies`` (valid-only), the invalid mass
    stays visible, so the bins sum to 1."""
    labels = _labels(labels)
    extra = torch.tensor([(labels == REJECTED).sum(), (labels == BLURRED).sum()]).double()
    return torch.cat([_counts(labels, n_modes), extra]) / max(labels.numel(), 1)


def label_histograms(labels_by_sample: list[list[int]], n_modes_by_sample: list[int],
                     ) -> dict[int, torch.Tensor]:
    """Label distribution for the mode-histogram plot: one histogram per mode count, its
    the labels of the samples concatenated. Same-``k`` label ids share only positional semantics
    (id 0 = first saved rollout), so this is the visual companion to ``mode_jsd``, not a
    per-sample score."""
    return {k: label_histogram([label for labels, kk in zip(labels_by_sample, n_modes_by_sample)
                                if kk == k for label in labels], k)
            for k in sorted(set(n_modes_by_sample))}


def coverage(labels, n_modes: int, min_fraction: float = 0.0) -> float:
    """Fraction of the ``n_modes`` ground-truth modes the outputs produced.

    A mode counts when at least one output hit it AND its share of ALL outputs reaches
    ``min_fraction`` -- a mode found once in a hundred trials is not a mode the model
    reliably produces. ``0.0`` (the default) asks only for the single hit."""
    labels = _labels(labels)
    counts = _counts(labels, n_modes)
    hit = (counts > 0) & (counts >= min_fraction * max(labels.numel(), 1))
    return float(hit.sum()) / n_modes


def rejected(labels) -> float:
    """Fraction of outputs that classified to no mode (any negative label)."""
    labels = _labels(labels)
    return float((labels < 0).double().mean()) if labels.numel() else 0.0


def blurred(labels) -> float:
    """Fraction of outputs rejected as mode-averaged (label ``BLURRED``)."""
    labels = _labels(labels)
    return float((labels == BLURRED).double().mean()) if labels.numel() else 0.0


def mode_report(labels_by_sample: list[list[int]], n_modes_by_sample: list[int],
                min_fraction: float = 0.0) -> dict[str, float]:
    """Aggregate per-sample mode labels into one metric dict.

    Every score is computed per sample over ITS trials, then averaged over the samples --
    overall AND resolved by the mode count of the sample (``coverage_k2``, ...). Single-mode
    samples stay in the report; their coverage is 0 when every trial was rejected.
    ``min_fraction`` is :func:`coverage`'s minimum share per mode.
    """
    per_sample = [
        {"coverage": coverage(labels, k, min_fraction), "mode_jsd": jsd_uniform(labels, k),
         "rejected": rejected(labels), "blurred": blurred(labels)}
        for labels, k in zip(labels_by_sample, n_modes_by_sample)]
    report = {name: _mean([s[name] for s in per_sample]) for name in per_sample[0]}
    for k in sorted(set(n_modes_by_sample)):
        of_k = [s for s, kk in zip(per_sample, n_modes_by_sample) if kk == k]
        report |= {f"{name}_k{k}": _mean([s[name] for s in of_k])
                   for name in ("coverage", "mode_jsd")}
    return report


def _mean(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).mean())




def jsd_uniform(labels, n_modes: int) -> float:
    """Jensen-Shannon divergence (natural log, in ``[0, ln(2)]``) against the ideal label
    distribution.

    0 = every output valid and the modes hit uniformly; ln(2) = worst. The distribution is over
    the ``n_modes`` PLUS an invalid sink, and the ideal is uniform over the modes with zero
    invalid mass, which makes the score MONOTONE in quality. Why a valid-only JSD instead rewards
    rejecting everything: docs/decisions.md. Using the natural logarithm keeps this metric on the
    same scale as ``angle_jsd`` and the Allen-Cahn symmetry JSD metrics.
    """
    labels = _labels(labels)
    counts = _counts(labels, n_modes)
    invalid = float(labels.numel()) - float(counts.sum())
    p = torch.cat([counts, torch.tensor([invalid], dtype=torch.float64)]) / max(labels.numel(), 1)
    q = torch.cat([torch.full((n_modes,), 1.0 / n_modes, dtype=torch.float64),
                   torch.zeros(1, dtype=torch.float64)])  # ideal: uniform over modes, none invalid
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * torch.log(a[nz] / b[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
