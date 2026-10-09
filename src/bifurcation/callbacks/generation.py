"""Generation callbacks: :class:`Generate` samples once per split per epoch (the expensive
part), every other callback is a pure reader of that result -- :class:`ModeCoverage` the
numbers (coverage/JSD/rejection + label histograms), :class:`ModeGallery`/:class:`ModePlot`/
:class:`Animate` the figures. WHAT is measured comes from ``configs/protocol/<dataset>.yaml``,
HOW the model draws from ``sampling``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import lightning as L
import torch

from bifurcation.datasets.utils.modes import DiscreteModes
from bifurcation.datasets.views import collate
from bifurcation.metrics.generic import masked_medae
from bifurcation.metrics.modes import BLURRED, REJECTED, label_histograms, mode_report
from bifurcation.utils.math import nanmean, nanmedian
from bifurcation.viz.gallery import mode_gallery
from bifurcation.viz.utils import log_figure, log_gif, plot_mode_histogram, save_gif


@dataclass
class Generated:
    """What the model produced for ONE sample, and everything a reader needs about it."""

    key: str                             # the h5 key of the sample (the CSV row id)
    sample: object | None                # the ground-truth sample; None with keep_outputs=false
    Y_gen: list[torch.Tensor]            # generated field rollouts (lang-ok); EMPTY with keep_outputs=false
    rollout_labels: list[int]            # rollout-level mode per output; negative = rejected
    frame_labels: list[list[int]]        # per output: mode at every scored frame; negative = rejected
    n_modes: int                         # modes this sample has (or the fixed bin count)
    n_frame_modes: int                   # the bin count of the frame histogram (max modes over frames)
    matched: list[int]                   # per output: the saved rollout it landed on
    mcon_mse: list[float]
    mcon_mae: list[float]
    mcon_medae: list[float]
    affine_mse: list[float]
    affine_mae: list[float]
    affine_medae: list[float]
    reference: dict[str, float] = field(default_factory=dict)


MODE_METRICS = ("coverage", "mode_jsd", "rejected", "blurred")
STORED_ERRORS = ("mcon_mse", "mcon_mae", "mcon_medae",
                 "affine_mse", "affine_mae", "affine_medae")
ERRORS = STORED_ERRORS + ("advantage_mse", "advantage_mae", "advantage_medae")
METRICS = MODE_METRICS + ERRORS


def sample_row(r: Generated, min_fraction: float = 0.0) -> dict:
    """The numbers of one sample as a flat row -- what the streamed CSV writes.

    nanmean over the trials, so a NaN trial is skipped rather than averaged in. What makes a
    trial NaN, and what each kind is counted as, is in docs/decisions.md. The raw per-trial
    errors ride along as space-joined ``*_trials`` columns, so the trial-level medians rebuild
    from the CSV alone.
    """
    from bifurcation.metrics.modes import blurred, coverage, jsd_uniform, rejected

    row = {"key": r.key, "n_modes": r.n_modes,
           "coverage": coverage(r.rollout_labels, r.n_modes, min_fraction),
           "mode_jsd": jsd_uniform(r.rollout_labels, r.n_modes),
           "rejected": rejected(r.rollout_labels), "blurred": blurred(r.rollout_labels)}
    row |= {name: nanmean(getattr(r, name)) for name in STORED_ERRORS}
    for error in ("mse", "mae", "medae"):  # error saved over predicting nothing
        row[f"advantage_{error}"] = row[f"affine_{error}"] - row[f"mcon_{error}"]
    if "mode_jsd" in r.reference:
        row["mode_jsd_invalid"] = row["mode_jsd"]
    row |= r.reference  # the published reference metrics, plain columns like the rest
    return row | {"labels": " ".join(str(label) for label in r.rollout_labels)} \
               | {f"{name}_trials": " ".join(f"{v:.6g}" for v in getattr(r, name))
                  for name in STORED_ERRORS}


def reference_names(rows: list[dict]) -> list[str]:
    """The columns ``protocol.reference_report`` contributed to these rows (empty when the
    protocol defines none). Discovered from the rows rather than threaded through configs:
    the dataset chooses the keys of the report."""
    fixed = {"key", "n_modes", "labels", *METRICS}
    return [k for k in rows[0] if k not in fixed and not k.endswith("_trials")]


def trial_values(rows: list[dict], name: str) -> list[float]:
    """The per-trial values of one error, flat across ``rows`` -- the pool the trial-level medians
    reduce. Reads the ``*_trials`` columns, so it works on streamed rows and on rows read
    back from the CSV alike. Advantage has no saved column: it is the per-trial difference
    of its affine and mcon columns (NaN for the same trials, so the pairing survives)."""
    def parse(row: dict, name: str) -> list[float]:
        return [float(x) for x in row[f"{name}_trials"].split()]

    if name.startswith("advantage_"):
        kind = name.split("_", 1)[1]
        return [a - m for row in rows
                for a, m in zip(parse(row, f"affine_{kind}"), parse(row, f"mcon_{kind}"))]
    return [v for row in rows for v in parse(row, name)]


def aggregate(rows: list[dict]) -> dict[str, float]:
    """Every metric as a mean and a median, overall and per mode count (``mode_jsd_k2``, ...).

    The MEAN reduces over SAMPLES, of the per-sample mean over trials, so every sample weighs
    the same however many of its trials were rejected. The error MEDIANS reduce over ALL trials
    of the subset, so ``mcon_mse_median`` is the median trajectory. The mode metrics have no
    per-trial value, so their median stays over samples.
    """
    def stats(name: str, subset: list[dict], suffix: str = "") -> dict[str, float]:
        values = [row[name] for row in subset]
        pool = trial_values(subset, name) if name in ERRORS else values
        return {f"{name}{suffix}": nanmean(values),
                f"{name}{suffix}_median": nanmedian(pool)}

    names = list(METRICS) + reference_names(rows)
    report: dict[str, float] = {}
    for name in names:
        report |= stats(name, rows)
    for k in sorted({row["n_modes"] for row in rows}):
        of_k = [row for row in rows if row["n_modes"] == k]
        for name in names:
            report |= stats(name, of_k, f"_k{k}")
    return report


def summary_text(tag: str, rows: list[dict]) -> str:
    """The rows as a human-readable table: the mode metrics, every error with its mean and
    median (the median over all trials, like :func:`aggregate`), and the same resolved by
    mode count. Reads only ``rows``, so it renders just as well for the samples done so far
    as for the whole split."""
    W, LABEL = 16, 22  # widest header is "advantage_medae"; widest label "modes by mode count"

    def cell(x) -> str:
        """One right-aligned column. ``.4g`` so 0.0013 and 2.3e+06 both fit the width."""
        return ("--" if isinstance(x, float) and x != x else
                f"{x:.4g}" if isinstance(x, float) else str(x)).rjust(W)

    def line(label: str, *values) -> str:
        return label.ljust(LABEL) + "".join(cell(v) for v in values)

    def of(k: int) -> list[dict]:
        return [row for row in rows if row["n_modes"] == k]

    def col(name: str, reduce=nanmean, subset=None):
        return reduce([row[name] for row in (rows if subset is None else subset)])

    def median(name: str, subset=None):
        subset = rows if subset is None else subset
        return nanmedian(trial_values(subset, name) if name in ERRORS
                         else [row[name] for row in subset])

    n_modes = sorted({row["n_modes"] for row in rows})
    rule = "-" * (LABEL + W * 7)  # the widest table: "samples" + the 6 columns of a 3-error group
    lines = [f"{tag}: {len(rows)} samples, mode counts {n_modes}", "=" * len(rule), "",
             line("mode metrics", "mean"), rule]
    lines += [line(name, col(name)) for name in ("coverage", "mode_jsd", "rejected", "blurred")]

    lines += ["", line("error", "mean", "median"), rule]
    lines += [line(name, col(name), median(name)) for name in ERRORS]

    if reference_names(rows):  # the published reference metrics, when the protocol has them
        lines += ["", line("reference metric", "mean", "median"), rule]
        lines += [line(name, col(name), median(name)) for name in reference_names(rows)]

    for group in (("coverage", "mode_jsd"), ("rejected", "blurred"),
                  ("mcon_mse", "mcon_mae", "mcon_medae"),
                  ("affine_mse", "affine_mae", "affine_medae"),
                  ("advantage_mse", "advantage_mae", "advantage_medae")):
        head = [x for name in group for x in (name, "median")]
        lines += ["", line("by mode count", "samples", *head), rule]
        for k in n_modes:
            values = [x for name in group for x in (col(name, subset=of(k)),
                                                    median(name, of(k)))]
            lines.append(line(f"k={k}", len(of(k)), *values))

    worst = sorted(rows, key=lambda r: (r["advantage_mse"] != r["advantage_mse"],
                                        r["advantage_mse"]))[:5]
    lines += ["", "worst 5 samples by advantage_mse (affine minus model; negative = the "
                  "model is worse than predicting no fluctuation)", rule,
              line("key", "n_modes", "mcon_mse", "affine_mse", "advantage")]
    lines += [line(row["key"], row["n_modes"], row["mcon_mse"], row["affine_mse"],
                   row["advantage_mse"]) for row in worst]
    return "\n".join(lines) + "\n"


class _Stopwatch:
    """Cumulative wall clock per phase of the generation loop, shown on the progress bar --
    the phase to blame for a slow eval should be readable off the bar, not profiled after."""

    def __init__(self):
        self.seconds: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, phase: str) -> None:
        """Charge everything since the previous lap to ``phase``."""
        now = time.perf_counter()
        self.seconds[phase] = self.seconds.get(phase, 0.0) + now - self._last
        self._last = now

    def lap_split(self, split: dict[str, float], rest: str) -> None:
        """A lap the callee already broke down (it timed its own stages): charge its numbers
        as-is and whatever of the lap they do not explain to ``rest``."""
        now = time.perf_counter()
        for phase, s in split.items():
            self.seconds[phase] = self.seconds.get(phase, 0.0) + s
        remainder = now - self._last - sum(split.values())
        if remainder > 0:
            self.seconds[rest] = self.seconds.get(rest, 0.0) + remainder
        self._last = now

    def text(self) -> str:
        """``gen 63%/121s label 20%/39s ...`` -- every phase, biggest share first."""
        total = max(sum(self.seconds.values()), 1e-9)
        return " ".join(f"{name} {100 * s / total:.0f}%/{s:.1f}s"
                        for name, s in sorted(self.seconds.items(), key=lambda kv: -kv[1]))


def producer(trainer, split: str) -> "Generate | None":
    """The :class:`Generate` callback for ``split``, or ``None`` if there is not one.
    """
    for cb in trainer.callbacks:
        if isinstance(cb, Generate) and cb.split == split:
            return cb
    return None


class Generate(L.Callback):
    """Draws the trials the other callbacks read. The only place generation happens.

    One instance per split, all with the same settings: the val/train instances fire at
    validation-epoch end on their ``every_n_epochs`` cadence, the test instance once at test
    time. Samples are picked once and never change: evenly spaced over the scoreable
    samples. An explicit ``samples`` list (h5 keys or indices) overrides the pick.
    """

    def __init__(self, n_samples: int | None, n_trials: int, num_steps: int, method: str,
                 every_n_epochs: int, split: str,
                 max_abs_distance: float | None, max_rel_distance: float | None,
                 n_modes: int | None,
                 label: Callable, representative_rollouts: Callable,
                 reference_report: Callable | None = None,
                 relax: Callable | None = None,
                 residual: Callable | None = None,
                 distance_metric: str = "mse",
                 vary_context_rollout: bool = False,
                 first_scored_frame: int = 0,
                 samples: list | None = None, samples_per_batch: int = 1,
                 keep_outputs: bool = True, min_fraction: float = 0.0,
                 evaluate_on_rejected: bool = False, evaluate_on_blurred: bool = False,
                 autocast: bool = False, report_dir: str | None = None):
        self.n_samples = n_samples  # evaluated samples; None: the whole split
        self.n_trials = n_trials    # generated rollouts per sample
        self.num_steps = num_steps
        self.method = method
        self.every_n_epochs = every_n_epochs
        self.split = split
        self.max_abs_distance = max_abs_distance
        self.max_rel_distance = max_rel_distance
        self.n_modes = n_modes  # None: the own mode count of the sample; int: fixed bins (continuous modes)
        self.label = label      # label(sample, Y, cutoffs...) -> (label, matched k, matched ref)
        self.representative_rollouts = representative_rollouts
        self.reference_report = reference_report
        self.relax = relax
        self.residual = residual
        self.distance_metric = distance_metric
        self.vary_context_rollout = vary_context_rollout
        self.first_scored_frame = first_scored_frame
        self.samples = samples  # explicit pick (h5 keys or indices); None: automatic
        self.samples_per_batch = samples_per_batch
        self.keep_outputs = keep_outputs
        self.min_fraction = min_fraction  # coverage  minimum share -- see metrics.coverage
        self.evaluate_on_rejected = evaluate_on_rejected
        self.evaluate_on_blurred = evaluate_on_blurred
        self.autocast = autocast
        self.report_dir = report_dir
        self._cache: dict = {}  # epoch -> list[Generated]


    def due(self, trainer) -> bool:
        return (not trainer.sanity_checking
                and (trainer.current_epoch + 1) % self.every_n_epochs == 0)

    def results(self, trainer, module) -> list[Generated] | None:
        """The trials of this epoch, generated on first ask; ``None`` on non-generating epochs
        (the cue for the readers to skip). The test instance is not epoch-gated -- it runs once."""
        if self.split != "test" and not self.due(trainer):
            return None
        e = trainer.current_epoch
        if e not in self._cache:
            self._cache = {e: self._generate_all(trainer, module)}  # nothing accumulates
        return self._cache[e]


    def _pick_samples(self) -> list[int]:
        if self.samples is not None:
            return self.dataset.indices_of(list(self.samples))
        pool = [i for i in range(len(self.dataset)) if self.dataset.train_mask(i).any()]
        if len(pool) < len(self.dataset):
            print(f"[gen] skipped {len(self.dataset) - len(pool)} sample(s) "
                  f"with no scoreable frame", flush=True)
        n = len(pool) if self.n_samples is None else min(self.n_samples, len(pool))
        return [pool[j] for j in torch.linspace(0, len(pool) - 1, n).long()]

    def _at_view_time(self, item: dict) -> dict:
        """A rollout item on the time axis of the view, so generated and saved rollouts line up."""
        s = self.view.time_stride
        return {k: v[::s] if s > 1 and k in ("Y", "X", "U", "H", "valid_mask") else v
                for k, v in item.items()}

    def _sample_at_view_time(self, sample):
        """Return ground truth on the same time axis as generated fields.

        Generation has always applied ``view.time_stride`` to model inputs, but the metric
        path previously retained the full-resolution Sample.  That made any stride greater
        than one fail when discrete-mode metrics compared generated and saved rollouts.
        """
        s = self.view.time_stride
        if s <= 1:
            return sample
        rollouts = sample.rollouts[:, ::s]
        valid_mask = None if sample.valid_mask is None else sample.valid_mask[:, ::s]
        modes = sample.modes
        if isinstance(modes, DiscreteModes):
            mode_valid = None if modes.valid_mask is None else modes.valid_mask[:, ::s]
            modes = DiscreteModes(
                rollouts=rollouts,
                rollout_labels=modes.rollout_labels,
                frame_labels=modes.frame_labels[::s],
                valid_mask=mode_valid,
            )
        return replace(
            sample,
            rollouts=rollouts,
            modes=modes,
            valid_mask=valid_mask,
            U=None if sample.U is None else sample.U[::s],
            H=None if sample.H is None else sample.H[::s],
            bifurcation=(None if sample.bifurcation is None else sample.bifurcation[::s]),
        )

    @staticmethod
    def _sample_at_last_frame(sample):
        """Reduce a discrete rollout sample to its last frame for final-state evaluation."""
        rollouts = sample.rollouts[:, -1:]
        valid_mask = None if sample.valid_mask is None else sample.valid_mask[:, -1:]
        modes = sample.modes
        if isinstance(modes, DiscreteModes):
            mode_valid = None if modes.valid_mask is None else modes.valid_mask[:, -1:]
            modes = DiscreteModes(
                rollouts=rollouts,
                rollout_labels=modes.rollout_labels,
                frame_labels=modes.frame_labels[-1:],
                valid_mask=mode_valid,
            )
        return replace(
            sample,
            rollouts=rollouts,
            modes=modes,
            valid_mask=valid_mask,
            U=None if sample.U is None else sample.U[-1:],
            H=None if sample.H is None else sample.H[-1:],
            bifurcation=(None if sample.bifurcation is None else sample.bifurcation[-1:]),
        )

    def _full_res_batch(self, samples: list, device, rollout: int = 0,
                        sample_indices: list[int] | None = None) -> dict:
        """The conditions of the samples as ONE model batch at FULL node resolution. The view may
        subsample nodes for prototyping (``num_nodes``), but the metrics always compare full
        clouds, so generation runs on them regardless. ``rollout``: which saved rollout supplies
        the conditioning frames -- 0 (the default query) unless a caller asks otherwise."""
        rows = []
        for b, s in enumerate(samples):
            item = s.rollout(rollout) | {"index": torch.tensor([b, 0])}
            if sample_indices is not None:
                item["valid_mask"] = self.dataset.train_mask(sample_indices[b])[rollout]
            rows.append(self.view.normalizer.normalize(self._at_view_time(item)))
        return {k: v.to(device) for k, v in collate(rows).items()}

    def _generate_varying_context(self, module, batch_samples: list, group: list[int],
                                  n_trials: int) -> torch.Tensor:
        """Like one batched ``module.generate_rollouts(batch, n_trials, ...)`` call, but trial
        ``t`` conditions on branch ``t % sample.K_`` of each sample instead of every trial
        sharing branch 0 -- see ``self.vary_context_rollout``. One sampler call per trial
        (batched across ``group``, same as the shared-context path would be per trial), so this
        costs the same total sampler work, just organized as ``n_trials`` smaller calls."""
        per_trial = []
        for t in range(n_trials):
            rows = [self._full_res_batch([sample], module.device, rollout=t % sample.K_,
                                         sample_indices=[idx])
                    for idx, sample in zip(group, batch_samples)]
            batch = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            fields = module.generate_rollouts(batch, 1, self.num_steps, self.method)
            per_trial.append(fields[0])  # [B_, T_, N_, C_]
        return torch.stack(per_trial)  # [n_trials, B_, T_, N_, C_]

    def _relax_dt(self) -> float:
        """Per-frame timestep at the active view's time resolution, for ``self.relax``.

        Only meaningful (and only called) for datasets whose metadata carries a solver
        ``dt``/``save_every``, i.e. Allen-Cahn -- other datasets never set ``protocol.relax``.
        """
        meta = self.dataset.metadata
        return meta["dt"] * meta["save_every"] * self.view.time_stride

    def _distance_metric_kwargs(self) -> dict:
        """``{"distance_metric": ...}`` when the protocol overrides it, else empty -- so
        ``self.label``/``self.reference_report`` are called with their old signature
        unless a protocol (currently only Allen-Cahn) actually opts into a non-default
        nearest-rollout distance."""
        return {} if self.distance_metric == "mse" else {"distance_metric": self.distance_metric}

    def _scored_frames(self, i: int) -> list[int]:
        """Physically valid frames at or after the protocol's ``first_scored_frame``."""
        if getattr(self, "final_frame_only", False):
            return [0]
        mask = self.dataset.train_mask(i).any(0)[::self.view.time_stride].clone()
        mask[: self.first_scored_frame] = False
        return torch.nonzero(mask).flatten().tolist()

    def _frame_labels(self, sample, Y: torch.Tensor, frames: list[int]) -> list[int]:
        """Frame-level mode of ONE output at every scored frame."""
        modes = sample.modes
        if isinstance(modes, DiscreteModes):
            return [int(modes.classify_frame(Y[t], t, self.max_abs_distance,
                                             self.max_rel_distance)) for t in frames]
        width = 2.0 * math.pi / self.n_modes
        return [int(modes.classify_frame(Y[t], t) / width) % self.n_modes for t in frames]

    def _n_frame_modes(self, sample, frames: list[int]) -> int:
        """Bin count of the frame histogram: the most frame-level modes over the scored frames
        (discrete), or the shared bin count (continuous)."""
        if isinstance(sample.modes, DiscreteModes):
            return max([sample.modes.n_modes_at(t) for t in frames], default=1)
        return self.n_modes

    def _scores(self, label: int) -> bool:
        """Whether a trial with this label is error-scored: valid always, REJECTED/BLURRED
        only when the matching ``evaluate_on_*`` flag opts them in."""
        return (label >= 0 or (label == REJECTED and self.evaluate_on_rejected)
                or (label == BLURRED and self.evaluate_on_blurred))

    def _score(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
        """``(mse, mae, medae)`` of one prediction against one reference. MSE sums over
        coordinates and averages over frames x nodes, MAE averages over all three -- the
        published scales; MedAE is :func:`masked_medae` over the same elements. NaN
        components where nothing is finite."""
        d = pred - target
        all_nodes = torch.ones(target.shape[:2], dtype=torch.bool)  # already cut to real nodes
        stats = ((d ** 2).sum(-1).mean(), d.abs().mean(),
                 masked_medae(pred, target, all_nodes))
        return tuple(float(s) if s.isfinite() else float("nan") for s in stats)

    def _errors(self, Y: torch.Tensor, ref: torch.Tensor,
                frames: list[int]) -> tuple[float, float, float]:
        """``(mse, mae, medae)`` of ONE output against the mode it landed on, over the
        scored frames. NaN when nothing is scoreable."""
        if not frames:
            return (float("nan"),) * 3
        return self._score(Y[frames], ref.cpu()[frames])

    def _affine_errors(self, sample, frames: list[int]) -> tuple[float, float, float]:
        """The affine baseline, mode-conditional like the error of the model: predicting zero
        fluctuation is scored against ITS nearest mode -- the min displacement^2 over the
        modes of the sample -- not against whichever mode a trial happened to land on. One triple
        per sample; the mode that wins the MSE also supplies MAE and MedAE, so all three
        describe the same prediction. NaN when nothing is scoreable."""
        if not frames:
            return (float("nan"),) * 3
        triples = [self._score(torch.zeros_like(t), t)
                   for ref in self.representative_rollouts(sample)
                   for t in [torch.nan_to_num(ref.cpu()[frames])]]
        return min(triples, key=lambda e: (e[0] != e[0], e[0]),  # NaNs lose to any number
                   default=(float("nan"),) * 3)

    @torch.no_grad()
    def _generate_all(self, trainer, module) -> list[Generated]:
        view = trainer.datamodule.views[self.split]
        self.view, self.dataset = view, view.dataset
        samples = self._pick_samples()
        return self.generate_for(module, view, samples, desc=self.split)

    @torch.no_grad()
    def generate_for(self, module, view, samples: list[int], desc: str = "",
                     n_trials: int | None = None) -> list[Generated]:
        """Generate ``n_trials`` rollouts per sample and label what came out. No Lightning in
        it: a notebook can call this on a loaded checkpoint and get exactly what the training
        callbacks see."""
        from tqdm import tqdm

        self.view, self.dataset = view, view.dataset
        n_trials = n_trials or self.n_trials
        bar = tqdm(total=len(samples), desc=f"{desc}/generate ({n_trials} trials, "
                                            f"{self.num_steps} steps)", unit="sample",
                   disable=len(samples) < 8)

        out: list[Generated] = []
        stream = self._open_stream(desc)
        watch = _Stopwatch()  # cumulative seconds per phase, shown on the bar and log line
        group_n = max(1, self.samples_per_batch)
        for g0 in range(0, len(samples), group_n):
            group = [int(i) for i in samples[g0:g0 + group_n]]
            batch_samples = [self.dataset[i] for i in group]
            watch.lap("data")
            with torch.autocast(module.device.type, dtype=torch.bfloat16,
                                enabled=self.autocast):
                if self.vary_context_rollout:
                    fields = self._generate_varying_context(module, batch_samples, group,
                                                            n_trials)
                else:
                    batch = self._full_res_batch(batch_samples, module.device, sample_indices=group)
                    fields = module.generate_rollouts(batch, n_trials, self.num_steps, self.method)
            if fields.is_cuda:  # else the async kernels bill their time to the first .cpu()
                torch.cuda.synchronize()
            split = getattr(module, "generate_seconds", None)
            if split:  # the latent model reports its sample/decode breakdown
                watch.lap_split(split, rest="gen")
            else:
                watch.lap("gen")

            for b, (i, sample) in enumerate(zip(group, batch_samples)):
                sample = self._sample_at_view_time(sample)
                if getattr(self, "final_frame_only", False):
                    sample = self._sample_at_last_frame(sample)
                N = sample.p.shape[0]  # collate zero-pads different node counts; cut back
                generated_fields = (fields[:, b, -1:, :N] if
                                    getattr(self, "final_frame_only", False) else
                                    fields[:, b, :, :N])
                Y_gen = [view.normalizer.denorm("y", generated_fields[s].float().cpu())
                         for s in range(n_trials)]
                watch.lap("denorm")
                dt = self._relax_dt() if (self.relax is not None or self.residual is not None) else None
                residual_before = ([self.residual(sample, Y, dt=dt) for Y in Y_gen]
                                   if self.residual is not None else None)
                Y_gen_relaxed = ([self.relax(sample, Y, dt=dt) for Y in Y_gen]
                                 if self.relax is not None else Y_gen)
                if self.relax is not None:
                    watch.lap("relax")
                residual_after = ([self.residual(sample, Y, dt=dt) for Y in Y_gen_relaxed]
                                  if self.residual is not None else None)
                if self.residual is not None:
                    watch.lap("residual")

                labeled = [self.label(sample, Y, max_abs_distance=self.max_abs_distance,
                                      max_rel_distance=self.max_rel_distance,
                                      **self._distance_metric_kwargs()) for Y in Y_gen]
                watch.lap("label")
                frames = self._scored_frames(i)
                errors = [self._errors(Y, ref, frames) if self._scores(label)
                          else (float("nan"),) * 3
                          for Y, (label, _, ref) in zip(Y_gen, labeled)]
                affine = self._affine_errors(sample, frames)
                paired = [tuple(a if e == e else float("nan") for a, e in zip(affine, err))
                          for err in errors]
                report_kwargs = ({"Y_gen_relaxed": Y_gen_relaxed} if self.relax is not None
                                 else {})
                reference = (self.reference_report(sample, Y_gen, **report_kwargs)
                             if self.reference_report is not None else {})
                if self.residual is not None:
                    reference = reference | {"relax_residual_before": nanmean(residual_before),
                                             "relax_residual_after": nanmean(residual_after)}
                watch.lap("score")
                frame_labels = [self._frame_labels(sample, Y, frames) for Y in Y_gen]
                watch.lap("frames")
                out.append(Generated(
                    key=str(sample.key),
                    sample=sample if self.keep_outputs else None,
                    Y_gen=Y_gen if self.keep_outputs else [],
                    rollout_labels=[int(label) for label, _, _ in labeled],
                    frame_labels=frame_labels,
                    n_modes=self.n_modes if self.n_modes is not None else sample.modes.n_modes,
                    n_frame_modes=self._n_frame_modes(sample, frames),
                    matched=[int(k) for _, k, _ in labeled],
                    mcon_mse=[e[0] for e in errors], mcon_mae=[e[1] for e in errors],
                    mcon_medae=[e[2] for e in errors],
                    affine_mse=[a[0] for a in paired], affine_mae=[a[1] for a in paired],
                    affine_medae=[a[2] for a in paired],
                    reference=reference))

                if stream is not None:  # on disk before the next sample runs
                    stream(sample_row(out[-1], self.min_fraction))
                running = mode_report([g.rollout_labels for g in out],
                                      [g.n_modes for g in out])
                nearest_jsd = [g.reference.get("mode_jsd") for g in out
                               if "mode_jsd" in g.reference]
                if nearest_jsd:
                    running["mode_jsd_invalid"] = running["mode_jsd"]
                    running["mode_jsd"] = nanmean(nearest_jsd)
                running["mcon_mse"] = nanmean([nanmean(g.mcon_mse) for g in out])
                running["adv_mse"] = nanmean([nanmean(g.affine_mse) - nanmean(g.mcon_mse)
                                              for g in out])
                watch.lap("report")  # streaming CSV + summary rebuild + running projection
                bar.set_postfix({name: f"{running[name]:.3f}"
                                 for name in ("coverage", "mode_jsd", "rejected")}
                                | {name: f"{running[name]:.4f}"
                                   for name in ("mcon_mse", "adv_mse")}
                                | {"t": watch.text()})
                bar.update(1)
                print(f"[gen:{desc}] sample {len(out)}/{len(samples)}"
                      f" (cov {running['coverage']:.2f} jsd {running['mode_jsd']:.2f}"
                      f" rej {running['rejected']:.2f}"
                      f" mcon_mse {running['mcon_mse']:.4f}"
                      f" adv {running['adv_mse']:+.4f})"
                      f" [{watch.text()}]", flush=True)
        bar.close()
        return out

    def _open_stream(self, desc: str):
        """A ``row -> None`` appender that keeps ``<report_dir>/<desc>_samples.csv`` and
        ``<desc>_summary.txt`` current as the eval runs -- the CSV flushed per row, the
        summary rebuilt from the samples done so far. None when no report_dir is set."""
        if self.report_dir is None:
            return None
        import csv

        path = Path(self.report_dir)
        path.mkdir(parents=True, exist_ok=True)
        handle = open(path / f"{desc}_samples.csv", "w", newline="")
        writer = csv.writer(handle)

        rows: list[dict] = []

        def append(row: dict) -> None:
            if not rows:
                writer.writerow(row)      # a dict writes its keys: the header
            writer.writerow(row.values())
            handle.flush()
            rows.append(row)
            (path / f"{desc}_summary.txt").write_text(summary_text(desc, rows))

        print(f"[gen:{desc}] streaming to {path}/{desc}_{{samples.csv,summary.txt}}", flush=True)
        return append


    def on_validation_epoch_end(self, trainer, module):
        if self.split != "test":
            self.results(trainer, module)  # warm the cache so the readers all share it

    def on_test_epoch_end(self, trainer, module):
        if self.split == "test":
            self.results(trainer, module)


class _Reader(L.Callback):
    """Base class for callbacks that consume generated trials instead of producing them.

    Each reader is bound to one split and pulls the trials from :class:`Generate` on the epochs
    the producer generated; it never triggers generation itself. Readers are cheap, so they run
    on every generating epoch; a reader whose output is expensive can set ``every_n_epochs``.
    """

    def __init__(self, split: str, every_n_epochs: int = 1, prefix: str | None = None):
        self.split = split
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix  # metric-name prefix override (e.g. a curated "val_sub" group)

    def _due(self, trainer) -> bool:
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def _run(self, trainer, module):
        gen = producer(trainer, self.split)
        results = gen.results(trainer, module) if gen is not None else None
        if results:
            self.run(trainer, module, self.prefix or self.split, results)

    def on_validation_epoch_end(self, trainer, module):
        if self.split != "test" and self._due(trainer):
            self._run(trainer, module)

    def on_test_epoch_end(self, trainer, module):
        if self.split == "test":
            self._run(trainer, module)

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        raise NotImplementedError


class ModeCoverage(_Reader):
    """The numbers: ``gen_coverage`` (fraction of the modes of a sample produced), ``gen_mode_jsd``
    (JSD over each trials of a sample, averaged over samples), ``gen_rejected`` / ``gen_blurred``
    (trials that are no branch at all, or an average of several), plus the rollout- and
    frame-level label histograms. Microstructures additionally reports the gated value as
    ``gen_mode_jsd_invalid`` and uses ``gen_mode_jsd`` for its nearest-mode, no-invalid-class
    counterpart."""

    def __init__(self, split: str, min_fraction: float = 0.0, every_n_epochs: int = 1,
                 prefix: str | None = None, report_dir: str | None = None):
        super().__init__(split, every_n_epochs=every_n_epochs, prefix=prefix)
        self.min_fraction = min_fraction
        self.report_dir = report_dir  # where the CSVs and summary go; None: the log_dir

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        rollout = [r.rollout_labels for r in results]
        n_modes = [r.n_modes for r in results]
        rows = [sample_row(r, self.min_fraction) for r in results]
        report = aggregate(rows)
        logged = {f"{tag}/gen_{name}": v for name, v in report.items()}
        if "mc_mse" in report:
            logged[f"{tag}/overall_mc_mse"] = report["mc_mse"]
        module.log_dict(logged, on_epoch=True)

        self._write_report(trainer, tag, rows, report)
        self._histogram(trainer, f"{tag}/gen_mode_hist", rollout, n_modes, f"{tag}_mode_hist")
        frame = [[label for trial in r.frame_labels for label in trial] for r in results]
        self._histogram(trainer, f"{tag}/gen_frame_mode_hist", frame,
                        [r.n_frame_modes for r in results], f"{tag}_frame_mode_hist")

    def _out_dir(self, trainer) -> Path:
        return Path(self.report_dir or (Path(trainer.log_dir or ".") / "csv"))

    def _write_report(self, trainer, tag, rows: list[dict], report: dict) -> None:
        """Three files in the report directory: ``<tag>_samples.csv`` (one row per sample),
        ``<tag>_report.csv`` (the aggregates, one metric per row) and ``<tag>_summary.txt``
        (the same aggregates as a table you can read)."""
        import csv

        out_dir = self._out_dir(trainer)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{tag}_samples.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rows[0])
            writer.writerows(row.values() for row in rows)
        with open(out_dir / f"{tag}_report.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(report.items())
        (out_dir / f"{tag}_summary.txt").write_text(summary_text(tag, rows))
        print(f"[eval:{tag}] wrote {out_dir}/{tag}_{{samples.csv,report.csv,summary.txt}}",
              flush=True)


    def _histogram(self, trainer, tag, labels, n_modes, save_as) -> None:
        import matplotlib.pyplot as plt

        fig = plot_mode_histogram(label_histograms(labels, n_modes), title=tag)
        out_dir = Path(trainer.log_dir or ".") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{save_as}_e{trainer.current_epoch:04d}.png", dpi=120)
        log_figure(trainer, tag, fig)
        plt.close(fig)


class SymmetryCoverage(_Reader):
    """Allen-Cahn only: the 3 symmetry/coverage JSD metrics of
    :mod:`bifurcation.metrics.allencahn_symmetry` (mode coverage over the sample's GT branches,
    periodic-translation coverage, octahedral rotation/reflection coverage), each averaged over
    every generated sample of the split -- the ``ModeCoverage`` of Step 0's FFT-based matching.

    Needs ``keep_outputs=true`` on the producing :class:`Generate` (it reads ``sample``/``Y_gen``
    directly, unlike ``ModeCoverage`` which only needs the cheaper ``rollout_labels``) -- raise
    ``n_samples``/``n_trials`` in ``protocol.<dataset>.yaml`` cautiously: every sample's full
    ``Y_gen`` (each trial ``[T_,N_,1]`` at Allen-Cahn's 64^3 resolution) stays in memory until
    every reader of this epoch has run.
    """

    def __init__(self, split: str, bins_per_axis: int = 8, mode: int | None = None,
                every_n_epochs: int = 1, prefix: str | None = None, report_dir: str | None = None):
        super().__init__(split, every_n_epochs=every_n_epochs, prefix=prefix)
        self.bins_per_axis = bins_per_axis
        self.mode = mode  # fix one GT branch for translation/rotation; None pools every branch
        self.report_dir = report_dir  # where the CSV/summary go; None: log_dir/csv

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        from bifurcation.metrics.allencahn_symmetry import symmetry_report

        if not results[0].Y_gen:  # producer ran with keep_outputs=false
            raise ValueError("SymmetryCoverage needs keep_outputs=true on the producing Generate")
        reports = [symmetry_report(r.sample, r.Y_gen, bins_per_axis=self.bins_per_axis, k=self.mode)
                  for r in results]
        names = ("mode_coverage", "translation", "rotation")
        means = {name: nanmean([rep[name]["jsd"] for rep in reports]) for name in names}
        module.log_dict({f"{tag}/gen_{name}_jsd": v for name, v in means.items()}, on_epoch=True)
        self._write_report(trainer, tag, results, reports, means)

    def _write_report(self, trainer, tag, results: list[Generated], reports: list[dict],
                      means: dict[str, float]) -> None:
        import csv

        out_dir = Path(self.report_dir or (Path(trainer.log_dir or ".") / "csv"))
        out_dir.mkdir(parents=True, exist_ok=True)
        names = ("mode_coverage", "translation", "rotation")
        rows = [{"key": r.key, "K_": r.sample.K_,
                **{f"{name}_jsd": rep[name]["jsd"] for name in names},
                **{f"{name}_n": rep[name]["n"] for name in names}}
               for r, rep in zip(results, reports)]
        with open(out_dir / f"{tag}_symmetry_samples.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rows[0])
            writer.writerows(row.values() for row in rows)
        lines = [f"{tag}: {len(rows)} samples", "-" * 60]
        lines += [f"{name:>14}_jsd: mean = {means[name]:.4f}" for name in names]
        (out_dir / f"{tag}_symmetry_summary.txt").write_text("\n".join(lines) + "\n")
        print(f"[eval:{tag}] wrote {out_dir}/{tag}_symmetry_{{samples.csv,summary.txt}}",
             flush=True)


class SymmetryGallery(_Reader):
    writes_files = 1

    """Allen-Cahn only: the picture behind ``SymmetryCoverage``'s ``gen_mode_coverage_jsd`` --
    one trial per sample plotted against every one of its ``K_`` GT branches, headed by Metric
    1's cosine similarity score, the winning match starred
    (:func:`bifurcation.viz.allencahn.plot_trial_vs_modes`). Same relationship to
    ``SymmetryCoverage`` as :class:`ModeGallery` has to ``ModeCoverage`` -- numbers and picture
    are separate readers of the same :class:`Generate` results. Needs ``keep_outputs=true``,
    like ``SymmetryCoverage``.
    """

    def __init__(self, split: str, trial: int = 0, every_n_epochs: int = 1, prefix: str | None = None):
        super().__init__(split, every_n_epochs=every_n_epochs, prefix=prefix)
        self.trial = trial  # which of the sample's trials to plot; 0: stable across epochs

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        import matplotlib.pyplot as plt

        from bifurcation.metrics.allencahn_symmetry import mode_similarity_scores, precompute_gt
        from bifurcation.viz.allencahn import plot_trial_vs_modes

        if not results[0].Y_gen:  # producer ran with keep_outputs=false
            return
        out_dir = Path(trainer.log_dir or ".") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        for j, r in enumerate(results):
            gt_grids, _, gt_spectra, gt_sign = precompute_gt(r.sample)
            trial = min(self.trial, len(r.Y_gen) - 1)
            sim = mode_similarity_scores(r.Y_gen[trial], gt_grids, gt_spectra, gt_sign)
            best_k = int(sim.argmax())
            fig = plot_trial_vs_modes(r.sample, r.Y_gen[trial], sim, best_k, key=r.key)
            fig.savefig(out_dir / f"{tag}_symmetry_{r.key}_e{trainer.current_epoch:04d}.png",
                       dpi=120)
            log_figure(trainer, f"{tag}/symmetry_gallery/{j}", fig)
            plt.close(fig)


class ModeGallery(_Reader):
    writes_files = 1

    """The picture behind ``gen_coverage``: one row per ground-truth mode with the trials that
    landed on it beside it (an empty row = mode collapse), and a final row for the rejected
    trials. ``draw(ax, sample, Y)`` renders one rollout; ``representative_rollouts(sample)``
    supplies the ground truth of every shown mode."""

    def __init__(self, split: str, draw: Callable, representative_rollouts: Callable,
                 layout: str = "pairs", max_per_mode: int = 4, n_cols: int = 3,
                 distance_metric: str = "mse",
                 every_n_epochs: int = 1, prefix: str | None = None):
        super().__init__(split, every_n_epochs=every_n_epochs, prefix=prefix)
        self.draw = draw
        self.representative_rollouts = representative_rollouts
        self.layout = layout                # pairs | by_mode -- see viz.gallery.mode_gallery
        self.max_per_mode = max_per_mode    # by_mode: outputs shown beside each mode
        self.n_cols = n_cols                # pairs: mode pairs per row
        self.distance_metric = distance_metric

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        import matplotlib.pyplot as plt

        if not results[0].Y_gen:  # producer ran with keep_outputs=false
            return
        out_dir = Path(trainer.log_dir or ".") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        for j, r in enumerate(results):
            fig = mode_gallery(r.sample, r.Y_gen, r.rollout_labels,
                               self.representative_rollouts(r.sample),
                               self.draw, layout=self.layout, max_per_mode=self.max_per_mode,
                               n_cols=self.n_cols, distance_metric=self.distance_metric)
            fig.savefig(out_dir / f"{tag}_gallery_{r.key}_e{trainer.current_epoch:04d}.png",
                        dpi=120)
            log_figure(trainer, f"{tag}/gallery/{j}", fig)
            plt.close(fig)


class ModePlot(_Reader):
    writes_files = 1

    """One figure per sample from an injected ``plotter(sample, Y_gen, labels, meta)``.

    This is the picture for datasets whose modes are CONTINUOUS: beam3d has a whole circle of
    buckling directions, so a gallery with one row per mode makes no sense -- what you want is
    all the outputs drawn together, positioned by the direction they chose. Discrete-mode
    datasets use :class:`ModeGallery` instead.
    """

    def __init__(self, split: str, plotter: Callable, tag: str = "mode_paths",
                 max_samples: int | None = None, every_n_epochs: int = 1,
                 prefix: str | None = None):
        super().__init__(split, every_n_epochs=every_n_epochs, prefix=prefix)
        self.plotter = plotter
        self.tag = tag
        self.max_samples = max_samples  # None: one figure per sample; else a random subset

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        import random

        import matplotlib.pyplot as plt

        if not results[0].Y_gen:  # producer ran with keep_outputs=false
            return
        out_dir = Path(trainer.log_dir or ".") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        chosen = results if self.max_samples is None \
            else random.sample(results, min(self.max_samples, len(results)))
        for j, r in enumerate(chosen):
            Y_gen = [torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0) for Y in r.Y_gen]
            fig = self.plotter(r.sample, Y_gen, r.rollout_labels, {"key": r.key})
            fig.savefig(out_dir / f"{tag}_{self.tag}_{r.key}_e{trainer.current_epoch:04d}.png",
                        dpi=120)
            log_figure(trainer, f"{tag}/{self.tag}/{j}", fig)
            plt.close(fig)


class Animate(_Reader):
    """A GIF per sample: the first trial against the branch it landed on."""

    writes_files = 1

    def __init__(self, split: str, animator: Callable, tag: str = "gen",
                 every_n_epochs: int = 10, prefix: str | None = None):
        super().__init__(split, every_n_epochs=every_n_epochs, prefix=prefix)
        self.animator = animator
        self.tag = tag  # distinguishes several animators on one split

    def run(self, trainer, module, tag: str, results: list[Generated]) -> None:
        if not results[0].Y_gen:  # producer ran with keep_outputs=false
            return
        out_dir = Path(trainer.log_dir or ".") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            k = r.matched[0]
            Y_pred = torch.nan_to_num(r.Y_gen[0], nan=0.0, posinf=0.0, neginf=0.0)
            anim = self.animator(r.sample, k=k, Y_pred=Y_pred)
            path = out_dir / f"{tag}_{self.tag}_{r.key}_vs_k{k}_e{trainer.current_epoch:04d}.gif"
            save_gif(anim, path)
            log_gif(trainer, f"{tag}/{self.tag}_{r.key}", path)
