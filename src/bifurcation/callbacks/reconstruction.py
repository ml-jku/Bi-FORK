"""First-stage reconstruction evaluation on fixed, FULL-resolution validation samples.

Both callbacks reconstruct the same deterministic selection (bifurcating samples preferred,
or an explicit ``samples`` list of h5 keys/indices; one representative rollout each,
``n_frames`` evenly spaced trainable frames incl. the last) and work on DENORMALIZED
fields (lang-ok). Frames are decoded one by one -- full-res clouds are large.

- :class:`ReconstructionMetrics`: ``(pred, target, mask) -> scalar`` metrics from the config,
  logged as ``val/{name}`` over all evaluated frames and ``val/lf_{name}`` at last frames.
- :class:`ReconstructionPlot`: a ``plotter(p, x, y_true, y_pred, meta) -> Figure`` from the config,
  written to TensorBoard and ``<log_dir>/plots/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import lightning as L
import torch

from bifurcation.datasets.views import collate
from bifurcation.viz.utils import log_figure


def _mean(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).mean())


class _ReconstructionCallback(L.Callback):
    def __init__(self, n_samples: int, n_frames: int, every_n_epochs: int, seed: int,
                 samples: list | None = None, prefix: str | None = None, split: str = "val"):
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.samples = samples  # explicit h5 keys/indices; None = automatic pick
        self.prefix = prefix    # metric-name prefix override (e.g. "val_sub")
        self.split = split
        self._selections: dict[str, list] = {}

    def _tag(self, split: str) -> str:
        """Metric-name prefix: ``prefix`` on the configured split -- a second instance can
        score a curated subset into its own wandb group (val_sub/...) alongside the plain
        val/ one -- else the running split."""
        return self.prefix if self.prefix is not None and split == self.split else split

    def _selection(self, trainer, split: str) -> list:
        """The curated ``samples`` when given (in order), else a seeded pick with
        bifurcating samples first; one representative rollout and ``n_frames`` each.
        Curated keys are split-specific, so any other split this callback runs on (test)
        falls back to the automatic pick."""
        if split not in self._selections:
            view = trainer.datamodule.views[split]
            self.normalizer = view.normalizer
            dataset = view.dataset
            if self.samples is not None and split == self.split:
                order = dataset.indices_of(list(self.samples))
            else:
                rng = torch.Generator().manual_seed(self.seed)
                shuffled = torch.randperm(len(dataset), generator=rng).tolist()
                order = [i for i in shuffled if dataset.bifurcates(i)]
                order += [i for i in torch.randperm(len(dataset), generator=rng).tolist()
                          if i not in order]
                order = order[: self.n_samples]
            selection = []
            for i in order:
                k = dataset.representatives(int(i))[-1]  # a non-trivial mode where present
                valid = torch.nonzero(dataset.train_mask(int(i))[k]).flatten()
                picks = torch.linspace(len(valid) - 1, 0, self.n_frames).round().long()
                frames = valid[picks.unique()]
                selection.append((dataset, int(i), int(k), frames.tolist()))
            self._selections[split] = selection
        return self._selections[split]

    @torch.no_grad()
    def _reconstruct(self, pl_module, dataset, i: int, k: int, frames: list[int]) -> dict:
        """Denormalized GT and prediction at full resolution: ``y_* [F_, N_, C_]``."""
        y_true, y_pred, items = [], [], []
        for t in frames:  # one frame per forward: full-res clouds are big
            item = dataset.frame(i, k, t)
            items.append(item)
            batch = {key: v.to(pl_module.device) for key, v in
                     collate([self.normalizer.normalize(item)]).items()}
            pred = pl_module.backbone(batch)["y"][0].cpu()
            y_pred.append(self.normalizer.denorm("y", pred))
            y_true.append(item["y"])
        meta = {"key": dataset.keys[i], "k": k, "frames": frames, "f": items[0].get("f")}
        x = torch.stack([it["x"] for it in items]) if "x" in items[0] else None
        return {"p": items[0]["p"], "x": x,
                "y_true": torch.stack(y_true), "y_pred": torch.stack(y_pred), "meta": meta}

    def _due(self, trainer) -> bool:
        return (not trainer.sanity_checking
                and (trainer.current_epoch + 1) % self.every_n_epochs == 0)


class ReconstructionMetrics(_ReconstructionCallback):
    def __init__(self, n_samples: int, n_frames: int, every_n_epochs: int, seed: int,
                 metrics: dict[str, Callable], samples: list | None = None,
                 prefix: str | None = None):
        super().__init__(n_samples, n_frames, every_n_epochs, seed, samples, prefix)
        self.metrics = metrics

    def _run(self, trainer, pl_module, split: str):
        scores = {name: [] for name in self.metrics}
        last = {name: [] for name in self.metrics}
        for dataset, i, k, frames in self._selection(trainer, split):
            rec = self._reconstruct(pl_module, dataset, i, k, frames)
            pred, target = rec["y_pred"], rec["y_true"]
            mask = torch.ones(pred.shape[0], pred.shape[1], dtype=torch.bool)
            for name, fn in self.metrics.items():
                scores[name].append(float(fn(pred, target, mask)))
                last[name].append(float(fn(pred[-1:], target[-1:], mask[-1:])))
        tag = self._tag(split)
        pl_module.log_dict(
            {f"{tag}/{n}": _mean(s) for n, s in scores.items()}
            | {f"{tag}/lf_{n}": _mean(s) for n, s in last.items()},
            on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._due(trainer):
            self._run(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._run(trainer, pl_module, "test")  # test runs once -- no cadence gate


class ReconstructionPlot(_ReconstructionCallback):
    @property
    def writes_files(self) -> int:
        return self.n_samples

    def __init__(self, n_samples: int, n_frames: int, every_n_epochs: int, seed: int,
                 plotter: Callable, tag: str, split: str, samples: list | None = None,
                 prefix: str | None = None):
        super().__init__(n_samples, n_frames, every_n_epochs, seed, samples, prefix, split)
        self.plotter = plotter
        self.tag = tag

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._due(trainer):
            return
        import matplotlib.pyplot as plt

        out_dir = Path(trainer.log_dir or ".") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self._tag(self.split)
        for j, (dataset, i, k, frames) in enumerate(self._selection(trainer, self.split)):
            rec = self._reconstruct(pl_module, dataset, i, k, frames)
            fig = self.plotter(rec["p"], rec["x"], rec["y_true"], rec["y_pred"], rec["meta"])
            fig.savefig(out_dir / f"{name}_{self.tag}_{rec['meta']['key']}_e{trainer.current_epoch:04d}.png",
                        dpi=120)
            log_figure(trainer, f"{name}/{self.tag}/{j}", fig)
            plt.close(fig)
