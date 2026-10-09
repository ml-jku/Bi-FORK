"""Evaluate a second-stage checkpoint with particle-guided Azula sampling."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from bifurcation.callbacks.generation import aggregate, sample_row, summary_text
from bifurcation.datasets.views import collate
from bifurcation.models.transport.azula import PGFlexGaussianDenoiser


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = str(ROOT / "configs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("microstructures", "beam3d", "allencahn"), required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--env", default="local")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--w-score-hat", type=float, default=1.0)
    parser.add_argument("--w-score-repulsive", type=float)
    parser.add_argument("--schedule-power", type=float, default=1.0)
    parser.add_argument("--schedule-min-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("particle_guidance_results"))
    args = parser.parse_args()
    if args.w_score_repulsive is None:
        args.w_score_repulsive = {
            "microstructures": 0.01,
            "beam3d": 1.0,
            "allencahn": 0.05,
        }[args.dataset]
    return args


def find_training_config(checkpoint: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    for parent in checkpoint.resolve().parents:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def disable_compilation(node) -> None:
    if not isinstance(node, DictConfig):
        return
    with open_dict(node):
        if "compile" in node:
            node.compile = False
        if "compile_dynamic" in node:
            node.compile_dynamic = False
    for value in node.values():
        disable_compilation(value)


def build(args: argparse.Namespace):
    GlobalHydra.instance().clear()
    overrides = [
        f"experiment={args.dataset}/eval",
        f"env={args.env}",
        f"ckpt_path={args.ckpt.resolve()}",
        f"data.test_split={args.split}",
        "callbacks.generate_test.keep_outputs=false",
    ]
    if args.data_dir is not None:
        overrides.append(f"paths.data_dir={args.data_dir.resolve()}")
    if args.dataset == "allencahn":
        overrides.append("protocol.vary_context_rollout=false")
    with hydra.initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        cfg = hydra.compose(config_name="eval", overrides=overrides)

    training_config = find_training_config(args.ckpt, args.model_config)
    if training_config is not None:
        saved = OmegaConf.load(training_config)
        with open_dict(cfg):
            cfg.model = saved.model
    disable_compilation(cfg.model)
    with open_dict(cfg.model):
        cfg.model.first_stage_ckpt = None
        if "init_ckpt" in cfg.model:
            cfg.model.init_ckpt = None
        if "freeze_except" in cfg.model:
            cfg.model.freeze_except = None

    datamodule = hydra.utils.instantiate(cfg.data)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)["state_dict"]
    state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    with open_dict(cfg.model):
        if "backbone.bifurcation_head.qkv.weight" in state:
            cfg.model.backbone.bifurcation_attention = True
        elif "backbone.bifurcation_head.weight" in state:
            cfg.model.backbone.bifurcation_attention = False
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(state)
    model = model.to(args.device).eval()

    datamodule.setup(args.split)
    view = datamodule.views[args.split]
    kwargs = {"report_dir": None, "keep_outputs": False}
    if args.n_samples is not None:
        kwargs["n_samples"] = args.n_samples
    if args.n_trials is not None:
        kwargs["n_trials"] = args.n_trials
    if args.num_steps is not None:
        kwargs["num_steps"] = args.num_steps
    generator = hydra.utils.instantiate(cfg.callbacks.generate_test, **kwargs)
    return model, view, generator


def bifurcation_frames(model, view, sample_index: int, device: str) -> torch.Tensor:
    dataset = view.dataset
    valid = dataset.train_mask(sample_index)[0][:: view.time_stride].to(device)
    if not model._predicts_bifurcation():
        return valid.float()

    rollout = dataset.representatives(sample_index)[0]
    sample = dataset[sample_index]
    item = sample.rollout(rollout) | {
        "index": torch.tensor([sample_index, rollout]),
        "valid_mask": dataset.train_mask(sample_index)[rollout],
    }
    if view.time_stride > 1:
        item = {
            key: value[:: view.time_stride] if key in {"Y", "X", "U", "H", "valid_mask", "bifurcation"} else value
            for key, value in item.items()
        }
    batch = {
        key: value.to(device)
        for key, value in collate([view.normalizer.normalize(item)]).items()
    }
    z0 = model.mask_invalid_latents(model.encode(batch), batch)
    t = torch.full((z0.shape[0],), 1.0 - model.transport.train_eps, device=z0.device)
    noisy = model.transport.interpolate(z0, torch.randn_like(z0), t)
    x_cond, cond_mask = model.conditioning(z0)
    condition = model.conditioner(batch) if model.conditioner is not None else None
    _, logits = model.predict(
        noisy, x_cond, cond_mask, t, condition, batch["valid_mask"].bool(),
        return_bifurcation_logits=True,
    )
    return ((logits[0].sigmoid() >= 0.5) & batch["valid_mask"][0].bool()).float()


def write_results(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = aggregate(rows)
    with (out_dir / "report.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(report.items())
    (out_dir / "summary.txt").write_text(summary_text("particle_guidance", rows))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model, view, generator = build(args)
    generator.view, generator.dataset = view, view.dataset
    indices = generator._pick_samples()
    rows = []
    for index in indices:
        frames = bifurcation_frames(model, view, index, args.device)
        if not frames.any():
            raise RuntimeError(f"sample {index} has no bifurcation frames")
        model.azula_denoiser_factory = partial(
            PGFlexGaussianDenoiser,
            w_score_hat=args.w_score_hat,
            w_score_repulsive=args.w_score_repulsive,
            bifurcation_frames=frames,
            schedule_power=args.schedule_power,
            schedule_min_weight=args.schedule_min_weight,
        )
        generated = generator.generate_for(model, view, [index], desc="particle_guidance")
        rows.append(sample_row(generated[0], generator.min_fraction))
    if not rows:
        raise RuntimeError("the selected split contains no scoreable samples")
    write_results(args.out, rows)
    print(summary_text("particle_guidance", rows))


if __name__ == "__main__":
    main()
