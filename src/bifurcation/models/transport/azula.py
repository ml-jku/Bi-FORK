"""Azula sampling backend for the second stage -- the reference `old_stage2.json` path.

The native `Transport.ddim_step` reproduces this math for eta=0; this wires the actual `azula`
library, so the reference run is reproducible with its own sampler. Imported lazily. azula runs
t=1 (noise) -> t=0 (clean) and the SiT backbone the other way, so the denoiser calls it at `1 - t`.

Follows https://github.com/probabilists/azula. azula is a dependency here, not a copy.
"""

from __future__ import annotations

import math
import einops
import torch
from azula.denoise import Denoiser, GaussianPosterior
from azula.noise import Schedule
from torch import Tensor


class GVPAzulaSchedule(Schedule):
    """GVP schedule in azula time (t=0 clean, t=1 noisy). Mirrors the `gvp` path of the transport under
    the t -> 1-t flip: alpha(t)=cos(t pi/2) (data weight), sigma(t)=sin(t pi/2) (noise weight)."""

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


class LinearAzulaSchedule(Schedule):
    """Linear schedule in Azula time: clean at ``t=0``, noise at ``t=1``."""

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - t, t


class BackboneDenoiser(Denoiser):
    """Wraps the SiT backbone of the second stage as an azula denoiser returning ``x0_hat``.

    The reference GVP run uses ``data`` prediction. Other standard diffusion parameterizations
    can be converted to data estimates for compatible, separately trained backbones. Path-velocity
    models still sample with the native Euler/midpoint path.
    Conditioning is threaded through as sampler keyword arguments.
    """

    def __init__(self, backbone, schedule: Schedule, prediction: str = "data"):
        super().__init__()
        if prediction not in {"data", "noise", "v", "score"}:
            raise ValueError(f"Invalid prediction: {prediction!r}")
        self.backbone = backbone
        self.schedule = schedule
        self.prediction = prediction

    def _raw_to_x_hat(
        self, x_t: Tensor, raw: Tensor, alpha_t: Tensor, sigma_t: Tensor
    ) -> Tensor:
        match self.prediction:
            case "data":
                x0_hat = raw
            case "noise":
                eps_hat = raw
                x0_hat = (x_t - sigma_t * eps_hat) / alpha_t
            case "v":
                v_hat = raw
                den = alpha_t**2 + sigma_t**2
                x0_hat = (alpha_t * x_t - sigma_t * v_hat) / den
            case "score":
                score_hat = raw
                x0_hat = (x_t + sigma_t**2 * score_hat) / alpha_t
            case _:
                raise ValueError(f"Invalid prediction: {self.prediction}")
        return x0_hat

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, *, x_cond, cond_mask, y=None,
                frame_mask=None, spatial_coords=None, **_) -> GaussianPosterior:
        alpha_t, sigma_t = self.schedule(t)
        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]
        t_model = (1 - t).reshape(1).expand(x_t.shape[0])  # azula time -> backbone (data) time
        ctx = {} if frame_mask is None else {"frame_mask": frame_mask}
        if spatial_coords is not None:
            ctx["spatial_coords"] = spatial_coords
        raw = self.backbone(x_t, x_cond, cond_mask, t_model, y, **ctx)  # data prediction: raw = x0_hat
        x0_hat = self._raw_to_x_hat(x_t=x_t, raw=raw, alpha_t=alpha_t, sigma_t=sigma_t)
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)
        return GaussianPosterior(mean=x0_hat, var=c_var)


# Implementation taken from: https://github.com/gcorso/particle-guidance
class PGFlexGaussianDenoiser(BackboneDenoiser):
    """Particle-guided sampling adapter for a data-prediction backbone.

    This changes sampling only: the checkpoint remains trained against the ordinary data target.
    The leading batch dimension must contain the alternative trials to repel from one another.
    """

    def __init__(
        self, backbone, schedule: Schedule, prediction: str = "data",
        w_score_hat: float = 0.5,
        w_score_repulsive: float = 0.5,
        bifurcation_frames: None | Tensor = None,  # per-frame weight in [0, 1] (e.g. bifurcation prob), not a hard mask
        schedule_power: float = 1.0,
        schedule_min_weight: float = 0.0,
        reverse: bool = False,
    ):
        if prediction != "data":
            raise ValueError("particle guidance requires a data-prediction checkpoint")
        if bifurcation_frames is None:
            raise ValueError("bifurcation_frames must be positive")
        super().__init__(backbone, schedule, prediction=prediction)
        self.w_score_hat = w_score_hat
        self.w_score_repulsive = w_score_repulsive
        self.bifurcation_frames = bifurcation_frames
        self.schedule_power = schedule_power
        self.schedule_min_weight = schedule_min_weight
        self.reverse = reverse


    def _raw_to_x_hat(
        self, x_t: Tensor, raw: Tensor, alpha_t: Tensor, sigma_t: Tensor
    ) -> Tensor:
        x0_hat = raw
        score_hat = -(x_t - alpha_t * x0_hat) / (sigma_t**2)
        score_repulsive = self._calculate_repulsive_score(x0_hat, sigma_t)
        score = self.w_score_hat * score_hat + score_repulsive
        alpha_safe = alpha_t.clamp_min(torch.finfo(x0_hat.dtype).eps)
        return (x_t + sigma_t**2 * score) / alpha_safe

    def _calculate_repulsive_score(self, x0_hat: Tensor, sigma_t: Tensor) -> Tensor:
        # Adapted from: https://github.com/gcorso/particle-guidance

        B, T, n_latents, h_dim = x0_hat.shape
        repulsive_score = torch.zeros_like(x0_hat)
        eps = torch.finfo(x0_hat.dtype).eps

        if not torch.any(self.bifurcation_frames > 0):
            raise ValueError("No bifurcation frames found in the current batch; cannot compute repulsive score.")
        frame_idx = torch.where(self.bifurcation_frames > 0)[0]
        frame_weight = self.bifurcation_frames[frame_idx].to(x0_hat.dtype)
        n_steps = int(frame_idx.numel())

        x = einops.rearrange(x0_hat[:, frame_idx], "B t n_latents h_dim -> B (t n_latents h_dim)")

        # Adapted from https://github.com/gcorso/particle-guidance/blob/main/synthetic.ipynb
        # Same as the reference's x[:, None] - x[None, :], without the [B, B, D] tensor: at 360
        # trials that is ~80 GB per copy for beam3d. Exact distances, O(B^2 + B*D) memory.
        dist = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")  # (B, B)
        d2_matrix = dist.square() / math.sqrt(x.shape[-1])

        med2 = d2_matrix.median()
        h = (med2 / max(math.log(B), 1.0)).clamp_min(eps)

        k = torch.exp(-d2_matrix / h)
        k_sum = k.sum(dim=1, keepdim=True).clamp_min(eps)

        # sum_j w_ij (x_i - x_j) = x_i * sum_j w_ij - (w @ x)_i. Pairs closer than eps (the
        # diagonal, duplicate trials) add ~nothing there but 1/eps here, which would swamp the sum.
        w = torch.where(dist > eps, k / dist.clamp_min(eps), 0.0)
        force_x0 = (x * w.sum(dim=1, keepdim=True) - w @ x) / k_sum

        sigma_t_max = 1.0
        sigma_t_scalar = sigma_t.detach().reshape(-1)[0]
        if self.reverse:
            progress = (sigma_t_scalar / sigma_t_max).clamp(0.0, 1.0)
        else: 
            progress = (1.0 - (sigma_t_scalar / sigma_t_max).clamp(0.0, 1.0))
        schedule_weight = progress.clamp_min(0.0) ** self.schedule_power
        schedule_weight = schedule_weight.clamp_min(self.schedule_min_weight)

        sigma_t_sq_safe = (sigma_t**2).clamp_min(eps)
        score_repulsive_val = force_x0 / sigma_t_sq_safe.reshape(-1)[0]

        repulsive_score[:, frame_idx] = (
            self.w_score_repulsive * schedule_weight * frame_weight[None, :, None, None] * einops.rearrange(
                score_repulsive_val,
                "B (t n_latents h_dim) -> B t n_latents h_dim",
                n_latents=n_latents,
                t=n_steps,
            )
        )

        return repulsive_score
