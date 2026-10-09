"""Interpolation path + prediction target for latent flow matching -- pluggable via config.

Convention (shared with the whole second stage): ``z0`` = data, ``z1`` = noise, and flow time
``t`` runs 0 (noise) -> 1 (data). The interpolant is ``zt = alpha_t z0 + sigma_t z1`` with

- ``linear`` (rectified flow):  ``alpha_t = t``,          ``sigma_t = 1 - t``
- ``gvp`` (old_stage2.json):    ``alpha_t = sin(pi t/2)``, ``sigma_t = cos(pi t/2)``

The network regresses either the ``velocity`` field ``d/dt zt = alpha_t' z0 + sigma_t' z1`` or the  lang-ok
``data`` ``z0`` directly. ``linear`` + ``velocity`` reproduces the original rectified-flow second
stage exactly (t ~ U(0,1), target ``z0 - z1``); ``gvp`` + ``data`` reproduces the old GVP run
(t ~ U(eps, 1-eps), target ``z0``, DDIM sampling). Data prediction clips the training/sampling
time away from the endpoints (``train_eps``) to avoid the ``sigma_t -> 0`` division.

Follows https://github.com/willisma/SiT (`transport/path.py`), whose interpolant paths
these are.
No code taken.
"""

from __future__ import annotations

import math

import torch


class Transport:
    def __init__(self, path: str = "linear", prediction: str = "velocity",
                 train_eps: float | None = None):
        if path not in ("linear", "gvp"):
            raise ValueError(f"path must be 'linear' or 'gvp', got {path!r}")
        if prediction not in ("velocity", "data"):
            raise ValueError(f"prediction must be 'velocity' or 'data', got {prediction!r}")
        self.path = path
        self.prediction = prediction
        self.train_eps = (0.0 if prediction == "velocity" else 1e-3) if train_eps is None else train_eps


    def _coeffs(self, t: torch.Tensor, ndim: int):
        """``alpha, dalpha, sigma, dsigma`` (data/noise coefficients and their time derivatives),
        broadcast to ``ndim`` dims from a per-sample time ``t [B_]``."""
        tb = t.reshape(-1, *([1] * (ndim - 1)))
        if self.path == "linear":
            one = torch.ones_like(tb)
            return tb, one, 1 - tb, -one
        h = math.pi / 2
        return torch.sin(h * tb), h * torch.cos(h * tb), torch.cos(h * tb), -h * torch.sin(h * tb)

    def interpolate(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha, _, sigma, _ = self._coeffs(t, z0.ndim)
        return alpha * z0 + sigma * z1

    def target(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What the network regresses at ``zt``: the data ``z0`` or the velocity field.  lang-ok"""
        if self.prediction == "data":
            return z0
        _, dalpha, _, dsigma = self._coeffs(t, z0.ndim)
        return dalpha * z0 + dsigma * z1

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        lo, hi = self.train_eps, 1.0 - self.train_eps
        return torch.rand(batch_size, device=device) * (hi - lo) + lo


    def predict_x0(self, zt: torch.Tensor, pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Data estimate ``x0_hat`` from the raw network output, for either prediction mode."""
        if self.prediction == "data":
            return pred
        alpha, dalpha, sigma, dsigma = self._coeffs(t, zt.ndim)
        det = alpha * dsigma - sigma * dalpha  # -1 for linear; -pi/2 for gvp
        return (dsigma * zt - sigma * pred) / det

    def ddim_step(self, zt: torch.Tensor, pred: torch.Tensor,
                  t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM (eta=0) update from time ``t0`` toward data at ``t1`` (> t0):
        recover the implied noise at ``t0``, then re-noise at ``t1``."""
        x0 = self.predict_x0(zt, pred, t0)
        alpha0, _, sigma0, _ = self._coeffs(t0, zt.ndim)
        alpha1, _, sigma1, _ = self._coeffs(t1, zt.ndim)
        eps = (zt - alpha0 * x0) / sigma0
        return alpha1 * x0 + sigma1 * eps
