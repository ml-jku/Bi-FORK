"""Allen--Cahn uses the repository's shared discrete-mode evaluation protocol."""

from __future__ import annotations

import math

import torch

from bifurcation.metrics.discrete import (  # noqa: F401
    classify_Y,
    label_Y,
    match_rollout,
    representative_rollouts,
)
from bifurcation.metrics.discrete import _matched_final_mae


def reference_report(sample, Y_gen: list[torch.Tensor],
                     Y_gen_relaxed: list[torch.Tensor] | None = None) -> dict[str, float]:
    """Allen-Cahn's ``mc_mse``/``mc_mae``/``mc_mae_overall``: self-consistency between the
    raw model output and its PDE-relaxed projection, NOT fidelity to a saved rollout like
    every other dataset's same-named fields -- see docs/decisions.md.

    ``Y_gen_relaxed`` (``None``: same as ``Y_gen``, so every field is exactly 0) is
    index-aligned with ``Y_gen``; ``sample`` is unused, kept for the protocol call signature.
    """
    if Y_gen_relaxed is None:
        Y_gen_relaxed = Y_gen
    mse, mae_final, mae_overall = [], [], []
    for Y, Y_relaxed in zip(Y_gen, Y_gen_relaxed):
        d = Y_relaxed - Y
        mse.append(float(torch.nanmean(d.square().sum(-1))))
        mae_overall.append(float(torch.nanmean(d.abs())))
        mae_final.append(_matched_final_mae(d))
    return {"mc_mse": sum(mse) / len(mse),
            "mc_mae": sum(mae_final) / len(mae_final),
            "mc_mae_overall": sum(mae_overall) / len(mae_overall)}


def _laplacian3d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Periodic 6-point-stencil Laplacian over the trailing (X, Y, Z) dims of ``u``.

    Matrix-free: a dense Laplacian over a 64^3 grid would be a (262144 x 262144) matrix, far
    too large to form.
    """
    lap = -6.0 * u
    for dim in (-3, -2, -1):
        lap = lap + torch.roll(u, 1, dims=dim) + torch.roll(u, -1, dims=dim)
    return lap / dx**2


def _compute_residual(u: torch.Tensor, eps: float, mu: float, dx: float, dt: float) -> torch.Tensor:
    lhs1 = (u[1:] - u[:-1]) / dt
    rhs1a = eps**2 * _laplacian3d(u[1:], dx)
    rhs1b = -(u[:-1] ** 3 - mu * u[:-1])
    return lhs1 - (rhs1a + rhs1b)


def _apply_J(v: torch.Tensor, reaction_jac: torch.Tensor, eps: float, dx: float, dt: float) -> torch.Tensor:
    return (v[1:] - v[:-1]) / dt - eps**2 * _laplacian3d(v[1:], dx) + reaction_jac * v[:-1]


def _apply_JT(w: torch.Tensor, reaction_jac: torch.Tensor, eps: float, dx: float, dt: float,
              T: int, grid_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False), torch.enable_grad():
        reaction_jac = reaction_jac.clone()
        v = torch.zeros(T, *grid_shape, device=device, requires_grad=True)
        (_apply_J(v, reaction_jac, eps, dx, dt) * w.clone()).sum().backward()
        return v.grad.detach()


def _normal_op(du, reaction_jac, eps, dx, dt, lam, T, grid_shape, device):
    return _apply_JT(_apply_J(du, reaction_jac, eps, dx, dt), reaction_jac, eps, dx, dt,
                     T, grid_shape, device) + lam * du


def _cg_solve(rhs, reaction_jac, eps, dx, dt, lam, T, grid_shape, device,
             n_iter: int, tol: float) -> torch.Tensor:
    du = torch.zeros_like(rhs)
    r = rhs - _normal_op(du, reaction_jac, eps, dx, dt, lam, T, grid_shape, device)
    p = r.clone()
    rs_old = (r * r).sum()
    for _ in range(n_iter):
        Ap = _normal_op(p, reaction_jac, eps, dx, dt, lam, T, grid_shape, device)
        alpha = rs_old / (p * Ap).sum()
        du = du + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return du


def relax_allencahn(u: torch.Tensor, eps: float, mu: float, dx: float, dt: float, *,
                    step_size: float = 0.4, lam: float = 0.0,
                    n_gn_iter: int = 100, cg_iter: int = 100, tol: float = 0.002) -> torch.Tensor:
    """Gauss-Newton relax ``u`` (``[T,X,Y,Z]``) onto the nearest trajectory that satisfies the
    Allen-Cahn PDE ``du/dt = eps^2 * laplacian(u) - (u^3 - mu*u)``.

    ``dx``/``dt`` must match the solver that generated the dataset -- see :func:`relax_Y` for
    how they're derived from the dataset's config. Runs under ``no_grad`` except for the
    internal autograd-adjoint trick in :func:`_apply_JT`.
    """
    T, grid_shape, device = u.shape[0], u.shape[1:], u.device
    u = u.clone()
    with torch.no_grad():
        residual = _compute_residual(u, eps, mu, dx, dt)
        for _ in range(n_gn_iter):
            reaction_jac = 3 * u[:-1] ** 2 - mu  # linearization of the cubic reaction term
            rhs = -_apply_JT(residual, reaction_jac, eps, dx, dt, T, grid_shape, device)
            du = _cg_solve(rhs, reaction_jac, eps, dx, dt, lam, T, grid_shape, device,
                           n_iter=cg_iter, tol=1e-8)
            u = u + step_size * du
            residual = _compute_residual(u, eps, mu, dx, dt)
            if torch.linalg.norm(residual).item() < tol:
                break
    return u


def _cube_and_params(sample, Y: torch.Tensor, domain_length: float):
    """Shared prep for ``relax_Y``/``residual_Y``: reshape ``[T_,N_,1]`` -> ``[T,X,Y,Z]`` on
    whatever device the physics should run on, plus this sample's own eps/mu/dx."""
    T_, N_, C_ = Y.shape
    if C_ != 1:
        raise ValueError(f"expected a scalar field, got {C_} channels")
    n = round(N_ ** (1 / 3))
    if n**3 != N_:
        raise ValueError(f"expected a cubic grid, got N_={N_}")

    device = torch.device("cuda") if torch.cuda.is_available() else Y.device
    u = Y[..., 0].to(device=device, dtype=torch.float32).reshape(T_, n, n, n)
    eps, mu = float(sample.U[0, 0]), float(sample.U[0, 1])
    dx = domain_length / n
    return u, eps, mu, dx


def relax_Y(sample, Y: torch.Tensor, dt: float, *, domain_length: float = 2.0 * math.pi,
           step_size: float = 0.4, lam: float = 0.0,
           n_gn_iter: int = 100, cg_iter: int = 100, tol: float = 0.002) -> torch.Tensor:
    """Protocol hook (``sample, Y`` convention, matching ``label_Y``/``draw_mode``): relax one
    generated Allen-Cahn rollout ``Y`` (``[T_,N_,1]``, physical units) onto the nearest
    PDE-consistent trajectory before it is labeled/scored.

    ``dt`` is the effective per-frame timestep at ``Y``'s time resolution -- the caller
    (``Generate``) derives it from the dataset's solver config and the active view's
    ``time_stride``, since neither is available from ``sample``/``Y`` alone. ``domain_length``
    follows the convention in ``bifurcation.datasets.allencahn`` (``2*pi`` for 3D); the solver
    config has no explicit domain field.
    """
    T_, N_, _ = Y.shape
    u, eps, mu, dx = _cube_and_params(sample, Y, domain_length)

    relaxed = relax_allencahn(u, eps, mu, dx, dt, step_size=step_size, lam=lam,
                              n_gn_iter=n_gn_iter, cg_iter=cg_iter, tol=tol)
    return relaxed.reshape(T_, N_, 1).to(device=Y.device, dtype=Y.dtype)


def residual_Y(sample, Y: torch.Tensor, dt: float, *, domain_length: float = 2.0 * math.pi) -> float:
    """Protocol hook (``sample, Y`` convention): the L2 norm of the Allen-Cahn PDE residual of
    one rollout ``Y`` (``[T_,N_,1]``, physical units) -- how far it is from actually solving the
    equation for this sample's own eps/mu, with no projection (unlike :func:`relax_Y`).

    Called both on the raw generated field and again on its relaxed counterpart, so the
    ``reference_report`` gap it fills before/after residual is visible: how inconsistent the
    model's raw output is, and how much of that the relaxation step actually closes.
    """
    u, eps, mu, dx = _cube_and_params(sample, Y, domain_length)
    with torch.no_grad():
        residual = _compute_residual(u, eps, mu, dx, dt)
    return float(torch.linalg.norm(residual).item())
