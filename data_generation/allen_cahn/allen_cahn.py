"""Reference (NumPy/SciPy) solvers for the Allen-Cahn equation in 1D/2D/3D.

    du/dt = eps^2 * Laplace(u) - (u^3 - mu*u)

All solvers use the same semi-implicit (IMEX) splitting as the original
notebooks in data/allencahn/: the stiff diffusion term is treated
implicitly, the nonlinear reaction term explicitly:

    (I - dt*eps^2*Laplace) u_{n+1} = u_n - dt*(u_n^3 - mu*u_n)

1D solves this with a sparse finite-difference Laplacian (periodic or
Neumann BCs); 2D/3D solve it in Fourier space (periodic BCs only), where
the implicit step is a pointwise division.

These implementations are the *ground truth* the fast C++/CUDA backends in
fast.py are validated against (see tests/test_consistency.py).
"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------- initial conditions

def random_ic_1d(N, rng, amplitude=1e-3):
    """Small uniform noise around 0, as in the original 1D notebook."""
    return rng.uniform(-1, 1, N) * amplitude


def random_ic_nd(shape, rng, amplitude=0.1):
    """Uniform noise in [-amplitude, amplitude], as in the original 2D notebook."""
    return amplitude * rng.uniform(-1, 1, shape)


# ---------------------------------------------------------------- 1D (finite differences)

def solve_ac_1d(u0, dt=0.1, steps=1000, epsilon=0.01, mu=1.0, L=1.0,
                bc='periodic', save_every=1):
    """Semi-implicit FD solver, identical scheme to the original notebook.

    The only change vs. the notebook is that the implicit matrix is
    LU-factorized once (splu) instead of re-factorized every step
    (spsolve) - same algorithm, same result, ~100x faster.

    Returns snapshots of shape (steps//save_every + 1, N), snapshot 0 = u0.
    """
    u = np.asarray(u0, dtype=np.float64).copy()
    N = u.shape[0]
    dx = L / N

    main = -2.0 * np.ones(N)
    off = np.ones(N - 1)
    # LIL format supports element assignment for the BC entries below; the
    # original notebook went dense via .toarray() instead (diags' default
    # DIA format cannot be indexed). Same matrix entries either way.
    lap = diags([off, main, off], offsets=[-1, 0, 1], format='lil')
    if bc.lower() == 'neumann':
        lap[0, 1] = 2.0
        lap[-1, -2] = 2.0
    elif bc.lower() == 'periodic':
        lap[0, -1] = lap[-1, 0] = 1.0
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")
    # A is constant in time (only the RHS changes), so LU-factorize it once.
    # The notebook called spsolve(A, rhs) every step, re-factorizing the same
    # matrix 1000x. spsolve uses SuperLU internally just like splu, so the
    # floating-point operations - and hence the results - are bit-identical
    # (tests/test_consistency.py), only ~100x faster.
    A = (eye(N) - dt * epsilon**2 / dx**2 * lap).tocsc()
    solve = splu(A).solve

    n_save = steps // save_every + 1
    snapshots = np.empty((n_save, N))
    snapshots[0] = u
    for n in range(1, steps + 1):
        u = solve(u - dt * (u**3 - mu * u))
        if n % save_every == 0:
            snapshots[n // save_every] = u
    return snapshots


# ---------------------------------------------------------------- 2D/3D (Fourier spectral)

def _spectral_solve(u0, dt, steps, epsilon, mu, lengths, save_every):
    u = np.asarray(u0, dtype=np.float64).copy()
    # -|k|^2, the Laplacian symbol on the periodic grid
    ks = [2 * np.pi * np.fft.fftfreq(n, d=Ld / n)
          for n, Ld in zip(u.shape, lengths)]
    K = np.meshgrid(*ks, indexing='ij')
    lap_k = -sum(k**2 for k in K)
    denom = 1 - dt * epsilon**2 * lap_k

    n_save = steps // save_every + 1
    snapshots = np.empty((n_save,) + u.shape)
    snapshots[0] = u
    for n in range(1, steps + 1):
        w = u + dt * (mu * u - u**3)            # explicit reaction
        u = np.real(np.fft.ifftn(np.fft.fftn(w) / denom))  # implicit diffusion
        if n % save_every == 0:
            snapshots[n // save_every] = u
    return snapshots


def solve_ac_2d(u0, dt=0.01, steps=1500, epsilon=0.01, mu=1.0,
                Lx=2 * np.pi, Ly=2 * np.pi, save_every=1):
    """Semi-implicit Fourier-spectral solver (periodic BCs), identical
    scheme to the original 2D notebook, generalized to mu != 1.
    u0 has shape (ny, nx); lengths are matched axis-by-axis (Ly, Lx)."""
    return _spectral_solve(u0, dt, steps, epsilon, mu, (Ly, Lx), save_every)


def solve_ac_3d(u0, dt=0.01, steps=500, epsilon=0.01, mu=1.0,
                Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi, save_every=1):
    """3D extension of the 2D spectral scheme. u0 has shape (nz, ny, nx)."""
    return _spectral_solve(u0, dt, steps, epsilon, mu, (Lz, Ly, Lx), save_every)


# ---------------------------------------------------------------- parameter sampling

def sample_parameters(n_samples, rng, eps_range=(-3, -1), mu_range=(0.0, 1.0),
                      mu_neg_fraction=1 / 50):
    """Sample (epsilon, mu) pairs as in the original createDataset notebook:
    epsilon log-uniform in 10^eps_range; a small fraction of mu in
    [-0.1, 0] (less interesting regime), the rest uniform in mu_range.

    eps_range is dimension/grid-agnostic here; pick it relative to the grid
    spacing dx=L/grid you'll solve on. The interface width is
    xi = epsilon*sqrt(2/mu), and xi/dx should be >~3 to resolve an
    interface at all - e.g. on a 64^3 grid over [0, 2pi] (dx~0.098),
    the default (-3, -1) => epsilon in [1e-3, 0.1] is under-resolved for
    most of its range (see data_generation/AllenCahn_3D.ipynb), leaving
    many samples looking like pure noise regardless of --steps.

    mu also sets the growth timescale (~1/mu) for the noise to separate
    into the two phases: mu near mu_range's lower bound (esp. near 0)
    grows so slowly it may still look unseparated at the end of the run
    even when epsilon is well resolved - raise the lower bound of
    mu_range (rather than adding --steps) to cut down on those.
    """
    epsilon = 10 ** rng.uniform(*eps_range, n_samples)
    n_neg = int(n_samples * mu_neg_fraction)
    mu = np.concatenate([rng.uniform(-0.1, 0.0, n_neg),
                         rng.uniform(*mu_range, n_samples - n_neg)])
    return epsilon, mu
