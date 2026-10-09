#!/usr/bin/env python3
"""Generate Allen-Cahn phase-separation datasets (1D/2D/3D).

Reproduces the sampling scheme of the original AllenCahn_createDataset.ipynb
(epsilon log-uniform in [1e-3, 1e-1], mu mostly uniform in [0, 1]) and the
same output format: a pickle with float32 torch tensors
    {'solutions': (B, T, *grid), 'epsilon': (B,), 'mu': (B,)}
plus the run configuration under 'config'.

``--realizations R`` (default 1) draws R independent noise initial conditions per
sampled (epsilon, mu) pair instead of just one, structure-major: rows
``i*R : (i+1)*R`` of 'solutions'/'epsilon'/'mu' are the R siblings of structure i, all
sharing the same (epsilon, mu) but different noise seeds, so B = n-samples * R. Since
Allen-Cahn is a gradient flow, different noise seeds at the same parameters generally
relax into genuinely different phase-separation morphologies -- these siblings are the
actual multiplicity of solutions at that (epsilon, mu), which src/datasets/allencahn.py
groups back into one dataset sample and runs mode detection over (together with each
realization's analytic u <-> -u sign flip). R=1 (the default) reproduces the previous
one-trajectory-per-pair behavior exactly.

Examples:
    python generate_dataset.py --dim 1 --n-samples 2000 --steps 1000 --grid 200
    python generate_dataset.py --dim 2 --n-samples 100 --steps 1500 --grid 100 --save-every 10
    python generate_dataset.py --dim 3 --n-samples 20  --steps 500  --grid 64 --save-every 10 --backend cuda
    python generate_dataset.py --dim 2 --n-samples 50 --realizations 4 --grid 100  # 4 noise
        seeds per (epsilon, mu) pair, for genuine multi-solution mode detection
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

import allen_cahn as ac
import fast


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dim', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--n-samples', type=int, default=100)
    p.add_argument('--grid', type=int, default=None,
                   help='points per axis (default: 200 for 1D, 100 for 2D, 64 for 3D)')
    p.add_argument('--steps', type=int, default=None,
                   help='time steps (default: 1000 / 1500 / 500 for 1D/2D/3D)')
    p.add_argument('--dt', type=float, default=None,
                   help='time step (default: 0.1 for 1D, 0.01 for 2D/3D)')
    p.add_argument('--save-every', type=int, default=1,
                   help='store every k-th time step')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--chunk', type=int, default=64,
                   help='samples solved per batch (memory control)')
    p.add_argument('--eps-range', type=float, nargs=2, default=(-3, -1),
                   metavar=('LO', 'HI'),
                   help='epsilon ~ 10^Uniform(LO, HI) (default: -3 -1, i.e. '
                        'epsilon in [1e-3, 0.1]). Pick relative to the grid '
                        'spacing dx=2*pi/grid - too small vs. dx leaves the '
                        'interface unresolved and samples look like noise '
                        'regardless of --steps; see allen_cahn.sample_parameters.')
    p.add_argument('--mu-range', type=float, nargs=2, default=(0.0, 1.0),
                   metavar=('LO', 'HI'),
                   help='mu ~ Uniform(LO, HI) for the non-negative slice (default: '
                        '0.0 1.0). mu sets the growth timescale (~1/mu); raising LO '
                        'above 0 cuts down on samples that are still unseparated '
                        'noise at the end of the run because mu was too small - '
                        'see allen_cahn.sample_parameters.')
    p.add_argument('--mu-neg-fraction', type=float, default=1 / 50,
                   help='fraction of mu drawn from [-0.1, 0] instead of --mu-range '
                        '(default: 1/50)')
    p.add_argument('--realizations', type=int, default=1,
                   help='independent noise initial conditions simulated per (epsilon, mu) '
                        'pair (default: 1, i.e. one trajectory per pair). >1 gives the '
                        'actual multi-solution siblings at fixed parameters, see the '
                        'module docstring.')
    p.add_argument('--out', type=str, default="../data")
    args = p.parse_args()

    dim = args.dim
    grid = args.grid or {1: 200, 2: 100, 3: 64}[dim]
    steps = args.steps or {1: 1000, 2: 1500, 3: 500}[dim]
    dt = args.dt or {1: 0.1, 2: 0.01, 3: 0.01}[dim]
    filename = (f'AllenCahn_{dim}D_periodic_{args.n_samples}'
                f'_seed{args.seed}.pkl')
    out = Path(args.out)
    if args.out == "../data":
        out = Path(__file__).resolve().parents[2] / "data"
    if out.suffix != ".pkl":
        out = out / filename

    R = args.realizations
    rng = np.random.default_rng(args.seed)
    epsilon, mu = ac.sample_parameters(args.n_samples, rng, eps_range=args.eps_range,
                                       mu_range=args.mu_range,
                                       mu_neg_fraction=args.mu_neg_fraction)
    # Repeat each (epsilon, mu) pair across its R noise siblings, structure-major, so rows
    # i*R:(i+1)*R of every per-trajectory array below belong to structure i.
    epsilon = np.repeat(epsilon, R)
    mu = np.repeat(mu, R)
    n_total = args.n_samples * R
    shape = (grid,) * dim
    # ICs drawn after the parameters, matching the original notebook's order (R=1 draws them
    # in the same order as before; R>1 draws one extra IC per sibling, structure-major).
    ic = ac.random_ic_1d if dim == 1 else ac.random_ic_nd
    u0 = np.stack([ic(shape if dim > 1 else grid, rng)
                   for _ in range(n_total)])

    solver = {1: fast.solve_ac_1d_batch,
              2: fast.solve_ac_2d_batch,
              3: fast.solve_ac_3d_batch}[dim]
    kwargs = dict(dt=dt, steps=steps, save_every=args.save_every)
    if dim > 1:
        kwargs['backend'] = args.backend

    n_save = steps // args.save_every + 1
    solutions = np.empty((n_total, n_save) + shape, dtype=np.float32)
    t0 = time.perf_counter()
    for lo in range(0, n_total, args.chunk):
        hi = min(lo + args.chunk, n_total)
        solutions[lo:hi] = solver(u0[lo:hi], epsilon=epsilon[lo:hi],
                                  mu=mu[lo:hi], **kwargs)
        print(f'  {hi}/{n_total} samples '
              f'({time.perf_counter() - t0:.1f}s)', flush=True)

    import torch
    payload = {
        'solutions': torch.from_numpy(solutions),
        'epsilon': torch.tensor(epsilon, dtype=torch.float32),
        'mu': torch.tensor(mu, dtype=torch.float32),
        'config': {**vars(args), 'grid': grid, 'steps': steps, 'dt': dt,
                   'bc': 'periodic', 'realizations': R},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump(payload, f)
    gb = solutions.nbytes / 1e9
    print(f'Wrote {out} ({gb:.2f} GB, solutions shape {tuple(solutions.shape)}, '
          f'{args.n_samples} structures x {R} realizations)')


if __name__ == '__main__':
    main()
