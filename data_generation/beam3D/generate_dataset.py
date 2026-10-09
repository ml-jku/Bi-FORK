#!/usr/bin/env python3
"""Generate the BucklingBeams3D dataset.

Scripts are based on the files from: https://github.com/FHendriks11/bifurcationML/tree/main/test4_buckling_beams

To execut the data generation script, run:
    uv run python data_generation/beam3D/generate_dataset.py --n-samples 1000 --workers 16

"""

import argparse
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.optimize as scopt
import torch
import torch_geometric as tg


REPOSITORY = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Derivatives of the Lagrangian
# x[0] is lambda, x[1:N+1] are the angles q, x[N+1:] are the strains eps
# ---------------------------------------------------------------------------

def dLdq(x, K, C, L, d):
    lamb = x[0]
    n = (len(x) - 1) // 2
    q = np.asanyarray(x)[1:n + 1]
    e = np.asanyarray(x)[n + 1:]
    temp = np.empty(len(x), dtype=float)

    # dL/dlambda
    temp[0] = d - np.sum(L * (1 - np.exp(e) * np.cos(q)))
    # dL/dq, case i = 0
    temp[1] = (K[0] + K[1]) * q[0] - K[1] * q[1]
    # dL/dq, case i = 1 to n-2
    temp[2:n] = (K[1:-1] + K[2:]) * q[1:-1] - K[1:-1] * q[:-2] - K[2:] * q[2:]
    # dL/dq, case i = n - 1
    temp[n] = K[-1] * (q[-1] - q[-2])
    temp[1:n + 1] -= lamb * L * np.exp(e) * np.sin(q)
    # dL/de
    temp[n + 1:] = L * C * e + lamb * L * np.exp(e) * np.cos(q)
    return temp


def d2Ldq2(x, K, C, L, d):
    lamb = x[0]
    n = (len(x) - 1) // 2
    q = np.asanyarray(x)[1:n + 1]
    e = np.asanyarray(x)[n + 1:]
    temp = np.zeros((len(x), len(x)), dtype=float)

    # lambda, q
    temp[0, 1:n + 1] = -L * np.exp(e) * np.sin(q)
    temp[1:n + 1, 0] = temp[0, 1:n + 1]
    # lambda, e
    temp[0, n + 1:] = L * np.exp(e) * np.cos(q)
    temp[n + 1:, 0] = temp[0, n + 1:]
    # q, q: diagonal, off-diagonal, final element
    np.fill_diagonal(temp[1:n, 1:n], K[:-1] + K[1:] - L[:-1] * lamb * np.exp(e[:-1]) * np.cos(q[:-1]))
    np.fill_diagonal(temp[2:n + 1, 1:n + 1], -K[1:])
    np.fill_diagonal(temp[1:n + 1, 2:n + 1], -K[1:])
    temp[n, n] = K[-1] - L[-1] * np.exp(e[-1]) * lamb * np.cos(q[-1])
    # e, q
    np.fill_diagonal(temp[n + 1:, 1:n + 1], -lamb * L * np.exp(e) * np.sin(q))
    np.fill_diagonal(temp[1:n + 1, n + 1:], -lamb * L * np.exp(e) * np.sin(q))
    # e, e
    np.fill_diagonal(temp[n + 1:, n + 1:], L * C + lamb * L * np.exp(e) * np.cos(q))
    return temp


def d2Udq2(x, K, C, L, d):
    """Hessian of the strain energy U w.r.t. (q, eps)."""
    n = (len(x) - 1) // 2
    temp = np.zeros((len(x) - 1, len(x) - 1), dtype=float)
    np.fill_diagonal(temp[0:n - 1, 0:n - 1], K[:-1] + K[1:])
    np.fill_diagonal(temp[1:n, 0:n], -K[1:])
    np.fill_diagonal(temp[0:n, 1:n], -K[1:])
    temp[n - 1, n - 1] = K[-1]
    np.fill_diagonal(temp[n:, n:], L * C)
    return temp


def displacement(angles, strains, L):
    return np.sum(L * (1 - np.exp(strains) * np.cos(angles)))


# ---------------------------------------------------------------------------
# Stability and solver
# ---------------------------------------------------------------------------

def is_stable(K):
    # Test 1: leading principal minors of the bordered Hessian
    stable1 = True
    for i in range(3, len(K) + 1):
        if np.linalg.det(K[:i, :i]) > 0:
            stable1 = False
            break
    # Test 2: exactly one negative eigenvalue
    stable2 = np.sum(np.linalg.eigvals(K) < 0) == 1

    if stable1 != stable2:
        warnings.warn('Stability tests are inconsistent')
    return stable2


def get_eigenmodes(K, K_orig, k=5):
    """Lowest eigenmodes of the energy Hessian ``K_orig`` restricted to the constraint
    tangent space (given by the first row of the Lagrangian Hessian ``K``)."""
    C = K[[0], 1:]  # constraint matrix: dg/dq = d2L/dlambda dq
    Q, _ = np.linalg.qr(C.T, mode='complete')
    Q2 = Q[:, 1:]
    # eigh, not eigsh: ARPACK starts from a random vector, which makes the sign random.
    _, eigenvectors = np.linalg.eigh(Q2.T @ K_orig @ Q2)
    eigenvectors = eigenvectors[:, :k]
    pivot = np.argmax(np.abs(eigenvectors), axis=0)
    eigenvectors = eigenvectors * np.sign(eigenvectors[pivot, np.arange(k)])
    return Q2 @ eigenvectors


def sol_valid(q_sol_temp, KK, L, d_temp, verbose=False):
    N = len(L)
    if np.isnan(q_sol_temp).any():
        if verbose:
            print(f'encountered np.nan in solution: {q_sol_temp}')
        return False
    elif not is_stable(KK):
        if verbose:
            print(f'q_sol={q_sol_temp} is unstable')
        return False
    elif np.abs(displacement(q_sol_temp[1:N + 1], q_sol_temp[N + 1:], L) - d_temp) > 1e-6:
        if verbose:
            print(f'q_sol={q_sol_temp} does not have displacement d={d_temp}')
        return False
    # lambda must be positive
    elif q_sol_temp[0] <= 1e-7 and d_temp != 0:
        return False
    # not all strains zero
    elif np.all(np.abs(q_sol_temp[N + 1:]) < 1e-9) and d_temp != 0:
        return False
    # not all angles equal
    elif max(q_sol_temp[1:N + 1]) - min(q_sol_temp[1:N + 1]) < 1e-7 and np.max(q_sol_temp[1:N + 1]) > 1e-6:
        return False
    return True


def _perturbed_solve(q0, q_sol_temp, KK, K, C, L, d_temp, magn, verbose):
    """Re-solve from ``q0`` perturbed along the lowest constrained eigenmode."""
    N = len(L)
    K_orig = d2Udq2(q_sol_temp, K, C, L, d_temp)
    mode = get_eigenmodes(KK, K_orig, k=min(1, N)).T[0]
    if verbose:
        print(f'eigenvector={mode}')
    q0_pert = np.copy(q0)
    q0_pert[1:] += magn * mode / np.max(np.abs(mode))
    q_sol_temp = scopt.root(dLdq, q0_pert, args=(K, C, L, d_temp), jac=d2Ldq2).x
    return q_sol_temp, d2Ldq2(q_sol_temp, K, C, L, d_temp)


def solve_buckling_beam(N, K, C, L, d, verbose=False):
    """Solve all increments of ``d``. Returns ``q_sol [T, 2, 2N+1]``, the solution and its
    mirror image (angles negated)."""
    q_sol = np.empty((len(d), 2 * N + 1), dtype=float)

    for i, d_temp in enumerate(d):
        if i == 0:
            q0 = np.array([0.1] + [0.0] * 2 * N)  # all angles and strains zero, lambda = 0.1
        else:
            q0 = np.copy(q_sol[i - 1])  # previous increment as initial guess
            if np.abs(q0[0]) < 1e-4:   # lambda close to zero causes problems
                q0[0] = 0.1

        q_sol_temp = scopt.root(dLdq, q0, args=(K, C, L, d_temp)).x
        KK = d2Ldq2(q_sol_temp, K, C, L, d_temp)

        if sol_valid(q_sol_temp, KK, L, d_temp, verbose=verbose):
            q_sol[i] = q_sol_temp
        elif not is_stable(KK):
            # unstable: perturb along the buckling mode, retry with a larger magnitude
            for magn in (0.1, 0.5):
                q_sol_temp, KK = _perturbed_solve(q0, q_sol_temp, KK, K, C, L, d_temp, magn, verbose)
                if sol_valid(q_sol_temp, KK, L, d_temp, verbose=verbose):
                    q_sol[i] = q_sol_temp
                    break
            else:
                raise ValueError('Perturbed solution is invalid')
        else:
            raise ValueError('Solution is invalid')

    mirrored = np.copy(q_sol)
    mirrored[:, 1:N + 1] = -mirrored[:, 1:N + 1]
    return np.stack((q_sol, mirrored), axis=1)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def sample_beam(rng, n_min, n_max, log2_range):
    """N ~ U{n_min..n_max}; L, C, K ~ 2^U(log2_range) per element."""
    N = int(rng.integers(n_min, n_max + 1))
    lo, hi = log2_range
    L = 2 ** rng.uniform(lo, hi, N)
    C = 2 ** rng.uniform(lo, hi, N)
    K = 2 ** rng.uniform(lo, hi, N)
    return {'N': N, 'K': K, 'C': C, 'L': L}


def solve_job(job):
    """Worker: solve one beam, or return None if the solver fails."""
    d = np.linspace(0, np.sum(job['L']) * job['max_d_frac'], job['n_steps'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            q_sol = solve_buckling_beam(job['N'], job['K'], job['C'], job['L'], d)
        except ValueError as err:
            return job['index'], None, str(err)
    return job['index'], {'N': job['N'], 'K': job['K'], 'C': job['C'], 'L': job['L'],
                          'd': d, 'q_sol': q_sol}, None


def make_graph(result):
    """2D graph from a solved beam; pos has shape [N+1, T, 2, n_solutions=2]."""
    N = result['N']
    L = result['L']
    d = result['d']

    node_attr = torch.tensor(np.append(result['K'], 0)).unsqueeze(-1)

    temp = torch.arange(N)
    edge_index = torch.stack((temp, temp + 1), dim=0)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)

    edge_attr = torch.tensor(np.stack((L, result['C']), axis=1))
    edge_attr = torch.cat((edge_attr, edge_attr), dim=0)

    angles = result['q_sol'][:, :, 1:N + 1]  # [T, 2, N]
    strains = result['q_sol'][:, :, N + 1:]
    new_lens = L.reshape(1, 1, -1) * np.exp(strains)
    pos = np.zeros((len(d), 2, N + 1, 2))   # [T, n_solutions, N+1, dim]
    pos[:, :, 1:, 0] = np.cumsum(new_lens * np.sin(angles), axis=-1)
    pos[:, :, 1:, 1] = np.cumsum(new_lens * np.cos(angles), axis=-1)
    pos = torch.tensor(np.transpose(pos, (2, 0, 3, 1)))  # [N+1, T, dim, n_solutions]

    lamb = result['q_sol'][:, :, 0]
    return tg.data.Data(edge_index=edge_index,
                        node_attr=node_attr,
                        pos=pos,
                        edge_attr=edge_attr,
                        d=torch.tensor(d).reshape(1, -1, 1),
                        N=torch.tensor([N]).reshape(1, 1),
                        lamb=torch.tensor(lamb).reshape(1, -1, 2, 1))


def make_branch_consistent(graph):
    """Mirror time steps whose tip bends to the other side than at the final step, so that
    solution branch 0 does not jump between the two mirror solutions over time."""
    x_tip = graph.pos[-1, :, 0, 0]  # [T]
    signs = torch.sign(x_tip[torch.abs(x_tip) > 1e-5])
    if torch.any(signs == 1) and torch.any(signs == -1):
        flip = torch.sign(x_tip) != torch.sign(x_tip[-1])
        graph.pos[:, flip, 0, :] *= -1
    return graph


def lift_to_3d(graph, phi):
    """Rotate the bending plane of solution branch 0 by azimuth ``phi`` about the vertical."""
    x, y = graph.pos[..., 0, 0], graph.pos[..., 1, 0]  # [N+1, T]
    graph.pos = torch.stack([x * np.cos(phi), x * np.sin(phi), y], dim=-1)  # [N+1, T, 3]
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--n-samples', type=int, default=1000,
                   help='beams to sample; beams whose solver fails are dropped')
    p.add_argument('--n-steps', type=int, default=200, help='displacement increments per beam')
    p.add_argument('--max-d-frac', type=float, default=1.0,
                   help='final tip displacement as a fraction of the total beam length')
    p.add_argument('--n-min', type=int, default=2, help='minimum number of elements')
    p.add_argument('--n-max', type=int, default=10, help='maximum number of elements')
    p.add_argument('--log2-range', type=float, nargs=2, default=(-1.0, 1.0), metavar=('LO', 'HI'),
                   help='L, C, K ~ 2^Uniform(LO, HI) (default: -1 1, i.e. [0.5, 2])')
    p.add_argument('--train-fraction', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', type=Path, default=REPOSITORY / 'data' / 'BucklingBeams3D_data.pkl')
    p.add_argument('--out-2d', type=Path, default=None,
                   help='optionally also save the 2D graphs (both solution branches)')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    jobs = [sample_beam(rng, args.n_min, args.n_max, args.log2_range)
            | {'index': i, 'n_steps': args.n_steps, 'max_d_frac': args.max_d_frac}
            for i in range(args.n_samples)]
    azimuths = rng.uniform(0, 2 * np.pi, args.n_samples)

    t0 = time.time()
    results = [None] * args.n_samples
    n_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, result, error) in enumerate(pool.map(solve_job, jobs, chunksize=4), 1):
            results[index] = result
            if error is not None:
                n_failed += 1
                beam = jobs[index]
                print(f'[{index}] failed ({error}): N={beam["N"]}, K={beam["K"]}, '
                      f'C={beam["C"]}, L={beam["L"]}')
            if done % 50 == 0 or done == args.n_samples:
                print(f'{done}/{args.n_samples} beams solved ({n_failed} failed), '
                      f'{time.time() - t0:.0f}s')

    graphs_2d, graphs_3d = [], []
    for result, phi in zip(results, azimuths):
        if result is None:
            continue
        graph = make_branch_consistent(make_graph(result))
        graphs_2d.append(graph)
        graphs_3d.append(lift_to_3d(graph.clone(), phi))

    n_train = int(args.train_fraction * len(graphs_3d))
    print(f'Training data size: {n_train}')
    print(f'Test data size: {len(graphs_3d) - n_train}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'data_tr': graphs_3d[:n_train], 'data_te': graphs_3d[n_train:]}, f)
    print(f'Wrote {args.out}')

    if args.out_2d is not None:
        args.out_2d.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_2d, 'wb') as f:
            pickle.dump({'data_tr': graphs_2d[:n_train], 'data_te': graphs_2d[n_train:]}, f)
        print(f'Wrote {args.out_2d}')


if __name__ == '__main__':
    main()
