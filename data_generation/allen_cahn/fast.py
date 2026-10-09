"""Fast C++/CUDA backends for the Allen-Cahn solvers in allen_cahn.py.

Same numerics and signatures as the reference solvers, but batched
(one call solves many samples, each with its own epsilon/mu) and
implemented in C++ (OpenMP + FFTW) or CUDA (cuFFT).

    solve_ac_1d_batch(u0, ...)              # C++ (tridiagonal, OpenMP)
    solve_ac_2d_batch(u0, ..., backend=...) # 'cpu' (FFTW) or 'cuda' (cuFFT)
    solve_ac_3d_batch(u0, ..., backend=...)

The shared libraries are built automatically on first import (csrc/Makefile)
if missing - but NOT rebuilt after source edits;
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_CSRC = Path(__file__).resolve().parent / "csrc"
_BUILD = _CSRC / "build"

_c_double_p = ctypes.POINTER(ctypes.c_double)


def _ensure_built():
    if not (_BUILD / "liballencahn_cpu.so").exists():
        subprocess.run(["make"], cwd=_CSRC, check=True, capture_output=True)


def _preload_bundled_cufft():
    """liballencahn_cuda.so links against libcufft.so.11 but has no rpath.
    In this project's uv venv that lib isn't on the system loader path, but
    it's already vendored as a dependency of the pinned cu129 torch wheel
    (nvidia-cufft-cu12); preload it by absolute path so the dynamic linker
    resolves the dependency without needing LD_LIBRARY_PATH set."""
    try:
        import nvidia.cufft
    except ImportError:
        return
    for lib_dir in nvidia.cufft.__path__:
        for so in sorted(Path(lib_dir, "lib").glob("libcufft.so*")):
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)


def _load(name):
    _ensure_built()
    path = _BUILD / name
    return ctypes.CDLL(str(path)) if path.exists() else None


_cpu = _load("liballencahn_cpu.so")
_preload_bundled_cufft()
_cuda = _load("liballencahn_cuda.so")
cuda_available = _cuda is not None

if _cuda is not None:
    _cuda.ac_cuda_last_error.restype = ctypes.c_char_p


def _as_batch(u0, ndim):
    """Accept a single sample or a batch; return C-contiguous float64 batch."""
    u0 = np.ascontiguousarray(u0, dtype=np.float64)
    if u0.ndim == ndim:
        u0 = u0[None]
    assert u0.ndim == ndim + 1, f"expected {ndim}D samples, got shape {u0.shape}"
    return u0


def _params(x, B):
    x = np.ascontiguousarray(np.broadcast_to(np.float64(x), (B,)))
    return x, x.ctypes.data_as(_c_double_p)


def solve_ac_1d_batch(u0, dt=0.1, steps=1000, epsilon=0.01, mu=1.0, L=1.0,
                      bc='periodic', save_every=1):
    """Batched 1D solver (C++/OpenMP)."""
    u0 = _as_batch(u0, 1)
    B, N = u0.shape
    eps_a, eps_p = _params(epsilon, B)
    mu_a, mu_p = _params(mu, B)
    n_save = steps // save_every + 1
    out = np.empty((B, n_save, N))
    bc_code = {'periodic': 0, 'neumann': 1}[bc.lower()]
    _cpu.ac1d_solve_batch(
        u0.ctypes.data_as(_c_double_p), ctypes.c_int(B), ctypes.c_int(N),
        ctypes.c_double(L), ctypes.c_double(dt), ctypes.c_int(steps),
        eps_p, mu_p, ctypes.c_int(bc_code), ctypes.c_int(save_every),
        out.ctypes.data_as(_c_double_p))
    return out


def _solve_spectral_batch(u0, ndim, lengths, dt, steps, epsilon, mu,
                          save_every, backend):
    u0 = _as_batch(u0, ndim)
    B = u0.shape[0]
    dims = u0.shape[1:]
    eps_a, eps_p = _params(epsilon, B)
    mu_a, mu_p = _params(mu, B)
    n_save = steps // save_every + 1
    out = np.empty((B, n_save) + dims)

    if backend == 'cuda' and not cuda_available:
        raise RuntimeError("CUDA backend not built (nvcc unavailable?)")
    lib = _cuda if backend == 'cuda' else _cpu
    suffix = '_cuda' if backend == 'cuda' else ''
    fn = getattr(lib, f"ac{ndim}d_solve_batch{suffix}")

    args = ([u0.ctypes.data_as(_c_double_p), ctypes.c_int(B)]
            + [ctypes.c_int(d) for d in dims]
            + [ctypes.c_double(Ld) for Ld in lengths]
            + [ctypes.c_double(dt), ctypes.c_int(steps), eps_p, mu_p,
               ctypes.c_int(save_every), out.ctypes.data_as(_c_double_p)])
    ret = fn(*args)
    if backend == 'cuda' and ret != 0:
        raise RuntimeError(_cuda.ac_cuda_last_error().decode())
    return out


def solve_ac_2d_batch(u0, dt=0.01, steps=1500, epsilon=0.01, mu=1.0,
                      Lx=2 * np.pi, Ly=2 * np.pi, save_every=1,
                      backend='cpu'):
    """Batched 2D spectral solver."""
    return _solve_spectral_batch(u0, 2, (Ly, Lx), dt, steps, epsilon, mu,
                                 save_every, backend)


def solve_ac_3d_batch(u0, dt=0.01, steps=500, epsilon=0.01, mu=1.0,
                      Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi,
                      save_every=1, backend='cpu'):
    """Batched 3D spectral solver."""
    return _solve_spectral_batch(u0, 3, (Lz, Ly, Lx), dt, steps, epsilon, mu,
                                 save_every, backend)
