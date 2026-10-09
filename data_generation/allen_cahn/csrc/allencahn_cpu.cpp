// Batched Allen-Cahn solvers (CPU), exposed via extern "C" for ctypes.
//
// Numerics are identical to the reference Python implementation in
// allen_cahn.py:
//   1D: semi-implicit finite differences, tridiagonal solve
//       (Thomas algorithm; Sherman-Morrison correction for periodic BCs)
//   2D/3D: semi-implicit Fourier spectral (FFTW, real-to-complex)
//
// All arithmetic in double precision. Each sample in a batch has its own
// (epsilon, mu); batches are parallelized with OpenMP.

#include <cmath>
#include <cstring>
#include <vector>
#include <fftw3.h>
#include <omp.h>

namespace {

// Tridiagonal factorization (Thomas algorithm) for constant coefficients
// sub = a, diag = b (b[0], b[n-1] may differ), super = c.
struct Tridiag {
    std::vector<double> w, bp;  // elimination multipliers, modified diagonal
    double a, c;
    int n;

    Tridiag(int n_, double a_, const std::vector<double>& b, double c_)
        : w(n_), bp(n_), a(a_), c(c_), n(n_) {
        bp[0] = b[0];
        for (int i = 1; i < n; ++i) {
            w[i] = a / bp[i - 1];
            bp[i] = b[i] - w[i] * c;
        }
    }

    void solve(const double* d, double* x) const {
        x[0] = d[0];
        for (int i = 1; i < n; ++i) x[i] = d[i] - w[i] * x[i - 1];
        x[n - 1] /= bp[n - 1];
        for (int i = n - 2; i >= 0; --i) x[i] = (x[i] - c * x[i + 1]) / bp[i];
    }
};

}  // namespace

extern "C" {

// 1D batch solver.
//   u0:  (B, N) initial conditions
//   eps, mu: (B,) parameters
//   bc:  0 = periodic, 1 = Neumann
//   out: (B, steps/save_every + 1, N), snapshot 0 = u0
void ac1d_solve_batch(const double* u0, int B, int N, double L, double dt,
                      int steps, const double* eps, const double* mu, int bc,
                      int save_every, double* out) {
    const double dx = L / N;
    const int n_save = steps / save_every + 1;

#pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; ++b) {
        const double r = dt * eps[b] * eps[b] / (dx * dx);
        // A = I - dt*eps^2/dx^2 * Lap: diag 1+2r, off-diag -r
        std::vector<double> diag(N, 1.0 + 2.0 * r);
        double sub = -r, sup = -r;

        // Periodic corners A[0][N-1] = A[N-1][0] = -r handled via
        // Sherman-Morrison: A = T + p q^T with p = (gamma,0,..,beta)^T,
        // q = (1,0,..,alpha/gamma)^T, alpha = beta = -r.
        const double alpha = -r, beta = -r, gamma = -diag[0];
        std::vector<double> z(N);
        if (bc == 0) {
            diag[0] -= gamma;
            diag[N - 1] -= alpha * beta / gamma;
        }

        std::vector<double> rhs(N), x(N);
        const double* ic = u0 + (size_t)b * N;
        std::vector<double> u(ic, ic + N);

        double* o = out + (size_t)b * n_save * N;
        std::memcpy(o, u.data(), N * sizeof(double));

        if (bc == 0) {
            Tridiag T(N, sub, diag, sup);
            // z = T^{-1} p is constant across time steps
            std::vector<double> p(N, 0.0);
            p[0] = gamma;
            p[N - 1] = beta;
            T.solve(p.data(), z.data());
            const double zden = 1.0 + z[0] + alpha * z[N - 1] / gamma;

            for (int n = 1; n <= steps; ++n) {
                for (int i = 0; i < N; ++i)
                    rhs[i] = u[i] - dt * (u[i] * u[i] * u[i] - mu[b] * u[i]);
                T.solve(rhs.data(), x.data());
                const double fact =
                    (x[0] + alpha * x[N - 1] / gamma) / zden;
                for (int i = 0; i < N; ++i) u[i] = x[i] - fact * z[i];
                if (n % save_every == 0)
                    std::memcpy(o + (size_t)(n / save_every) * N, u.data(),
                                N * sizeof(double));
            }
        } else {
            // Neumann: general Thomas with per-row super/sub for rows 0, N-1
            // A[0][1] = A[N-1][N-2] = -2r; interior off-diagonals -r.
            std::vector<double> a(N, -r), c(N, -r), w(N), bp(N);
            c[0] = -2.0 * r;
            a[N - 1] = -2.0 * r;
            bp[0] = diag[0];
            for (int i = 1; i < N; ++i) {
                w[i] = a[i] / bp[i - 1];
                bp[i] = diag[i] - w[i] * c[i - 1];
            }
            for (int n = 1; n <= steps; ++n) {
                for (int i = 0; i < N; ++i)
                    rhs[i] = u[i] - dt * (u[i] * u[i] * u[i] - mu[b] * u[i]);
                x[0] = rhs[0];
                for (int i = 1; i < N; ++i) x[i] = rhs[i] - w[i] * x[i - 1];
                x[N - 1] /= bp[N - 1];
                for (int i = N - 2; i >= 0; --i)
                    x[i] = (x[i] - c[i] * x[i + 1]) / bp[i];
                u = x;
                if (n % save_every == 0)
                    std::memcpy(o + (size_t)(n / save_every) * N, u.data(),
                                N * sizeof(double));
            }
        }
    }
}

// Shared spectral stepper for 2D/3D. dims/lengths are ordered slowest-to-
// fastest axis (matching C-contiguous numpy arrays), e.g. (ny, nx) or
// (nz, ny, nx).
static void spectral_solve_batch(const double* u0, int B, int ndim,
                                 const int* dims, const double* lengths,
                                 double dt, int steps, const double* eps,
                                 const double* mu, int save_every,
                                 double* out) {
    size_t ntot = 1;
    for (int d = 0; d < ndim; ++d) ntot *= dims[d];
    const int nx = dims[ndim - 1];
    const size_t nc = ntot / nx * (nx / 2 + 1);  // r2c output size
    const int n_save = steps / save_every + 1;

    // -|k|^2 on the r2c grid
    std::vector<double> lapk(nc);
    {
        std::vector<std::vector<double>> k2(ndim);
        for (int d = 0; d < ndim; ++d) {
            const int n = dims[d];
            k2[d].resize(n);
            for (int i = 0; i < n; ++i) {
                const int f = (i < (n + 1) / 2) ? i : i - n;  // np.fft.fftfreq
                double k = 2.0 * M_PI * f / lengths[d];
                k2[d][i] = k * k;
            }
        }
        const int nxh = nx / 2 + 1;
        for (size_t j = 0; j < nc; ++j) {
            size_t rem = j;
            const int ix = rem % nxh;
            rem /= nxh;
            double s = k2[ndim - 1][ix];
            for (int d = ndim - 2; d >= 0; --d) {
                s += k2[d][rem % dims[d]];
                rem /= dims[d];
            }
            lapk[j] = -s;
        }
    }

    // Plans are created once on scratch buffers and reused with the
    // new-array execute API (all buffers come from fftw_malloc -> same
    // alignment). Plan creation is not thread-safe; execution is.
    double* scratch_r = fftw_alloc_real(ntot);
    fftw_complex* scratch_c = fftw_alloc_complex(nc);
    fftw_plan fwd = fftw_plan_dft_r2c(ndim, dims, scratch_r, scratch_c,
                                      FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_c2r(ndim, dims, scratch_c, scratch_r,
                                      FFTW_ESTIMATE);

#pragma omp parallel
    {
        double* u = fftw_alloc_real(ntot);
        fftw_complex* uhat = fftw_alloc_complex(nc);
        std::vector<double> fac(nc);

#pragma omp for schedule(dynamic)
        for (int b = 0; b < B; ++b) {
            const double e2 = eps[b] * eps[b], m = mu[b];
            for (size_t j = 0; j < nc; ++j)
                fac[j] = 1.0 / ((1.0 - dt * e2 * lapk[j]) * (double)ntot);

            std::memcpy(u, u0 + (size_t)b * ntot, ntot * sizeof(double));
            double* o = out + (size_t)b * n_save * ntot;
            std::memcpy(o, u, ntot * sizeof(double));

            for (int n = 1; n <= steps; ++n) {
                for (size_t i = 0; i < ntot; ++i)
                    u[i] += dt * (m * u[i] - u[i] * u[i] * u[i]);
                fftw_execute_dft_r2c(fwd, u, uhat);
                for (size_t j = 0; j < nc; ++j) {
                    uhat[j][0] *= fac[j];
                    uhat[j][1] *= fac[j];
                }
                fftw_execute_dft_c2r(bwd, uhat, u);
                if (n % save_every == 0)
                    std::memcpy(o + (size_t)(n / save_every) * ntot, u,
                                ntot * sizeof(double));
            }
        }
        fftw_free(u);
        fftw_free(uhat);
    }

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(scratch_r);
    fftw_free(scratch_c);
}

// 2D batch solver. u0: (B, ny, nx); out: (B, n_save, ny, nx).
void ac2d_solve_batch(const double* u0, int B, int ny, int nx, double Ly,
                      double Lx, double dt, int steps, const double* eps,
                      const double* mu, int save_every, double* out) {
    int dims[2] = {ny, nx};
    double lengths[2] = {Ly, Lx};
    spectral_solve_batch(u0, B, 2, dims, lengths, dt, steps, eps, mu,
                         save_every, out);
}

// 3D batch solver. u0: (B, nz, ny, nx); out: (B, n_save, nz, ny, nx).
void ac3d_solve_batch(const double* u0, int B, int nz, int ny, int nx,
                      double Lz, double Ly, double Lx, double dt, int steps,
                      const double* eps, const double* mu, int save_every,
                      double* out) {
    int dims[3] = {nz, ny, nx};
    double lengths[3] = {Lz, Ly, Lx};
    spectral_solve_batch(u0, B, 3, dims, lengths, dt, steps, eps, mu,
                         save_every, out);
}

}  // extern "C"
