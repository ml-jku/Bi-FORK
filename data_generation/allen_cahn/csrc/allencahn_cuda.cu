// Batched Allen-Cahn 2D/3D spectral solver on the GPU (cuFFT, double
// precision). Same numerics as the CPU/reference versions; the whole batch
// is advanced in lockstep with batched FFTs.

#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::snprintf(err_msg, sizeof(err_msg), "CUDA error %s at %s:%d", \
                          cudaGetErrorString(err__), __FILE__, __LINE__);  \
            return -1;                                                     \
        }                                                                  \
    } while (0)

#define CUFFT_CHECK(call)                                                  \
    do {                                                                   \
        cufftResult err__ = (call);                                        \
        if (err__ != CUFFT_SUCCESS) {                                      \
            std::snprintf(err_msg, sizeof(err_msg), "cuFFT error %d at %s:%d", \
                          (int)err__, __FILE__, __LINE__);                 \
            return -1;                                                     \
        }                                                                  \
    } while (0)

static char err_msg[256] = "";

extern "C" const char* ac_cuda_last_error() { return err_msg; }

// w = u + dt*(mu_b*u - u^3), batched
__global__ void react_kernel(double* u, const double* mu, double dt,
                             size_t ntot, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const double m = mu[i / ntot];
    const double v = u[i];
    u[i] = v + dt * (m * v - v * v * v);
}

// uhat *= 1 / ((1 - dt*eps_b^2*lapk) * ntot), batched
__global__ void implicit_kernel(cufftDoubleComplex* uhat, const double* lapk,
                                const double* eps, double dt, double inv_ntot,
                                size_t nc, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const double e = eps[i / nc];
    const double fac = inv_ntot / (1.0 - dt * e * e * lapk[i % nc]);
    uhat[i].x *= fac;
    uhat[i].y *= fac;
}

// Shared batched spectral solver; dims ordered slowest-to-fastest
// (C-contiguous), ndim in {2, 3}. Returns 0 on success, -1 on error
// (message via ac_cuda_last_error).
static int spectral_solve_batch_cuda(const double* u0, int B, int ndim,
                                     const int* dims, const double* lengths,
                                     double dt, int steps, const double* eps,
                                     const double* mu, int save_every,
                                     double* out) {
    size_t ntot = 1;
    for (int d = 0; d < ndim; ++d) ntot *= dims[d];
    const int nx = dims[ndim - 1];
    const size_t nc = ntot / nx * (nx / 2 + 1);
    const int n_save = steps / save_every + 1;

    // -|k|^2 on the r2c grid (computed on host, copied once)
    std::vector<double> lapk_h(nc);
    {
        std::vector<std::vector<double>> k2(ndim);
        for (int d = 0; d < ndim; ++d) {
            const int n = dims[d];
            k2[d].resize(n);
            for (int i = 0; i < n; ++i) {
                const int f = (i < (n + 1) / 2) ? i : i - n;
                const double k = 2.0 * M_PI * f / lengths[d];
                k2[d][i] = k * k;
            }
        }
        const int nxh = nx / 2 + 1;
        for (size_t j = 0; j < nc; ++j) {
            size_t rem = j;
            double s = k2[ndim - 1][rem % nxh];
            rem /= nxh;
            for (int d = ndim - 2; d >= 0; --d) {
                s += k2[d][rem % dims[d]];
                rem /= dims[d];
            }
            lapk_h[j] = -s;
        }
    }

    double *d_u, *d_lapk, *d_eps, *d_mu;
    cufftDoubleComplex* d_uhat;
    CUDA_CHECK(cudaMalloc(&d_u, (size_t)B * ntot * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uhat, (size_t)B * nc * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_lapk, nc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eps, B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, B * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_u, u0, (size_t)B * ntot * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lapk, lapk_h.data(), nc * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eps, eps, B * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu, B * sizeof(double),
                          cudaMemcpyHostToDevice));

    cufftHandle fwd, bwd;
    std::vector<int> n(dims, dims + ndim);
    CUFFT_CHECK(cufftPlanMany(&fwd, ndim, n.data(), nullptr, 1, 0, nullptr, 1,
                              0, CUFFT_D2Z, B));
    CUFFT_CHECK(cufftPlanMany(&bwd, ndim, n.data(), nullptr, 1, 0, nullptr, 1,
                              0, CUFFT_Z2D, B));

    const size_t total_r = (size_t)B * ntot, total_c = (size_t)B * nc;
    const int threads = 256;
    const size_t blocks_r = (total_r + threads - 1) / threads;
    const size_t blocks_c = (total_c + threads - 1) / threads;
    const double inv_ntot = 1.0 / (double)ntot;

    // snapshot 0 = initial condition (strided copy host-side)
    for (int b = 0; b < B; ++b)
        std::memcpy(out + (size_t)b * n_save * ntot, u0 + (size_t)b * ntot,
                    ntot * sizeof(double));

    std::vector<double> buf((size_t)B * ntot);
    for (int s = 1; s <= steps; ++s) {
        react_kernel<<<blocks_r, threads>>>(d_u, d_mu, dt, ntot, total_r);
        CUFFT_CHECK(cufftExecD2Z(fwd, d_u, d_uhat));
        implicit_kernel<<<blocks_c, threads>>>(d_uhat, d_lapk, d_eps, dt,
                                               inv_ntot, nc, total_c);
        CUFFT_CHECK(cufftExecZ2D(bwd, d_uhat, d_u));
        if (s % save_every == 0) {
            CUDA_CHECK(cudaMemcpy(buf.data(), d_u, total_r * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            const size_t snap = s / save_every;
            for (int b = 0; b < B; ++b)
                std::memcpy(out + ((size_t)b * n_save + snap) * ntot,
                            buf.data() + (size_t)b * ntot,
                            ntot * sizeof(double));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cufftDestroy(fwd);
    cufftDestroy(bwd);
    cudaFree(d_u);
    cudaFree(d_uhat);
    cudaFree(d_lapk);
    cudaFree(d_eps);
    cudaFree(d_mu);
    return 0;
}

extern "C" {

int ac2d_solve_batch_cuda(const double* u0, int B, int ny, int nx, double Ly,
                          double Lx, double dt, int steps, const double* eps,
                          const double* mu, int save_every, double* out) {
    int dims[2] = {ny, nx};
    double lengths[2] = {Ly, Lx};
    return spectral_solve_batch_cuda(u0, B, 2, dims, lengths, dt, steps, eps,
                                     mu, save_every, out);
}

int ac3d_solve_batch_cuda(const double* u0, int B, int nz, int ny, int nx,
                          double Lz, double Ly, double Lx, double dt,
                          int steps, const double* eps, const double* mu,
                          int save_every, double* out) {
    int dims[3] = {nz, ny, nx};
    double lengths[3] = {Lz, Ly, Lx};
    return spectral_solve_batch_cuda(u0, B, 3, dims, lengths, dt, steps, eps,
                                     mu, save_every, out);
}

}  // extern "C"
