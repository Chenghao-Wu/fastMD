#include "pme.cuh"
#include "../core/reduce.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

#define CUFFT_CHECK(call)                                                          \
    do {                                                                           \
        cufftResult r_ = (call);                                                   \
        if (r_ != CUFFT_SUCCESS)                                                   \
            throw std::runtime_error("cuFFT error " + std::to_string((int)r_) +   \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } while (0)

namespace {

constexpr int kMaxOrder = 8;

// Cardinal B-spline weights th[0..N-1] and derivatives dth for fractional
// offset w in [0, 1); the particle contributes th[j] to grid point
// floor(u) + j (OpenMM convention; the constant shift of the stencil is
// immaterial because only |b(m)|^2 enters the influence function).
template <int N, typename T>
__host__ __device__ __forceinline__ void bspline(T w, T* th, T* dth) {
    th[N - 1] = 0;
    th[1] = w;
    th[0] = 1 - w;
    for (int j = 3; j < N; j++) {
        T div = T(1) / (j - 1);
        th[j - 1] = div * w * th[j - 2];
        for (int k = 1; k < j - 1; k++)
            th[j - k - 1] = div * ((w + k) * th[j - k - 2] + (j - k - w) * th[j - k - 1]);
        th[0] = div * (1 - w) * th[0];
    }
    dth[0] = -th[0];
    for (int j = 1; j < N; j++) dth[j] = th[j - 1] - th[j];
    T div = T(1) / (N - 1);
    th[N - 1] = div * w * th[N - 2];
    for (int k = 1; k < N - 1; k++)
        th[N - k - 1] = div * ((w + k) * th[N - k - 2] + (N - k - w) * th[N - k - 1]);
    th[0] = div * (1 - w) * th[0];
}

template <int N>
__device__ __forceinline__ void stencil(float x, float scale, int K, int& g0, float* th, float* dth) {
    float u = x * scale;             // grid coordinate in [0, K)
    int iu = __float2int_rd(u);
    float w = u - iu;
    iu %= K;
    if (iu < 0) iu += K;
    g0 = iu;
    bspline<N, float>(w, th, dth);
}

template <int N>
__global__ void pme_spread_kernel(const float4* __restrict__ pos, const float* __restrict__ q,
                                  float* __restrict__ grid, int natoms, int K,
                                  const Box* __restrict__ d_box) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    float qi = q[i];
    if (qi == 0.0f) return;
    float4 r = pos[i];
    const float scale = K * d_box->inv_L.x;
    int gx, gy, gz;
    float tx[N], ty[N], tz[N], d[N];
    stencil<N>(r.x, scale, K, gx, tx, d);
    stencil<N>(r.y, scale, K, gy, ty, d);
    stencil<N>(r.z, scale, K, gz, tz, d);
    for (int a = 0; a < N; a++) {
        int x = gx + a; x -= x >= K ? K : 0;
        for (int b = 0; b < N; b++) {
            int y = gy + b; y -= y >= K ? K : 0;
            float qxy = qi * tx[a] * ty[b];
            int base = (x * K + y) * K;
            for (int c = 0; c < N; c++) {
                int z = gz + c; z -= z >= K ? K : 0;
                atomicAdd(&grid[base + z], qxy * tz[c]);
            }
        }
    }
}

// Multiply the transformed charge grid by the influence function
// G(m) = exp(-pi^2 m^2 / alpha^2) / (pi V m^2 |b(m)|^2) and accumulate the
// reciprocal energy and virial (half-complex grid: interior z planes count
// twice).
__global__ void pme_convolve_kernel(cufftComplex* __restrict__ cgrid,
                                    const float* __restrict__ bsp_mod, int K, float alpha,
                                    float coul_const, const Box* __restrict__ d_box,
                                    double* __restrict__ energy, double* __restrict__ virial) {
    const int Kh = K / 2 + 1;
    const int total = K * K * Kh;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double e[1] = {0.0};
    double v[6] = {0, 0, 0, 0, 0, 0};
    if (idx < total) {
        int z = idx % Kh;
        int y = (idx / Kh) % K;
        int x = idx / (Kh * K);
        cufftComplex val = cgrid[idx];
        if (x == 0 && y == 0 && z == 0) {
            cgrid[idx] = make_cuFloatComplex(0.0f, 0.0f);
        } else {
            const double inv_L = d_box->inv_L.x;
            const double V = 1.0 / (inv_L * inv_L * inv_L);
            double mx = (x <= K / 2 ? x : x - K) * inv_L;
            double my = (y <= K / 2 ? y : y - K) * inv_L;
            double mz = z * inv_L;
            double m2 = mx * mx + my * my + mz * mz;
            const double pi = 3.14159265358979323846;
            double g = exp(-pi * pi * m2 / ((double)alpha * alpha)) /
                       (pi * V * m2 * bsp_mod[x] * bsp_mod[y] * bsp_mod[z]);
            double s2 = (double)val.x * val.x + (double)val.y * val.y;
            double wgt = (z == 0 || (K % 2 == 0 && z == K / 2)) ? 1.0 : 2.0;
            double ek = 0.5 * coul_const * wgt * g * s2;
            e[0] = ek;
            double f = -2.0 * (1.0 / m2 + pi * pi / ((double)alpha * alpha));
            v[0] = ek * (1.0 + f * mx * mx);
            v[1] = ek * f * mx * my;
            v[2] = ek * f * mx * mz;
            v[3] = ek * (1.0 + f * my * my);
            v[4] = ek * f * my * mz;
            v[5] = ek * (1.0 + f * mz * mz);
            cgrid[idx] = make_cuFloatComplex((float)(val.x * g), (float)(val.y * g));
        }
    }
    block_reduce_atomic_add<1, double>(e, energy);
    block_reduce_atomic_add<6, double>(v, virial);
}

// Interpolate the convolved potential psi back to the particles:
// F_i = -C q_i sum_k grad theta_i(k) psi(k), E_i = 0.5 C q_i sum_k theta_i(k) psi(k).
template <int N>
__global__ void pme_gather_kernel(const float4* __restrict__ pos, const float* __restrict__ q,
                                  const float* __restrict__ psi, float4* __restrict__ force,
                                  int natoms, int K, float coul_const,
                                  const Box* __restrict__ d_box) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    float qi = q[i];
    if (qi == 0.0f) return;
    float4 r = pos[i];
    const float scale = K * d_box->inv_L.x;
    int gx, gy, gz;
    float tx[N], ty[N], tz[N], dx[N], dy[N], dz[N];
    stencil<N>(r.x, scale, K, gx, tx, dx);
    stencil<N>(r.y, scale, K, gy, ty, dy);
    stencil<N>(r.z, scale, K, gz, tz, dz);
    float fx = 0.0f, fy = 0.0f, fz = 0.0f, ep = 0.0f;
    for (int a = 0; a < N; a++) {
        int x = gx + a; x -= x >= K ? K : 0;
        for (int b = 0; b < N; b++) {
            int y = gy + b; y -= y >= K ? K : 0;
            const float* row = psi + (x * K + y) * K;
            float txy = tx[a] * ty[b], dxy = dx[a] * ty[b], txdy = tx[a] * dy[b];
            for (int c = 0; c < N; c++) {
                int z = gz + c; z -= z >= K ? K : 0;
                float p = __ldg(&row[z]);
                ep += txy * tz[c] * p;
                fx += dxy * tz[c] * p;
                fy += txdy * tz[c] * p;
                fz += txy * dz[c] * p;
            }
        }
    }
    const float cq = coul_const * qi;
    float4 f = force[i];
    f.x -= cq * scale * fx;
    f.y -= cq * scale * fy;
    f.z -= cq * scale * fz;
    f.w += 0.5f * cq * ep;
    force[i] = f;
}

template <int N>
void launch_order(const float4* pos, const float* q, float4* force, double* virial,
                  const Box* d_box, cudaStream_t stream, PME& p) {
    const int blocks = div_ceil(p.natoms, 256);
    CUDA_CHECK(cudaMemsetAsync(p.d_grid, 0, sizeof(float) * (size_t)p.K * p.K * p.K, stream));
    CUDA_CHECK(cudaMemsetAsync(p.d_energy, 0, sizeof(double), stream));
    pme_spread_kernel<N><<<blocks, 256, 0, stream>>>(pos, q, p.d_grid, p.natoms, p.K, d_box);
    CUFFT_CHECK(cufftExecR2C(p.plan_r2c, p.d_grid, p.d_cgrid));
    const int ctotal = p.K * p.K * (p.K / 2 + 1);
    pme_convolve_kernel<<<div_ceil(ctotal, 256), 256, 0, stream>>>(
        p.d_cgrid, p.d_bsp_mod, p.K, p.alpha, p.coul_const, d_box, p.d_energy, virial);
    CUFFT_CHECK(cufftExecC2R(p.plan_c2r, p.d_cgrid, p.d_grid));
    pme_gather_kernel<N><<<blocks, 256, 0, stream>>>(pos, q, p.d_grid, force, p.natoms, p.K,
                                                     p.coul_const, d_box);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int pme_fft_friendly(int n) {
    for (int k = std::max(n, 1);; k++) {
        int m = k;
        for (int p : {2, 3, 5, 7})
            while (m % p == 0) m /= p;
        if (m == 1) return k;
    }
}

float ewald_alpha_for_tolerance(float tol, float rc) {
    return std::sqrt(-std::log(2.0 * tol)) / rc;
}

int pme_grid_for_tolerance(float tol, float alpha, float L) {
    int k = (int)std::ceil(2.0 * alpha * L / (3.0 * std::pow((double)tol, 0.2)));
    return pme_fft_friendly(std::max(k, 8));
}

int ewald_kmax_for_tolerance(float tol, float alpha, float L) {
    return std::max(1, (int)std::ceil(alpha * L * std::sqrt(-std::log((double)tol)) / M_PI));
}

void PME::allocate(int n, int K_in, int order_in, float alpha_in, float coul) {
    if (order_in < 3 || order_in > kMaxOrder)
        throw std::runtime_error("PME order must be in [3, " + std::to_string(kMaxOrder) + "]");
    if (K_in < order_in) throw std::runtime_error("PME grid must have at least `order` points");
    natoms = n;
    K = K_in;
    order = order_in;
    alpha = alpha_in;
    coul_const = coul;
    const size_t nreal = (size_t)K * K * K, ncomplex = (size_t)K * K * (K / 2 + 1);
    CUDA_CHECK(cudaMalloc(&d_grid, nreal * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cgrid, ncomplex * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_bsp_mod, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));
    CUFFT_CHECK(cufftPlan3d(&plan_r2c, K, K, K, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&plan_c2r, K, K, K, CUFFT_C2R));

    // B-spline moduli |b(m)|^-2 (Essmann eq. 4.4), identical in x, y, z.
    std::vector<double> th(kMaxOrder), dth(kMaxOrder);
    switch (order) {
        case 3: bspline<3, double>(0.0, th.data(), dth.data()); break;
        case 4: bspline<4, double>(0.0, th.data(), dth.data()); break;
        case 5: bspline<5, double>(0.0, th.data(), dth.data()); break;
        case 6: bspline<6, double>(0.0, th.data(), dth.data()); break;
        case 7: bspline<7, double>(0.0, th.data(), dth.data()); break;
        default: bspline<8, double>(0.0, th.data(), dth.data()); break;
    }
    std::vector<double> mod(K);
    for (int m = 0; m < K; m++) {
        double sc = 0, ss = 0;
        for (int j = 0; j < order; j++) {
            double arg = 2.0 * M_PI * m * j / K;
            sc += th[j] * std::cos(arg);
            ss += th[j] * std::sin(arg);
        }
        mod[m] = sc * sc + ss * ss;
    }
    for (int m = 0; m < K; m++)   // odd orders vanish at m = K/2: interpolate
        if (mod[m] < 1e-7) mod[m] = 0.5 * (mod[(m - 1 + K) % K] + mod[(m + 1) % K]);
    std::vector<float> modf(mod.begin(), mod.end());
    CUDA_CHECK(cudaMemcpy(d_bsp_mod, modf.data(), K * sizeof(float), cudaMemcpyHostToDevice));
}

void PME::free() {
    if (plan_r2c) cufftDestroy(plan_r2c);
    if (plan_c2r) cufftDestroy(plan_c2r);
    for (void* p : {(void*)d_grid, (void*)d_cgrid, (void*)d_bsp_mod, (void*)d_energy})
        if (p) CUDA_CHECK(cudaFree(p));
    *this = PME{};
}

void PME::compute(const float4* pos, const float* charges, float4* force, double* virial,
                  const Box* d_box, cudaStream_t stream) {
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));
    switch (order) {
        case 3: launch_order<3>(pos, charges, force, virial, d_box, stream, *this); break;
        case 4: launch_order<4>(pos, charges, force, virial, d_box, stream, *this); break;
        case 5: launch_order<5>(pos, charges, force, virial, d_box, stream, *this); break;
        case 6: launch_order<6>(pos, charges, force, virial, d_box, stream, *this); break;
        case 7: launch_order<7>(pos, charges, force, virial, d_box, stream, *this); break;
        default: launch_order<8>(pos, charges, force, virial, d_box, stream, *this); break;
    }
}

double PME::last_energy(cudaStream_t stream) const {
    double e = 0;
    CUDA_CHECK(cudaMemcpyAsync(&e, d_energy, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return e;
}
