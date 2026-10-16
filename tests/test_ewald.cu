#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/lj_coul_long.cuh"
#include "force/ewald_recip.cuh"
#include "force/ewald_excl.cuh"
#include "neighbor/verlet_list.cuh"
#include "../reference/cpu_reference.hpp"
#include <cstdlib>
#include <random>
#include <cmath>

// 6-atom charged cluster with one bond chain 0-1-2-3-4 (gives 1-2, 1-3,
// 1-4 relationships), plus an isolated atom 5. All Ewald components
// (real, reciprocal, erf subtraction, scaled 1-4, self) are exercised.
TEST(EwaldOPLS, SmallClusterMatchesBruteForce) {
    const float L = 2.5f;            // nm
    const float inv_L = 1.0f / L;
    const float alpha = 3.0f;        // 1/nm
    const int kmax = 4;
    const float coul_const = 138.9354576f;
    const float rc = 1.1f;           // nm
    const float lj14 = 0.5f, coul14 = 0.5f;

    std::vector<float4> h_pos = {
        {0.50f, 0.50f, 0.50f, pack_type_id(0)},
        {0.65f, 0.50f, 0.50f, pack_type_id(0)},
        {0.80f, 0.55f, 0.50f, pack_type_id(1)},
        {0.95f, 0.52f, 0.55f, pack_type_id(0)},
        {1.10f, 0.58f, 0.52f, pack_type_id(1)},
        {1.80f, 1.80f, 1.80f, pack_type_id(1)},
    };
    const int N = h_pos.size();
    std::vector<float> h_q = {0.3f, -0.2f, 0.4f, -0.3f, 0.2f, -0.4f};

    // bonds 0-1,1-2,2-3,3-4  =>  1-4 pairs (0,3) and (1,4)
    std::vector<int2> h_bonds = {{0,1},{1,2},{2,3},{3,4}};

    // build excl13 (1-2,1-3) CSR and pairs14 like build_exclusions_opls
    std::vector<std::vector<int>> adj(N);
    for (auto& b : h_bonds) { adj[b.x].push_back(b.y); adj[b.y].push_back(b.x); }
    std::vector<int> ex13_off(N + 1, 0), ex13_list;
    for (int i = 0; i < N; i++) {
        std::vector<int> e = adj[i];
        for (int j : adj[i]) for (int k : adj[j]) if (k != i) e.push_back(k);
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());
        ex13_off[i] = (int)ex13_list.size();
        ex13_list.insert(ex13_list.end(), e.begin(), e.end());
    }
    ex13_off[N] = (int)ex13_list.size();
    std::vector<int2> h_p14 = { {0,3}, {1,4} };

    // full exclusion CSR for the verlet build: 1-2,1-3,1-4
    std::vector<int> ex_off(N + 1, 0), ex_list;
    for (int i = 0; i < N; i++) {
        std::vector<int> e;
        for (int ee = ex13_off[i]; ee < ex13_off[i + 1]; ee++) e.push_back(ex13_list[ee]);
        for (auto& p : h_p14) {
            if (p.x == i) e.push_back(p.y);
            else if (p.y == i) e.push_back(p.x);
        }
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());
        ex_off[i] = (int)ex_list.size();
        ex_list.insert(ex_list.end(), e.begin(), e.end());
    }
    ex_off[N] = (int)ex_list.size();

    // LJ params (2 types), geometric mixing
    std::vector<float2> lj(4);
    float eps0 = 0.5f, sig0 = 0.30f, eps1 = 0.3f, sig1 = 0.35f;
    lj[0] = {eps0, sig0};
    lj[3] = {eps1, sig1};
    lj[1] = lj[2] = {sqrtf(eps0 * eps1), sqrtf(sig0 * sig1)};
    int ntypes = 2;

    // --- CPU reference (double precision) ---
    auto ref = ref::brute_force_opls_full(h_pos, h_q, lj, ntypes,
                                           coul_const, alpha, kmax,
                                           ex13_off, ex13_list, h_p14,
                                           lj14, coul14, rc, L, inv_L);

    // --- GPU ---
    int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> h_pos_pad(np, make_float4(0,0,0,pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());
    std::vector<float> h_q_pad(np, 0.0f);
    std::copy(h_q.begin(), h_q.end(), h_q_pad.begin());

    float4* d_pos = to_device(h_pos_pad);
    float* d_q = to_device(h_q_pad);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    double* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(double)));
    float2* d_lj = to_device(lj);

    // verlet neighbor list (skin 0.2)
    int* d_ex_off = to_device(ex_off);
    int* d_ex_list = to_device(ex_list);
    VerletList verlet;
    verlet.allocate(N, rc + 0.2f, L);
    verlet.build(d_pos, N, L, inv_L, d_ex_off, d_ex_list, rc + 0.2f);

    // 1. fused LJ + real-space erfc (overwrites force)
    launch_lj_coul_long_kernel(d_pos, d_q, d_force, d_virial, d_lj,
                                verlet.neighbors, verlet.num_neighbors,
                                N, ntypes, coul_const, alpha, rc * rc, L, inv_L);

    // 2. reciprocal space
    std::vector<int3> kvecs = ewald_build_kvectors(kmax, L, alpha);
    int kcount = (int)kvecs.size();
    int3* d_kvecs = to_device(kvecs);
    float* d_sfac;
    CUDA_CHECK(cudaMalloc(&d_sfac, 2 * kcount * sizeof(float)));
    launch_ewald_sf_kernel(d_pos, d_q, d_kvecs, d_sfac, N, kcount, L);
    launch_ewald_recip_kernel(d_pos, d_q, d_kvecs, d_sfac, d_force, d_virial,
                               N, kcount, coul_const, alpha, L);

    // 3. erf subtraction on 1-2/1-3
    int* d_ex13_off = to_device(ex13_off);
    int* d_ex13_list = to_device(ex13_list);
    launch_ewald_excl_kernel(d_pos, d_q, d_force, d_virial,
                              d_ex13_off, d_ex13_list, nullptr,
                              N, (int)ex13_list.size(),
                              coul_const, alpha, L, inv_L);

    // 4. scaled 1-4 pairs
    int2* d_p14 = to_device(h_p14);
    launch_pair14_kernel(d_pos, d_q, d_force, d_virial, d_p14, d_lj,
                          (int)h_p14.size(), ntypes,
                          coul_const, alpha, lj14, coul14, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- compare forces ---
    auto h_force_out = to_host(d_force, N);
    double max_rel = 0;
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            double g = (d == 0) ? h_force_out[i].x : (d == 1) ? h_force_out[i].y : h_force_out[i].z;
            double r = (d == 0) ? ref.fx[i] : (d == 1) ? ref.fy[i] : ref.fz[i];
            double denom = fmax(fabs(g), fabs(r));
            double rel = (denom < 1e-6) ? fabs(g - r) : fabs(g - r) / denom;
            max_rel = fmax(max_rel, rel);
            EXPECT_LT(rel, 2e-3f) << "atom " << i << " dim " << d
                                  << " gpu=" << g << " ref=" << r;
        }
    }

    // --- compare PE (GPU per-atom sum + self offset) ---
    double pe_gpu = 0;
    for (int i = 0; i < N; i++) pe_gpu += h_force_out[i].w;
    double self = ewald_self_offset(h_q, coul_const, alpha, L);
    pe_gpu += self;
    EXPECT_NEAR(pe_gpu, ref.pe, fmax(2e-3 * fabs(ref.pe), 1e-2))
        << "gpu=" << pe_gpu << " ref=" << ref.pe;

    // --- compare virial diagonal (off-diagonals not accumulated in recip kernel) ---
    auto h_vir = to_host(d_virial, 6);
    for (int c : {0, 3, 5}) {
        double denom = fmax(fabs((double)h_vir[c]), fabs(ref.virial[c]));
        double rel = (denom < 1e-6) ? fabs(h_vir[c] - ref.virial[c])
                                    : fabs(h_vir[c] - ref.virial[c]) / denom;
        EXPECT_LT(rel, 5e-3f) << "virial[" << c << "] gpu=" << h_vir[c]
                              << " ref=" << ref.virial[c];
    }

    free_device(d_pos); free_device(d_q); free_device(d_force);
    free_device(d_virial); free_device(d_lj);
    free_device(d_ex_off); free_device(d_ex_list);
    free_device(d_kvecs); free_device(d_sfac);
    free_device(d_ex13_off); free_device(d_ex13_list);
    free_device(d_p14);
    verlet.free();
}

namespace {
// fp64 reciprocal energy under a homogeneous deformation F of the box and
// particles: k' = F^-T k, V' = det(F) V, and k'.r' = k.r (S(k) invariant).
double recip_energy(const std::vector<float4>& pos, const std::vector<float>& q,
                    const std::vector<int3>& kv, double L, double alpha, double C,
                    const double F[3][3]) {
    double det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
               - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
               + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
    double inv[3][3];   // F^-1
    inv[0][0] = (F[1][1] * F[2][2] - F[1][2] * F[2][1]) / det;
    inv[0][1] = (F[0][2] * F[2][1] - F[0][1] * F[2][2]) / det;
    inv[0][2] = (F[0][1] * F[1][2] - F[0][2] * F[1][1]) / det;
    inv[1][0] = (F[1][2] * F[2][0] - F[1][0] * F[2][2]) / det;
    inv[1][1] = (F[0][0] * F[2][2] - F[0][2] * F[2][0]) / det;
    inv[1][2] = (F[0][2] * F[1][0] - F[0][0] * F[1][2]) / det;
    inv[2][0] = (F[1][0] * F[2][1] - F[1][1] * F[2][0]) / det;
    inv[2][1] = (F[0][1] * F[2][0] - F[0][0] * F[2][1]) / det;
    inv[2][2] = (F[0][0] * F[1][1] - F[0][1] * F[1][0]) / det;
    double V = det * L * L * L;
    double E = 0.0;
    for (const int3& n : kv) {
        double k0[3] = {2 * M_PI * n.x / L, 2 * M_PI * n.y / L, 2 * M_PI * n.z / L};
        double kp[3];
        for (int a = 0; a < 3; a++)   // F^-T k: (F^-T)_ab = inv[b][a]
            kp[a] = inv[0][a] * k0[0] + inv[1][a] * k0[1] + inv[2][a] * k0[2];
        double k2 = kp[0] * kp[0] + kp[1] * kp[1] + kp[2] * kp[2];
        double re = 0, im = 0;
        for (size_t j = 0; j < pos.size(); j++) {
            double ph = k0[0] * pos[j].x + k0[1] * pos[j].y + k0[2] * pos[j].z;
            re += q[j] * std::cos(ph);
            im += q[j] * std::sin(ph);
        }
        // half-shell sum: each +/-k pair counted once with weight 4 pi / V
        E += C * (4 * M_PI / V) * std::exp(-k2 / (4 * alpha * alpha)) / k2 * (re * re + im * im);
    }
    return E;
}
}  // namespace

// All six reciprocal-space virial components equal -dE_recip/dF_ab.
// Previously only the diagonal was accumulated; Pxy/Pxz/Pyz missed the
// k-space part although it is non-zero for any configuration.
TEST(EwaldOPLS, ReciprocalVirialMatchesStrainDerivative) {
    const float L = 2.5f, alpha = 3.0f, C = 138.9354576f;
    const int kmax = 5, N = 37;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(0.0f, L);
    std::vector<float4> pos(N);
    std::vector<float> q(N);
    for (int i = 0; i < N; i++) {
        pos[i] = make_float4(u(rng), u(rng), u(rng), pack_type_id(0));
        q[i] = (i % 2 ? 0.4f : -0.4f) * (1.0f + 0.1f * (i % 3));
    }
    double qsum = 0;
    for (float v : q) qsum += v;
    q[N - 1] -= (float)qsum;   // neutral

    std::vector<int3> kv = ewald_build_kvectors(kmax, L, alpha);
    int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> pos_pad(np, make_float4(0, 0, 0, pack_type_id(-1)));
    std::copy(pos.begin(), pos.end(), pos_pad.begin());
    std::vector<float> q_pad(np, 0.0f);
    std::copy(q.begin(), q.end(), q_pad.begin());
    float4* d_pos = to_device(pos_pad);
    float* d_q = to_device(q_pad);
    int3* d_kv = to_device(kv);
    float* d_sfac;
    CUDA_CHECK(cudaMalloc(&d_sfac, 2 * kv.size() * sizeof(float)));
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    double* d_vir;
    CUDA_CHECK(cudaMalloc(&d_vir, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vir, 0, 6 * sizeof(double)));
    launch_ewald_sf_kernel(d_pos, d_q, d_kv, d_sfac, N, (int)kv.size(), L);
    launch_ewald_recip_kernel(d_pos, d_q, d_kv, d_sfac, d_force, d_vir, N, (int)kv.size(), C, alpha, L);
    CUDA_CHECK(cudaDeviceSynchronize());
    double vir[6];
    CUDA_CHECK(cudaMemcpy(vir, d_vir, sizeof(vir), cudaMemcpyDeviceToHost));

    const int comp[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};  // xx xy xz yy yz zz
    double scale = 0;
    double ref[6];
    for (int c = 0; c < 6; c++) {
        const double h = 1e-6;
        double Fp[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, Fm[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        Fp[comp[c][0]][comp[c][1]] += h;
        Fm[comp[c][0]][comp[c][1]] -= h;
        ref[c] = -(recip_energy(pos, q, kv, L, alpha, C, Fp) -
                   recip_energy(pos, q, kv, L, alpha, C, Fm)) / (2 * h);
        scale = std::max(scale, std::fabs(ref[c]));
    }
    for (int c = 0; c < 6; c++)
        EXPECT_NEAR(vir[c], ref[c], 2e-4 * scale) << "virial component " << c;
    // the configuration is generic: shear components must be non-trivial
    EXPECT_GT(std::fabs(ref[1]) + std::fabs(ref[2]) + std::fabs(ref[4]), 1e-3 * scale);

    free_device(d_pos); free_device(d_q); free_device(d_kv);
    free_device(d_sfac); free_device(d_force); free_device(d_vir);
}
