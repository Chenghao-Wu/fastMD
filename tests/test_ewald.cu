#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/lj_coul_long.cuh"
#include "force/ewald_recip.cuh"
#include "force/ewald_excl.cuh"
#include "neighbor/verlet_list.cuh"
#include "../reference/cpu_reference.hpp"
#include <cstdlib>

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
    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
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
                              d_ex13_off, d_ex13_list,
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
