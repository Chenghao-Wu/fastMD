#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/lj.cuh"
#include "neighbor/verlet_list.cuh"
#include "../reference/cpu_reference.hpp"

TEST(LJ, SmallSystemMatchesCPU) {
    const int N = 64;
    const float L = 8.0f;
    const float inv_L = 1.0f / L;
    const float rc = 2.5f;
    const float rc2 = rc * rc;
    const float skin = 0.5f;
    const int ntypes = 1;
    const int ntiles = div_ceil(N, TILE_SIZE);
    const int np = ntiles * TILE_SIZE;

    std::vector<float4> h_pos(N);
    srand(123);
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            pack_type_id(0)
        );
    }

    std::vector<float2> h_lj = {{1.0f, 1.0f}};

    auto ref = ref::brute_force_lj(h_pos, h_lj, ntypes, rc, L, inv_L);

    std::vector<float4> h_pos_pad(np, make_float4(0,0,0, pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());

    float4* d_pos = to_device(h_pos_pad);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));

    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));

    float2* d_lj = to_device(h_lj);

    VerletList verlet;
    verlet.allocate(N, rc + skin, L);
    verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);

    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                      verlet.neighbors, verlet.num_neighbors,
                      N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_force = to_host(d_force, N);
    for (int i = 0; i < N; i++) {
        float3 gpu_f = make_float3(h_force[i].x, h_force[i].y, h_force[i].z);
        float3 ref_f = make_float3(ref.fx[i], ref.fy[i], ref.fz[i]);
        assert_float3_near(gpu_f, ref_f, 1e-3f);
    }

    float h_virial[6];
    CUDA_CHECK(cudaMemcpy(h_virial, d_virial, 6 * sizeof(float),
                           cudaMemcpyDeviceToHost));
    for (int k = 0; k < 6; k++) {
        float denom = fmaxf(fabsf(ref.virial[k]), 1e-6f);
        float rel = fabsf(h_virial[k] - ref.virial[k]) / denom;
        EXPECT_LT(rel, 1e-2f) << "virial component " << k;
    }

    verlet.free();
    free_device(d_pos);
    free_device(d_force);
    free_device(d_virial);
    free_device(d_lj);
}

// Optional force cap: F = -dU/dr must hold everywhere (the old clamp kept
// the LJ energy while capping the force: at r = 0.75 sigma it returned 750
// against -dU/dr = 1840.6).
TEST(LJ, ForceCapIsEnergyConsistent) {
    const float L = 10.0f, rc = 2.5f, Fcap = 750.0f;
    std::vector<float2> lj = {{1.0f, 1.0f}};
    auto cap = lj_force_cap_table(lj, 1, Fcap);
    ASSERT_GT(cap[0].x, 0.7f);
    ASSERT_LT(cap[0].x, 1.0f);
    float2* d_lj = to_device(lj);
    float2* d_cap = to_device(cap);
    auto eval = [&](float r, float* energy) {
        std::vector<float4> pos(32, make_float4(0, 0, 0, pack_type_id(-1)));
        pos[0] = make_float4(5.0f, 5.0f, 5.0f, pack_type_id(0));
        pos[1] = make_float4(5.0f + r, 5.0f, 5.0f, pack_type_id(0));
        float4* d_pos = to_device(pos);
        float4* d_f;
        float* d_v;
        CUDA_CHECK(cudaMalloc(&d_f, 32 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_v, 6 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v, 0, 6 * sizeof(float)));
        VerletList v;
        v.allocate(2, rc + 0.3f, L);
        v.build(d_pos, 2, L, 1.0f / L, nullptr, nullptr, rc + 0.3f);
        launch_lj_kernel(d_pos, d_f, d_v, d_lj, v.neighbors, v.num_neighbors, 2, 1, rc * rc,
                         L, 1.0f / L, 0, d_cap, Fcap);
        auto f = to_host(d_f, 2);
        *energy = f[0].w + f[1].w;
        v.free();
        free_device(d_pos); free_device(d_f); free_device(d_v);
        return f[1].x;   // particle 1 sits at +r: radial force, positive = repulsive
    };
    for (float r : {0.6f, 0.75f, 0.85f, 0.95f, 1.05f, 1.3f}) {
        float e0, ep, em;
        float F = eval(r, &e0);
        const float h = 1e-3f;
        eval(r + h, &ep);
        eval(r - h, &em);
        float dUdr = (ep - em) / (2 * h);
        EXPECT_NEAR(F, -dUdr, 2e-3f * std::max(1.0f, std::fabs(F))) << "r = " << r;
        if (r < cap[0].x) EXPECT_NEAR(F, Fcap, 1e-2f) << "r = " << r;
    }
    free_device(d_lj);
    free_device(d_cap);
}
