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

    double* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(double)));

    float2* d_lj = to_device(h_lj);

    VerletList verlet;
    verlet.allocate(N, rc + skin, cubic_box(L));
    verlet.build(d_pos, N, cubic_box(L), nullptr, nullptr, rc + skin);

    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                      verlet.neighbors, verlet.num_neighbors,
                      N, ntypes, rc2, cubic_box(L));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_force = to_host(d_force, N);
    for (int i = 0; i < N; i++) {
        float3 gpu_f = make_float3(h_force[i].x, h_force[i].y, h_force[i].z);
        float3 ref_f = make_float3(ref.fx[i], ref.fy[i], ref.fz[i]);
        assert_float3_near(gpu_f, ref_f, 1e-3f);
    }

    double h_virial[6];
    CUDA_CHECK(cudaMemcpy(h_virial, d_virial, 6 * sizeof(double),
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
        double* d_v;
        CUDA_CHECK(cudaMalloc(&d_f, 32 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_v, 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_v, 0, 6 * sizeof(double)));
        VerletList v;
        v.allocate(2, rc + 0.3f, cubic_box(L));
        v.build(d_pos, 2, cubic_box(L), nullptr, nullptr, rc + 0.3f);
        launch_lj_kernel(d_pos, d_f, d_v, d_lj, v.neighbors, v.num_neighbors, 2, 1, rc * rc,
                         cubic_box(L), 0, d_cap, Fcap);
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

// pair_shift: U(r) - U(rc) for every pair (forces unchanged); the forces-only
// variant gives the same forces and leaves force.w = 0.
TEST(LJ, EnergyShiftAndForcesOnlyVariant) {
    const float L = 10.0f, rc = 2.5f;
    std::vector<float2> lj = {{1.3f, 1.1f}};
    float2* d_lj = to_device(lj);
    auto eval = [&](float r, bool ev, bool shift, float* energy) {
        std::vector<float4> pos(32, make_float4(0, 0, 0, pack_type_id(-1)));
        pos[0] = make_float4(5.0f, 5.0f, 5.0f, pack_type_id(0));
        pos[1] = make_float4(5.0f + r, 5.0f, 5.0f, pack_type_id(0));
        float4* d_pos = to_device(pos);
        float4* d_f;
        double* d_v;
        CUDA_CHECK(cudaMalloc(&d_f, 32 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_v, 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_v, 0, 6 * sizeof(double)));
        VerletList v;
        v.allocate(2, rc + 0.3f, cubic_box(L));
        v.build(d_pos, 2, cubic_box(L), nullptr, nullptr, rc + 0.3f);
        launch_lj_kernel(d_pos, d_f, d_v, d_lj, v.neighbors, v.num_neighbors, 2, 1, rc * rc,
                         cubic_box(L), 0, nullptr, 0.0f, 1, nullptr, ev, shift);
        auto f = to_host(d_f, 2);
        *energy = f[0].w + f[1].w;
        v.free();
        free_device(d_pos); free_device(d_f); free_device(d_v);
        return f[1].x;
    };
    auto U = [](double r) { double s6 = std::pow(1.1 / r, 6); return 4.0 * 1.3 * (s6 * s6 - s6); };
    for (float r : {1.0f, 1.4f, 2.0f, 2.45f}) {
        float e_plain, e_shift, e_none;
        float f_plain = eval(r, true, false, &e_plain);
        float f_shift = eval(r, true, true, &e_shift);
        float f_none = eval(r, false, true, &e_none);
        EXPECT_NEAR(e_plain, U(r), 1e-4 * std::max(1.0, std::fabs(U(r))));
        EXPECT_NEAR(e_shift, U(r) - U(rc), 1e-4 * std::max(1.0, std::fabs(U(r))));
        EXPECT_EQ(f_plain, f_shift);
        EXPECT_EQ(f_plain, f_none);
        EXPECT_EQ(e_none, 0.0f);
    }
    free_device(d_lj);
}

#include "force/soft.cuh"

// Soft repulsion: F = -dE/dr, finite at overlap, amplitude ramps with the
// device step counter (A(step) = A0 + (A1 - A0) (step - 1) / nsteps).
TEST(Soft, ForceEnergyRampAndOverlap) {
    const float L = 10.0f, rc = 1.12246f;
    auto eval = [&](float r, int step, float* energy) {
        std::vector<float4> pos(32, make_float4(0, 0, 0, pack_type_id(-1)));
        pos[0] = make_float4(5.0f, 5.0f, 5.0f, pack_type_id(0));
        pos[1] = make_float4(5.0f + r, 5.0f, 5.0f, pack_type_id(0));
        float4* d_pos = to_device(pos);
        int* d_step = to_device(std::vector<int>{step});
        float4* d_f;
        double* d_v;
        CUDA_CHECK(cudaMalloc(&d_f, 32 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_v, 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_v, 0, 6 * sizeof(double)));
        VerletList v;
        v.allocate(2, rc + 0.3f, cubic_box(L));
        v.build(d_pos, 2, cubic_box(L), nullptr, nullptr, rc + 0.3f);
        launch_soft_kernel(d_pos, d_f, d_v, v.neighbors, v.num_neighbors, 2, rc, 10.0f, 50.0f, 100,
                           d_step, cubic_box(L), 0, 1, nullptr, true);
        auto f = to_host(d_f, 2);
        *energy = f[0].w + f[1].w;
        v.free();
        free_device(d_pos); free_device(d_f); free_device(d_v); free_device(d_step);
        return f[1].x;
    };
    for (int step : {1, 51, 101, 500}) {
        double A = 10.0 + 40.0 * std::min(1.0, (step - 1) / 100.0);
        for (float r : {0.05f, 0.3f, 0.7f, 1.0f}) {
            float e, ep, em;
            float F = eval(r, step, &e);
            const float h = 1e-3f;
            eval(r + h, step, &ep);
            eval(r - h, step, &em);
            EXPECT_NEAR(e, A * (1.0 + std::cos(M_PI * r / rc)), 1e-4 * A) << "step " << step << " r " << r;
            EXPECT_NEAR(F, -(ep - em) / (2 * h), 2e-3 * A) << "step " << step << " r " << r;
            EXPECT_GT(F, 0.0f);
        }
    }
    float e;
    EXPECT_EQ(eval(1.2f, 1, &e), 0.0f);   // beyond rc
    EXPECT_EQ(e, 0.0f);
}
