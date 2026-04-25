#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/lj.cuh"
#include "neighbor/tile_list.cuh"
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

    TileList tl;
    tl.allocate(ntiles, ntiles * ntiles);
    tl.build(d_pos, N, ntiles, rc + skin, L, inv_L);

    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                      tl.offsets, tl.tile_neighbors,
                      ntiles, N, ntypes, rc2, L, inv_L);
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

    tl.free();
    free_device(d_pos);
    free_device(d_force);
    free_device(d_virial);
    free_device(d_lj);
}
