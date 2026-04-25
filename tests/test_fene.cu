#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/fene.cuh"
#include "../reference/cpu_reference.hpp"

TEST(FENE, TwoBondChainMatchesCPU) {
    const int N = 3;
    const float L = 10.0f;
    const float inv_L = 1.0f / L;

    std::vector<float4> h_pos = {
        {2.0f, 5.0f, 5.0f, pack_type_id(0)},
        {3.0f, 5.0f, 5.0f, pack_type_id(0)},
        {4.2f, 5.0f, 5.0f, pack_type_id(0)},
    };
    std::vector<int2> h_bonds = {{0, 1}, {1, 2}};
    std::vector<int> h_types = {0, 0};
    std::vector<float> h_k = {30.0f}, h_R0 = {1.5f}, h_eps = {1.0f}, h_sig = {1.0f};

    auto ref = ref::brute_force_fene(h_pos, h_bonds, h_types,
                                      h_k, h_R0, h_eps, h_sig, L, inv_L);

    int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> h_pos_pad(np, make_float4(0,0,0,pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());

    float4* d_pos = to_device(h_pos_pad);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));

    int2* d_bonds = to_device(h_bonds);
    int* d_btypes = to_device(h_types);
    FENEParams h_params = {30.0f, 1.5f, 1.0f, 1.0f};
    FENEParams* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(FENEParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(FENEParams),
                           cudaMemcpyHostToDevice));

    launch_fene_kernel(d_pos, d_force, d_virial, d_bonds, d_btypes, d_params,
                        2, 1, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_force = to_host(d_force, N);
    for (int i = 0; i < N; i++) {
        float3 gpu_f = make_float3(h_force[i].x, h_force[i].y, h_force[i].z);
        float3 ref_f = make_float3(ref.fx[i], ref.fy[i], ref.fz[i]);
        assert_float3_near(gpu_f, ref_f, 1e-3f);
    }

    free_device(d_pos); free_device(d_force); free_device(d_virial);
    free_device(d_bonds); free_device(d_btypes); free_device(d_params);
}
