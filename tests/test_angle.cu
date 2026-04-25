#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/angle.cuh"
#include "../reference/cpu_reference.hpp"

TEST(Angle, TriangleMatchesCPU) {
    const int N = 3;
    const float L = 10.0f;
    const float inv_L = 1.0f / L;

    std::vector<float4> h_pos = {
        {3.0f, 5.0f, 5.0f, pack_type_id(0)},
        {5.0f, 5.0f, 5.0f, pack_type_id(0)},
        {5.0f, 7.0f, 5.0f, pack_type_id(0)},
    };
    std::vector<int4> h_angles = {{0, 1, 2, 0}};
    std::vector<float> h_k = {100.0f};
    float theta0 = M_PI * 120.0f / 180.0f;
    std::vector<float> h_theta0 = {theta0};

    auto ref = ref::brute_force_angle(h_pos, h_angles, h_k, h_theta0, L, inv_L);

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

    int4* d_angles = to_device(h_angles);
    AngleParams h_params = {100.0f, theta0};
    AngleParams* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(AngleParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(AngleParams),
                           cudaMemcpyHostToDevice));

    launch_angle_kernel(d_pos, d_force, d_virial, d_angles, d_params,
                         1, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_force_out = to_host(d_force, N);
    for (int i = 0; i < N; i++) {
        float3 gpu_f = make_float3(h_force_out[i].x, h_force_out[i].y, h_force_out[i].z);
        float3 ref_f = make_float3(ref.fx[i], ref.fy[i], ref.fz[i]);
        assert_float3_near(gpu_f, ref_f, 1e-3f);
    }

    float total_fx = h_force_out[0].x + h_force_out[1].x + h_force_out[2].x;
    float total_fy = h_force_out[0].y + h_force_out[1].y + h_force_out[2].y;
    float total_fz = h_force_out[0].z + h_force_out[1].z + h_force_out[2].z;
    EXPECT_NEAR(total_fx, 0.0f, 1e-3f);
    EXPECT_NEAR(total_fy, 0.0f, 1e-3f);
    EXPECT_NEAR(total_fz, 0.0f, 1e-3f);

    free_device(d_pos); free_device(d_force); free_device(d_virial);
    free_device(d_angles); free_device(d_params);
}
