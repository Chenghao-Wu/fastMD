#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "core/pbc.cuh"
#include "../reference/cpu_reference.hpp"

__global__ void test_min_image_kernel(const float4* r1, const float4* r2,
                                       float3* dr_out, int n,
                                       float L, float inv_L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dr_out[i] = min_image_dr(r1[i], r2[i], L, inv_L);
    }
}

__global__ void test_wrap_kernel(float4* pos, int n, float L, float inv_L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pos[i] = wrap_position(pos[i], L, inv_L);
    }
}

TEST(PBC, MinImageBasicCases) {
    const float L = 10.0f;
    const float inv_L = 1.0f / L;
    const int N = 5;

    std::vector<float4> h_r1 = {
        {1.0f, 2.0f, 3.0f, 0.0f},
        {9.5f, 0.5f, 5.0f, 0.0f},
        {0.5f, 0.5f, 0.5f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        {0.1f, 9.9f, 5.0f, 0.0f},
    };
    std::vector<float4> h_r2 = {
        {2.0f, 3.0f, 4.0f, 0.0f},
        {0.5f, 9.5f, 5.0f, 0.0f},
        {9.5f, 9.5f, 9.5f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        {9.9f, 0.1f, 5.0f, 0.0f},
    };

    float4* d_r1 = to_device(h_r1);
    float4* d_r2 = to_device(h_r2);
    float3* d_dr;
    CUDA_CHECK(cudaMalloc(&d_dr, N * sizeof(float3)));

    test_min_image_kernel<<<1, N>>>(d_r1, d_r2, d_dr, N, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_dr = to_host(d_dr, N);

    for (int i = 0; i < N; i++) {
        float ref_dx = ref::min_image(h_r1[i].x - h_r2[i].x, L, inv_L);
        float ref_dy = ref::min_image(h_r1[i].y - h_r2[i].y, L, inv_L);
        float ref_dz = ref::min_image(h_r1[i].z - h_r2[i].z, L, inv_L);
        EXPECT_FLOAT_EQ(h_dr[i].x, ref_dx) << "pair " << i << " dx";
        EXPECT_FLOAT_EQ(h_dr[i].y, ref_dy) << "pair " << i << " dy";
        EXPECT_FLOAT_EQ(h_dr[i].z, ref_dz) << "pair " << i << " dz";
    }

    free_device(d_r1);
    free_device(d_r2);
    free_device(d_dr);
}

TEST(PBC, WrapPositions) {
    const float L = 10.0f;
    const float inv_L = 1.0f / L;
    const int N = 4;

    std::vector<float4> h_pos = {
        {-0.5f, 10.5f, 5.0f, 0.0f},
        {20.3f, -3.7f, 0.0f, 0.0f},
        {5.0f,  5.0f,  5.0f, 0.0f},
        {10.0f, 0.0f, -0.01f, 0.0f},
    };

    float4* d_pos = to_device(h_pos);
    test_wrap_kernel<<<1, N>>>(d_pos, N, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto result = to_host(d_pos, N);

    for (int i = 0; i < N; i++) {
        EXPECT_GE(result[i].x, 0.0f) << "atom " << i;
        EXPECT_LT(result[i].x, L)    << "atom " << i;
        EXPECT_GE(result[i].y, 0.0f) << "atom " << i;
        EXPECT_LT(result[i].y, L)    << "atom " << i;
        EXPECT_GE(result[i].z, 0.0f) << "atom " << i;
        EXPECT_LT(result[i].z, L)    << "atom " << i;
    }

    free_device(d_pos);
}
