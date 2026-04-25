#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "core/morton.cuh"
#include "../reference/cpu_reference.hpp"

TEST(Morton, SortPreservesLocalitySmall) {
    const int N = 64;
    const float L = 10.0f;
    const float inv_L = 1.0f / L;

    std::vector<float4> h_pos(N);
    std::vector<float4> h_vel(N);
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(
            fmodf(i * 3.7f, L),
            fmodf(i * 5.3f, L),
            fmodf(i * 7.1f, L),
            pack_type_id(0)
        );
        h_vel[i] = make_float4(float(i), 0.0f, 0.0f, 0.0f);
    }

    auto ref_perm = ref::morton_sort_ref(h_pos, inv_L);

    int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> h_pos_padded(np, make_float4(0,0,0,0));
    std::vector<float4> h_vel_padded(np, make_float4(0,0,0,0));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_padded.begin());
    std::copy(h_vel.begin(), h_vel.end(), h_vel_padded.begin());

    float4* d_pos = to_device(h_pos_padded);
    float4* d_vel = to_device(h_vel_padded);

    MortonSorter sorter;
    sorter.allocate(np);
    sorter.sort_and_permute(d_pos, d_vel, N, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result_pos = to_host(d_pos, N);
    auto result_vel = to_host(d_vel, N);

    for (int i = 0; i < N; i++) {
        int ref_idx = ref_perm[i];
        EXPECT_FLOAT_EQ(result_pos[i].x, h_pos[ref_idx].x) << "atom " << i;
        EXPECT_FLOAT_EQ(result_pos[i].y, h_pos[ref_idx].y) << "atom " << i;
        EXPECT_FLOAT_EQ(result_pos[i].z, h_pos[ref_idx].z) << "atom " << i;
        EXPECT_FLOAT_EQ(result_vel[i].x, float(ref_idx)) << "vel mismatch atom " << i;
    }

    sorter.free();
    free_device(d_pos);
    free_device(d_vel);
}
