#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "neighbor/tile_list.cuh"
#include "../reference/cpu_reference.hpp"
#include <algorithm>
#include <set>

TEST(TileList, SmallSystemMatchesBruteForce) {
    const int N = 128;
    const float L = 10.0f;
    const float inv_L = 1.0f / L;
    const float rc = 2.5f;
    const float skin = 0.5f;
    const float rc_skin = rc + skin;
    const int ntiles = div_ceil(N, TILE_SIZE);

    std::vector<float4> h_pos(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            pack_type_id(0)
        );
    }

    auto ref_pairs = ref::brute_force_tile_list(h_pos, ntiles, rc_skin, L, inv_L);
    std::set<std::pair<int,int>> ref_set;
    for (auto& p : ref_pairs) ref_set.insert({p.tile_a, p.tile_b});

    int np = ntiles * TILE_SIZE;
    std::vector<float4> h_pos_pad(np, make_float4(0,0,0,0));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());
    float4* d_pos = to_device(h_pos_pad);

    TileList tl;
    tl.allocate(ntiles, ntiles * ntiles);
    tl.build(d_pos, N, ntiles, rc_skin, L, inv_L);

    auto h_offsets = to_host(tl.offsets, ntiles + 1);
    auto h_neighbors = to_host(tl.tile_neighbors, tl.npairs);

    std::set<std::pair<int,int>> gpu_set;
    for (int ta = 0; ta < ntiles; ta++) {
        for (int j = h_offsets[ta]; j < h_offsets[ta + 1]; j++) {
            gpu_set.insert({ta, h_neighbors[j]});
        }
    }

    for (auto& p : ref_set) {
        EXPECT_TRUE(gpu_set.count(p))
            << "Missing pair (" << p.first << ", " << p.second << ")";
    }

    tl.free();
    free_device(d_pos);
}
