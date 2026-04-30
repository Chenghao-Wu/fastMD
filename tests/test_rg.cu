#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "analysis/rg.cuh"

TEST(Rg, TwoChainsUniform) {
    const int natoms = 8;
    const float L = 10.0f;

    std::vector<float4> h_pos(natoms);
    for (int i = 0; i < natoms; i++) {
        h_pos[i] = make_float4(0, 0, 0, pack_type_id(0));
    }
    h_pos[0] = make_float4(1.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[1] = make_float4(3.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[2] = make_float4(5.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[3] = make_float4(7.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[4] = make_float4(2.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[5] = make_float4(4.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[6] = make_float4(6.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[7] = make_float4(8.0f, 0.0f, 0.0f, pack_type_id(0));

    std::vector<int> h_image(natoms * 3, 0);

    std::vector<int> chain_offsets = {0, 4, 8};
    std::vector<int> chain_lengths = {4, 4};
    int nchains = 2;
    int max_len = 4;

    float4* d_pos = to_device(h_pos);
    int* d_image = to_device(h_image);

    RgBuffers bufs;
    bufs.allocate(chain_offsets, chain_lengths, nchains, max_len, "/dev/null");

    float h_rg;
    compute_rg(d_pos, d_image, bufs, L, 1, &h_rg);

    EXPECT_NEAR(h_rg, 2.23607f, 0.01f);

    bufs.free();
    free_device(d_pos);
    free_device(d_image);
}

TEST(Rg, SingleChainWithImageFlags) {
    const int natoms = 3;
    const float L = 10.0f;

    std::vector<float4> h_pos(natoms);
    h_pos[0] = make_float4(9.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[1] = make_float4(1.0f, 0.0f, 0.0f, pack_type_id(0));
    h_pos[2] = make_float4(3.0f, 0.0f, 0.0f, pack_type_id(0));

    std::vector<int> h_image(natoms * 3, 0);
    h_image[1 * 3 + 0] = 1;
    h_image[2 * 3 + 0] = 1;

    std::vector<int> chain_offsets = {0, 3};
    std::vector<int> chain_lengths = {3};
    int nchains = 1;
    int max_len = 3;

    float4* d_pos = to_device(h_pos);
    int* d_image = to_device(h_image);

    RgBuffers bufs;
    bufs.allocate(chain_offsets, chain_lengths, nchains, max_len, "/dev/null");

    float h_rg;
    compute_rg(d_pos, d_image, bufs, L, 1, &h_rg);

    EXPECT_NEAR(h_rg, 1.63299f, 0.01f);

    bufs.free();
    free_device(d_pos);
    free_device(d_image);
}
