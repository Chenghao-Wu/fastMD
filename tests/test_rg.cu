#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "analysis/rg.cuh"
#include <cmath>
#include <numeric>
#include <random>
#include <map>
#include <algorithm>

namespace {

float run_rg(const std::vector<float4>& pos, const std::vector<int>& image,
             const std::vector<int>& mol_ids, const std::vector<int>* rank, float L) {
    float4* d_pos = to_device(pos);
    int* d_image = to_device(image);
    int* d_rank = rank ? to_device(*rank) : nullptr;
    RgBuffers bufs;
    bufs.allocate(mol_ids, nullptr);
    float rg = 0.0f;
    compute_rg(d_pos, d_image, d_rank, bufs, cubic_box(L), 1, &rg);
    bufs.free();
    free_device(d_pos);
    free_device(d_image);
    if (d_rank) free_device(d_rank);
    return rg;
}

// fp64 reference: mean over molecules of sqrt(<|r - r_com|^2>)
double ref_rg(const std::vector<float4>& pos, const std::vector<int>& image,
              const std::vector<int>& mol_ids, float L) {
    std::map<int, std::vector<int>> mols;
    for (int t = 0; t < (int)mol_ids.size(); t++) mols[mol_ids[t]].push_back(t);
    double sum = 0.0;
    for (auto& [m, members] : mols) {
        double c[3] = {0, 0, 0};
        auto unwrap = [&](int t, int d) {
            const float* p = &pos[t].x;
            return (double)p[d] + image[3 * t + d] * (double)L;
        };
        for (int t : members) for (int d = 0; d < 3; d++) c[d] += unwrap(t, d);
        for (double& v : c) v /= members.size();
        double s = 0.0;
        for (int t : members) for (int d = 0; d < 3; d++) {
            double x = unwrap(t, d) - c[d];
            s += x * x;
        }
        sum += std::sqrt(s / members.size());
    }
    return sum / mols.size();
}

}  // namespace

TEST(Rg, TwoChainsUniform) {
    const float L = 10.0f;
    std::vector<float4> pos = {
        make_float4(1, 0, 0, pack_type_id(0)), make_float4(3, 0, 0, pack_type_id(0)),
        make_float4(5, 0, 0, pack_type_id(0)), make_float4(7, 0, 0, pack_type_id(0)),
        make_float4(2, 0, 0, pack_type_id(0)), make_float4(4, 0, 0, pack_type_id(0)),
        make_float4(6, 0, 0, pack_type_id(0)), make_float4(8, 0, 0, pack_type_id(0))};
    std::vector<int> image(8 * 3, 0);
    std::vector<int> mol = {1, 1, 1, 1, 2, 2, 2, 2};
    EXPECT_NEAR(run_rg(pos, image, mol, nullptr, L), 2.23607f, 1e-4f);
}

TEST(Rg, SingleChainWithImageFlags) {
    const float L = 10.0f;
    std::vector<float4> pos = {make_float4(9, 0, 0, pack_type_id(0)),
                               make_float4(1, 0, 0, pack_type_id(0)),
                               make_float4(3, 0, 0, pack_type_id(0))};
    std::vector<int> image(9, 0);
    image[1 * 3 + 0] = 1;
    image[2 * 3 + 0] = 1;
    EXPECT_NEAR(run_rg(pos, image, {7, 7, 7}, nullptr, L), 1.63299f, 1e-4f);
}

// Long, non-multiple-of-32 chains exercise the multi-warp block reduction
// (the previous kernel had a shared-memory race once a block exceeded 32
// threads); molecules are interleaved and particles permuted to check that
// membership is taken from tags, not from contiguity.
TEST(Rg, LongInterleavedPermutedChainsMatchReference) {
    const float L = 20.0f;
    const int nmol = 5;
    const int lengths[nmol] = {1, 31, 33, 257, 997};
    std::mt19937 rng(1234);
    std::normal_distribution<float> step(0.0f, 0.55f);

    std::vector<float4> pos;
    std::vector<int> image, mol;
    for (int m = 0; m < nmol; m++) {
        float p[3] = {10, 10, 10};
        for (int k = 0; k < lengths[m]; k++) {
            float4 w = make_float4(0, 0, 0, pack_type_id(0));
            int im[3];
            float* wp = &w.x;
            for (int d = 0; d < 3; d++) {
                p[d] += step(rng);
                im[d] = (int)std::floor(p[d] / L);
                wp[d] = p[d] - im[d] * L;
            }
            pos.push_back(w);
            image.insert(image.end(), im, im + 3);
            mol.push_back(100 + m);
        }
    }
    const int n = (int)pos.size();
    double ref = ref_rg(pos, image, mol, L);

    EXPECT_NEAR(run_rg(pos, image, mol, nullptr, L), ref, 1e-4 * ref);

    // Store particles in a shuffled order: index i holds tag perm[i].
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<float4> pos_p(n);
    std::vector<int> image_p(3 * n), rank(n);
    for (int i = 0; i < n; i++) {
        pos_p[i] = pos[perm[i]];
        for (int d = 0; d < 3; d++) image_p[3 * i + d] = image[3 * perm[i] + d];
        rank[perm[i]] = i;
    }
    EXPECT_NEAR(run_rg(pos_p, image_p, mol, &rank, L), ref, 1e-4 * ref);
}
