#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "neighbor/verlet_list.cuh"
#include <algorithm>
#include <numeric>
#include <random>
#include <set>

namespace {

double min_img(double d, double L) { return d - L * std::nearbyint(d / L); }

// Brute-force neighbor sets (by index) within rcut, minus excluded tag pairs.
std::vector<std::set<int>> brute_neighbors(const std::vector<float4>& pos, float L, float rcut,
                                           const std::vector<int>& tag,
                                           const std::set<std::pair<int, int>>& excl_tags) {
    int n = (int)pos.size();
    std::vector<std::set<int>> nb(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = min_img(pos[i].x - pos[j].x, L), dy = min_img(pos[i].y - pos[j].y, L),
                   dz = min_img(pos[i].z - pos[j].z, L);
            if (dx * dx + dy * dy + dz * dz >= (double)rcut * rcut) continue;
            int a = tag[i], b = tag[j];
            if (excl_tags.count({std::min(a, b), std::max(a, b)})) continue;
            nb[i].insert(j);
        }
    return nb;
}

// Pairs within float round-off of the cutoff may legitimately differ.
void expect_list_matches(const VerletList& v, const std::vector<float4>& pos, float L, float rcut,
                         const std::vector<int>& tag,
                         const std::set<std::pair<int, int>>& excl_tags) {
    int n = (int)pos.size();
    auto ref = brute_neighbors(pos, L, rcut, tag, excl_tags);
    auto num = to_host(v.num_neighbors, n);
    auto flat = to_host(v.neighbors, (size_t)v.max_neighbors * n);
    int missing = 0, extra = 0;
    for (int i = 0; i < n; i++) {
        std::set<int> got;
        for (int k = 0; k < num[i]; k++) got.insert(flat[(size_t)k * n + i]);
        for (int j : ref[i]) {
            if (got.count(j)) continue;
            double dx = min_img(pos[i].x - pos[j].x, L), dy = min_img(pos[i].y - pos[j].y, L),
                   dz = min_img(pos[i].z - pos[j].z, L);
            if (std::fabs(std::sqrt(dx * dx + dy * dy + dz * dz) - rcut) > 1e-4) missing++;
        }
        for (int j : got) if (!ref[i].count(j)) {
            double dx = min_img(pos[i].x - pos[j].x, L), dy = min_img(pos[i].y - pos[j].y, L),
                   dz = min_img(pos[i].z - pos[j].z, L);
            if (std::fabs(std::sqrt(dx * dx + dy * dy + dz * dz) - rcut) > 1e-4) extra++;
        }
    }
    EXPECT_EQ(missing, 0) << "pairs missing from the neighbor list";
    EXPECT_EQ(extra, 0) << "pairs wrongly present in the neighbor list";
}

std::vector<float4> random_box(int n, float L, unsigned seed, float cluster_frac = 0.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, L);
    std::uniform_real_distribution<float> c(0.0f, 0.15f * L);
    std::vector<float4> pos(n);
    for (int i = 0; i < n; i++) {
        bool clustered = i < cluster_frac * n;
        pos[i] = make_float4(clustered ? c(rng) : u(rng), clustered ? c(rng) : u(rng),
                             clustered ? c(rng) : u(rng), pack_type_id(0));
    }
    return pos;
}

}  // namespace

// Half the particles packed into a small corner: local neighbor counts far
// exceed the density-based capacity estimate. The list must grow instead of
// silently dropping neighbors.
TEST(VerletList, GrowsOnOverflowAndStaysComplete) {
    const int n = 1500;
    const float L = 12.0f, rcut = 2.8f;
    auto pos = random_box(n, L, 3, 0.5f);
    std::vector<int> tag(n);
    std::iota(tag.begin(), tag.end(), 0);
    float4* d_pos = to_device(pos);
    VerletList v;
    v.allocate(n, rcut, cubic_box(L));
    int cap0 = v.max_neighbors;
    v.build(d_pos, n, cubic_box(L), nullptr, nullptr, rcut);
    EXPECT_GT(v.max_count, cap0) << "test does not exercise overflow";
    EXPECT_GE(v.n_resizes, 1);
    EXPECT_GE(v.max_neighbors, v.max_count);
    expect_list_matches(v, pos, L, rcut, tag, {});
    v.free();
    free_device(d_pos);
}

// NPT changes the box between builds; the cell grid must follow it.
TEST(VerletList, CompleteAfterBoxExpansionAndCompression) {
    const int n = 1200;
    const float L0 = 12.0f, rcut = 2.8f;
    auto pos0 = random_box(n, L0, 9);
    std::vector<int> tag(n);
    std::iota(tag.begin(), tag.end(), 0);
    VerletList v;
    v.allocate(n, rcut, cubic_box(L0));
    // 0.5: fewer than 2R+1 cells per side (stencil cells repeat under PBC)
    for (float s : {1.0f, 1.3f, 0.75f, 1.05f, 0.5f}) {
        float L = L0 * s;
        std::vector<float4> pos(pos0);
        for (auto& p : pos) { p.x *= s; p.y *= s; p.z *= s; }
        float4* d_pos = to_device(pos);
        v.build(d_pos, n, cubic_box(L), nullptr, nullptr, rcut);
        EXPECT_EQ(v.nx, int(L * v.cell_div / rcut)) << "grid not derived from the current box";
        expect_list_matches(v, pos, L, rcut, tag, {});
        free_device(d_pos);
    }
    v.free();
}

// Exclusions are stored by tag; particles may sit at any index.
TEST(VerletList, TagSpaceExclusionsAfterPermutation) {
    const int n = 600;
    const float L = 9.0f, rcut = 2.8f;
    auto pos = random_box(n, L, 21);
    std::mt19937 rng(4);
    std::vector<int> tag(n);
    std::iota(tag.begin(), tag.end(), 0);
    std::shuffle(tag.begin(), tag.end(), rng);   // index i holds particle tag[i]
    // exclude each tag from its tag+1 and tag+2 neighbors (chain-like)
    std::vector<std::vector<int>> rows(n);
    std::set<std::pair<int, int>> excl;
    for (int t = 0; t < n; t++)
        for (int d = 1; d <= 2; d++)
            if (t + d < n) {
                rows[t].push_back(t + d);
                rows[t + d].push_back(t);
                excl.insert({t, t + d});
            }
    std::vector<int> off(n + 1, 0), list;
    for (int t = 0; t < n; t++) {
        std::sort(rows[t].begin(), rows[t].end());
        off[t] = (int)list.size();
        list.insert(list.end(), rows[t].begin(), rows[t].end());
    }
    off[n] = (int)list.size();
    // put excluded partners close together so exclusions matter
    for (int i = 0; i < n; i++) {
        int t = tag[i];
        float4& p = pos[i];
        p.x = std::fmod(0.37f * t, L); p.y = std::fmod(0.11f * t, L); p.z = std::fmod(0.05f * t, L);
    }
    float4* d_pos = to_device(pos);
    int* d_tag = to_device(tag);
    int* d_off = to_device(off);
    int* d_list = to_device(list);
    VerletList v;
    v.allocate(n, rcut, cubic_box(L));
    v.build(d_pos, n, cubic_box(L), d_tag, d_off, d_list, rcut);
    expect_list_matches(v, pos, L, rcut, tag, excl);
    v.free();
    free_device(d_pos); free_device(d_tag); free_device(d_off); free_device(d_list);
}

// Orthorhombic boxes: independent cell counts per axis, including an axis
// with fewer than 2R+1 cells (duplicate stencil cells).
TEST(VerletList, CompleteInOrthorhombicBoxes) {
    const float rcut = 2.8f;
    for (float3 L : {make_float3(12.0f, 9.0f, 15.0f), make_float3(6.0f, 13.0f, 20.0f),
                     make_float3(25.0f, 6.5f, 8.0f)}) {
        const int n = (int)(0.6f * L.x * L.y * L.z);
        std::mt19937 rng(17);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::vector<float4> pos(n);
        for (auto& p : pos) p = make_float4(u(rng) * L.x, u(rng) * L.y, u(rng) * L.z, pack_type_id(0));
        std::vector<int> tag(n);
        std::iota(tag.begin(), tag.end(), 0);
        float4* d_pos = to_device(pos);
        VerletList v;
        const Box box = make_box(L);
        v.allocate(n, rcut, box);
        v.build(d_pos, n, box, nullptr, nullptr, rcut);
        EXPECT_EQ(v.nx, v.cells_along(L.x));
        EXPECT_EQ(v.nz, v.cells_along(L.z));
        // brute force with per-axis minimum image
        auto num = to_host(v.num_neighbors, n);
        auto flat = to_host(v.neighbors, (size_t)v.max_neighbors * n);
        int missing = 0, extra = 0;
        for (int i = 0; i < n; i++) {
            std::set<int> got;
            for (int k = 0; k < num[i]; k++) got.insert(flat[(size_t)k * n + i]);
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y, dz = pos[i].z - pos[j].z;
                dx -= L.x * std::nearbyint(dx / L.x);
                dy -= L.y * std::nearbyint(dy / L.y);
                dz -= L.z * std::nearbyint(dz / L.z);
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (std::fabs(r - rcut) < 1e-4) continue;
                bool in = r < rcut;
                if (in && !got.count(j)) missing++;
                if (!in && got.count(j)) extra++;
            }
        }
        EXPECT_EQ(missing, 0) << L.x << "x" << L.y << "x" << L.z;
        EXPECT_EQ(extra, 0) << L.x << "x" << L.y << "x" << L.z;
        v.free();
        free_device(d_pos);
    }
}

// Sheared (xy-tilted) boxes: cells in fractional coordinates; compare with
// the true minimum distance over lattice images.
TEST(VerletList, CompleteInTiltedBoxes) {
    const float rcut = 2.8f;
    for (float tilt : {0.1f, -0.35f, 0.5f}) {
        Box box = make_box(13.0f, 11.0f, 9.0f);
        box.xy = tilt * box.L.x;
        const int n = (int)(0.6f * box.volume());
        std::mt19937 rng(23);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::vector<float4> pos(n);
        for (auto& p : pos) {   // uniform in the sheared cell
            float sx = u(rng), sy = u(rng), sz = u(rng);
            p = make_float4(sx * box.L.x + sy * box.xy, sy * box.L.y, sz * box.L.z, pack_type_id(0));
        }
        float4* d_pos = to_device(pos);
        VerletList v;
        v.allocate(n, rcut, box);
        v.build(d_pos, n, box, nullptr, nullptr, rcut);
        auto num = to_host(v.num_neighbors, n);
        auto flat = to_host(v.neighbors, (size_t)v.max_neighbors * n);
        int missing = 0, extra = 0;
        for (int i = 0; i < n; i++) {
            std::set<int> got;
            for (int k = 0; k < num[i]; k++) got.insert(flat[(size_t)k * n + i]);
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double best = 1e30;
                for (int a = -1; a <= 1; a++)
                    for (int b = -1; b <= 1; b++)
                        for (int c = -1; c <= 1; c++) {
                            double dx = pos[i].x - pos[j].x + a * box.L.x + b * box.xy;
                            double dy = pos[i].y - pos[j].y + b * box.L.y;
                            double dz = pos[i].z - pos[j].z + c * box.L.z;
                            best = std::min(best, dx * dx + dy * dy + dz * dz);
                        }
                double r = std::sqrt(best);
                if (std::fabs(r - rcut) < 1e-4) continue;
                if (r < rcut && !got.count(j)) missing++;
                if (r >= rcut && got.count(j)) extra++;
            }
        }
        EXPECT_EQ(missing, 0) << "tilt " << tilt;
        EXPECT_EQ(extra, 0) << "tilt " << tilt;
        v.free();
        free_device(d_pos);
    }
}
