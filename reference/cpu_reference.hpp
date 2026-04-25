#pragma once
#include <cmath>
#include <vector>

namespace ref {

inline float min_image(float dr, float L, float inv_L) {
    return dr - L * rintf(dr * inv_L);
}

inline float dist2_pbc(float x1, float y1, float z1,
                        float x2, float y2, float z2,
                        float L, float inv_L) {
    float dx = min_image(x1 - x2, L, inv_L);
    float dy = min_image(y1 - y2, L, inv_L);
    float dz = min_image(z1 - z2, L, inv_L);
    return dx*dx + dy*dy + dz*dz;
}

inline float wrap_coord(float x, float L, float inv_L) {
    return x - L * floorf(x * inv_L);
}

// Morton code: interleave 10 bits from each of x, y, z into a 30-bit key
inline uint32_t expand_bits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

inline uint32_t morton3d(float x, float y, float z, float inv_L) {
    // Discretize to [0, 1023]
    uint32_t ix = min(max(uint32_t(x * inv_L * 1024.0f), 0u), 1023u);
    uint32_t iy = min(max(uint32_t(y * inv_L * 1024.0f), 0u), 1023u);
    uint32_t iz = min(max(uint32_t(z * inv_L * 1024.0f), 0u), 1023u);
    return (expand_bits(iz) << 2) | (expand_bits(iy) << 1) | expand_bits(ix);
}

// Reference sort: return permutation indices that sort atoms by Morton code
inline std::vector<int> morton_sort_ref(const std::vector<float4>& pos,
                                         float inv_L) {
    int n = pos.size();
    std::vector<std::pair<uint32_t, int>> keyed(n);
    for (int i = 0; i < n; i++) {
        keyed[i] = {morton3d(pos[i].x, pos[i].y, pos[i].z, inv_L), i};
    }
    std::sort(keyed.begin(), keyed.end());
    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = keyed[i].second;
    return perm;
}

struct TilePairRef {
    int tile_a, tile_b;
};

inline std::vector<TilePairRef> brute_force_tile_list(
    const std::vector<float4>& pos, int ntiles, float rc_skin,
    float L, float inv_L)
{
    std::vector<TilePairRef> pairs;
    int tile_size = 32;
    for (int ta = 0; ta < ntiles; ta++) {
        for (int tb = ta; tb < ntiles; tb++) {
            bool found = false;
            for (int ia = ta * tile_size; ia < (ta+1) * tile_size && ia < (int)pos.size(); ia++) {
                for (int ib = tb * tile_size; ib < (tb+1) * tile_size && ib < (int)pos.size(); ib++) {
                    float d2 = dist2_pbc(pos[ia].x, pos[ia].y, pos[ia].z,
                                          pos[ib].x, pos[ib].y, pos[ib].z,
                                          L, inv_L);
                    if (d2 < rc_skin * rc_skin) {
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (found) {
                pairs.push_back({ta, tb});
                if (ta != tb) pairs.push_back({tb, ta});
            }
        }
    }
    return pairs;
}

} // namespace ref
