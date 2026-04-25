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

} // namespace ref
