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

struct LJResult {
    std::vector<float> fx, fy, fz;
    std::vector<float> pe;
    float virial[6];
};

inline LJResult brute_force_lj(const std::vector<float4>& pos,
                                 const std::vector<float2>& lj_params,
                                 int ntypes, float rc, float L, float inv_L,
                                 const std::vector<int>* exclusion_offsets = nullptr,
                                 const std::vector<int>* exclusion_list = nullptr) {
    int N = pos.size();
    float rc2 = rc * rc;
    LJResult res;
    res.fx.resize(N, 0.0f);
    res.fy.resize(N, 0.0f);
    res.fz.resize(N, 0.0f);
    res.pe.resize(N, 0.0f);
    memset(res.virial, 0, sizeof(res.virial));

    for (int i = 0; i < N; i++) {
        int ti = unpack_type_id(pos[i].w);
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            if (exclusion_offsets && exclusion_list) {
                int start = (*exclusion_offsets)[i];
                int end = (*exclusion_offsets)[i + 1];
                bool skip = false;
                for (int e = start; e < end; e++) {
                    if ((*exclusion_list)[e] == j) { skip = true; break; }
                }
                if (skip) continue;
            }
            int tj = unpack_type_id(pos[j].w);
            float dx = min_image(pos[i].x - pos[j].x, L, inv_L);
            float dy = min_image(pos[i].y - pos[j].y, L, inv_L);
            float dz = min_image(pos[i].z - pos[j].z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= rc2) continue;

            float2 p = lj_params[ti * ntypes + tj];
            float eps = p.x, sig = p.y;
            float sig2 = sig * sig;
            float r2inv = 1.0f / r2;
            float sr2 = sig2 * r2inv;
            float sr6 = sr2 * sr2 * sr2;
            float sr12 = sr6 * sr6;
            float force_r = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);
            float pair_pe = 4.0f * eps * (sr12 - sr6);

            res.fx[i] += force_r * dx;
            res.fy[i] += force_r * dy;
            res.fz[i] += force_r * dz;
            res.pe[i] += 0.5f * pair_pe;

            res.virial[0] += 0.5f * force_r * dx * dx;
            res.virial[1] += 0.5f * force_r * dx * dy;
            res.virial[2] += 0.5f * force_r * dx * dz;
            res.virial[3] += 0.5f * force_r * dy * dy;
            res.virial[4] += 0.5f * force_r * dy * dz;
            res.virial[5] += 0.5f * force_r * dz * dz;
        }
    }
    return res;
}

struct BondResult {
    std::vector<float> fx, fy, fz;
    std::vector<float> pe;
    float virial[6];
};

inline BondResult brute_force_fene(
    const std::vector<float4>& pos,
    const std::vector<int2>& bonds,
    const std::vector<int>& bond_types,
    const std::vector<float>& bond_k,
    const std::vector<float>& bond_R0,
    const std::vector<float>& bond_eps,
    const std::vector<float>& bond_sig,
    float L, float inv_L)
{
    int N = pos.size();
    BondResult res;
    res.fx.resize(N, 0); res.fy.resize(N, 0); res.fz.resize(N, 0);
    res.pe.resize(N, 0);
    memset(res.virial, 0, sizeof(res.virial));

    for (size_t b = 0; b < bonds.size(); b++) {
        int i = bonds[b].x, j = bonds[b].y;
        int t = bond_types[b];
        float k = bond_k[t], R0 = bond_R0[t];
        float eps = bond_eps[t], sig = bond_sig[t];

        float dx = min_image(pos[i].x - pos[j].x, L, inv_L);
        float dy = min_image(pos[i].y - pos[j].y, L, inv_L);
        float dz = min_image(pos[i].z - pos[j].z, L, inv_L);
        float r2 = dx*dx + dy*dy + dz*dz;
        float R02 = R0 * R0;

        float fene_ff = -k * R02 / (R02 - r2);

        float sig2 = sig * sig;
        float r_cut2 = sig2 * 1.2599210498948732f;
        float wca_ff = 0.0f;
        float wca_pe = 0.0f;
        if (r2 < r_cut2) {
            float r2inv = 1.0f / r2;
            float sr2 = sig2 * r2inv;
            float sr6 = sr2 * sr2 * sr2;
            float sr12 = sr6 * sr6;
            wca_ff = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);
            wca_pe = 4.0f * eps * (sr12 - sr6) + eps;
        }

        float total_ff = fene_ff + wca_ff;
        float fix = total_ff * dx;
        float fiy = total_ff * dy;
        float fiz = total_ff * dz;

        res.fx[i] += fix;  res.fy[i] += fiy;  res.fz[i] += fiz;
        res.fx[j] -= fix;  res.fy[j] -= fiy;  res.fz[j] -= fiz;

        float fene_pe = -0.5f * k * R02 * logf(1.0f - r2 / R02);
        float pair_pe = fene_pe + wca_pe;
        res.pe[i] += 0.5f * pair_pe;
        res.pe[j] += 0.5f * pair_pe;

        res.virial[0] += fix * dx;
        res.virial[1] += fix * dy;
        res.virial[2] += fix * dz;
        res.virial[3] += fiy * dy;
        res.virial[4] += fiy * dz;
        res.virial[5] += fiz * dz;
    }
    return res;
}

struct AngleResult {
    std::vector<float> fx, fy, fz;
    std::vector<float> pe;
    float virial[6];
};

inline AngleResult brute_force_angle(
    const std::vector<float4>& pos,
    const std::vector<int4>& angles,
    const std::vector<float>& angle_k,
    const std::vector<float>& angle_theta0,
    float L, float inv_L)
{
    int N = pos.size();
    AngleResult res;
    res.fx.resize(N, 0); res.fy.resize(N, 0); res.fz.resize(N, 0);
    res.pe.resize(N, 0);
    memset(res.virial, 0, sizeof(res.virial));

    for (size_t a = 0; a < angles.size(); a++) {
        int i = angles[a].x, j = angles[a].y, k = angles[a].z;
        int t = angles[a].w;
        float ka = angle_k[t], theta0 = angle_theta0[t];

        float dxij = min_image(pos[i].x - pos[j].x, L, inv_L);
        float dyij = min_image(pos[i].y - pos[j].y, L, inv_L);
        float dzij = min_image(pos[i].z - pos[j].z, L, inv_L);
        float dxkj = min_image(pos[k].x - pos[j].x, L, inv_L);
        float dykj = min_image(pos[k].y - pos[j].y, L, inv_L);
        float dzkj = min_image(pos[k].z - pos[j].z, L, inv_L);

        float rij2 = dxij*dxij + dyij*dyij + dzij*dzij;
        float rkj2 = dxkj*dxkj + dykj*dykj + dzkj*dzkj;
        float rij = sqrtf(rij2);
        float rkj = sqrtf(rkj2);

        float cos_theta = (dxij*dxkj + dyij*dykj + dzij*dzkj) / (rij * rkj);
        cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
        float theta = acosf(cos_theta);
        float dtheta = theta - theta0;
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        if (sin_theta < 1e-6f) sin_theta = 1e-6f;

        float prefactor = 2.0f * ka * dtheta / sin_theta;

        float fi_x = prefactor * (dxkj / (rij * rkj) - cos_theta * dxij / rij2);
        float fi_y = prefactor * (dykj / (rij * rkj) - cos_theta * dyij / rij2);
        float fi_z = prefactor * (dzkj / (rij * rkj) - cos_theta * dzij / rij2);
        float fk_x = prefactor * (dxij / (rij * rkj) - cos_theta * dxkj / rkj2);
        float fk_y = prefactor * (dyij / (rij * rkj) - cos_theta * dykj / rkj2);
        float fk_z = prefactor * (dzij / (rij * rkj) - cos_theta * dzkj / rkj2);

        res.fx[i] += fi_x; res.fy[i] += fi_y; res.fz[i] += fi_z;
        res.fx[k] += fk_x; res.fy[k] += fk_y; res.fz[k] += fk_z;
        res.fx[j] -= (fi_x + fk_x);
        res.fy[j] -= (fi_y + fk_y);
        res.fz[j] -= (fi_z + fk_z);

        float pe = ka * dtheta * dtheta;
        res.pe[i] += pe / 3.0f;
        res.pe[j] += pe / 3.0f;
        res.pe[k] += pe / 3.0f;

        res.virial[0] += fi_x * dxij + fk_x * dxkj;
        res.virial[1] += fi_x * dyij + fk_x * dykj;
        res.virial[2] += fi_x * dzij + fk_x * dzkj;
        res.virial[3] += fi_y * dyij + fk_y * dykj;
        res.virial[4] += fi_y * dzij + fk_y * dzkj;
        res.virial[5] += fi_z * dzij + fk_z * dzkj;
    }
    return res;
}

} // namespace ref
