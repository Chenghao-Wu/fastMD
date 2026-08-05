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

// --- OPLS additions ---------------------------------------------------------

inline BondResult brute_force_harmonic_bond(
    const std::vector<float4>& pos,
    const std::vector<int2>& bonds,
    const std::vector<int>& bond_types,
    const std::vector<float>& bond_k,
    const std::vector<float>& bond_r0,
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
        float k = bond_k[t], r0 = bond_r0[t];

        float dx = min_image(pos[i].x - pos[j].x, L, inv_L);
        float dy = min_image(pos[i].y - pos[j].y, L, inv_L);
        float dz = min_image(pos[i].z - pos[j].z, L, inv_L);
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        float dr = r - r0;

        float ff = -2.0f * k * dr / r;
        float fix = ff * dx, fiy = ff * dy, fiz = ff * dz;

        res.fx[i] += fix;  res.fy[i] += fiy;  res.fz[i] += fiz;
        res.fx[j] -= fix;  res.fy[j] -= fiy;  res.fz[j] -= fiz;

        float pe = k * dr * dr;
        res.pe[i] += 0.5f * pe;
        res.pe[j] += 0.5f * pe;

        res.virial[0] += fix * dx;
        res.virial[1] += fix * dy;
        res.virial[2] += fix * dz;
        res.virial[3] += fiy * dy;
        res.virial[4] += fiy * dz;
        res.virial[5] += fiz * dz;
    }
    return res;
}

struct DihedralResult {
    std::vector<float> fx, fy, fz;
    std::vector<float> pe;
    float virial[6];
};

// OPLS dihedral: 0.5*K1(1+c) + 0.5*K2(1-cos2) + 0.5*K3(1+cos3) + 0.5*K4(1-cos4)
// Gradient as polynomial in c = cos(phi) — mirrors LAMMPS dihedral_opls.cpp.
inline DihedralResult brute_force_opls_dihedral(
    const std::vector<float4>& pos,
    const std::vector<int4>& dihedrals,
    const std::vector<int>& dihedral_types,
    const std::vector<float>& K1, const std::vector<float>& K2,
    const std::vector<float>& K3, const std::vector<float>& K4,
    float L, float inv_L)
{
    int N = pos.size();
    DihedralResult res;
    res.fx.resize(N, 0); res.fy.resize(N, 0); res.fz.resize(N, 0);
    res.pe.resize(N, 0);
    memset(res.virial, 0, sizeof(res.virial));

    for (size_t dd = 0; dd < dihedrals.size(); dd++) {
        int i1 = dihedrals[dd].x, i2 = dihedrals[dd].y;
        int i3 = dihedrals[dd].z, i4 = dihedrals[dd].w;
        int t = dihedral_types[dd];
        float k1 = 0.5f * K1[t], k2 = 0.5f * K2[t];
        float k3 = 0.5f * K3[t], k4 = 0.5f * K4[t];

        float vb1x = min_image(pos[i1].x - pos[i2].x, L, inv_L);
        float vb1y = min_image(pos[i1].y - pos[i2].y, L, inv_L);
        float vb1z = min_image(pos[i1].z - pos[i2].z, L, inv_L);
        float vb2x = min_image(pos[i3].x - pos[i2].x, L, inv_L);
        float vb2y = min_image(pos[i3].y - pos[i2].y, L, inv_L);
        float vb2z = min_image(pos[i3].z - pos[i2].z, L, inv_L);
        float vb3x = min_image(pos[i4].x - pos[i3].x, L, inv_L);
        float vb3y = min_image(pos[i4].y - pos[i3].y, L, inv_L);
        float vb3z = min_image(pos[i4].z - pos[i3].z, L, inv_L);

        float sb1 = 1.0f / (vb1x*vb1x + vb1y*vb1y + vb1z*vb1z);
        float sb2 = 1.0f / (vb2x*vb2x + vb2y*vb2y + vb2z*vb2z);
        float sb3 = 1.0f / (vb3x*vb3x + vb3y*vb3y + vb3z*vb3z);
        float rb1 = sqrtf(sb1), rb3 = sqrtf(sb3);
        float c0 = (vb1x*vb3x + vb1y*vb3y + vb1z*vb3z) * rb1 * rb3;

        float b1mag = 1.0f / rb1, b2mag = 1.0f / sqrtf(sb2), b3mag = 1.0f / rb3;
        float r12c1 = 1.0f / (b1mag * b2mag);
        float c1mag = (vb1x*vb2x + vb1y*vb2y + vb1z*vb2z) * r12c1;
        float r12c2 = 1.0f / (b2mag * b3mag);
        float c2mag = (-vb2x*vb3x - vb2y*vb3y - vb2z*vb3z) * r12c2;

        float sc1 = fmaxf(sqrtf(fmaxf(1.0f - c1mag*c1mag, 0.0f)), 1e-6f);
        float sc2 = fmaxf(sqrtf(fmaxf(1.0f - c2mag*c2mag, 0.0f)), 1e-6f);
        sc1 = 1.0f / sc1; sc2 = 1.0f / sc2;
        float s1 = sc1 * sc1, s2 = sc2 * sc2, s12 = sc1 * sc2;
        float c = (c0 + c1mag * c2mag) * s12;
        c = fminf(fmaxf(c, -1.0f), 1.0f);

        float c2 = 2.0f * c * c - 1.0f;
        float c3 = 4.0f * c * c * c - 3.0f * c;
        float c4 = 8.0f * c * c * c * c - 8.0f * c * c + 1.0f;
        float pe = k1 * (1.0f + c) + k2 * (1.0f - c2) +
                   k3 * (1.0f + c3) + k4 * (1.0f - c4);
        float pd = k1 - 2.0f * k2 * 2.0f * c
                      + 3.0f * k3 * (4.0f * c * c - 1.0f)
                      - 4.0f * k4 * 4.0f * c * (2.0f * c * c - 1.0f);

        float a = pd;
        c = c * a;
        s12 = s12 * a;
        float a11 = c * sb1 * s1;
        float a22 = -sb2 * (2.0f * c0 * s12 - c * (s1 + s2));
        float a33 = c * sb3 * s2;
        float a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
        float a13 = -rb1 * rb3 * s12;
        float a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

        float sx2 = a12*vb1x + a22*vb2x + a23*vb3x;
        float sy2 = a12*vb1y + a22*vb2y + a23*vb3y;
        float sz2 = a12*vb1z + a22*vb2z + a23*vb3z;

        float f1x = a11*vb1x + a12*vb2x + a13*vb3x;
        float f1y = a11*vb1y + a12*vb2y + a13*vb3y;
        float f1z = a11*vb1z + a12*vb2z + a13*vb3z;
        float f2x = -sx2 - f1x, f2y = -sy2 - f1y, f2z = -sz2 - f1z;
        float f4x = a13*vb1x + a23*vb2x + a33*vb3x;
        float f4y = a13*vb1y + a23*vb2y + a33*vb3y;
        float f4z = a13*vb1z + a23*vb2z + a33*vb3z;
        float f3x = sx2 - f4x, f3y = sy2 - f4y, f3z = sz2 - f4z;

        res.fx[i1] += f1x; res.fy[i1] += f1y; res.fz[i1] += f1z;
        res.fx[i2] += f2x; res.fy[i2] += f2y; res.fz[i2] += f2z;
        res.fx[i3] += f3x; res.fy[i3] += f3y; res.fz[i3] += f3z;
        res.fx[i4] += f4x; res.fy[i4] += f4y; res.fz[i4] += f4z;

        float pe4 = 0.25f * pe;
        res.pe[i1] += pe4; res.pe[i2] += pe4;
        res.pe[i3] += pe4; res.pe[i4] += pe4;

        res.virial[0] += vb1x * f1x + vb2x * f3x + (vb3x + vb2x) * f4x;
        res.virial[1] += vb1x * f1y + vb2x * f3y + (vb3x + vb2x) * f4y;
        res.virial[2] += vb1x * f1z + vb2x * f3z + (vb3x + vb2x) * f4z;
        res.virial[3] += vb1y * f1y + vb2y * f3y + (vb3y + vb2y) * f4y;
        res.virial[4] += vb1y * f1z + vb2y * f3z + (vb3y + vb2y) * f4z;
        res.virial[5] += vb1z * f1z + vb2z * f3z + (vb3z + vb2z) * f4z;
    }
    return res;
}

// Full OPLS pair potential: LJ + erfc real-space + reciprocal Ewald +
// 1-2/1-3 erf subtraction + scaled 1-4 + self-energy. Brute force O(N^2),
// intended for small validation systems only.
inline double erfc_ref(double x) { return std::erfc(x); }
inline double erf_ref(double x) { return std::erf(x); }

struct EwaldRefResult {
    std::vector<double> fx, fy, fz;
    double pe;          // total (not per atom)
    double virial[6];
};

inline EwaldRefResult brute_force_opls_full(
    const std::vector<float4>& pos,
    const std::vector<float>& charges,
    const std::vector<float2>& lj_params,   // ntypes*ntypes
    int ntypes, float coul_const, float alpha, int kmax,
    const std::vector<int>& excl13_offsets, // 1-2/1-3 CSR
    const std::vector<int>& excl13_list,
    const std::vector<int2>& pairs14,
    float lj14, float coul14,
    float rc, float L, float inv_L)
{
    int N = pos.size();
    EwaldRefResult res;
    res.fx.assign(N, 0.0); res.fy.assign(N, 0.0); res.fz.assign(N, 0.0);
    res.pe = 0.0;
    memset(res.virial, 0, sizeof(res.virial));
    double rc2 = (double)rc * rc;

    auto type_of = [&](int i) { int t; memcpy(&t, &pos[i].w, 4); return t; };

    // exclusion set: 1-2/1-3 (from CSR) plus 1-4 pairs — all omitted from
    // real-space erfc and LJ, exactly like the Verlet list with exclusions.
    std::vector<std::vector<int>> excluded(N);
    for (int i = 0; i < N; i++)
        for (int e = excl13_offsets[i]; e < excl13_offsets[i + 1]; e++)
            excluded[i].push_back(excl13_list[e]);
    for (auto& pr : pairs14) {
        excluded[pr.x].push_back(pr.y);
        excluded[pr.y].push_back(pr.x);
    }
    for (auto& v : excluded) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // --- real-space: LJ + erfc Coulomb over non-excluded i < j within rc ---
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (std::binary_search(excluded[i].begin(), excluded[i].end(), j))
                continue;
            double dx = min_image(pos[i].x - pos[j].x, L, inv_L);
            double dy = min_image(pos[i].y - pos[j].y, L, inv_L);
            double dz = min_image(pos[i].z - pos[j].z, L, inv_L);
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= rc2 || r2 < 1e-12) continue;
            double r = sqrt(r2);

            int ti = type_of(i), tj = type_of(j);
            double eps = lj_params[ti * ntypes + tj].x;
            double sig = lj_params[ti * ntypes + tj].y;
            double sr2 = sig * sig / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double lj_ff = 24.0 * eps * (2.0 * sr12 - sr6) / r2;
            double lj_pe = 4.0 * eps * (sr12 - sr6);

            double qiqj = coul_const * (double)charges[i] * charges[j];
            double ar = alpha * r;
            double erfc_ar = erfc_ref(ar);
            double expm2 = exp(-ar * ar);
            double coul_pe = qiqj * erfc_ar / r;
            double coul_ff = qiqj / (r * r * r) * (erfc_ar + 1.1283791670955126 * ar * expm2);

            double ff = lj_ff + coul_ff;
            double fx = ff * dx, fy = ff * dy, fz = ff * dz;
            res.fx[i] += fx; res.fy[i] += fy; res.fz[i] += fz;
            res.fx[j] -= fx; res.fy[j] -= fy; res.fz[j] -= fz;
            res.pe += lj_pe + coul_pe;
            res.virial[0] += fx * dx;
            res.virial[1] += fx * dy;
            res.virial[2] += fx * dz;
            res.virial[3] += fy * dy;
            res.virial[4] += fy * dz;
            res.virial[5] += fz * dz;
        }
    }

    // --- reciprocal space (half shell) ---
    double unitk = 2.0 * M_PI / L;
    double V = (double)L * L * L;
    double preu = 4.0 * M_PI / V;
    double inv4a2 = 0.25 / (alpha * alpha);
    for (int nx = 0; nx <= kmax; nx++)
        for (int ny = -kmax; ny <= kmax; ny++)
            for (int nz = -kmax; nz <= kmax; nz++) {
                int n2 = nx*nx + ny*ny + nz*nz;
                if (n2 == 0 || n2 > kmax * kmax) continue;
                // half shell: keep only one of (n, -n)
                if (nx == 0 && ny < 0) continue;
                if (nx == 0 && ny == 0 && nz < 0) continue;
                double kx = unitk * nx, ky = unitk * ny, kz = unitk * nz;
                double k2 = kx*kx + ky*ky + kz*kz;
                double ug = preu * exp(-k2 * inv4a2) / k2;
                double sf_rl = 0, sf_im = 0;
                for (int i = 0; i < N; i++) {
                    double ph = kx * pos[i].x + ky * pos[i].y + kz * pos[i].z;
                    sf_rl += charges[i] * cos(ph);
                    sf_im += charges[i] * sin(ph);
                }
                res.pe += coul_const * ug * (sf_rl*sf_rl + sf_im*sf_im);
                double vterm = -2.0 * (1.0 / k2 + inv4a2);
                double uk = coul_const * ug * (sf_rl*sf_rl + sf_im*sf_im);
                res.virial[0] += uk * (1.0 + vterm * kx * kx);
                res.virial[1] += uk * vterm * kx * ky;
                res.virial[2] += uk * vterm * kx * kz;
                res.virial[3] += uk * (1.0 + vterm * ky * ky);
                res.virial[4] += uk * vterm * ky * kz;
                res.virial[5] += uk * (1.0 + vterm * kz * kz);
                for (int i = 0; i < N; i++) {
                    double ph = kx * pos[i].x + ky * pos[i].y + kz * pos[i].z;
                    double s = sin(ph), c = cos(ph);
                    double partial = s * sf_rl - c * sf_im;
                    res.fx[i] += coul_const * charges[i] * partial * 2.0 * kx * ug;
                    res.fy[i] += coul_const * charges[i] * partial * 2.0 * ky * ug;
                    res.fz[i] += coul_const * charges[i] * partial * 2.0 * kz * ug;
                }
            }

    // --- self energy ---
    double qsq = 0, qsum = 0;
    for (int i = 0; i < N; i++) { qsq += (double)charges[i] * charges[i]; qsum += charges[i]; }
    res.pe += -coul_const * alpha / sqrt(M_PI) * qsq
              - coul_const * M_PI_2 * qsum * qsum / (alpha * alpha * V);

    // --- 1-2/1-3 erf subtraction ---
    for (int i = 0; i < N; i++) {
        for (int e = excl13_offsets[i]; e < excl13_offsets[i + 1]; e++) {
            int j = excl13_list[e];
            if (j <= i) continue;
            double dx = min_image(pos[i].x - pos[j].x, L, inv_L);
            double dy = min_image(pos[i].y - pos[j].y, L, inv_L);
            double dz = min_image(pos[i].z - pos[j].z, L, inv_L);
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            double qiqj = coul_const * (double)charges[i] * charges[j];
            double ar = alpha * r;
            double erf_ar = erf_ref(ar);
            double expm2 = exp(-ar * ar);
            double ff = qiqj / (r * r * r) * (1.1283791670955126 * ar * expm2 - erf_ar);
            double fx = ff * dx, fy = ff * dy, fz = ff * dz;
            res.fx[i] += fx; res.fy[i] += fy; res.fz[i] += fz;
            res.fx[j] -= fx; res.fy[j] -= fy; res.fz[j] -= fz;
            res.pe += -qiqj * erf_ar / r;
            res.virial[0] += fx * dx;
            res.virial[1] += fx * dy;
            res.virial[2] += fx * dz;
            res.virial[3] += fy * dy;
            res.virial[4] += fy * dz;
            res.virial[5] += fz * dz;
        }
    }

    // --- scaled 1-4 pairs ---
    for (auto& pr : pairs14) {
        int i = pr.x, j = pr.y;
        double dx = min_image(pos[i].x - pos[j].x, L, inv_L);
        double dy = min_image(pos[i].y - pos[j].y, L, inv_L);
        double dz = min_image(pos[i].z - pos[j].z, L, inv_L);
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        int ti = type_of(i), tj = type_of(j);
        double eps = lj_params[ti * ntypes + tj].x;
        double sig = lj_params[ti * ntypes + tj].y;
        double sr2 = sig * sig / (r * r);
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double lj_ff = lj14 * 24.0 * eps * (2.0 * sr12 - sr6) / (r * r);
        double lj_pe = lj14 * 4.0 * eps * (sr12 - sr6);

        double qiqj = coul_const * (double)charges[i] * charges[j];
        double ar = alpha * r;
        double erf_ar = erf_ref(ar);
        double expm2 = exp(-ar * ar);
        double coul_pe = qiqj * (coul14 - erf_ar) / r;
        double coul_ff = qiqj / (r * r * r) * (coul14 - erf_ar + 1.1283791670955126 * ar * expm2);

        double ff = lj_ff + coul_ff;
        double fx = ff * dx, fy = ff * dy, fz = ff * dz;
        res.fx[i] += fx; res.fy[i] += fy; res.fz[i] += fz;
        res.fx[j] -= fx; res.fy[j] -= fy; res.fz[j] -= fz;
        res.pe += lj_pe + coul_pe;
        res.virial[0] += fx * dx;
        res.virial[1] += fx * dy;
        res.virial[2] += fx * dz;
        res.virial[3] += fy * dy;
        res.virial[4] += fy * dz;
        res.virial[5] += fz * dz;
    }

    return res;
}

} // namespace ref
