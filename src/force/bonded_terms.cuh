#pragma once
#include "../core/types.cuh"
#include "../io/table_parser.hpp"
#include "fene.cuh"
#include "bond_harmonic.cuh"
#include "angle.cuh"
#include "dihedral_opls.cuh"

// Force/energy/virial of single bonded terms, shared by the per-term
// (scatter) kernels and the per-atom (gather) kernel so both evaluate the
// identical expressions. Displacements are already minimum-imaged.
// Virial: the term's full contribution, order xx xy xz yy yz zz.

struct BondTerm {           // force on the first atom; the second gets -fi
    float3 fi;
    float  e;
    float  vir[6];
};

struct AngleTerm {          // forces on the end atoms; the vertex gets -(fi + fk)
    float3 fi, fk;
    float  e;
    float  vir[6];
};

struct DihedralTerm {       // forces on atoms 1..4
    float3 f1, f2, f3, f4;
    float  e;
    float  vir[6];
};

__device__ __forceinline__ void pair_virial(float3 f, float3 d, float (&v)[6]) {
    v[0] = f.x * d.x; v[1] = f.x * d.y; v[2] = f.x * d.z;
    v[3] = f.y * d.y; v[4] = f.y * d.z; v[5] = f.z * d.z;
}

// FENE + WCA (LAMMPS bond_style fene). d = r_i - r_j. r^2 is clamped to
// [1e-6, 0.9999 R0^2] as in the original kernel.
__device__ __forceinline__ BondTerm fene_term(float3 d, FENEParams p) {
    float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
    float R02 = p.R0 * p.R0;
    float r2_safe = fminf(fmaxf(r2, 1e-6f), R02 * 0.9999f);
    float fene_ff = -p.k * R02 / (R02 - r2_safe);
    float sig2 = p.sig * p.sig;
    float r_cut2 = sig2 * 1.2599210498948732f;   // 2^(1/3) sig^2
    float wca_ff = 0.0f, wca_pe = 0.0f;
    if (r2_safe < r_cut2) {
        float r2inv = 1.0f / r2_safe;
        float sr2 = sig2 * r2inv;
        float sr6 = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6;
        wca_ff = 24.0f * p.eps * r2inv * (2.0f * sr12 - sr6);
        wca_pe = 4.0f * p.eps * (sr12 - sr6) + p.eps;
    }
    float ff = fene_ff + wca_ff;
    BondTerm t;
    t.fi = make_float3(ff * d.x, ff * d.y, ff * d.z);
    t.e = -0.5f * p.k * R02 * logf(1.0f - r2_safe / R02) + wca_pe;
    pair_virial(t.fi, d, t.vir);
    return t;
}

// Harmonic bond E = K (r - r0)^2 (LAMMPS bond_style harmonic).
__device__ __forceinline__ BondTerm harmonic_bond_term(float3 d, BondHarmonicParams p) {
    float r2 = fmaxf(d.x * d.x + d.y * d.y + d.z * d.z, 1e-12f);
    float r = sqrtf(r2);
    float dr = r - p.r0;
    float ff = -2.0f * p.k * dr / r;
    BondTerm t;
    t.fi = make_float3(ff * d.x, ff * d.y, ff * d.z);
    t.e = p.k * dr * dr;
    pair_virial(t.fi, d, t.vir);
    return t;
}

// Tabulated bond; table column y holds F(r)/r (see table_parser.hpp).
__device__ __forceinline__ BondTerm table_bond_term(float3 d, const TableParams& tp,
                                                    const float4* __restrict__ tbl) {
    float r = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    float rr = fminf(fmaxf(r, tp.rmin), tp.rmax);
    int idx = (int)((rr - tp.rmin) * tp.inv_dr);
    idx = max(0, min(idx, tp.npoints - 2));
    float u = (rr - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
    float4 p0 = tbl[idx], p1 = tbl[idx + 1];
    float f_over_r = p0.y + u * (p1.y - p0.y);
    BondTerm t;
    t.fi = make_float3(f_over_r * d.x, f_over_r * d.y, f_over_r * d.z);
    t.e = p0.z + u * (p1.z - p0.z);
    pair_virial(t.fi, d, t.vir);
    return t;
}

// Angle force projection shared by harmonic and tabulated angles:
// prefactor = (dU/dtheta) / sin(theta). dij = r_i - r_j, dkj = r_k - r_j.
__device__ __forceinline__ void angle_project(float3 dij, float3 dkj, float rij2, float rkj2,
                                              float rij_inv, float rkj_inv, float cos_theta,
                                              float prefactor, AngleTerm& t) {
    float rij2_inv = 1.0f / rij2, rkj2_inv = 1.0f / rkj2, rr_inv = rij_inv * rkj_inv;
    t.fi = make_float3(prefactor * (dkj.x * rr_inv - cos_theta * dij.x * rij2_inv),
                       prefactor * (dkj.y * rr_inv - cos_theta * dij.y * rij2_inv),
                       prefactor * (dkj.z * rr_inv - cos_theta * dij.z * rij2_inv));
    t.fk = make_float3(prefactor * (dij.x * rr_inv - cos_theta * dkj.x * rkj2_inv),
                       prefactor * (dij.y * rr_inv - cos_theta * dkj.y * rkj2_inv),
                       prefactor * (dij.z * rr_inv - cos_theta * dkj.z * rkj2_inv));
    t.vir[0] = t.fi.x * dij.x + t.fk.x * dkj.x;
    t.vir[1] = t.fi.x * dij.y + t.fk.x * dkj.y;
    t.vir[2] = t.fi.x * dij.z + t.fk.x * dkj.z;
    t.vir[3] = t.fi.y * dij.y + t.fk.y * dkj.y;
    t.vir[4] = t.fi.y * dij.z + t.fk.y * dkj.z;
    t.vir[5] = t.fi.z * dij.z + t.fk.z * dkj.z;
}

// Harmonic angle E = K (theta - theta0)^2 (LAMMPS angle_style harmonic).
__device__ __forceinline__ AngleTerm harmonic_angle_term(float3 dij, float3 dkj, AngleParams p) {
    float rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;
    float rkj2 = dkj.x * dkj.x + dkj.y * dkj.y + dkj.z * dkj.z;
    float rij_inv = rsqrtf(rij2), rkj_inv = rsqrtf(rkj2);
    float cos_theta = (dij.x * dkj.x + dij.y * dkj.y + dij.z * dkj.z) * rij_inv * rkj_inv;
    cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
    float dtheta = acosf(cos_theta) - p.theta0;
    float sin_theta = fmaxf(sqrtf(1.0f - cos_theta * cos_theta), 1e-6f);
    AngleTerm t;
    angle_project(dij, dkj, rij2, rkj2, rij_inv, rkj_inv, cos_theta,
                  2.0f * p.k_theta * dtheta / sin_theta, t);
    t.e = p.k_theta * dtheta * dtheta;
    return t;
}

// Cosine angle E = K (1 + cos theta) (LAMMPS angle_style cosine; minimum at
// theta = 180 degrees, the usual Kremer-Grest bending potential). dU/dtheta
// = -K sin(theta), so the projection prefactor is exactly -K: no division by
// sin(theta) near straight angles.
__device__ __forceinline__ AngleTerm cosine_angle_term(float3 dij, float3 dkj, AngleParams p) {
    float rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;
    float rkj2 = dkj.x * dkj.x + dkj.y * dkj.y + dkj.z * dkj.z;
    float rij_inv = rsqrtf(rij2), rkj_inv = rsqrtf(rkj2);
    float cos_theta = (dij.x * dkj.x + dij.y * dkj.y + dij.z * dkj.z) * rij_inv * rkj_inv;
    cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
    AngleTerm t;
    angle_project(dij, dkj, rij2, rkj2, rij_inv, rkj_inv, cos_theta, -p.k_theta, t);
    t.e = p.k_theta * (1.0f + cos_theta);
    return t;
}

// Tabulated angle; table column y holds -dU/dtheta per radian.
__device__ __forceinline__ AngleTerm table_angle_term(float3 dij, float3 dkj, const TableParams& tp,
                                                      const float4* __restrict__ tbl) {
    float rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;
    float rkj2 = dkj.x * dkj.x + dkj.y * dkj.y + dkj.z * dkj.z;
    float rij_inv = rsqrtf(rij2), rkj_inv = rsqrtf(rkj2);
    float cos_theta = (dij.x * dkj.x + dij.y * dkj.y + dij.z * dkj.z) * rij_inv * rkj_inv;
    cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
    float theta = acosf(cos_theta);
    float sin_theta = fmaxf(sqrtf(1.0f - cos_theta * cos_theta), 1e-6f);
    float th = fminf(fmaxf(theta, tp.rmin), tp.rmax);
    int idx = (int)((th - tp.rmin) * tp.inv_dr);
    idx = max(0, min(idx, tp.npoints - 2));
    float u = (th - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
    float4 p0 = tbl[idx], p1 = tbl[idx + 1];
    float f_theta = p0.y + u * (p1.y - p0.y);   // -dU/dtheta
    AngleTerm t;
    angle_project(dij, dkj, rij2, rkj2, rij_inv, rkj_inv, cos_theta, -f_theta / sin_theta, t);
    t.e = p0.z + u * (p1.z - p0.z);
    return t;
}

// OPLS dihedral, port of LAMMPS DihedralOPLS::compute().
// vb1 = r1 - r2, vb2 = r3 - r2, vb3 = r4 - r3.
__device__ __forceinline__ DihedralTerm opls_dihedral_term(float3 vb1, float3 vb2, float3 vb3,
                                                           OPLSDihedralParams p) {
    float k1 = 0.5f * p.k1, k2 = 0.5f * p.k2, k3 = 0.5f * p.k3, k4 = 0.5f * p.k4;
    float vb2xm = -vb2.x, vb2ym = -vb2.y, vb2zm = -vb2.z;

    float sb1 = 1.0f / (vb1.x * vb1.x + vb1.y * vb1.y + vb1.z * vb1.z);
    float sb2 = 1.0f / (vb2.x * vb2.x + vb2.y * vb2.y + vb2.z * vb2.z);
    float sb3 = 1.0f / (vb3.x * vb3.x + vb3.y * vb3.y + vb3.z * vb3.z);
    float rb1 = sqrtf(sb1), rb3 = sqrtf(sb3);
    float c0 = (vb1.x * vb3.x + vb1.y * vb3.y + vb1.z * vb3.z) * rb1 * rb3;

    float b1mag = rsqrtf(sb1), b2mag = rsqrtf(sb2), b3mag = rsqrtf(sb3);
    float r12c1 = 1.0f / (b1mag * b2mag);
    float c1mag = (vb1.x * vb2.x + vb1.y * vb2.y + vb1.z * vb2.z) * r12c1;
    float r12c2 = 1.0f / (b2mag * b3mag);
    float c2mag = (vb2xm * vb3.x + vb2ym * vb3.y + vb2zm * vb3.z) * r12c2;

    float sc1 = 1.0f / fmaxf(sqrtf(fmaxf(1.0f - c1mag * c1mag, 0.0f)), 1e-6f);
    float sc2 = 1.0f / fmaxf(sqrtf(fmaxf(1.0f - c2mag * c2mag, 0.0f)), 1e-6f);
    float s1 = sc1 * sc1, s2 = sc2 * sc2, s12 = sc1 * sc2;
    float c = fminf(fmaxf((c0 + c1mag * c2mag) * s12, -1.0f), 1.0f);

    float c2 = 2.0f * c * c - 1.0f;
    float c3 = 4.0f * c * c * c - 3.0f * c;
    float c4 = 8.0f * c * c * c * c - 8.0f * c * c + 1.0f;
    DihedralTerm t;
    t.e = k1 * (1.0f + c) + k2 * (1.0f - c2) + k3 * (1.0f + c3) + k4 * (1.0f - c4);
    // dE/dc via U_{n-1}(c) = sin(n phi) / sin(phi)
    float a = k1 - 2.0f * k2 * 2.0f * c + 3.0f * k3 * (4.0f * c * c - 1.0f)
              - 4.0f * k4 * 4.0f * c * (2.0f * c * c - 1.0f);
    c = c * a;
    s12 = s12 * a;
    float a11 = c * sb1 * s1;
    float a22 = -sb2 * (2.0f * c0 * s12 - c * (s1 + s2));
    float a33 = c * sb3 * s2;
    float a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
    float a13 = -rb1 * rb3 * s12;
    float a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

    float3 sx2 = make_float3(a12 * vb1.x + a22 * vb2.x + a23 * vb3.x,
                             a12 * vb1.y + a22 * vb2.y + a23 * vb3.y,
                             a12 * vb1.z + a22 * vb2.z + a23 * vb3.z);
    t.f1 = make_float3(a11 * vb1.x + a12 * vb2.x + a13 * vb3.x,
                       a11 * vb1.y + a12 * vb2.y + a13 * vb3.y,
                       a11 * vb1.z + a12 * vb2.z + a13 * vb3.z);
    t.f2 = make_float3(-sx2.x - t.f1.x, -sx2.y - t.f1.y, -sx2.z - t.f1.z);
    t.f4 = make_float3(a13 * vb1.x + a23 * vb2.x + a33 * vb3.x,
                       a13 * vb1.y + a23 * vb2.y + a33 * vb3.y,
                       a13 * vb1.z + a23 * vb2.z + a33 * vb3.z);
    t.f3 = make_float3(sx2.x - t.f4.x, sx2.y - t.f4.y, sx2.z - t.f4.z);
    // LAMMPS ev_tally convention: v = vb1 f1 + vb2 f3 + (vb3 + vb2) f4
    t.vir[0] = vb1.x * t.f1.x + vb2.x * t.f3.x + (vb3.x + vb2.x) * t.f4.x;
    t.vir[1] = vb1.x * t.f1.y + vb2.x * t.f3.y + (vb3.x + vb2.x) * t.f4.y;
    t.vir[2] = vb1.x * t.f1.z + vb2.x * t.f3.z + (vb3.x + vb2.x) * t.f4.z;
    t.vir[3] = vb1.y * t.f1.y + vb2.y * t.f3.y + (vb3.y + vb2.y) * t.f4.y;
    t.vir[4] = vb1.y * t.f1.z + vb2.y * t.f3.z + (vb3.y + vb2.y) * t.f4.z;
    t.vir[5] = vb1.z * t.f1.z + vb2.z * t.f3.z + (vb3.z + vb2.z) * t.f4.z;
    return t;
}

// Tabulated dihedral (dihedral_table). Same arguments as the OPLS term: vb1 = r1 - r2, vb2 = r3 - r2,
// vb3 = r4 - r3. phi = atan2(sin, cos) in (-pi, pi] with the IUPAC convention (cis 0, trans pi), forces
// from the Blondel-Karplus gradients (J. Comput. Chem. 17, 1132 (1996)): no 1/sin(phi) singularity.
// Table column y holds -dU/dphi per radian on [-pi, pi] (table_parser.hpp).
// With TABLE_DIHEDRAL_CBT (combined bending-torsion, Bulacu et al., J. Chem. Theory Comput. 9, 3282 (2013))
// E = sin^3(theta1) sin^3(theta2) U(phi), theta1 the angle 1-2-3 and theta2 the angle 2-3-4: the torsion
// gradient's 1/sin(theta) is cancelled by the prefactor, so forces stay finite (and vanish) as a bond angle
// straightens. The prefactor's own derivatives add bending forces on both angles.
__device__ __forceinline__ DihedralTerm table_dihedral_term(float3 vb1, float3 vb2, float3 vb3,
                                                            const TableParams& tp,
                                                            const float4* __restrict__ tbl) {
    // Blondel-Karplus notation: F = r1 - r2, G = r2 - r3, H = r4 - r3
    const float3 F = vb1, G = make_float3(-vb2.x, -vb2.y, -vb2.z), H = vb3;
    const float3 A = make_float3(F.y * G.z - F.z * G.y, F.z * G.x - F.x * G.z, F.x * G.y - F.y * G.x);
    const float3 B = make_float3(H.y * G.z - H.z * G.y, H.z * G.x - H.x * G.z, H.x * G.y - H.y * G.x);
    const float a2_raw = A.x * A.x + A.y * A.y + A.z * A.z, b2_raw = B.x * B.x + B.y * B.y + B.z * B.z;
    const float a2 = fmaxf(a2_raw, 1e-20f), b2 = fmaxf(b2_raw, 1e-20f);
    const float g2 = G.x * G.x + G.y * G.y + G.z * G.z;
    const float gn = fmaxf(sqrtf(g2), 1e-20f);
    const float3 BxA = make_float3(B.y * A.z - B.z * A.y, B.z * A.x - B.x * A.z, B.x * A.y - B.y * A.x);
    const float cos_part = A.x * B.x + A.y * B.y + A.z * B.z;                      // |A||B| cos(phi)
    const float sin_part = (BxA.x * G.x + BxA.y * G.y + BxA.z * G.z) / gn;       // |A||B| sin(phi)
    float phi = atan2f(sin_part, cos_part);

    const float ph = fminf(fmaxf(phi, tp.rmin), tp.rmax);
    int idx = (int)((ph - tp.rmin) * tp.inv_dr);
    idx = max(0, min(idx, tp.npoints - 2));
    const float u = (ph - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
    const float4 p0 = tbl[idx], p1 = tbl[idx + 1];
    const float f_phi = p0.y + u * (p1.y - p0.y);       // -dU/dphi
    const float e_phi = p0.z + u * (p1.z - p0.z);       // U(phi)

    // force on atom a = f_phi * w * dphi/dr_a, w = 1 (plain) or sin^3 theta1 sin^3 theta2 (CBT), through
    // ga = w |G| / |A|^2, gb = w |G| / |B|^2, fg = w F.G / (|A|^2 |G|), hg = w H.G / (|B|^2 |G|)
    const float fgd = F.x * G.x + F.y * G.y + F.z * G.z, hgd = H.x * G.x + H.y * G.y + H.z * G.z;
    const bool cbt = tp.flags & TABLE_DIHEDRAL_CBT;
    float w = 1.0f, ga, gb, fg, hg;
    float s1 = 0.0f, s2 = 0.0f, ff2 = 0.0f, hh2 = 0.0f, fgn2 = 1.0f, hgn2 = 1.0f;
    if (!cbt) {
        ga = gn / a2; gb = gn / b2;
        fg = fgd / (a2 * gn); hg = hgd / (b2 * gn);
    } else {
        // |A| = |F||G| sin(theta1), |B| = |H||G| sin(theta2): w / |A|^2 = s1 s2^3 / (|F|^2 |G|^2), no division by |A|^2
        ff2 = fmaxf(F.x * F.x + F.y * F.y + F.z * F.z, 1e-20f);
        hh2 = fmaxf(H.x * H.x + H.y * H.y + H.z * H.z, 1e-20f);
        fgn2 = ff2 * fmaxf(g2, 1e-20f);
        hgn2 = hh2 * fmaxf(g2, 1e-20f);
        s1 = sqrtf(fminf(a2_raw / fgn2, 1.0f));
        s2 = sqrtf(fminf(b2_raw / hgn2, 1.0f));
        const float s13 = s1 * s1 * s1, s23 = s2 * s2 * s2;
        w = s13 * s23;
        ga = s1 * s23 * gn / fgn2; gb = s13 * s2 * gn / hgn2;
        fg = s1 * s23 * fgd / (fgn2 * gn); hg = s13 * s2 * hgd / (hgn2 * gn);
    }
    DihedralTerm t;
    t.f1 = make_float3(-f_phi * ga * A.x, -f_phi * ga * A.y, -f_phi * ga * A.z);
    t.f4 = make_float3(f_phi * gb * B.x, f_phi * gb * B.y, f_phi * gb * B.z);
    t.f2 = make_float3(f_phi * ((ga + fg) * A.x - hg * B.x), f_phi * ((ga + fg) * A.y - hg * B.y),
                       f_phi * ((ga + fg) * A.z - hg * B.z));
    if (cbt) {
        // bending: dE/dtheta1 = 3 s1^2 cos(theta1) s2^3 U, projected with prefactor (dE/dtheta) / sin(theta)
        // (angle_project); theta1 at atom 2 (arms r1 - r2, r3 - r2), theta2 at atom 3 (arms r2 - r3, r4 - r3)
        const float c1 = -fgd * rsqrtf(fgn2), c2 = hgd * rsqrtf(hgn2);
        AngleTerm t1, t2;
        angle_project(F, vb2, ff2, fmaxf(g2, 1e-20f), rsqrtf(ff2), 1.0f / gn, c1,
                      3.0f * s1 * c1 * s2 * s2 * s2 * e_phi, t1);
        angle_project(G, H, fmaxf(g2, 1e-20f), hh2, 1.0f / gn, rsqrtf(hh2), c2,
                      3.0f * s2 * c2 * s1 * s1 * s1 * e_phi, t2);
        t.f1 = make_float3(t.f1.x + t1.fi.x, t.f1.y + t1.fi.y, t.f1.z + t1.fi.z);
        t.f2 = make_float3(t.f2.x - t1.fi.x - t1.fk.x + t2.fi.x, t.f2.y - t1.fi.y - t1.fk.y + t2.fi.y,
                           t.f2.z - t1.fi.z - t1.fk.z + t2.fi.z);
        t.f4 = make_float3(t.f4.x + t2.fk.x, t.f4.y + t2.fk.y, t.f4.z + t2.fk.z);
    }
    t.f3 = make_float3(-t.f1.x - t.f2.x - t.f4.x, -t.f1.y - t.f2.y - t.f4.y, -t.f1.z - t.f2.z - t.f4.z);
    t.e = w * e_phi;
    // same virial convention as the OPLS term: v = vb1 f1 + vb2 f3 + (vb3 + vb2) f4
    t.vir[0] = vb1.x * t.f1.x + vb2.x * t.f3.x + (vb3.x + vb2.x) * t.f4.x;
    t.vir[1] = vb1.x * t.f1.y + vb2.x * t.f3.y + (vb3.x + vb2.x) * t.f4.y;
    t.vir[2] = vb1.x * t.f1.z + vb2.x * t.f3.z + (vb3.x + vb2.x) * t.f4.z;
    t.vir[3] = vb1.y * t.f1.y + vb2.y * t.f3.y + (vb3.y + vb2.y) * t.f4.y;
    t.vir[4] = vb1.y * t.f1.z + vb2.y * t.f3.z + (vb3.y + vb2.y) * t.f4.z;
    t.vir[5] = vb1.z * t.f1.z + vb2.z * t.f3.z + (vb3.z + vb2.z) * t.f4.z;
    return t;
}
