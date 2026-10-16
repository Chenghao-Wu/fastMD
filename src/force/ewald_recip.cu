#include "ewald_recip.cuh"
#include "../core/reduce.cuh"
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Host: half-shell k-vector list matching LAMMPS Ewald::coeffs() ordering and
// spherical cutoff n_x^2 + n_y^2 + n_z^2 <= kmax^2.
// ---------------------------------------------------------------------------
std::vector<int3> ewald_build_kvectors(int kmax, float L, float alpha) {
    std::vector<int3> vecs;
    // unit cell reciprocal unit = 2*pi/L; gsqmx in units of (2*pi/L)^2
    float gsqmx = (float)kmax * kmax;

    auto push = [&](int nx, int ny, int nz) {
        vecs.push_back(make_int3(nx, ny, nz));
    };

    // (m,0,0), (0,m,0), (0,0,m)
    for (int m = 1; m <= kmax; m++) {
        if (m * m <= gsqmx) push(m, 0, 0);
        if (m * m <= gsqmx) push(0, m, 0);
        if (m * m <= gsqmx) push(0, 0, m);
    }
    // (k,l,0), (k,-l,0)
    for (int k = 1; k <= kmax; k++)
        for (int l = 1; l <= kmax; l++) {
            if (k * k + l * l <= gsqmx) { push(k, l, 0); push(k, -l, 0); }
        }
    // (0,l,m), (0,l,-m)
    for (int l = 1; l <= kmax; l++)
        for (int m = 1; m <= kmax; m++) {
            if (l * l + m * m <= gsqmx) { push(0, l, m); push(0, l, -m); }
        }
    // (k,0,m), (k,0,-m)
    for (int k = 1; k <= kmax; k++)
        for (int m = 1; m <= kmax; m++) {
            if (k * k + m * m <= gsqmx) { push(k, 0, m); push(k, 0, -m); }
        }
    // (k,l,m) with all four sign combos having product l*m handled via LAMMPS set:
    // (k,l,m), (k,-l,m), (k,l,-m), (k,-l,-m)
    for (int k = 1; k <= kmax; k++)
        for (int l = 1; l <= kmax; l++)
            for (int m = 1; m <= kmax; m++) {
                if (k * k + l * l + m * m <= gsqmx) {
                    push(k, l, m);
                    push(k, -l, m);
                    push(k, l, -m);
                    push(k, -l, -m);
                }
            }
    return vecs;
}

float ewald_self_offset(const std::vector<float>& charges,
                        float coul_const, float alpha, float L) {
    double qsqsum = 0.0, qsum = 0.0;
    for (float q : charges) { qsqsum += (double)q * q; qsum += q; }
    double V = (double)L * L * L;
    double off = -alpha / sqrt(M_PI) * qsqsum
                 - M_PI_2 * qsum * qsum / (alpha * alpha * V);
    return (float)(coul_const * off);
}

// ---------------------------------------------------------------------------
// Structure factor: one block (256 threads) per k-vector; block-strided sum
// over atoms, shared reduction, lane 0 writes.
// ---------------------------------------------------------------------------
__global__ void ewald_sf_kernel(
    const float4* __restrict__ pos,
    const float* __restrict__ charges,
    const int3* __restrict__ kvecs,
    float* __restrict__ sfac,
    int natoms, float L, const float* __restrict__ d_box)
{
    if (d_box) L = d_box[0];
    int k = blockIdx.x;
    int3 n = kvecs[k];
    float unitk = 2.0f * 3.14159265358979323846f / L;
    float kx = unitk * n.x, ky = unitk * n.y, kz = unitk * n.z;

    float rl = 0.0f, im = 0.0f;
    for (int i = threadIdx.x; i < natoms; i += blockDim.x) {
        float4 p = pos[i];
        float q = charges[i];
        float phase = kx * p.x + ky * p.y + kz * p.z;
        float s, c;
        __sincosf(phase, &s, &c);
        rl += q * c;
        im += q * s;
    }

    // block reduction in double: S(k) sums N terms of either sign
    double w[2] = {rl, im};
    __shared__ double s_part[2][32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int c = 0; c < 2; c++) {
        double t = warp_sum(w[c]);
        if (lane == 0) s_part[c][warp] = t;
    }
    __syncthreads();
    if (warp == 0) {
        int nwarps = blockDim.x >> 5;
        for (int c = 0; c < 2; c++) {
            double t = warp_sum(lane < nwarps ? s_part[c][lane] : 0.0);
            if (lane == 0) sfac[2 * k + c] = (float)t;
        }
    }
}

// ---------------------------------------------------------------------------
// Reciprocal force/energy/virial: one thread per atom, loops over k-vectors.
//   F_i += q_i C sum_k [ Im(e^{-ikr} S(k)) ] * eg_k
// where eg_k = 2 k ug, ug = (4pi/V) exp(-k2/(4 a2)) / k2.
// PE per atom: 0.5 q_i C sum_k ug [ Re(e^{-ikr} S(k)) ]  (half = per-pair share)
// Virial per LAMMPS: virial += C sum_k ug |S|^2 vg_k,
//   vg_ab = delta_ab - 2(1/k2 + 1/(4a2)) k_a k_b (all six components).
// ---------------------------------------------------------------------------
__global__ void ewald_recip_kernel(
    const float4* __restrict__ pos,
    const float* __restrict__ charges,
    const int3* __restrict__ kvecs,
    const float* __restrict__ sfac,
    float4* __restrict__ force,
    double* __restrict__ virial,
    int natoms, int kcount,
    float coul_const, float alpha, float L, const float* __restrict__ d_box)
{
    if (d_box) L = d_box[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float unitk = 2.0f * 3.14159265358979323846f / L;
    float V = L * L * L;
    float preu = 4.0f * 3.14159265358979323846f / V;
    float inv_4a2 = 0.25f / (alpha * alpha);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    // reciprocal virial is a global (per-k-grid) quantity: only block 0
    // accumulates it, strided over k; all other blocks contribute 0.
    bool do_virial = (blockIdx.x == 0);
    float v[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};   // xx xy xz yy yz zz

    if (i < natoms) {
        float4 p = pos[i];
        float qi = charges[i];

        for (int k = 0; k < kcount; k++) {
            int3 n = kvecs[k];
            float kx = unitk * n.x, ky = unitk * n.y, kz = unitk * n.z;
            float k2 = kx*kx + ky*ky + kz*kz;

            float ug = preu * __expf(-k2 * inv_4a2) / k2;

            float phase = kx * p.x + ky * p.y + kz * p.z;
            float s, c;
            __sincosf(phase, &s, &c);
            float sf_rl = __ldg(&sfac[2 * k]);
            float sf_im = __ldg(&sfac[2 * k + 1]);

            // Im(e^{+ik r} S*(k)): matches LAMMPS ewald.cpp
            // partial = expim*sfacrl - exprl*sfacim = s*sf_rl - c*sf_im
            float partial = s * sf_rl - c * sf_im;
            // Re(e^{-ikr} S) = c*sf_rl + s*sf_im
            float partial_pe = c * sf_rl + s * sf_im;

            float egx = 2.0f * kx * ug;
            float egy = 2.0f * ky * ug;
            float egz = 2.0f * kz * ug;

            fx += partial * egx;
            fy += partial * egy;
            fz += partial * egz;
            pe_i += qi * ug * partial_pe;
        }

        float qC = qi * coul_const;
        atomicAdd(&force[i].x, qC * fx);
        atomicAdd(&force[i].y, qC * fy);
        atomicAdd(&force[i].z, qC * fz);
        // per-atom reciprocal PE: q_i C sum_k ug Re(e^{-ikr_i} S(k)).
        // sum_i = C sum_k ug |S|^2 = full reciprocal PE (matches LAMMPS eatom,
        // which is NOT halved for kspace).
        atomicAdd(&force[i].w, coul_const * pe_i);
    }

    // Full reciprocal virial (LAMMPS Ewald::coeffs/compute):
    //   W_ab = C sum_k ug |S(k)|^2 (delta_ab - 2 (1/k^2 + 1/(4 alpha^2)) k_a k_b)
    // Off-diagonal terms are non-zero for any instantaneous configuration,
    // cubic box or not, and enter the shear-stress autocorrelation.
    if (do_virial) {
        for (int k = threadIdx.x; k < kcount; k += blockDim.x) {
            int3 n = kvecs[k];
            float kx = unitk * n.x, ky = unitk * n.y, kz = unitk * n.z;
            float k2 = kx*kx + ky*ky + kz*kz;
            float ug = preu * __expf(-k2 * inv_4a2) / k2;
            float sf_rl = __ldg(&sfac[2 * k]);
            float sf_im = __ldg(&sfac[2 * k + 1]);
            float uk = ug * (sf_rl * sf_rl + sf_im * sf_im);
            float vterm = -2.0f * (1.0f / k2 + inv_4a2);
            v[0] += uk * (1.0f + vterm * kx * kx);
            v[1] += uk * vterm * kx * ky;
            v[2] += uk * vterm * kx * kz;
            v[3] += uk * (1.0f + vterm * ky * ky);
            v[4] += uk * vterm * ky * kz;
            v[5] += uk * (1.0f + vterm * kz * kz);
        }
        // do_virial is uniform within the block, so the barrier inside is safe
        for (int c = 0; c < 6; c++) v[c] *= coul_const;
        block_reduce_virial(v, virial);
    }
}

void launch_ewald_sf_kernel(const float4* pos, const float* charges,
                             const int3* kvecs, float* sfac,
                             int natoms, int kcount, float L,
                             cudaStream_t stream, const float* d_box) {
    if (kcount == 0) return;
    ewald_sf_kernel<<<kcount, 256, 0, stream>>>(
        pos, charges, kvecs, sfac, natoms, L, d_box);
}

void launch_ewald_recip_kernel(const float4* pos, const float* charges,
                                const int3* kvecs, const float* sfac,
                                float4* force, double* virial,
                                int natoms, int kcount,
                                float coul_const, float alpha, float L,
                                cudaStream_t stream, const float* d_box) {
    if (kcount == 0) return;
    int blocks = div_ceil(natoms, 256);
    ewald_recip_kernel<<<blocks, 256, 0, stream>>>(
        pos, charges, kvecs, sfac, force, virial,
        natoms, kcount, coul_const, alpha, L, d_box);
}
