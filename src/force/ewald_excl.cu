#include "ewald_excl.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

// erf(x) via Abramowitz & Stegun 7.1.26 (same polynomial as erfc kernel)
__device__ inline float erf_poly(float x) {
    float t = __frcp_rn(1.0f + 0.3275911f * x);
    float poly = t * (0.254829592f
               + t * (-0.284496736f
               + t * (1.421413741f
               + t * (-1.453152027f
               + t * 1.061405429f))));
    return 1.0f - poly * __expf(-x * x);
}

// ---------------------------------------------------------------------------
// Fully-excluded pairs: subtract C q_i q_j erf(alpha r)/r (short-range part of
// the Ewald sum that must NOT be counted for excluded neighbors).
// CSR in tag space: thread t handles entry t of the row of tag ti, i.e. the
// pair (ti, excl_list[t]); indices are recovered through rank[] (nullptr =
// identity). Each pair appears in both rows and is handled once (tj > ti).
// ---------------------------------------------------------------------------
__global__ void ewald_excl_kernel(
    const float4* __restrict__ pos,
    const float* __restrict__ charges,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const int* __restrict__ excl_offsets,
    const int* __restrict__ excl_list,
    const int* __restrict__ rank,
    int natoms, int nexcl,
    float coul_const, float alpha,
    float L, float inv_L)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (t < nexcl) {
        // binary search: find i such that offsets[i] <= t < offsets[i+1]
        int lo = 0, hi = natoms;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (excl_offsets[mid + 1] <= t) lo = mid + 1;
            else hi = mid;
        }
        int ti = lo;
        int tj = excl_list[t];
        if (tj > ti) {   // each pair stored twice in CSR; handle once
            int i = rank ? rank[ti] : ti;
            int j = rank ? rank[tj] : tj;
            float4 ri = pos[i];
            float4 rj = pos[j];
            float dx = min_image(ri.x - rj.x, L, inv_L);
            float dy = min_image(ri.y - rj.y, L, inv_L);
            float dz = min_image(ri.z - rj.z, L, inv_L);
            float r2 = fmaxf(dx*dx + dy*dy + dz*dz, 1e-12f);
            float r = sqrtf(r2);
            float rinv = __frcp_rn(r);

            float qiqj = coul_const * charges[i] * charges[j];
            float ar = alpha * r;
            float erf_ar = erf_poly(ar);
            float expm2 = __expf(-ar * ar);

            // subtract E_sub = qiqj * erf(ar)/r:
            // F_i = -dE_sub/dr * (dr/r),  dr = r_i - r_j
            // dE_sub/dr = qiqj * (2a/sqrt(pi) e^{-a^2 r^2}/r - erf(ar)/r^2)
            // => ff (F per unit dr) = qiqj/r^3 (2 ar/sqrt(pi) e^{-a^2 r^2} - erf(ar))
            // (matches LAMMPS pair_lj_cut_coul_long factor-0 path:
            //  forcecoul = prefactor*(erfc + EWALD_F*grij*expm2 - 1))
            float r3inv = rinv * rinv * rinv;
            float ff = qiqj * r3inv
                * (1.1283791670955126f * ar * expm2 - erf_ar);
            float fix = ff * dx;
            float fiy = ff * dy;
            float fiz = ff * dz;

            atomicAdd(&force[i].x, fix);
            atomicAdd(&force[i].y, fiy);
            atomicAdd(&force[i].z, fiz);
            atomicAdd(&force[j].x, -fix);
            atomicAdd(&force[j].y, -fiy);
            atomicAdd(&force[j].z, -fiz);

            float pe = -qiqj * erf_ar * rinv;
            atomicAdd(&force[i].w, 0.5f * pe);
            atomicAdd(&force[j].w, 0.5f * pe);

            v0 = fix * dx;
            v1 = fix * dy;
            v2 = fix * dz;
            v3 = fiy * dy;
            v4 = fiy * dz;
            v5 = fiz * dz;
        }
    }

    float vir[6] = {v0, v1, v2, v3, v4, v5};
    block_reduce_virial(vir, virial);
}

// ---------------------------------------------------------------------------
// Scaled 1-4 pairs (stored once each):
//   LJ:   lj14 * [4 eps (sr12-sr6)],  F = lj14 * 24 eps r2inv (2 sr12 - sr6)
//   Coul: ewald14 * C qiqj/r  -  C qiqj erf(a r)/r
// ---------------------------------------------------------------------------
__global__ void pair14_kernel(
    const float4* __restrict__ pos,
    const float* __restrict__ charges,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const int2* __restrict__ pairs,
    const float2* __restrict__ lj_params,
    int npairs, int ntypes,
    float coul_const, float alpha,
    float lj14, float ewald14,
    float L, float inv_L)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (p < npairs) {
        int2 pr = pairs[p];
        float4 ri = pos[pr.x];
        float4 rj = pos[pr.y];
        int ti = __float_as_int(ri.w);
        int tj = __float_as_int(rj.w);

        float dx = min_image(ri.x - rj.x, L, inv_L);
        float dy = min_image(ri.y - rj.y, L, inv_L);
        float dz = min_image(ri.z - rj.z, L, inv_L);
        float r2 = fmaxf(dx*dx + dy*dy + dz*dz, 1e-12f);
        float r = sqrtf(r2);
        float rinv = __frcp_rn(r);
        float r2inv = rinv * rinv;

        // LJ scaled (no clamp — see lj_coul_long.cu)
        float2 lp = __ldg(&lj_params[ti * ntypes + tj]);
        float sig2 = lp.y * lp.y;
        float sr2 = sig2 * r2inv;
        float sr6 = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6;
        float lj_ff = lj14 * 24.0f * lp.x * r2inv * (2.0f * sr12 - sr6);
        float lj_pe = lj14 * 4.0f * lp.x * (sr12 - sr6);

        // Coulomb: ewald14 * full - erf subtraction
        float qiqj = coul_const * charges[pr.x] * charges[pr.y];
        float ar = alpha * r;
        float erf_ar = erf_poly(ar);
        float expm2 = __expf(-ar * ar);

        float coul_pe = qiqj * rinv * (ewald14 - erf_ar);
        // E = qiqj (ewald14 - erf(ar))/r
        // ff = qiqj/r^3 (ewald14 - erf(ar) + 2 ar/sqrt(pi) e^{-a^2 r^2})
        // (LAMMPS factor-0.5 path: forcecoul = prefactor*(erfc + F*grij*expm2 - 0.5))
        float coul_ff = qiqj * rinv * r2inv
            * (ewald14 - erf_ar + 1.1283791670955126f * ar * expm2);

        float ff = lj_ff + coul_ff;
        float fix = ff * dx;
        float fiy = ff * dy;
        float fiz = ff * dz;

        atomicAdd(&force[pr.x].x, fix);
        atomicAdd(&force[pr.x].y, fiy);
        atomicAdd(&force[pr.x].z, fiz);
        atomicAdd(&force[pr.y].x, -fix);
        atomicAdd(&force[pr.y].y, -fiy);
        atomicAdd(&force[pr.y].z, -fiz);

        float pe = lj_pe + coul_pe;
        atomicAdd(&force[pr.x].w, 0.5f * pe);
        atomicAdd(&force[pr.y].w, 0.5f * pe);

        v0 = fix * dx;
        v1 = fix * dy;
        v2 = fix * dz;
        v3 = fiy * dy;
        v4 = fiy * dz;
        v5 = fiz * dz;
    }

    float vir[6] = {v0, v1, v2, v3, v4, v5};
    block_reduce_virial(vir, virial);
}

void launch_ewald_excl_kernel(const float4* pos, const float* charges,
                               float4* force, double* virial,
                               const int* excl_offsets, const int* excl_list,
                               const int* rank,
                               int natoms, int nexcl,
                               float coul_const, float alpha,
                               float L, float inv_L,
                               cudaStream_t stream) {
    if (nexcl == 0) return;
    int blocks = div_ceil(nexcl, 256);
    int smem = 0;
    ewald_excl_kernel<<<blocks, 256, smem, stream>>>(
        pos, charges, force, virial, excl_offsets, excl_list, rank,
        natoms, nexcl, coul_const, alpha, L, inv_L);
}

void launch_pair14_kernel(const float4* pos, const float* charges,
                           float4* force, double* virial,
                           const int2* pairs, const float2* lj_params,
                           int npairs, int ntypes,
                           float coul_const, float alpha,
                           float lj14, float ewald14,
                           float L, float inv_L,
                           cudaStream_t stream) {
    if (npairs == 0) return;
    int blocks = div_ceil(npairs, 256);
    int smem = 0;
    pair14_kernel<<<blocks, 256, smem, stream>>>(
        pos, charges, force, virial, pairs, lj_params,
        npairs, ntypes, coul_const, alpha, lj14, ewald14, L, inv_L);
}
