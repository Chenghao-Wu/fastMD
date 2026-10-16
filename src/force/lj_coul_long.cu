#include "lj_coul_long.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"
#include "../core/launch.cuh"

// LAMMPS EWALD_P/A1..A5 erfc polynomial (Abramowitz & Stegun 7.1.26)
__device__ inline float erfc_poly(float x) {
    float t = __frcp_rn(1.0f + 0.3275911f * x);
    float poly = t * (0.254829592f
               + t * (-0.284496736f
               + t * (1.421413741f
               + t * (-1.453152027f
               + t * 1.061405429f))));
    return poly * __expf(-x * x);
}

template <int TPA>
__global__ void lj_coul_long_kernel(
    const float4* __restrict__ pos,
    const float* __restrict__ charges,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const float2* __restrict__ lj_params,
    const int* __restrict__ neighbors,
    const int* __restrict__ num_neighbors,
    int natoms, int ntypes,
    float coul_const, float alpha,
    float rc2, float L, float inv_L, const float* __restrict__ d_box)
{
    if (d_box) { L = d_box[0]; inv_L = d_box[1]; }   // box on the device (NPT)
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = t / TPA;
    const int sub = t & (TPA - 1);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    if (i < natoms) {
        float4 pos_i = pos[i];
        int type_i = __float_as_int(pos_i.w);
        float qi = charges[i];

        int nneigh = __ldg(&num_neighbors[i]);
        #pragma unroll 4
        for (int k = sub; k < nneigh; k += TPA) {
            int j = neighbors[k * natoms + i];
            float4 pos_j = __ldg(&pos[j]);
            int type_j = __float_as_int(pos_j.w);

            float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
            float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
            float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2 && r2 > 1e-10f) {
                float2 params = __ldg(&lj_params[type_i * ntypes + type_j]);
                float eps = params.x, sig = params.y;
                float sig2 = sig * sig;
                float r = sqrtf(r2);
                float rinv = __frcp_rn(r);
                float r2inv = rinv * rinv;

                // LJ 12-6 (no force clamp: all-atom OPLS has legitimate
                // forces > 1000 kJ/(mol nm^2) at close H contacts)
                float sr2 = sig2 * r2inv;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float lj_ff = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);
                float lj_pe = 4.0f * eps * (sr12 - sr6);

                // real-space Coulomb: C q_i q_j erfc(alpha r)/r
                float qiqj = coul_const * qi * charges[j];
                float ar = alpha * r;
                float erfc_ar = erfc_poly(ar);
                float expm2 = __expf(-ar * ar);
                float coul_pe = qiqj * erfc_ar * rinv;
                // F_i = C qiqj (erfc(ar)/r + 2a/sqrt(pi) e^{-a^2 r^2}) / r^3 * dr
                float r3inv = rinv * r2inv;
                float coul_ff = qiqj * r3inv
                    * (erfc_ar + 1.1283791670955126f * ar * expm2);

                float ff = lj_ff + coul_ff;
                float fxij = ff * dx;
                float fyij = ff * dy;
                float fzij = ff * dz;

                fx += fxij; fy += fyij; fz += fzij;
                pe_i += 0.5f * (lj_pe + coul_pe);
                vir_xx += 0.5f * fxij * dx;
                vir_xy += 0.5f * fxij * dy;
                vir_xz += 0.5f * fxij * dz;
                vir_yy += 0.5f * fyij * dy;
                vir_yz += 0.5f * fyij * dz;
                vir_zz += 0.5f * fzij * dz;
            }
        }
    }
    if (TPA > 1) {
#pragma unroll
        for (int o = TPA / 2; o > 0; o >>= 1) {
            fx += __shfl_down_sync(0xffffffffu, fx, o, TPA);
            fy += __shfl_down_sync(0xffffffffu, fy, o, TPA);
            fz += __shfl_down_sync(0xffffffffu, fz, o, TPA);
            pe_i += __shfl_down_sync(0xffffffffu, pe_i, o, TPA);
        }
    }
    if (i < natoms && sub == 0) force[i] = make_float4(fx, fy, fz, pe_i);

    float vir[6] = {vir_xx, vir_xy, vir_xz, vir_yy, vir_yz, vir_zz};
    block_reduce_virial(vir, virial);
}

void launch_lj_coul_long_kernel(const float4* pos, const float* charges,
                                 float4* force, double* virial,
                                 const float2* lj_params,
                                 const int* neighbors, const int* num_neighbors,
                                 int natoms, int ntypes,
                                 float coul_const, float alpha,
                                 float rc2, float L, float inv_L,
                                 cudaStream_t stream, int tpa, const float* d_box)
{
    if (tpa <= 0) tpa = pair_threads_per_atom(natoms);
    const int blocks = div_ceil(natoms * tpa, 256);
#define FASTMD_LAUNCH(T) lj_coul_long_kernel<T><<<blocks, 256, 0, stream>>>(       \
        pos, charges, force, virial, lj_params, neighbors, num_neighbors,          \
        natoms, ntypes, coul_const, alpha, rc2, L, inv_L, d_box)
    switch (tpa) {
        case 1: FASTMD_LAUNCH(1); break;
        case 2: FASTMD_LAUNCH(2); break;
        case 4: FASTMD_LAUNCH(4); break;
        default: FASTMD_LAUNCH(8); break;
    }
#undef FASTMD_LAUNCH
}
