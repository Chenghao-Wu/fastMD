#include "angle.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

template <bool COSINE>
__global__ void angle_kernel(const float4* __restrict__ pos, float4* __restrict__ force,
                             double* __restrict__ virial, const int4* __restrict__ angles,
                             const AngleParams* __restrict__ params,
                             int nangles, Box box)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (a < nangles) {
        int4 ang = angles[a];
        float4 rj = pos[ang.y];
        float3 dij = min_image_dr(pos[ang.x], rj, box);
        float3 dkj = min_image_dr(pos[ang.z], rj, box);
        AngleTerm t = COSINE ? cosine_angle_term(dij, dkj, params[ang.w])
                             : harmonic_angle_term(dij, dkj, params[ang.w]);
        atomicAdd(&force[ang.x].x, t.fi.x);
        atomicAdd(&force[ang.x].y, t.fi.y);
        atomicAdd(&force[ang.x].z, t.fi.z);
        atomicAdd(&force[ang.z].x, t.fk.x);
        atomicAdd(&force[ang.z].y, t.fk.y);
        atomicAdd(&force[ang.z].z, t.fk.z);
        atomicAdd(&force[ang.y].x, -(t.fi.x + t.fk.x));
        atomicAdd(&force[ang.y].y, -(t.fi.y + t.fk.y));
        atomicAdd(&force[ang.y].z, -(t.fi.z + t.fk.z));
        float third = t.e / 3.0f;
        atomicAdd(&force[ang.x].w, third);
        atomicAdd(&force[ang.y].w, third);
        atomicAdd(&force[ang.z].w, third);
        for (int c = 0; c < 6; c++) vir[c] = t.vir[c];
    }
    block_reduce_virial(vir, virial);
}

void launch_angle_kernel(const float4* pos, float4* force, double* virial,
                         const int4* angles, const AngleParams* params,
                         int nangles, Box box,
                         cudaStream_t stream, bool cosine) {
    if (nangles == 0) return;
    if (cosine)
        angle_kernel<true><<<div_ceil(nangles, 256), 256, 0, stream>>>(
            pos, force, virial, angles, params, nangles, box);
    else
        angle_kernel<false><<<div_ceil(nangles, 256), 256, 0, stream>>>(
            pos, force, virial, angles, params, nangles, box);
}
