#include "bond_harmonic.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

__global__ void bond_harmonic_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const int2* __restrict__ bonds,
    const int* __restrict__ bond_types,
    const BondHarmonicParams* __restrict__ params,
    int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (b < nbonds) {
        int2 bond = bonds[b];
        int type = bond_types[b];
        BondHarmonicParams p = params[type];

        float4 ri = pos[bond.x];
        float4 rj = pos[bond.y];

        float dx = min_image(ri.x - rj.x, L, inv_L);
        float dy = min_image(ri.y - rj.y, L, inv_L);
        float dz = min_image(ri.z - rj.z, L, inv_L);
        float r2 = fmaxf(dx*dx + dy*dy + dz*dz, 1e-12f);
        float r = sqrtf(r2);
        float dr = r - p.r0;

        // ff = -(1/r) dE/dr = -2 K (r - r0) / r
        // (LAMMPS bond_harmonic: fbond = -2.0*k*dr/r)
        float ff = -2.0f * p.k * dr / r;
        float fix = ff * dx;
        float fiy = ff * dy;
        float fiz = ff * dz;

        atomicAdd(&force[bond.x].x, fix);
        atomicAdd(&force[bond.x].y, fiy);
        atomicAdd(&force[bond.x].z, fiz);
        atomicAdd(&force[bond.y].x, -fix);
        atomicAdd(&force[bond.y].y, -fiy);
        atomicAdd(&force[bond.y].z, -fiz);

        float pe = p.k * dr * dr;
        atomicAdd(&force[bond.x].w, 0.5f * pe);
        atomicAdd(&force[bond.y].w, 0.5f * pe);

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

void launch_bond_harmonic_kernel(const float4* pos, float4* force, double* virial,
                                  const int2* bonds, const int* bond_types,
                                  const BondHarmonicParams* params,
                                  int nbonds, float L, float inv_L,
                                  cudaStream_t stream) {
    if (nbonds == 0) return;
    int blocks = div_ceil(nbonds, 256);
    int smem = 0;
    bond_harmonic_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, bonds, bond_types, params, nbonds, L, inv_L);
}
