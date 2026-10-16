#include "bond_harmonic.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

__global__ void bond_harmonic_kernel(const float4* __restrict__ pos, float4* __restrict__ force,
                                     double* __restrict__ virial, const int2* __restrict__ bonds,
                                     const int* __restrict__ bond_types,
                                     const BondHarmonicParams* __restrict__ params,
                                     int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (b < nbonds) {
        int2 bond = bonds[b];
        float3 d = min_image_dr(pos[bond.x], pos[bond.y], L, inv_L);
        BondTerm t = harmonic_bond_term(d, params[bond_types[b]]);
        atomicAdd(&force[bond.x].x, t.fi.x);
        atomicAdd(&force[bond.x].y, t.fi.y);
        atomicAdd(&force[bond.x].z, t.fi.z);
        atomicAdd(&force[bond.y].x, -t.fi.x);
        atomicAdd(&force[bond.y].y, -t.fi.y);
        atomicAdd(&force[bond.y].z, -t.fi.z);
        atomicAdd(&force[bond.x].w, 0.5f * t.e);
        atomicAdd(&force[bond.y].w, 0.5f * t.e);
        for (int c = 0; c < 6; c++) vir[c] = t.vir[c];
    }
    block_reduce_virial(vir, virial);
}

void launch_bond_harmonic_kernel(const float4* pos, float4* force, double* virial,
                                 const int2* bonds, const int* bond_types,
                                 const BondHarmonicParams* params,
                                 int nbonds, float L, float inv_L,
                                 cudaStream_t stream) {
    if (nbonds == 0) return;
    bond_harmonic_kernel<<<div_ceil(nbonds, 256), 256, 0, stream>>>(
        pos, force, virial, bonds, bond_types, params, nbonds, L, inv_L);
}
