#include "fene.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

__device__ __forceinline__ void add_force(float4* force, int i, float3 f, float e) {
    atomicAdd(&force[i].x, f.x);
    atomicAdd(&force[i].y, f.y);
    atomicAdd(&force[i].z, f.z);
    atomicAdd(&force[i].w, e);
}

__global__ void fene_kernel(const float4* __restrict__ pos, float4* __restrict__ force,
                            double* __restrict__ virial, const int2* __restrict__ bonds,
                            const int* __restrict__ bond_types,
                            const FENEParams* __restrict__ params,
                            int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (b < nbonds) {
        int2 bond = bonds[b];
        float3 d = min_image_dr(pos[bond.x], pos[bond.y], L, inv_L);
        BondTerm t = fene_term(d, params[bond_types[b]]);
        add_force(force, bond.x, t.fi, 0.5f * t.e);
        add_force(force, bond.y, make_float3(-t.fi.x, -t.fi.y, -t.fi.z), 0.5f * t.e);
        for (int c = 0; c < 6; c++) vir[c] = t.vir[c];
    }
    block_reduce_virial(vir, virial);
}

void launch_fene_kernel(const float4* pos, float4* force, double* virial,
                        const int2* bonds, const int* bond_types,
                        const FENEParams* params,
                        int nbonds, int /*nparamtypes*/,
                        float L, float inv_L, cudaStream_t stream) {
    if (nbonds == 0) return;
    fene_kernel<<<div_ceil(nbonds, 256), 256, 0, stream>>>(
        pos, force, virial, bonds, bond_types, params, nbonds, L, inv_L);
}
