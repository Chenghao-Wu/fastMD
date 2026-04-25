#include "fene.cuh"
#include "../core/pbc.cuh"

__global__ void fene_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int2* __restrict__ bonds,
    const int* __restrict__ bond_types,
    const FENEParams* __restrict__ params,
    int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbonds) return;

    int2 bond = bonds[b];
    int type = bond_types[b];
    FENEParams p = params[type];

    float4 ri = pos[bond.x];
    float4 rj = pos[bond.y];

    float dx = min_image(ri.x - rj.x, L, inv_L);
    float dy = min_image(ri.y - rj.y, L, inv_L);
    float dz = min_image(ri.z - rj.z, L, inv_L);
    float r2 = dx*dx + dy*dy + dz*dz;

    float R02 = p.R0 * p.R0;

    float fene_ff = -p.k * R02 / (R02 - r2);

    float sig2 = p.sig * p.sig;
    float r_cut2 = sig2 * 1.2599210498948732f;
    float wca_ff = 0.0f;
    float wca_pe = 0.0f;
    if (r2 < r_cut2) {
        float r2inv = 1.0f / r2;
        float sr2 = sig2 * r2inv;
        float sr6 = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6;
        wca_ff = 24.0f * p.eps * r2inv * (2.0f * sr12 - sr6);
        wca_pe = 4.0f * p.eps * (sr12 - sr6) + p.eps;
    }

    float total_ff = fene_ff + wca_ff;
    float fix = total_ff * dx;
    float fiy = total_ff * dy;
    float fiz = total_ff * dz;

    atomicAdd(&force[bond.x].x, fix);
    atomicAdd(&force[bond.x].y, fiy);
    atomicAdd(&force[bond.x].z, fiz);
    atomicAdd(&force[bond.y].x, -fix);
    atomicAdd(&force[bond.y].y, -fiy);
    atomicAdd(&force[bond.y].z, -fiz);

    float fene_pe = -0.5f * p.k * R02 * logf(1.0f - r2 / R02);
    float pair_pe = fene_pe + wca_pe;
    atomicAdd(&force[bond.x].w, 0.5f * pair_pe);
    atomicAdd(&force[bond.y].w, 0.5f * pair_pe);

    atomicAdd(&virial[0], fix * dx);
    atomicAdd(&virial[1], fix * dy);
    atomicAdd(&virial[2], fix * dz);
    atomicAdd(&virial[3], fiy * dy);
    atomicAdd(&virial[4], fiy * dz);
    atomicAdd(&virial[5], fiz * dz);
}

void launch_fene_kernel(const float4* pos, float4* force, float* virial,
                         const int2* bonds, const int* bond_types,
                         const FENEParams* params,
                         int nbonds, int nparamtypes,
                         float L, float inv_L, cudaStream_t stream) {
    if (nbonds == 0) return;
    int blocks = div_ceil(nbonds, 256);
    fene_kernel<<<blocks, 256, 0, stream>>>(
        pos, force, virial, bonds, bond_types, params,
        nbonds, L, inv_L);
}
