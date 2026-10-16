#include "bond_table.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

// Tabulated bond U(r); the table stores F(r)/r (see table_parser.hpp).
__global__ void bond_table_kernel(
    const float4* __restrict__ pos, float4* __restrict__ force,
    double* __restrict__ virial,
    const int2* __restrict__ bonds, const int* __restrict__ bond_types,
    const int* __restrict__ bond_table_idx,
    const TableParams* __restrict__ table_params,
    const float4* __restrict__ table_data,
    int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (b < nbonds) {
        int tidx = bond_table_idx[bond_types[b]];
        if (tidx >= 0) {
            TableParams tp = table_params[tidx];
            int2 bond = bonds[b];
            float3 d = min_image_dr(pos[bond.x], pos[bond.y], L, inv_L);
            BondTerm t = table_bond_term(d, tp, table_data + tp.data_offset);
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
    }
    block_reduce_virial(vir, virial);
}

void launch_bond_table_kernel(const float4* pos, float4* force, double* virial,
                              const int2* bonds, const int* bond_types,
                              const int* bond_table_idx,
                              const TableParams* table_params, const float4* table_data,
                              int nbonds, float L, float inv_L, cudaStream_t stream) {
    if (nbonds == 0) return;
    bond_table_kernel<<<div_ceil(nbonds, 256), 256, 0, stream>>>(
        pos, force, virial, bonds, bond_types, bond_table_idx,
        table_params, table_data, nbonds, L, inv_L);
}
