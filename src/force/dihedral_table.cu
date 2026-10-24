#include "dihedral_table.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

__global__ void dihedral_table_kernel(const float4* __restrict__ pos, float4* __restrict__ force,
                                      double* __restrict__ virial,
                                      const int4* __restrict__ dihedrals,
                                      const int* __restrict__ dihedral_types,
                                      const int* __restrict__ table_idx,
                                      const TableParams* __restrict__ table_params,
                                      const float4* __restrict__ table_data,
                                      int ndihedrals, Box box)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (d < ndihedrals) {
        int tidx = table_idx[dihedral_types[d]];
        if (tidx >= 0) {
            int4 dih = dihedrals[d];
            float4 r2 = pos[dih.y], r3 = pos[dih.z];
            TableParams tp = table_params[tidx];
            DihedralTerm t = table_dihedral_term(min_image_dr(pos[dih.x], r2, box),
                                                 min_image_dr(r3, r2, box),
                                                 min_image_dr(pos[dih.w], r3, box),
                                                 tp, table_data + tp.data_offset);
            const int idx[4] = {dih.x, dih.y, dih.z, dih.w};
            const float3 f[4] = {t.f1, t.f2, t.f3, t.f4};
            for (int a = 0; a < 4; a++) {
                atomicAdd(&force[idx[a]].x, f[a].x);
                atomicAdd(&force[idx[a]].y, f[a].y);
                atomicAdd(&force[idx[a]].z, f[a].z);
                atomicAdd(&force[idx[a]].w, 0.25f * t.e);
            }
            for (int c = 0; c < 6; c++) vir[c] = t.vir[c];
        }
    }
    block_reduce_virial(vir, virial);
}

void launch_dihedral_table_kernel(const float4* pos, float4* force, double* virial,
                                  const int4* dihedrals, const int* dihedral_types,
                                  const int* table_idx, const TableParams* table_params,
                                  const float4* table_data, int ndihedrals, Box box,
                                  cudaStream_t stream) {
    if (ndihedrals == 0) return;
    dihedral_table_kernel<<<div_ceil(ndihedrals, 256), 256, 0, stream>>>(
        pos, force, virial, dihedrals, dihedral_types, table_idx, table_params, table_data, ndihedrals, box);
}
