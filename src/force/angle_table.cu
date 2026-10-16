#include "angle_table.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

// Tabulated angle U(theta), theta in radians on [0, pi]; the table stores
// -dU/dtheta per radian (see table_parser.hpp).
__global__ void angle_table_kernel(
    const float4* __restrict__ pos, float4* __restrict__ force,
    double* __restrict__ virial,
    const int4* __restrict__ angles, const int* __restrict__ angle_table_idx,
    const TableParams* __restrict__ table_params, const float4* __restrict__ table_data,
    int nangles, Box box)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (a < nangles) {
        int4 ang = angles[a];
        int tidx = angle_table_idx[ang.w];
        if (tidx >= 0) {
            TableParams tp = table_params[tidx];
            float4 rj = pos[ang.y];
            AngleTerm t = table_angle_term(min_image_dr(pos[ang.x], rj, box),
                                           min_image_dr(pos[ang.z], rj, box),
                                           tp, table_data + tp.data_offset);
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
    }
    block_reduce_virial(vir, virial);
}

void launch_angle_table_kernel(const float4* pos, float4* force, double* virial,
                               const int4* angles, const int* angle_table_idx,
                               const TableParams* table_params, const float4* table_data,
                               int nangles, Box box, cudaStream_t stream) {
    if (nangles == 0) return;
    angle_table_kernel<<<div_ceil(nangles, 256), 256, 0, stream>>>(
        pos, force, virial, angles, angle_table_idx, table_params, table_data,
        nangles, box);
}
