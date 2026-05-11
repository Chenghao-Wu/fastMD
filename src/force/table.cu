#include "table.cuh"
#include "../core/pbc.cuh"

__global__ void table_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int* __restrict__ table_idx,
    const TableParams* __restrict__ table_params,
    const float4* __restrict__ table_data,
    const int* __restrict__ neighbors,
    const int* __restrict__ num_neighbors,
    int natoms, int ntypes,
    float rc2, float L, float inv_L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    if (i < natoms) {
        float4 pos_i = pos[i];
        int type_i = __float_as_int(pos_i.w);
        int nneigh = __ldg(&num_neighbors[i]);

        #pragma unroll 8
        for (int k = 0; k < nneigh; k++) {
            int j = neighbors[k * natoms + i];
            float4 pos_j = __ldg(&pos[j]);
            int type_j = __float_as_int(pos_j.w);

            float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
            float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
            float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2 && r2 > 1e-10f) {
                int tidx = __ldg(&table_idx[type_i * ntypes + type_j]);
                if (tidx >= 0) {
                    TableParams tp = table_params[tidx];
                    float r = sqrtf(r2);
                    if (r < tp.rmin) r = tp.rmin;
                    if (r > tp.rmax) r = tp.rmax;

                    int idx = (int)((r - tp.rmin) * tp.inv_dr);
                    if (idx < 0) idx = 0;
                    if (idx >= tp.npoints - 1) idx = tp.npoints - 2;

                    float t = (r - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
                    float4 p0 = __ldg(&table_data[tp.data_offset + idx]);
                    float4 p1 = __ldg(&table_data[tp.data_offset + idx + 1]);

                    float f_scalar = p0.y + t * (p1.y - p0.y);
                    float e = p0.z + t * (p1.z - p0.z);

                    float fxij = f_scalar * dx;
                    float fyij = f_scalar * dy;
                    float fzij = f_scalar * dz;

                    fx += fxij; fy += fyij; fz += fzij;
                    pe_i += 0.5f * e;
                    vir_xx += 0.5f * fxij * dx;
                    vir_xy += 0.5f * fxij * dy;
                    vir_xz += 0.5f * fxij * dz;
                    vir_yy += 0.5f * fyij * dy;
                    vir_yz += 0.5f * fyij * dz;
                    vir_zz += 0.5f * fzij * dz;
                }
            }
        }

        float4 f = force[i];
        f.x += fx; f.y += fy; f.z += fz; f.w += pe_i;
        force[i] = f;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        vir_xx += __shfl_down_sync(0xFFFFFFFF, vir_xx, offset);
        vir_xy += __shfl_down_sync(0xFFFFFFFF, vir_xy, offset);
        vir_xz += __shfl_down_sync(0xFFFFFFFF, vir_xz, offset);
        vir_yy += __shfl_down_sync(0xFFFFFFFF, vir_yy, offset);
        vir_yz += __shfl_down_sync(0xFFFFFFFF, vir_yz, offset);
        vir_zz += __shfl_down_sync(0xFFFFFFFF, vir_zz, offset);
    }

    extern __shared__ float s_virial[];
    if (lane == 0) {
        int wid = threadIdx.x >> 5;
        s_virial[wid * 6 + 0] = vir_xx;
        s_virial[wid * 6 + 1] = vir_xy;
        s_virial[wid * 6 + 2] = vir_xz;
        s_virial[wid * 6 + 3] = vir_yy;
        s_virial[wid * 6 + 4] = vir_yz;
        s_virial[wid * 6 + 5] = vir_zz;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x >> 5;
        float b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
        for (int w = 0; w < num_warps; w++) {
            b0 += s_virial[w * 6 + 0];
            b1 += s_virial[w * 6 + 1];
            b2 += s_virial[w * 6 + 2];
            b3 += s_virial[w * 6 + 3];
            b4 += s_virial[w * 6 + 4];
            b5 += s_virial[w * 6 + 5];
        }
        atomicAdd(&virial[0], b0);
        atomicAdd(&virial[1], b1);
        atomicAdd(&virial[2], b2);
        atomicAdd(&virial[3], b3);
        atomicAdd(&virial[4], b4);
        atomicAdd(&virial[5], b5);
    }
}

void launch_table_kernel(const float4* pos, float4* force, float* virial,
                         const int* table_idx,
                         const TableParams* table_params,
                         const float4* table_data,
                         const int* neighbors, const int* num_neighbors,
                         int natoms, int ntypes,
                         float rc2, float L, float inv_L,
                         cudaStream_t stream)
{
    int blocks = div_ceil(natoms, 256);
    int smem = (256 / 32) * 6 * sizeof(float);
    table_verlet_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, table_idx, table_params, table_data,
        neighbors, num_neighbors,
        natoms, ntypes, rc2, L, inv_L);
}
