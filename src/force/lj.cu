#include "lj.cuh"
#include "../core/pbc.cuh"

__global__ void lj_tile_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const float2* __restrict__ lj_params,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ tile_neighbors,
    int ntiles, int natoms, int ntypes,
    float rc2, float L, float inv_L)
{
    int tile_a = blockIdx.x;
    if (tile_a >= ntiles) return;
    int lane = threadIdx.x;
    int atom_i = tile_a * TILE_SIZE + lane;

    float4 pos_i = (atom_i < natoms) ? pos[atom_i] : make_float4(0,0,0, __int_as_float(-1));
    int type_i = __float_as_int(pos_i.w);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    __shared__ float4 smem_pos[TILE_SIZE];

    int start = tile_offsets[tile_a];
    int end   = tile_offsets[tile_a + 1];

    for (int nb = start; nb < end; nb++) {
        int tile_b = tile_neighbors[nb];

        int atom_j_load = tile_b * TILE_SIZE + lane;
        smem_pos[lane] = (atom_j_load < natoms) ?
            pos[atom_j_load] : make_float4(0,0,0, __int_as_float(-1));
        __syncwarp();

        if (atom_i < natoms) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                float4 pos_j = smem_pos[k];
                int type_j = __float_as_int(pos_j.w);
                int atom_j = tile_b * TILE_SIZE + k;

                if (atom_j >= natoms || atom_i == atom_j) continue;
                if (type_j < 0) continue;

                float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
                float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
                float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
                float r2 = dx*dx + dy*dy + dz*dz;

                if (r2 < rc2 && r2 > 1e-10f) {
                    float2 params = __ldg(&lj_params[type_i * ntypes + type_j]);
                    float eps = params.x, sig = params.y;
                    float sig2 = sig * sig;
                    float r2inv = 1.0f / r2;
                    float sr2 = sig2 * r2inv;
                    float sr6 = sr2 * sr2 * sr2;
                    float sr12 = sr6 * sr6;
                    float force_r = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);

                    float fxij = force_r * dx;
                    float fyij = force_r * dy;
                    float fzij = force_r * dz;

                    fx += fxij;
                    fy += fyij;
                    fz += fzij;

                    pe_i += 0.5f * 4.0f * eps * (sr12 - sr6);

                    vir_xx += 0.5f * fxij * dx;
                    vir_xy += 0.5f * fxij * dy;
                    vir_xz += 0.5f * fxij * dz;
                    vir_yy += 0.5f * fyij * dy;
                    vir_yz += 0.5f * fyij * dz;
                    vir_zz += 0.5f * fzij * dz;
                }
            }
        }
        __syncwarp();
    }

    if (atom_i < natoms) {
        force[atom_i] = make_float4(fx, fy, fz, pe_i);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        vir_xx += __shfl_down_sync(0xFFFFFFFF, vir_xx, offset);
        vir_xy += __shfl_down_sync(0xFFFFFFFF, vir_xy, offset);
        vir_xz += __shfl_down_sync(0xFFFFFFFF, vir_xz, offset);
        vir_yy += __shfl_down_sync(0xFFFFFFFF, vir_yy, offset);
        vir_yz += __shfl_down_sync(0xFFFFFFFF, vir_yz, offset);
        vir_zz += __shfl_down_sync(0xFFFFFFFF, vir_zz, offset);
    }
    if (lane == 0) {
        atomicAdd(&virial[0], vir_xx);
        atomicAdd(&virial[1], vir_xy);
        atomicAdd(&virial[2], vir_xz);
        atomicAdd(&virial[3], vir_yy);
        atomicAdd(&virial[4], vir_yz);
        atomicAdd(&virial[5], vir_zz);
    }
}

void launch_lj_kernel(const float4* pos, float4* force, float* virial,
                       const float2* lj_params,
                       const int* tile_offsets, const int* tile_neighbors,
                       int ntiles, int natoms, int ntypes,
                       float rc2, float L, float inv_L,
                       cudaStream_t stream) {
    lj_tile_kernel<<<ntiles, TILE_SIZE, 0, stream>>>(
        pos, force, virial, lj_params,
        tile_offsets, tile_neighbors,
        ntiles, natoms, ntypes, rc2, L, inv_L);
}
