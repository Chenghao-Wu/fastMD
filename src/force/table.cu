#include "table.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"
#include "../core/launch.cuh"

#define TABLE_MAX_SHARED 2048

template <int TPA>
__global__ void table_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const int* __restrict__ table_idx,
    const TableParams* __restrict__ table_params,
    const float4* __restrict__ table_data,
    const int* __restrict__ neighbors,
    const int* __restrict__ num_neighbors,
    int natoms, int ntypes,
    float rc2, float L, float inv_L)
{
    __shared__ float4 s_table[TABLE_MAX_SHARED];

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = t / TPA;
    const int sub = t & (TPA - 1);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    // --- Cooperative table load (once per block) ---
    int first = min(blockIdx.x * blockDim.x / TPA, natoms - 1);
    int rep_type = __float_as_int(pos[first].w);
    int tidx_rep = __ldg(&table_idx[rep_type * ntypes + rep_type]);
    bool use_smem = false;
    float rmin_s = 0.0f, rmax_s = 0.0f, dr_s = 0.0f, inv_dr_s = 0.0f;
    int npoints_s = 0;

    if (tidx_rep >= 0) {
        TableParams tp = table_params[tidx_rep];
        if (tp.npoints <= TABLE_MAX_SHARED) {
            rmin_s   = tp.rmin;
            rmax_s   = tp.rmax;
            dr_s     = tp.dr;
            inv_dr_s = tp.inv_dr;
            npoints_s = tp.npoints;
            use_smem  = true;
        }
    }
    if (use_smem) {
        for (int t = threadIdx.x; t < npoints_s; t += blockDim.x)
            s_table[t] = __ldg(&table_data[table_params[tidx_rep].data_offset + t]);
    }
    __syncthreads();

    if (i < natoms) {
        float4 pos_i = pos[i];
        int type_i = __float_as_int(pos_i.w);
        int nneigh = __ldg(&num_neighbors[i]);
        int type_i_offset = type_i * ntypes;

        #pragma unroll 8
        for (int k = sub; k < nneigh; k += TPA) {
            int j = neighbors[k * natoms + i];
            float4 pos_j = __ldg(&pos[j]);
            int type_j = __float_as_int(pos_j.w);

            float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
            float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
            float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2 && r2 > 1e-10f) {
                int tidx = __ldg(&table_idx[type_i_offset + type_j]);
                if (tidx >= 0) {
                    // Choose shared memory if the table matches, else global
                    const float4* tbl;
                    float rmin, rmax, dr, inv_dr;
                    int npoints;

                    if (use_smem && tidx == tidx_rep) {
                        tbl = s_table;
                        rmin = rmin_s; rmax = rmax_s;
                        dr = dr_s; inv_dr = inv_dr_s;
                        npoints = npoints_s;
                    } else {
                        TableParams tp = table_params[tidx];
                        tbl = table_data + tp.data_offset;
                        rmin = tp.rmin; rmax = tp.rmax;
                        dr = tp.dr; inv_dr = tp.inv_dr;
                        npoints = tp.npoints;
                    }

                    float r = sqrtf(r2);
                    if (r < rmin) r = rmin;
                    if (r > rmax) r = rmax;

                    int idx = (int)((r - rmin) * inv_dr);
                    if (idx < 0) idx = 0;
                    if (idx >= npoints - 1) idx = npoints - 2;

                    float t = (r - (rmin + idx * dr)) * inv_dr;
                    float4 p0 = tbl[idx];
                    float4 p1 = tbl[idx + 1];

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

    }
    if (TPA > 1) {
#pragma unroll
        for (int o = TPA / 2; o > 0; o >>= 1) {
            fx += __shfl_down_sync(0xffffffffu, fx, o, TPA);
            fy += __shfl_down_sync(0xffffffffu, fy, o, TPA);
            fz += __shfl_down_sync(0xffffffffu, fz, o, TPA);
            pe_i += __shfl_down_sync(0xffffffffu, pe_i, o, TPA);
        }
    }
    if (i < natoms && sub == 0) {
        float4 f = force[i];
        f.x += fx; f.y += fy; f.z += fz; f.w += pe_i;
        force[i] = f;
    }

    float vir[6] = {vir_xx, vir_xy, vir_xz, vir_yy, vir_yz, vir_zz};
    block_reduce_virial(vir, virial);
}

void launch_table_kernel(const float4* pos, float4* force, double* virial,
                         const int* table_idx,
                         const TableParams* table_params,
                         const float4* table_data,
                         const int* neighbors, const int* num_neighbors,
                         int natoms, int ntypes,
                         float rc2, float L, float inv_L,
                         cudaStream_t stream, int tpa)
{
    if (tpa <= 0) tpa = pair_threads_per_atom(natoms);
    const int blocks = div_ceil(natoms * tpa, 256);
#define FASTMD_LAUNCH(T) table_verlet_kernel<T><<<blocks, 256, 0, stream>>>(       \
        pos, force, virial, table_idx, table_params, table_data,                  \
        neighbors, num_neighbors, natoms, ntypes, rc2, L, inv_L)
    switch (tpa) {
        case 1: FASTMD_LAUNCH(1); break;
        case 2: FASTMD_LAUNCH(2); break;
        case 4: FASTMD_LAUNCH(4); break;
        default: FASTMD_LAUNCH(8); break;
    }
#undef FASTMD_LAUNCH
}
