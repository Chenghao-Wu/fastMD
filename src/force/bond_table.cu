#include "bond_table.cuh"
#include "../core/pbc.cuh"

// Tabulated bond: U(r) over the bond list. Reuses the nonbonded table format
// (parse_table_file). The force column is packed as F(r)/r so that
// force_vec = f_scalar * d_component matches the nonbonded table kernel.
__global__ void bond_table_kernel(
    const float4* __restrict__ pos, float4* __restrict__ force,
    float* __restrict__ virial,
    const int2* __restrict__ bonds, const int* __restrict__ bond_types,
    const int* __restrict__ bond_table_idx,
    const TableParams* __restrict__ table_params,
    const float4* __restrict__ table_data,
    int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float v0=0,v1=0,v2=0,v3=0,v4=0,v5=0;

    if (b < nbonds) {
        int tidx = bond_table_idx[bond_types[b]];
        if (tidx >= 0) {
            TableParams tp = table_params[tidx];
            const float4* tbl = table_data + tp.data_offset;
            int2 bond = bonds[b];
            float4 ri = pos[bond.x], rj = pos[bond.y];
            float dx = min_image(ri.x - rj.x, L, inv_L);
            float dy = min_image(ri.y - rj.y, L, inv_L);
            float dz = min_image(ri.z - rj.z, L, inv_L);
            float r = sqrtf(dx*dx + dy*dy + dz*dz);

            float rr = fminf(fmaxf(r, tp.rmin), tp.rmax);
            int idx = (int)((rr - tp.rmin) * tp.inv_dr);
            idx = max(0, min(idx, tp.npoints - 2));
            float t = (rr - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
            float4 p0 = tbl[idx], p1 = tbl[idx + 1];
            float f_scalar = p0.y + t * (p1.y - p0.y);   // = F(r)/r
            float e        = p0.z + t * (p1.z - p0.z);

            float fix = f_scalar * dx, fiy = f_scalar * dy, fiz = f_scalar * dz;
            atomicAdd(&force[bond.x].x, fix); atomicAdd(&force[bond.x].y, fiy); atomicAdd(&force[bond.x].z, fiz);
            atomicAdd(&force[bond.y].x, -fix); atomicAdd(&force[bond.y].y, -fiy); atomicAdd(&force[bond.y].z, -fiz);
            atomicAdd(&force[bond.x].w, 0.5f * e); atomicAdd(&force[bond.y].w, 0.5f * e);

            v0 = fix*dx; v1 = fix*dy; v2 = fix*dz; v3 = fiy*dy; v4 = fiy*dz; v5 = fiz*dz;
        }
    }

    extern __shared__ float s_virial[];
    s_virial[tid*6+0]=v0; s_virial[tid*6+1]=v1; s_virial[tid*6+2]=v2;
    s_virial[tid*6+3]=v3; s_virial[tid*6+4]=v4; s_virial[tid*6+5]=v5;
    __syncthreads();
    if (tid == 0) {
        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0;
        for (int u=0; u<blockDim.x; u++){ s0+=s_virial[u*6+0]; s1+=s_virial[u*6+1]; s2+=s_virial[u*6+2];
            s3+=s_virial[u*6+3]; s4+=s_virial[u*6+4]; s5+=s_virial[u*6+5]; }
        atomicAdd(&virial[0],s0); atomicAdd(&virial[1],s1); atomicAdd(&virial[2],s2);
        atomicAdd(&virial[3],s3); atomicAdd(&virial[4],s4); atomicAdd(&virial[5],s5);
    }
}

void launch_bond_table_kernel(const float4* pos, float4* force, float* virial,
                              const int2* bonds, const int* bond_types,
                              const int* bond_table_idx,
                              const TableParams* table_params, const float4* table_data,
                              int nbonds, float L, float inv_L, cudaStream_t stream) {
    if (nbonds == 0) return;
    int blocks = div_ceil(nbonds, 256);
    int smem = 256 * 6 * sizeof(float);
    bond_table_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, bonds, bond_types, bond_table_idx,
        table_params, table_data, nbonds, L, inv_L);
}
