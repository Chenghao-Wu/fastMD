#include "angle_table.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

// Tabulated angle: U(theta) over the angle list, theta in radians on [0, pi].
// Geometry/force-projection mirrors the harmonic angle kernel; the table force
// column stores f = -dU/dtheta and prefactor = -f / sin(theta) = (dU/dtheta)/sin(theta).
__global__ void angle_table_kernel(
    const float4* __restrict__ pos, float4* __restrict__ force,
    double* __restrict__ virial,
    const int4* __restrict__ angles, const int* __restrict__ angle_table_idx,
    const TableParams* __restrict__ table_params, const float4* __restrict__ table_data,
    int nangles, float L, float inv_L)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float v0=0,v1=0,v2=0,v3=0,v4=0,v5=0;

    if (a < nangles) {
        int4 ang = angles[a];
        int tidx = angle_table_idx[ang.w];
        if (tidx >= 0) {
            int i = ang.x, j = ang.y, k = ang.z;
            float4 ri = pos[i], rj = pos[j], rk = pos[k];
            float dxij = min_image(ri.x-rj.x, L, inv_L);
            float dyij = min_image(ri.y-rj.y, L, inv_L);
            float dzij = min_image(ri.z-rj.z, L, inv_L);
            float dxkj = min_image(rk.x-rj.x, L, inv_L);
            float dykj = min_image(rk.y-rj.y, L, inv_L);
            float dzkj = min_image(rk.z-rj.z, L, inv_L);
            float rij2 = dxij*dxij+dyij*dyij+dzij*dzij;
            float rkj2 = dxkj*dxkj+dykj*dykj+dzkj*dzkj;
            float rij_inv = rsqrtf(rij2), rkj_inv = rsqrtf(rkj2);
            float cos_theta = (dxij*dxkj+dyij*dykj+dzij*dzkj)*rij_inv*rkj_inv;
            cos_theta = fminf(fmaxf(cos_theta,-1.0f),1.0f);
            float theta = acosf(cos_theta);
            float sin_theta = fmaxf(sqrtf(1.0f-cos_theta*cos_theta), 1e-6f);

            TableParams tp = table_params[tidx];
            const float4* tbl = table_data + tp.data_offset;
            float th = fminf(fmaxf(theta, tp.rmin), tp.rmax);
            int idx = (int)((th - tp.rmin) * tp.inv_dr);
            idx = max(0, min(idx, tp.npoints - 2));
            float t = (th - (tp.rmin + idx*tp.dr)) * tp.inv_dr;
            float4 p0 = tbl[idx], p1 = tbl[idx+1];
            float f_theta = p0.y + t*(p1.y - p0.y);   // = -dU/dtheta
            float e       = p0.z + t*(p1.z - p0.z);
            float prefactor = -f_theta / sin_theta;   // = (dU/dtheta)/sin_theta

            float rij2_inv = 1.0f/rij2, rkj2_inv = 1.0f/rkj2, rij_rkj_inv = rij_inv*rkj_inv;
            float fi_x = prefactor*(dxkj*rij_rkj_inv - cos_theta*dxij*rij2_inv);
            float fi_y = prefactor*(dykj*rij_rkj_inv - cos_theta*dyij*rij2_inv);
            float fi_z = prefactor*(dzkj*rij_rkj_inv - cos_theta*dzij*rij2_inv);
            float fk_x = prefactor*(dxij*rij_rkj_inv - cos_theta*dxkj*rkj2_inv);
            float fk_y = prefactor*(dyij*rij_rkj_inv - cos_theta*dykj*rkj2_inv);
            float fk_z = prefactor*(dzij*rij_rkj_inv - cos_theta*dzkj*rkj2_inv);
            atomicAdd(&force[i].x,fi_x); atomicAdd(&force[i].y,fi_y); atomicAdd(&force[i].z,fi_z);
            atomicAdd(&force[k].x,fk_x); atomicAdd(&force[k].y,fk_y); atomicAdd(&force[k].z,fk_z);
            atomicAdd(&force[j].x,-(fi_x+fk_x)); atomicAdd(&force[j].y,-(fi_y+fk_y)); atomicAdd(&force[j].z,-(fi_z+fk_z));
            float third = e/3.0f;
            atomicAdd(&force[i].w,third); atomicAdd(&force[j].w,third); atomicAdd(&force[k].w,third);
            v0 = fi_x*dxij+fk_x*dxkj; v1 = fi_x*dyij+fk_x*dykj; v2 = fi_x*dzij+fk_x*dzkj;
            v3 = fi_y*dyij+fk_y*dykj; v4 = fi_y*dzij+fk_y*dzkj; v5 = fi_z*dzij+fk_z*dzkj;
        }
    }

    float vir[6] = {v0, v1, v2, v3, v4, v5};
    block_reduce_virial(vir, virial);
}

void launch_angle_table_kernel(const float4* pos, float4* force, double* virial,
                               const int4* angles, const int* angle_table_idx,
                               const TableParams* table_params, const float4* table_data,
                               int nangles, float L, float inv_L, cudaStream_t stream) {
    if (nangles == 0) return;
    int blocks = div_ceil(nangles, 256);
    int smem = 0;
    angle_table_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, angles, angle_table_idx,
        table_params, table_data, nangles, L, inv_L);
}
