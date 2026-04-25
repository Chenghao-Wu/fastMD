#include "angle.cuh"
#include "../core/pbc.cuh"

__global__ void angle_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int4* __restrict__ angles,
    const AngleParams* __restrict__ params,
    int nangles, float L, float inv_L)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nangles) return;

    int4 ang = angles[a];
    int i = ang.x, j = ang.y, k = ang.z, type = ang.w;
    AngleParams p = params[type];

    float4 ri = pos[i], rj = pos[j], rk = pos[k];

    float dxij = min_image(ri.x - rj.x, L, inv_L);
    float dyij = min_image(ri.y - rj.y, L, inv_L);
    float dzij = min_image(ri.z - rj.z, L, inv_L);
    float dxkj = min_image(rk.x - rj.x, L, inv_L);
    float dykj = min_image(rk.y - rj.y, L, inv_L);
    float dzkj = min_image(rk.z - rj.z, L, inv_L);

    float rij2 = dxij*dxij + dyij*dyij + dzij*dzij;
    float rkj2 = dxkj*dxkj + dykj*dykj + dzkj*dzkj;
    float rij_inv = rsqrtf(rij2);
    float rkj_inv = rsqrtf(rkj2);
    float rij = rij2 * rij_inv;
    float rkj = rkj2 * rkj_inv;

    float cos_theta = (dxij*dxkj + dyij*dykj + dzij*dzkj) * rij_inv * rkj_inv;
    cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
    float theta = acosf(cos_theta);
    float dtheta = theta - p.theta0;
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    sin_theta = fmaxf(sin_theta, 1e-6f);

    float prefactor = -p.k_theta * dtheta / sin_theta;

    float rij2_inv = 1.0f / rij2;
    float rkj2_inv = 1.0f / rkj2;
    float rij_rkj_inv = rij_inv * rkj_inv;

    float fi_x = prefactor * (dxkj * rij_rkj_inv - cos_theta * dxij * rij2_inv);
    float fi_y = prefactor * (dykj * rij_rkj_inv - cos_theta * dyij * rij2_inv);
    float fi_z = prefactor * (dzkj * rij_rkj_inv - cos_theta * dzij * rij2_inv);
    float fk_x = prefactor * (dxij * rij_rkj_inv - cos_theta * dxkj * rkj2_inv);
    float fk_y = prefactor * (dyij * rij_rkj_inv - cos_theta * dykj * rkj2_inv);
    float fk_z = prefactor * (dzij * rij_rkj_inv - cos_theta * dzkj * rkj2_inv);

    atomicAdd(&force[i].x, fi_x);
    atomicAdd(&force[i].y, fi_y);
    atomicAdd(&force[i].z, fi_z);
    atomicAdd(&force[k].x, fk_x);
    atomicAdd(&force[k].y, fk_y);
    atomicAdd(&force[k].z, fk_z);
    atomicAdd(&force[j].x, -(fi_x + fk_x));
    atomicAdd(&force[j].y, -(fi_y + fk_y));
    atomicAdd(&force[j].z, -(fi_z + fk_z));

    float pe = 0.5f * p.k_theta * dtheta * dtheta;
    float pe_third = pe / 3.0f;
    atomicAdd(&force[i].w, pe_third);
    atomicAdd(&force[j].w, pe_third);
    atomicAdd(&force[k].w, pe_third);

    atomicAdd(&virial[0], fi_x * dxij + fk_x * dxkj);
    atomicAdd(&virial[1], fi_x * dyij + fk_x * dykj);
    atomicAdd(&virial[2], fi_x * dzij + fk_x * dzkj);
    atomicAdd(&virial[3], fi_y * dyij + fk_y * dykj);
    atomicAdd(&virial[4], fi_y * dzij + fk_y * dzkj);
    atomicAdd(&virial[5], fi_z * dzij + fk_z * dzkj);
}

void launch_angle_kernel(const float4* pos, float4* force, float* virial,
                          const int4* angles, const AngleParams* params,
                          int nangles, float L, float inv_L,
                          cudaStream_t stream) {
    if (nangles == 0) return;
    int blocks = div_ceil(nangles, 256);
    angle_kernel<<<blocks, 256, 0, stream>>>(
        pos, force, virial, angles, params, nangles, L, inv_L);
}
