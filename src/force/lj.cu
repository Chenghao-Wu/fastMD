#include "lj.cuh"
#include "../core/pbc.cuh"
#include <cmath>

template <bool CAP>
__global__ void lj_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const float2* __restrict__ lj_params,
    const float2* __restrict__ lj_cap,
    float force_cap,
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
                float2 params = __ldg(&lj_params[type_i * ntypes + type_j]);
                float eps = params.x, sig = params.y;
                float sig2 = sig * sig;
                float r2inv = __fdividef(1.0f, r2);
                float sr2 = sig2 * r2inv;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                // force_r = F(r) / r, so that F_vec = force_r * dr
                float force_r = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);
                float pe_pair = 4.0f * eps * (sr12 - sr6);
                if (CAP) {
                    float2 cap = __ldg(&lj_cap[type_i * ntypes + type_j]);  // (r_cap, U(r_cap))
                    if (r2 < cap.x * cap.x) {
                        float r = sqrtf(r2);
                        force_r = force_cap / r;
                        pe_pair = cap.y + force_cap * (cap.x - r);
                    }
                }

                float fxij = force_r * dx;
                float fyij = force_r * dy;
                float fzij = force_r * dz;

                fx += fxij; fy += fyij; fz += fzij;
                pe_i += 0.5f * pe_pair;
                vir_xx += 0.5f * fxij * dx;
                vir_xy += 0.5f * fxij * dy;
                vir_xz += 0.5f * fxij * dz;
                vir_yy += 0.5f * fyij * dy;
                vir_yz += 0.5f * fyij * dz;
                vir_zz += 0.5f * fzij * dz;
            }
        }

        force[i] = make_float4(fx, fy, fz, pe_i);
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

void launch_lj_kernel(const float4* pos, float4* force, float* virial,
                       const float2* lj_params,
                       const int* neighbors, const int* num_neighbors,
                       int natoms, int ntypes,
                       float rc2, float L, float inv_L,
                       cudaStream_t stream,
                       const float2* lj_cap, float force_cap)
{
    int blocks = div_ceil(natoms, 256);
    int smem = (256 / 32) * 6 * sizeof(float);
    if (lj_cap && force_cap > 0.0f)
        lj_verlet_kernel<true><<<blocks, 256, smem, stream>>>(
            pos, force, virial, lj_params, lj_cap, force_cap,
            neighbors, num_neighbors, natoms, ntypes, rc2, L, inv_L);
    else
        lj_verlet_kernel<false><<<blocks, 256, smem, stream>>>(
            pos, force, virial, lj_params, nullptr, 0.0f,
            neighbors, num_neighbors, natoms, ntypes, rc2, L, inv_L);
}

std::vector<float2> lj_force_cap_table(const std::vector<float2>& lj_params,
                                       int ntypes, float force_cap) {
    std::vector<float2> cap(lj_params.size(), make_float2(0.0f, 0.0f));
    for (int k = 0; k < ntypes * ntypes; k++) {
        double eps = lj_params[k].x, sig = lj_params[k].y;
        if (eps <= 0.0 || sig <= 0.0) continue;
        auto F = [&](double r) {  // radial LJ force, positive = repulsive
            double sr6 = std::pow(sig / r, 6);
            return 24.0 * eps / r * (2.0 * sr6 * sr6 - sr6);
        };
        // F decreases monotonically on (0, 2^(1/6) sig], where it reaches 0.
        double lo = 1e-3 * sig, hi = std::pow(2.0, 1.0 / 6.0) * sig;
        if (F(lo) <= force_cap) continue;   // cap never reached
        for (int it = 0; it < 200; it++) {
            double mid = 0.5 * (lo + hi);
            (F(mid) > force_cap ? lo : hi) = mid;
        }
        double rcap = 0.5 * (lo + hi);
        double sr6 = std::pow(sig / rcap, 6);
        cap[k] = make_float2((float)rcap, (float)(4.0 * eps * (sr6 * sr6 - sr6)));
    }
    return cap;
}
