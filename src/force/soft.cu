#include "soft.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"
#include "../core/launch.cuh"

template <int TPA, bool EV>
__global__ void soft_kernel(const float4* __restrict__ pos, float4* __restrict__ force,
                            double* __restrict__ virial, const int* __restrict__ neighbors,
                            const int* __restrict__ num_neighbors, int natoms, float rc,
                            float A_start, float A_stop, int nsteps, const int* __restrict__ d_step,
                            Box box, const Box* __restrict__ d_box) {
    if (d_box) box = *d_box;
    const int step = d_step ? *d_step : 1;
    const float A = A_start + (A_stop - A_start) *
                    (nsteps > 0 ? fminf((float)(step - 1) / (float)nsteps, 1.0f) : 0.0f);
    const float k = 3.14159265358979f / rc, rc2 = rc * rc;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = t / TPA;
    const int sub = t & (TPA - 1);
    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe = 0.0f;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    if (i < natoms) {
        const float4 pi = pos[i];
        const int nn = __ldg(&num_neighbors[i]);
        for (int q = sub; q < nn; q += TPA) {
            const int j = neighbors[q * natoms + i];
            const float4 pj = __ldg(&pos[j]);
            const float3 dd_ = min_image_dr(pi, pj, box);
            float dx = dd_.x, dy = dd_.y, dz = dd_.z;
            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rc2 && r2 > 1e-12f) {
                float r = sqrtf(r2);
                float s, c;
                __sincosf(k * r, &s, &c);
                float f_over_r = A * k * s / r;
                float fxij = f_over_r * dx, fyij = f_over_r * dy, fzij = f_over_r * dz;
                fx += fxij; fy += fyij; fz += fzij;
                if (EV) {
                    pe += 0.5f * A * (1.0f + c);
                    vir[0] += 0.5f * fxij * dx; vir[1] += 0.5f * fxij * dy; vir[2] += 0.5f * fxij * dz;
                    vir[3] += 0.5f * fyij * dy; vir[4] += 0.5f * fyij * dz; vir[5] += 0.5f * fzij * dz;
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
            if (EV) pe += __shfl_down_sync(0xffffffffu, pe, o, TPA);
        }
    }
    if (i < natoms && sub == 0) force[i] = make_float4(fx, fy, fz, pe);
    if (EV) block_reduce_virial(vir, virial);
}

void launch_soft_kernel(const float4* pos, float4* force, double* virial,
                        const int* neighbors, const int* num_neighbors,
                        int natoms, float rc, float A_start, float A_stop, int nsteps,
                        const int* d_step, Box box, cudaStream_t stream,
                        int tpa, const Box* d_box, bool ev) {
    if (tpa <= 0) tpa = pair_threads_per_atom(natoms);
    const int blocks = div_ceil(natoms * tpa, 256);
#define FASTMD_SOFT(T, E) soft_kernel<T, E><<<blocks, 256, 0, stream>>>(                 \
        pos, force, virial, neighbors, num_neighbors, natoms, rc, A_start, A_stop, nsteps, \
        d_step, box, d_box)
#define FASTMD_SOFT_T(T) do { if (ev) FASTMD_SOFT(T, true); else FASTMD_SOFT(T, false); } while (0)
    switch (tpa) {
        case 1: FASTMD_SOFT_T(1); break;
        case 2: FASTMD_SOFT_T(2); break;
        case 4: FASTMD_SOFT_T(4); break;
        default: FASTMD_SOFT_T(8); break;
    }
#undef FASTMD_SOFT_T
#undef FASTMD_SOFT
    CUDA_CHECK(cudaGetLastError());
}
