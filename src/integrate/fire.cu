#include "fire.cuh"
#include "../core/reduce.cuh"
#include "../core/system.cuh"
#include "../neighbor/skin_trigger.cuh"

namespace {

__global__ void fire_reduce_kernel(const float4* __restrict__ vel, const float4* __restrict__ force,
                                   int natoms, FireState* __restrict__ s, FireParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v3[3] = {0.0, 0.0, 0.0};
    float f2 = 0.0f;
    if (i < natoms) {
        const float4 v = vel[i], f = force[i];
        v3[0] = (double)(f.x * v.x + f.y * v.y + f.z * v.z);
        v3[1] = (double)(v.x * v.x + v.y * v.y + v.z * v.z);
        f2 = f.x * f.x + f.y * f.y + f.z * f.z;
        v3[2] = (double)f2;
    }
    block_reduce_atomic_add<3, double>(v3, &s->p);
    for (int o = 16; o > 0; o >>= 1) f2 = fmaxf(f2, __shfl_down_sync(0xffffffffu, f2, o));
    if ((threadIdx.x & 31) == 0) atomicMax((int*)&s->fmax2, __float_as_int(f2));   // f2 >= 0
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = (atomicAdd(&s->ticket, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    const double P = atomicAdd(&s->p, 0.0), vv = atomicAdd(&s->vv, 0.0), ff = atomicAdd(&s->ff, 0.0);
    s->p = s->vv = s->ff = 0.0;
    s->ticket = 0u;
    s->iterations++;
    if (P > 0.0) {
        s->uphill = 0;
        s->mix_v = 1.0f - s->alpha;
        s->mix_f = ff > 0.0 ? (float)(s->alpha * sqrt(vv / ff)) : 0.0f;
        if (++s->n_pos > p.n_delay) {
            s->dt = fminf(s->dt * p.f_inc, p.dt_max);
            s->alpha *= p.f_alpha;
        }
    } else {
        s->uphill = 1;
        s->mix_v = 0.0f;
        s->mix_f = 0.0f;
        s->n_pos = 0;
        s->dt = fmaxf(s->dt * p.f_dec, p.dt_min);
        s->alpha = p.alpha_start;
    }
}

__global__ void fire_update_kernel(float4* __restrict__ pos, float4* __restrict__ vel,
                                   const float4* __restrict__ force, float4* __restrict__ pos_ref,
                                   int* __restrict__ d_max_dr2, int* __restrict__ image, int natoms,
                                   const Box* __restrict__ d_box, const FireState* __restrict__ s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    const Box box = *d_box;
    const float dt = s->dt;
    float4 r = pos[i], v = vel[i];
    const float4 f = force[i];
    if (s->uphill) {   // FIRE 2.0: undo half of the last step, stop
        r.x -= 0.5f * dt * v.x;
        r.y -= 0.5f * dt * v.y;
        r.z -= 0.5f * dt * v.z;
        v.x = v.y = v.z = 0.0f;
    } else {
        v.x = s->mix_v * v.x + s->mix_f * f.x;
        v.y = s->mix_v * v.y + s->mix_f * f.y;
        v.z = s->mix_v * v.z + s->mix_f * f.z;
    }
    const float dt_m = dt / c_masses[unpack_type_id(r.w)];
    v.x += dt_m * f.x;
    v.y += dt_m * f.y;
    v.z += dt_m * f.z;
    r.x += dt * v.x;
    r.y += dt * v.y;
    r.z += dt * v.z;
    wrap_into_box(r, image ? image + 3 * i : nullptr, box);
    update_max_displacement(r, pos_ref[i], d_max_dr2, box);
    pos[i] = r;
    vel[i] = v;
}

}  // namespace

void fire_init(FireState* d_state, const FireParams& p, cudaStream_t stream) {
    FireState h{};
    h.dt = p.dt_start;
    h.alpha = p.alpha_start;
    CUDA_CHECK(cudaMemcpyAsync(d_state, &h, sizeof(FireState), cudaMemcpyHostToDevice, stream));
}

void launch_fire_reduce(const float4* vel, const float4* force, int natoms, FireState* d_state,
                        const FireParams& p, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(&d_state->fmax2, 0, sizeof(float), stream));
    fire_reduce_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(vel, force, natoms, d_state, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fire_update(float4* pos, float4* vel, const float4* force, float4* pos_ref,
                        int* d_max_dr2_bits, int* image, int natoms, const Box* d_box,
                        const FireState* d_state, cudaStream_t stream) {
    fire_update_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(
        pos, vel, force, pos_ref, d_max_dr2_bits, image, natoms, d_box, d_state);
    CUDA_CHECK(cudaGetLastError());
}
