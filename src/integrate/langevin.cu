#include "langevin.cuh"
#include "../core/pbc.cuh"
#include "../core/system.cuh"
#include "../neighbor/skin_trigger.cuh"
#include <curand_kernel.h>

void LangevinState::init(int /*natoms_padded*/, float Tdamp, float dt,
                         float T_start_in, uint64_t seed_in, float T_stop_in, int nsteps_in) {
    float gamma = 1.0f / Tdamp;
    c1 = expf(-gamma * dt);
    c2 = sqrtf(1.0f - c1 * c1);
    half_dt = 0.5f * dt;
    T_start = T_start_in;
    T_stop = T_stop_in < 0.0f ? T_start_in : T_stop_in;
    nsteps = nsteps_in;
    seed = seed_in;
}

__global__ void integrator_pre_force_kernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float4* __restrict__ pos_ref,
    int* __restrict__ d_max_dr2_int,
    int* __restrict__ image,
    const int* __restrict__ tag,
    uint64_t seed, const int* __restrict__ d_step,
    int natoms, float L, float inv_L,
    float half_dt, float c1, float c2,
    float T_start, float T_stop, int nsteps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    const int step = *d_step;
    const float kT = T_start + (T_stop - T_start) *
                     (nsteps > 0 ? (float)(step - 1) / (float)nsteps : 0.0f);

    float4 r = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    int type_i = unpack_type_id(r.w);
    float m_i = c_masses[type_i];
    float half_dt_over_m = half_dt / m_i;

    v.x += half_dt_over_m * f.x;
    v.y += half_dt_over_m * f.y;
    v.z += half_dt_over_m * f.z;

    r.x += half_dt * v.x;
    r.y += half_dt * v.y;
    r.z += half_dt * v.z;

    if (c2 > 0.0f) {   // c2 = 0: plain velocity Verlet (NVE), no random numbers drawn
        // Philox stream: subsequence = particle tag, one 4-word block per step
        curandStatePhilox4_32_10_t rs;
        curand_init(seed, (unsigned long long)(tag ? tag[i] : i), (unsigned long long)step * 4ull, &rs);
        float4 rand4 = curand_normal4(&rs);

        float noise_scale = c2 * sqrtf(kT / m_i);
        v.x = c1 * v.x + noise_scale * rand4.x;
        v.y = c1 * v.y + noise_scale * rand4.y;
        v.z = c1 * v.z + noise_scale * rand4.z;
    }

    r.x += half_dt * v.x;
    r.y += half_dt * v.y;
    r.z += half_dt * v.z;

    int ix = (int)floorf(r.x * inv_L);
    int iy = (int)floorf(r.y * inv_L);
    int iz = (int)floorf(r.z * inv_L);
    r.x -= ix * L;
    r.y -= iy * L;
    r.z -= iz * L;
    if (image != nullptr) {
        int i3 = i * 3;
        image[i3 + 0] += ix;
        image[i3 + 1] += iy;
        image[i3 + 2] += iz;
    }

    update_max_displacement(r, pos_ref[i], d_max_dr2_int, L, inv_L);

    pos[i] = r;
    vel[i] = v;
}

void launch_integrator_pre_force(float4* pos, float4* vel, const float4* force,
                                  float4* pos_ref, int* d_max_dr2_int,
                                  int* image, const int* tag, const int* d_step,
                                  const LangevinState& lang,
                                  int natoms, float L, float inv_L,
                                  float half_dt, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    integrator_pre_force_kernel<<<blocks, 256, 0, stream>>>(
        pos, vel, force, pos_ref, d_max_dr2_int, image, tag,
        lang.seed, d_step, natoms, L, inv_L,
        half_dt, lang.c1, lang.c2, lang.T_start, lang.T_stop, lang.nsteps);
}

__global__ void integrator_post_force_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ pos,
    int natoms, float half_dt, int* __restrict__ d_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && d_step) (*d_step)++;   // no other thread reads the counter here
    if (i >= natoms) return;

    float4 v = vel[i];
    float4 f = force[i];
    float4 r = pos[i];

    int type_i = unpack_type_id(r.w);
    float m_i = c_masses[type_i];
    float half_dt_over_m = half_dt / m_i;

    v.x += half_dt_over_m * f.x;
    v.y += half_dt_over_m * f.y;
    v.z += half_dt_over_m * f.z;
    vel[i] = v;
}

void launch_integrator_post_force(float4* vel, const float4* force,
                                   const float4* pos,
                                   int natoms, float half_dt,
                                   cudaStream_t stream, int* d_step) {
    int blocks = div_ceil(natoms, 256);
    integrator_post_force_kernel<<<blocks, 256, 0, stream>>>(
        vel, force, pos, natoms, half_dt, d_step);
}
