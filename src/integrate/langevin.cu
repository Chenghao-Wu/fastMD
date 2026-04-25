#include "langevin.cuh"
#include "../core/pbc.cuh"
#include "../neighbor/skin_trigger.cuh"
#include <curand_kernel.h>

__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* states,
                                 uint64_t seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curand_init(seed, i, 0, &states[i]);
    }
}

void LangevinState::init(int natoms_padded, float gamma, float dt,
                          float temperature, uint64_t seed) {
    c1 = expf(-gamma * dt);
    c2 = sqrtf(1.0f - c1 * c1);
    half_dt = 0.5f * dt;
    kT = temperature;

    CUDA_CHECK(cudaMalloc(&rng_states,
                           natoms_padded * sizeof(curandStatePhilox4_32_10_t)));
    int blocks = div_ceil(natoms_padded, 256);
    init_rng_kernel<<<blocks, 256>>>(rng_states, seed, natoms_padded);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LangevinState::free() {
    CUDA_CHECK(cudaFree(rng_states));
}

__global__ void integrator_pre_force_kernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float4* __restrict__ pos_ref,
    int* __restrict__ d_max_dr2_int,
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    int natoms, float L, float inv_L,
    float half_dt, float c1, float c2, float kT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 r = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    v.x += half_dt * f.x;
    v.y += half_dt * f.y;
    v.z += half_dt * f.z;

    r.x += half_dt * v.x;
    r.y += half_dt * v.y;
    r.z += half_dt * v.z;

    curandStatePhilox4_32_10_t local_state = rng_states[i];
    float4 rand4 = curand_normal4(&local_state);
    rng_states[i] = local_state;

    float noise_scale = c2 * sqrtf(kT);
    v.x = c1 * v.x + noise_scale * rand4.x;
    v.y = c1 * v.y + noise_scale * rand4.y;
    v.z = c1 * v.z + noise_scale * rand4.z;

    r.x += half_dt * v.x;
    r.y += half_dt * v.y;
    r.z += half_dt * v.z;

    r = wrap_position(r, L, inv_L);

    update_max_displacement(r, pos_ref[i], d_max_dr2_int);

    pos[i] = r;
    vel[i] = v;
}

void launch_integrator_pre_force(float4* pos, float4* vel, const float4* force,
                                  float4* pos_ref, int* d_max_dr2_int,
                                  const LangevinState& lang,
                                  int natoms, float L, float inv_L,
                                  float half_dt, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    integrator_pre_force_kernel<<<blocks, 256, 0, stream>>>(
        pos, vel, force, pos_ref, d_max_dr2_int,
        lang.rng_states, natoms, L, inv_L,
        half_dt, lang.c1, lang.c2, lang.kT);
}

__global__ void integrator_post_force_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    int natoms, float half_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 v = vel[i];
    float4 f = force[i];
    v.x += half_dt * f.x;
    v.y += half_dt * f.y;
    v.z += half_dt * f.z;
    vel[i] = v;
}

void launch_integrator_post_force(float4* vel, const float4* force,
                                   int natoms, float half_dt,
                                   cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    integrator_post_force_kernel<<<blocks, 256, 0, stream>>>(
        vel, force, natoms, half_dt);
}
