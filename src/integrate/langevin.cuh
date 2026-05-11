#pragma once
#include "../core/types.cuh"
#include <curand_kernel.h>

struct LangevinState {
    curandStatePhilox4_32_10_t* rng_states;
    float c1;
    float c2;
    float half_dt;
    float kT;

    void init(int natoms_padded, float Tdamp, float dt, float temperature,
              uint64_t seed);
    void free();
};

void launch_integrator_pre_force(float4* pos, float4* vel, const float4* force,
                                  float4* pos_ref, int* d_max_dr2_int,
                                  int* image,
                                  const LangevinState& lang,
                                  int natoms, float L, float inv_L,
                                  float half_dt,
                                  cudaStream_t stream = 0);

void launch_integrator_post_force(float4* vel, const float4* force,
                                   const float4* pos,
                                   int natoms, float half_dt,
                                   cudaStream_t stream = 0);
