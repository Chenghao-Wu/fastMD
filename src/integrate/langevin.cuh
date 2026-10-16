#pragma once
#include "../core/types.cuh"

// Langevin thermostat (BAOAB-style splitting of the velocity-Verlet step).
//
// The random force uses counter-based Philox: the normal variates for a
// particle at a given step depend only on (seed, tag, step). There is no
// per-index RNG state, so the noise follows the particle under any
// reordering and a run is reproducible for a fixed seed.
struct LangevinState {
    uint64_t seed = 0;
    float c1 = 1.0f;
    float c2 = 0.0f;
    float half_dt = 0.0f;
    float kT = 0.0f;

    void init(int natoms_padded, float Tdamp, float dt, float temperature,
              uint64_t seed);
    void free() {}
};

// tag: particle tags (nullptr = identity); step: current step number (>= 1).
void launch_integrator_pre_force(float4* pos, float4* vel, const float4* force,
                                  float4* pos_ref, int* d_max_dr2_int,
                                  int* image, const int* tag, uint64_t step,
                                  const LangevinState& lang,
                                  int natoms, float L, float inv_L,
                                  float half_dt,
                                  cudaStream_t stream = 0);

void launch_integrator_post_force(float4* vel, const float4* force,
                                   const float4* pos,
                                   int natoms, float half_dt,
                                   cudaStream_t stream = 0);
