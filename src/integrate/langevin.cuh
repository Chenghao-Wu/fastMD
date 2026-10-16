#pragma once
#include "../core/types.cuh"

// Langevin thermostat (BAOAB-style splitting of the velocity-Verlet step).
//
// The random force uses counter-based Philox: the normal variates for a
// particle at a given step depend only on (seed, tag, step). There is no
// per-index RNG state, so the noise follows the particle under any
// reordering and a run is reproducible for a fixed seed.
//
// The target temperature ramps linearly, kT(step) = T_start + (T_stop -
// T_start) (step - 1) / nsteps, evaluated on the device from a device step
// counter that the post-force kernel advances (no per-step host input, so
// the step can be captured in a CUDA graph).
struct LangevinState {
    uint64_t seed = 0;
    float c1 = 1.0f;
    float c2 = 0.0f;
    float half_dt = 0.0f;
    float T_start = 0.0f, T_stop = 0.0f;
    int   nsteps = 0;

    void init(int natoms_padded, float Tdamp, float dt, float T_start,
              uint64_t seed, float T_stop = -1.0f, int nsteps = 0);
    void free() {}
};

// tag: particle tags (nullptr = identity); d_step: device step counter
// (current step, >= 1).
void launch_integrator_pre_force(float4* pos, float4* vel, const float4* force,
                                  float4* pos_ref, int* d_max_dr2_int,
                                  int* image, const int* tag, const int* d_step,
                                  const LangevinState& lang,
                                  int natoms, float L, float inv_L,
                                  float half_dt,
                                  cudaStream_t stream = 0);

// Ends the step: advances *d_step when d_step is not null.
void launch_integrator_post_force(float4* vel, const float4* force,
                                   const float4* pos,
                                   int natoms, float half_dt,
                                   cudaStream_t stream = 0,
                                   int* d_step = nullptr);
