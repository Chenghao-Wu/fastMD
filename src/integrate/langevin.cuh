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
//
// Depth-dependent friction (`friction_profile`): gamma = gamma_b f(d), with
// d the particle's unwrapped coordinate along the film normal minus the
// film reference plane. The O step then uses c1 = exp(-gamma dt), c2 =
// sqrt(1 - c1^2) per particle at the midpoint position, which keeps the
// Maxwell-Boltzmann velocity distribution exact for any profile.
// f is either
//   analytic: f = exp(a [exp(-s / ell) + exp(-(h - s) / ell)]),
//             s = d + h/2 clamped to [0, h] (distance from one surface)
//   table:    f(d) linearly interpolated on a uniform grid (clamped ends).
//
// Andersen thermostat (`nvt_andersen`): velocity Verlet with velocities
// redrawn from the Maxwell-Boltzmann distribution with probability
// nu f(d) dt per particle and step (same profile mechanism).
struct FrictionProfile {
    int   mode = 0;              // 0 uniform, 1 analytic, 2 table
    int   axis = 2;              // film normal
    float a = 0.0f, ell = 1.0f, h = 0.0f;
    int   npoints = 0;
    float zmin = 0.0f, inv_dz = 0.0f;
    const float* d_table = nullptr;
};

__host__ __device__ inline float friction_factor(const FrictionProfile& p, float d) {
    if (p.mode == 1) {
        float s = fminf(fmaxf(d + 0.5f * p.h, 0.0f), p.h);
        return expf(p.a * (expf(-s / p.ell) + expf(-(p.h - s) / p.ell)));
    }
    if (p.mode == 2) {
        float x = (d - p.zmin) * p.inv_dz;
        x = fminf(fmaxf(x, 0.0f), (float)(p.npoints - 1));
        int k = (int)x < p.npoints - 2 ? (int)x : p.npoints - 2;
        float u = x - k;
        return p.d_table[k] + u * (p.d_table[k + 1] - p.d_table[k]);
    }
    return 1.0f;
}

struct LangevinState {
    uint64_t seed = 0;
    float c1 = 1.0f;
    float c2 = 0.0f;
    float half_dt = 0.0f;
    float T_start = 0.0f, T_stop = 0.0f;
    int   nsteps = 0;
    float gamma_dt = 0.0f;       // gamma_b dt (profile)
    int   andersen = 0;          // 1: Andersen collisions instead of friction
    float nu_dt = 0.0f;          // Andersen collision probability per step (times f)
    FrictionProfile profile;
    const float* d_film_center = nullptr;   // film reference plane (profile)

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
                                  int natoms, const Box& box,
                                  float half_dt,
                                  cudaStream_t stream = 0);

// Ends the step: advances *d_step when d_step is not null.
void launch_integrator_post_force(float4* vel, const float4* force,
                                   const float4* pos,
                                   int natoms, float half_dt,
                                   cudaStream_t stream = 0,
                                   int* d_step = nullptr);
