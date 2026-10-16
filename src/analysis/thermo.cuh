#pragma once
#include "../core/types.cuh"
#include <cstdio>

struct ThermoOutput {
    float kinetic_energy;
    float potential_energy;
    float temperature;
    float stress[6];
    float max_vel;      // max velocity magnitude
    int   max_vel_idx;  // atom index with max velocity
    float max_force;    // max force magnitude
    int   max_force_idx;// atom index with max force
};

struct ThermoBuffers {
    float* d_kin_stress;
    float* d_pe;
    float* d_max_vel;     // [2]: {max_vel_mag, float(idx)}
    float* d_max_force;   // [2]: {max_force_mag, float(idx)}
    bool allocated;
    FILE* fp;

    void allocate();
    void free();
    void open_file(const char* path);
    void close_file();
};

// Pressure tensor sampled entirely on the device (no host transfer), for
// the stress autocorrelation: P_ab = (sum_i m v_a v_b * s^2 + W_ab) / V,
// written to d_stress (order xx xy xz yy yz zz). d_vscale (optional) is a
// pending velocity scale factor s not yet applied to vel; d_L the box length.
struct StressSampler {
    double* d_acc = nullptr;          // [6] kinetic tensor accumulator
    unsigned int* d_done = nullptr;   // last-block ticket
    float* d_stress = nullptr;        // [6] result
    void allocate();
    void free();
    void sample(const float4* vel, const float4* pos, const double* virial, int natoms,
                const float* d_L, const float* d_vscale, cudaStream_t stream);
};

void compute_thermo(const float4* vel, const float4* pos, const float4* force,
                     const double* virial,
                     int natoms, float box_L,
                     ThermoOutput* h_output,
                     ThermoBuffers& bufs,
                     int step, FILE* fp,
                     cudaStream_t stream = 0,
                     float pe_offset_per_atom = 0.0f);
