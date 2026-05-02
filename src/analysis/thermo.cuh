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

void compute_thermo(const float4* vel, const float4* force,
                     const float* virial,
                     int natoms, float box_L,
                     ThermoOutput* h_output,
                     ThermoBuffers& bufs,
                     int step, FILE* fp,
                     cudaStream_t stream = 0);
