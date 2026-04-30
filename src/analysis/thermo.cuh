#pragma once
#include "../core/types.cuh"
#include <cstdio>

struct ThermoOutput {
    float kinetic_energy;
    float potential_energy;
    float temperature;
    float stress[6];
};

struct ThermoBuffers {
    float* d_kin_stress;
    float* d_pe;
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
