#pragma once
#include "../core/types.cuh"

struct ThermoOutput {
    float kinetic_energy;
    float potential_energy;
    float temperature;
    float stress[6];
};

void compute_thermo(const float4* vel, const float4* force,
                     const float* virial,
                     int natoms, float box_L,
                     ThermoOutput* h_output,
                     cudaStream_t stream = 0);
