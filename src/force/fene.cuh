#pragma once
#include "../core/types.cuh"

struct FENEParams {
    float k;
    float R0;
    float eps;
    float sig;
};

void launch_fene_kernel(const float4* __restrict__ pos,
                         float4* __restrict__ force,
                         float* __restrict__ virial,
                         const int2* __restrict__ bonds,
                         const int* __restrict__ bond_types,
                         const FENEParams* __restrict__ params,
                         int nbonds, int nparamtypes,
                         float L, float inv_L,
                         cudaStream_t stream = 0);
