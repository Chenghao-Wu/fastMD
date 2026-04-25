#pragma once
#include "../core/types.cuh"

struct AngleParams {
    float k_theta;
    float theta0;
};

void launch_angle_kernel(const float4* __restrict__ pos,
                          float4* __restrict__ force,
                          float* __restrict__ virial,
                          const int4* __restrict__ angles,
                          const AngleParams* __restrict__ params,
                          int nangles, float L, float inv_L,
                          cudaStream_t stream = 0);
