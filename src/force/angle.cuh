#pragma once
#include "../core/types.cuh"

struct AngleParams {
    float k_theta;   // harmonic: K in K (theta - theta0)^2; cosine: K in K (1 + cos theta)
    float theta0;    // radians (harmonic only)
};

void launch_angle_kernel(const float4* __restrict__ pos,
                          float4* __restrict__ force,
                          double* __restrict__ virial,
                          const int4* __restrict__ angles,
                          const AngleParams* __restrict__ params,
                          int nangles, float L, float inv_L,
                          cudaStream_t stream = 0,
                          bool cosine = false);   // E = K (1 + cos theta) instead of harmonic
