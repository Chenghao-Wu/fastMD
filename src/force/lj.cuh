#pragma once
#include "../core/types.cuh"

void launch_lj_kernel(const float4* __restrict__ pos,
                       float4* __restrict__ force,
                       float* __restrict__ virial,
                       const float2* __restrict__ lj_params,
                       const int* __restrict__ tile_offsets,
                       const int* __restrict__ tile_neighbors,
                       int ntiles, int natoms, int ntypes,
                       float rc2, float L, float inv_L,
                       cudaStream_t stream = 0);
