#pragma once
#include "io/table_parser.hpp"
#include "../core/types.cuh"

void launch_table_kernel(const float4* __restrict__ pos,
                         float4* __restrict__ force,
                         float* __restrict__ virial,
                         const int* __restrict__ table_idx,
                         const TableParams* __restrict__ table_params,
                         const float4* __restrict__ table_data,
                         const int* __restrict__ neighbors,
                         const int* __restrict__ num_neighbors,
                         int natoms, int ntypes,
                         float rc2, float L, float inv_L,
                         cudaStream_t stream = 0);
