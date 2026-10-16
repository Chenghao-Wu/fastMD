#pragma once
#include "../io/table_parser.hpp"
#include "../core/types.cuh"

void launch_angle_table_kernel(const float4* pos, float4* force, double* virial,
                               const int4* angles, const int* angle_table_idx,
                               const TableParams* table_params, const float4* table_data,
                               int nangles, Box box,
                               cudaStream_t stream = 0);
