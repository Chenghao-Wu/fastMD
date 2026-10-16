#pragma once
#include "../io/table_parser.hpp"
#include "../core/types.cuh"

void launch_bond_table_kernel(const float4* pos, float4* force, double* virial,
                              const int2* bonds, const int* bond_types,
                              const int* bond_table_idx,
                              const TableParams* table_params,
                              const float4* table_data,
                              int nbonds, Box box,
                              cudaStream_t stream = 0);
