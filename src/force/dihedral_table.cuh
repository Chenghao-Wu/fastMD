#pragma once
#include "../core/types.cuh"
#include "../io/table_parser.hpp"

// Tabulated dihedral (dihedral_table), one thread per dihedral (scatter mode).
void launch_dihedral_table_kernel(const float4* __restrict__ pos,
                                  float4* __restrict__ force,
                                  double* __restrict__ virial,
                                  const int4* __restrict__ dihedrals,
                                  const int* __restrict__ dihedral_types,
                                  const int* __restrict__ table_idx,
                                  const TableParams* __restrict__ table_params,
                                  const float4* __restrict__ table_data,
                                  int ndihedrals,
                                  Box box,
                                  cudaStream_t stream = 0);
