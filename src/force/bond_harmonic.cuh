#pragma once
#include "../core/types.cuh"

// Harmonic bond: E = K * (r - r0)^2   (LAMMPS bond_style harmonic convention)
struct BondHarmonicParams {
    float k;
    float r0;
};

void launch_bond_harmonic_kernel(const float4* __restrict__ pos,
                                  float4* __restrict__ force,
                                  double* __restrict__ virial,
                                  const int2* __restrict__ bonds,
                                  const int* __restrict__ bond_types,
                                  const BondHarmonicParams* __restrict__ params,
                                  int nbonds,
                                  Box box,
                                  cudaStream_t stream = 0);
