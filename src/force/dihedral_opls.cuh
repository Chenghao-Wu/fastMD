#pragma once
#include "../core/types.cuh"

// OPLS dihedral: E = 0.5*K1*(1+cos phi) + 0.5*K2*(1-cos 2phi)
//                + 0.5*K3*(1+cos 3phi) + 0.5*K4*(1-cos 4phi)
// (LAMMPS dihedral_style opls convention; Ki are full Fourier coeffs)
struct OPLSDihedralParams {
    float k1;
    float k2;
    float k3;
    float k4;
};

void launch_dihedral_opls_kernel(const float4* __restrict__ pos,
                                  float4* __restrict__ force,
                                  double* __restrict__ virial,
                                  const int4* __restrict__ dihedrals,
                                  const int* __restrict__ dihedral_types,
                                  const OPLSDihedralParams* __restrict__ params,
                                  int ndihedrals,
                                  Box box,
                                  cudaStream_t stream = 0);
