#pragma once
#include "../core/types.cuh"

// Fused LJ 12-6 + real-space Ewald Coulomb: C*q_i*q_j*erfc(alpha*r)/r.
// Neighbor-list kernel; one thread per atom (writes force[i] directly like
// lj.cu), 0.5 factor on PE/virial because the full neighbor list visits
// each pair twice. Requires a separate overwrite/reset when combined with
// additive kernels — see main.cu wiring.
void launch_lj_coul_long_kernel(const float4* __restrict__ pos,
                                 const float* __restrict__ charges,
                                 float4* __restrict__ force,
                                 float* __restrict__ virial,
                                 const float2* __restrict__ lj_params,
                                 const int* __restrict__ neighbors,
                                 const int* __restrict__ num_neighbors,
                                 int natoms, int ntypes,
                                 float coul_const, float alpha,
                                 float rc2, float L, float inv_L,
                                 cudaStream_t stream = 0);
