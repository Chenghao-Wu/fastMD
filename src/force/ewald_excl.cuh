#pragma once
#include "../core/types.cuh"

// Subtract the real-space Ewald term for fully-excluded pairs (1-2, 1-3):
//   F -= grad[ C q_i q_j erf(alpha r)/r ],  PE -= C q_i q_j erf(alpha r)/r
// One thread per CSR entry. The CSR is in tag space (rows indexed by tag,
// entries are tags); rank maps tag -> current index (nullptr = identity).
void launch_ewald_excl_kernel(const float4* __restrict__ pos,
                               const float* __restrict__ charges,
                               float4* __restrict__ force,
                               float* __restrict__ virial,
                               const int* __restrict__ excl_offsets,
                               const int* __restrict__ excl_list,
                               const int* __restrict__ rank,
                               int natoms, int nexcl,
                               float coul_const, float alpha,
                               float L, float inv_L,
                               cudaStream_t stream = 0);

// Scaled 1-4 pairs (each stored once):
//   LJ part:   lj14  * 4 eps (sr12 - sr6)
//   Coul part: ewald14 * C q_i q_j/r - C q_i q_j erf(alpha r)/r
//   (full scaled Coulomb minus the spurious short-range term)
void launch_pair14_kernel(const float4* __restrict__ pos,
                           const float* __restrict__ charges,
                           float4* __restrict__ force,
                           float* __restrict__ virial,
                           const int2* __restrict__ pairs,
                           const float2* __restrict__ lj_params,
                           int npairs, int ntypes,
                           float coul_const, float alpha,
                           float lj14, float ewald14,
                           float L, float inv_L,
                           cudaStream_t stream = 0);
