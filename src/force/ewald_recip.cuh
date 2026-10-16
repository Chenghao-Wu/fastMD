#pragma once
#include "../core/types.cuh"
#include <vector>

// Reciprocal-space Ewald (orthorhombic cubic box, half-shell k-vector list).
// k-vectors stored as integer triplets; actual k = 2*pi*n/L computed on the
// fly so NPT box changes are handled without regenerating the list.
// Coefficients follow LAMMPS Ewald::coeffs() (src/KSPACE/ewald.cpp).

// Build the half-shell k-vector list on the host (returns int3 triplets).
std::vector<int3> ewald_build_kvectors(int kmax, float L, float alpha);

// Structure factor: sfac[2k] = Re S(k), sfac[2k+1] = Im S(k), block per k.
void launch_ewald_sf_kernel(const float4* __restrict__ pos,
                             const float* __restrict__ charges,
                             const int3* __restrict__ kvecs,
                             float* __restrict__ sfac,
                             int natoms, int kcount,
                             float L,
                             cudaStream_t stream = 0, const Box* d_box = nullptr);

// Per-atom reciprocal force + PE + reciprocal virial.
// Adds into force[] and virial[] (no overwrite).
void launch_ewald_recip_kernel(const float4* __restrict__ pos,
                                const float* __restrict__ charges,
                                const int3* __restrict__ kvecs,
                                const float* __restrict__ sfac,
                                float4* __restrict__ force,
                                double* __restrict__ virial,
                                int natoms, int kcount,
                                float coul_const, float alpha, float L,
                                cudaStream_t stream = 0, const Box* d_box = nullptr);

// Self + charged-system energy offset (host): returns the constant that must
// be ADDED to the PE sum to match LAMMPS (offset is negative or zero):
//   offset = -C * [ alpha/sqrt(pi) * sum(q^2) + pi/(2 alpha^2 V) * (sum q)^2 ]
float ewald_self_offset(const std::vector<float>& charges,
                        float coul_const, float alpha, float L);
