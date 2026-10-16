#pragma once
#include "../core/types.cuh"
#include <vector>

// Plain 12-6 Lennard-Jones over a full neighbor list (overwrites force[i]).
//
// lj_cap (optional, nullptr = exact LJ): per type pair (r_cap, U(r_cap)).
// For r < r_cap the force magnitude is held at force_cap and the energy is
// continued linearly, U(r) = U(r_cap) + force_cap (r_cap - r), so that
// F = -dU/dr holds everywhere.
void launch_lj_kernel(const float4* __restrict__ pos,
                       float4* __restrict__ force,
                       float* __restrict__ virial,
                       const float2* __restrict__ lj_params,
                       const int* __restrict__ neighbors,
                       const int* __restrict__ num_neighbors,
                       int natoms, int ntypes,
                       float rc2, float L, float inv_L,
                       cudaStream_t stream = 0,
                       const float2* __restrict__ lj_cap = nullptr,
                       float force_cap = 0.0f);

// Host: (r_cap, U(r_cap)) per type pair for a force cap; r_cap = 0 when the
// pair never reaches the cap (eps == 0).
std::vector<float2> lj_force_cap_table(const std::vector<float2>& lj_params,
                                       int ntypes, float force_cap);
