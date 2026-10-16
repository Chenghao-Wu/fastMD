#pragma once
#include "types.cuh"
#include <vector>

// Per-type masses live in constant memory, read by the integrators and the
// kinetic-energy reductions as c_masses[type].
static constexpr int MAX_TYPES = 16;
extern __constant__ float c_masses[MAX_TYPES];

// Upload per-type masses (types beyond masses.size() get mass 1). Throws if
// ntypes exceeds MAX_TYPES or a mass is not positive.
void set_type_masses(const std::vector<float>& masses, int ntypes);
