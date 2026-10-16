#include "system.cuh"
#include <stdexcept>
#include <string>

__constant__ float c_masses[MAX_TYPES];

void set_type_masses(const std::vector<float>& masses, int ntypes) {
    if (ntypes < 1 || ntypes > MAX_TYPES)
        throw std::runtime_error("ntypes must be in [1, " + std::to_string(MAX_TYPES) + "]");
    float h[MAX_TYPES];
    for (int t = 0; t < MAX_TYPES; t++) {
        h[t] = t < (int)masses.size() ? masses[t] : 1.0f;
        if (t < ntypes && !(h[t] > 0.0f))
            throw std::runtime_error("mass of type " + std::to_string(t + 1) + " must be > 0");
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h, sizeof(h)));
}
