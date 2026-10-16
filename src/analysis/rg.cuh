#pragma once
#include "../core/types.cuh"
#include <cstdio>
#include <vector>

// Ensemble-averaged radius of gyration (mean over molecules of Rg).
//
// Molecule membership is a CSR in tag space (mol_offsets / mol_members hold
// particle tags), so particles need not be contiguous or in any order; the
// kernel maps tags to current indices through rank[] (nullptr = identity).
// Coordinates are unwrapped with the periodic image counters.
struct RgBuffers {
    int*   d_mol_offsets = nullptr;   // [nmols + 1]
    int*   d_mol_members = nullptr;   // [natoms in molecules] particle tags
    float* d_ensemble_sum = nullptr;
    int    nmols = 0;
    FILE*  fp = nullptr;

    // mol_ids[t] = molecule id of tag t (any integers, grouped on the host)
    void allocate(const std::vector<int>& mol_ids, const char* output_file);
    void free();
};

void compute_rg(const float4* pos, const int* image, const int* rank,
                const RgBuffers& bufs, float box_L,
                int step, float* h_rg,
                cudaStream_t stream = 0);
