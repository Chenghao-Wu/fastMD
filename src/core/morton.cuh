#pragma once
#include "types.cuh"

struct MortonSorter {
    uint32_t* d_keys;
    uint32_t* d_keys_sorted;
    int*      d_indices;
    int*      d_indices_sorted;
    float4*   d_temp;
    void*     d_cub_temp;
    size_t    cub_temp_bytes;
    int       capacity;

    void allocate(int natoms_padded);
    void free();

    // Compute the Morton order of the first natoms particles. Returns a device
    // array new_order[k] = old index of the particle placed at k, covering the
    // padded range (padding keeps its position). Valid until the next call.
    const int* compute_order(const float4* pos, int natoms, float inv_L,
                             cudaStream_t stream = 0);

    // Sort and permute pos/vel only (kept for tests; simulations must use
    // compute_order + ParticleData::apply_permutation).
    void sort_and_permute(float4* pos, float4* vel, int natoms,
                          float inv_L, cudaStream_t stream = 0);
};
