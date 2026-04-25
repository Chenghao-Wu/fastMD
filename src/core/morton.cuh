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

    void sort_and_permute(float4* pos, float4* vel, int natoms,
                          float inv_L, cudaStream_t stream = 0);
};
