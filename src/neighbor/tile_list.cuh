#pragma once
#include "../core/types.cuh"

struct TileList {
    int* offsets;
    int* tile_neighbors;
    int  ntiles;
    int  npairs;
    int  capacity;

    float3* tile_min;
    float3* tile_max;
    int*    d_flags;
    int*    d_npairs;
    void*   d_cub_temp;
    size_t  cub_temp_bytes;

    void allocate(int ntiles, int max_pairs_estimate);
    void free();

    void build(const float4* pos, int natoms, int ntiles,
               float rc_skin, float L, float inv_L,
               cudaStream_t stream = 0);
};
