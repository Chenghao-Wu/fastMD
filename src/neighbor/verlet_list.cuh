#pragma once
#include "../core/types.cuh"
#include <cub/cub.cuh>

#define MAX_SHARED_ATOMS 96

struct VerletList {
    int*  neighbors;       // [max_neighbors * natoms] transposed
    int*  num_neighbors;   // [natoms]
    int*  cell_ids;        // [natoms] temp
    int*  cell_ids_out;    // [natoms] temp for sort output keys
    int*  atom_ids;        // [natoms] temp for sort input values
    int*  sorted_atoms;    // [natoms] permutation from cell sort
    int*  cell_starts;     // [ncells]
    int*  cell_ends;       // [ncells]
    int   max_neighbors;
    int   ncells;
    int   nx, ny, nz;
    bool  dedup_needed;
    float cell_size;
    void* d_cub_temp;
    size_t cub_temp_bytes;
    int*  h_cell_max;
    int   max_cell_atoms;
    int*  h_max_nneigh;    // pinned host: max num_neighbors after each build
    int   max_nneigh;

    void allocate(int natoms, float rc_skin, float box_L);
    void free();
    void build(const float4* pos, int natoms, float box_L, float inv_L,
               const int* exclusion_offsets, const int* exclusion_list,
               float rc_skin, cudaStream_t stream = 0);
};
