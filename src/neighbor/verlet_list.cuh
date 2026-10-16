#pragma once
#include "../core/types.cuh"

// Cell-list-built Verlet (neighbor) list, stored transposed:
// neighbors[k * natoms + i] is the k-th neighbor of particle i.
//
// Cells are keyed by the Morton code of their (cx, cy, cz) coordinates, so
// particles sorted by cell key are ordered along a space-filling curve and
// the 27 cells around any cell lie close together in memory.
//
// build() always derives the cell grid from the *current* box, so it stays
// valid when NPT changes the box. Neighbor counts are measured exactly; if
// any particle has more neighbors than the current capacity the list is
// reallocated and rebuilt, so a returned list is never truncated.
//
// Two-phase use (spatial sorting): bin() sorts particles by cell; the caller
// may then permute its particle arrays with order_padded() and call
// build_from_bins(..., /*pos_in_cell_order=*/true).
struct VerletList {
    int*  neighbors = nullptr;      // [max_neighbors * natoms]
    int*  num_neighbors = nullptr;  // [natoms]
    int*  cell_ids = nullptr;       // [natoms] cell key per particle
    int*  cell_ids_out = nullptr;   // [natoms] cell keys in sorted order
    int*  atom_ids = nullptr;       // [natoms] temp sort input values
    int*  sorted_atoms = nullptr;   // [natoms] particles ordered by cell key
    int*  order_pad = nullptr;      // [natoms_padded] sorted_atoms + identity padding
    float4* pos_sorted = nullptr;   // [natoms] positions in cell order
    int*  cell_starts = nullptr;    // [nkeys]
    int*  cell_ends = nullptr;      // [nkeys]
    int*  d_stats = nullptr;        // [2] {max neighbor count, max cell occupancy}
    int*  h_stats = nullptr;        // pinned mirror of d_stats
    int   max_neighbors = 0;        // current capacity per particle
    int   natoms = 0, natoms_padded = 0;
    int   ncells = 0;               // nx * ny * nz
    int   nkeys = 0, nkeys_alloc = 0;  // Morton key range (power-of-two cube)
    int   key_bits = 0;
    int   nx = 0, ny = 0, nz = 0;
    bool  dedup_needed = false;
    float cell_size = 0.0f;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;
    int   max_cell_atoms = 0;       // diagnostics from the last build
    int   max_count = 0;            // largest neighbor count seen at the last build
    int   n_resizes = 0;            // capacity growth events (diagnostics)

    void allocate(int natoms, float rc_skin, float box_L);
    void free();

    // tag: per-index particle tag used for tag-space exclusion lookups
    // (nullptr = identity). exclusion CSR rows are indexed by tag.
    void build(const float4* pos, int natoms, float box_L, float inv_L,
               const int* tag,
               const int* exclusion_offsets, const int* exclusion_list,
               float rc_skin, cudaStream_t stream = 0);

    // Backwards-compatible overload: identity tags.
    void build(const float4* pos, int natoms, float box_L, float inv_L,
               const int* exclusion_offsets, const int* exclusion_list,
               float rc_skin, cudaStream_t stream = 0) {
        build(pos, natoms, box_L, inv_L, nullptr, exclusion_offsets,
              exclusion_list, rc_skin, stream);
    }

    // Phase 1: assign cells and sort particles by cell key.
    void bin(const float4* pos, float box_L, float rc_skin, cudaStream_t stream = 0);
    // new_order for ParticleData::apply_permutation (valid after bin()).
    const int* order_padded() const { return order_pad; }
    // Phase 2: build the list. pos_in_cell_order: the caller applied
    // order_padded() to its particles, so index k already holds the k-th
    // particle in cell order.
    void build_from_bins(const float4* pos, bool pos_in_cell_order, float box_L, float inv_L,
                         const int* tag, const int* exclusion_offsets,
                         const int* exclusion_list, float rc_skin, cudaStream_t stream = 0);

private:
    void set_grid(float box_L, float rc_skin);
    void launch_list(const float4* pos, bool pos_in_cell_order, float box_L, float inv_L,
                     const int* tag, const int* exclusion_offsets, const int* exclusion_list,
                     float rc_skin, cudaStream_t stream);
};
