#pragma once
#include "../core/types.cuh"

// Cell-list-built Verlet (neighbor) list, stored transposed:
// neighbors[k * natoms + i] is the k-th neighbor of particle i.
//
// The list can be maintained entirely on the device. Each step two kernels
// run: nl_bin evaluates the exact rebuild criterion from device state (the
// maximum displacement since the last build and the box scaling) and exits
// at once when no rebuild is needed; otherwise it bins particles into cells
// (counting sort; the last block scans the cell counts, scatters and
// gathers positions in cell order) and resets the displacement reference.
// nl_build then builds the list (or just ages it). Nothing is read back per
// step: overflow of the neighbor capacity and an invalid cell grid (box
// compressed below the grid's cell size) are recorded in DeviceNeighborState
// for the caller to check periodically (and roll back if set).
//
// Cells are dense (cz * ny + cy) * nx + cx, stored in Morton order of their
// coordinates (cell_rank), so particles sorted by cell follow a space-filling
// curve. Cells are (rc + skin) / cell_div wide; dense systems use cell_div 2.
struct DeviceNeighborState {
    int   force_rebuild;   // host request: rebuild at the next nl_bin
    int   rebuild;         // set by nl_bin when this step rebuilds
    int   age;             // steps since the last build
    int   n_builds;        // total builds
    int   max_count;       // largest neighbor count of the last build
    int   overflow;        // > 0: a build exceeded capacity (largest count seen)
    int   grid_error;      // 1: cell size < rc + skin or L < 2 rc at some build
    int   max_cell;        // largest cell occupancy of the last build
    float L_build;         // box length at the last build
    unsigned int bin_done, build_done;   // grid-wide "last block" tickets
};

struct VerletList {
    int*  neighbors = nullptr;      // [max_neighbors * natoms]
    int*  num_neighbors = nullptr;  // [natoms]
    int*  cell_key = nullptr;       // [natoms] cell (Morton rank) of each particle
    int*  cell_slot = nullptr;      // [natoms] slot within its cell
    int*  cell_count = nullptr;     // [ncells] (kept zeroed between builds)
    int*  cell_starts = nullptr;    // [ncells]
    int*  cell_ends = nullptr;      // [ncells]
    int*  cell_rank = nullptr;      // [ncells] dense cell index -> Morton rank
    int*  order = nullptr;          // [natoms_padded] particles in cell order (+ identity padding)
    float4* pos_sorted = nullptr;   // [natoms] positions in cell order
    DeviceNeighborState* d_state = nullptr;
    DeviceNeighborState* h_state = nullptr;   // pinned mirror (read_state)
    int   max_neighbors = 0;        // current capacity per particle
    int   natoms = 0, natoms_padded = 0;
    int   nx = 0, ny = 0, nz = 0, ncells = 0, ncells_alloc = 0;
    float cell_size = 0.0f;         // box_L / nx at the time the grid was set
    int   cell_div = 1;             // cells per rc + skin (stencil reach): 1 or 2
    float rc = 0.0f, skin = 0.0f;
    float rc_min_image = 0.0f;      // interaction cutoff that must stay <= L/2 (0 = unchecked)
    int   max_age = 0;              // forced rebuild age (0 = off)
    float grid_margin = 1.0f;       // cells sized for box compression down to L / margin
    int   n_resizes = 0;
    int   build_tpa = 4;                // threads per particle in the list build (1,2,4,8)
    int   max_count = 0, max_cell_atoms = 0;   // host copies after a synchronous build

    void allocate(int natoms, float rc, float skin, float box_L, float grid_margin = 1.0f,
                  int max_age = 0);
    void free();

    // Choose the cell grid for box length L (host); requests a rebuild.
    void set_grid(float box_L);
    bool grid_ok_for(float box_L) const;
    int  cells_per_side(float box_L) const;

    // Device-side maintenance (enqueue only). d_box_L: current box length on
    // the device; d_max_dr2_bits: max squared displacement since the last
    // build (float bits), reset by a rebuild. Positions must be wrapped.
    void step(const float4* pos, float4* pos_ref, const float* d_box_L, int* d_max_dr2_bits,
              const int* tag, const int* exclusion_offsets, const int* exclusion_list,
              cudaStream_t stream);
    // Bin only (forced). Afterwards order_padded() gives the cell order.
    void bin(const float4* pos, float4* pos_ref, const float* d_box_L, int* d_max_dr2_bits,
             cudaStream_t stream);
    const int* order_padded() const { return order; }

    void request_rebuild(cudaStream_t stream);
    void read_state(cudaStream_t stream);           // synchronous copy into *h_state
    void clear_errors(cudaStream_t stream);
    void grow(int needed);                          // capacity for at least `needed`
    void set_capacity(int cap);                     // exact capacity (testing)

    // Synchronous host build (tests and setup): bins, builds and grows the
    // capacity until nothing overflows.
    void build(const float4* pos, int natoms, float box_L, float inv_L,
               const int* tag,
               const int* exclusion_offsets, const int* exclusion_list,
               float rc_skin, cudaStream_t stream = 0);
    void build(const float4* pos, int natoms, float box_L, float inv_L,
               const int* exclusion_offsets, const int* exclusion_list,
               float rc_skin, cudaStream_t stream = 0) {
        build(pos, natoms, box_L, inv_L, nullptr, exclusion_offsets,
              exclusion_list, rc_skin, stream);
    }

    // Legacy allocate (tests): list radius rc_skin, no minimum-image check.
    void allocate(int natoms, float rc_skin, float box_L) {
        allocate(natoms, rc_skin, 0.0f, box_L);
        rc_min_image = 0.0f;
    }

private:
    float* d_box_scratch_ = nullptr;   // box length for synchronous builds
    int*   d_dr2_scratch_ = nullptr;
    float4* d_ref_scratch_ = nullptr;
    void launch_bin(const float4* pos, float4* pos_ref, const float* d_box_L,
                    int* d_max_dr2_bits, cudaStream_t stream);
    void launch_build(const float4* pos, const float* d_box_L, const int* tag,
                      const int* exclusion_offsets, const int* exclusion_list,
                      cudaStream_t stream);
};
