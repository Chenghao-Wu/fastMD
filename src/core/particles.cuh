#pragma once
#include "types.cuh"
#include <vector>

// All per-particle device state.
//
// Every array is indexed by the current particle index i in
// [0, natoms_padded). Particles may be reordered at run time (spatial
// sorting), so the index is not an identity: tag[i] is the particle's
// permanent identity (its 0-based position in the input data file) and
// rank[t] is the inverse map, tag -> current index.
//
// Reordering must go through apply_permutation(), which moves *every*
// per-particle field together. Code that needs a stable order (output,
// topology, analysis) works in tag space and maps through rank.
struct ParticleData {
    int natoms = 0;
    int natoms_padded = 0;

    float4* pos = nullptr;      // xyz + packed type id (padding: type -1)
    float4* vel = nullptr;      // xyz
    float4* force = nullptr;    // xyz + per-atom potential energy
    float4* pos_ref = nullptr;  // positions at the last neighbor-list build
    int*    image = nullptr;    // [3 * natoms_padded] periodic image counters
    float*  charge = nullptr;   // always allocated (zero when uncharged)
    int*    mol_id = nullptr;
    int*    tag = nullptr;
    int*    rank = nullptr;

    void allocate(int natoms);
    void free();

    // Upload host data given in tag order (index == tag after upload).
    // velocities / images / charges / mol_ids may be empty (zero-filled).
    void upload(const std::vector<float4>& positions,
                const std::vector<float4>& velocities,
                const std::vector<int>& images,
                const std::vector<float>& charges,
                const std::vector<int>& mol_ids);

    // new_order[k] = old index of the particle that moves to index k, for
    // k in [0, natoms_padded); padding entries must map to themselves.
    void apply_permutation(const int* d_new_order, cudaStream_t stream = 0);

    // Device-to-device copy of every array from an allocated twin (same natoms).
    void copy_from(const ParticleData& src, cudaStream_t stream = 0);

    // Gather a per-particle array into tag order on the device.
    void gather_float4_by_tag(float4* d_out, const float4* d_in,
                              cudaStream_t stream = 0) const;

    // Host copies in tag order (synchronous).
    std::vector<float4> download_positions() const;
    std::vector<float4> download_velocities() const;
    std::vector<float4> download_forces() const;
    std::vector<int>    download_images() const;
    std::vector<float>  download_charges() const;
    std::vector<int>    download_mol_ids() const;

private:
    void* scratch_ = nullptr;   // natoms_padded * sizeof(float4)
    template <typename T> std::vector<T> download_by_tag(const T* d, int width) const;
};
