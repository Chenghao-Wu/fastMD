#pragma once
#include "types.cuh"
#include <vector>

// Device-side bonded topology.
//
// The canonical lists are stored in tag space (*_tag arrays) and never change.
// The index-space copies (bonds, angles, dihedrals, pairs14) are what force
// kernels read; remap() regenerates them through rank[] after every particle
// permutation. Exclusion CSRs stay in tag space: kernels look up the row of
// tag[i] and compare against tag[j].
struct DeviceTopology {
    int nbonds = 0, nangles = 0, ndihedrals = 0, npairs14 = 0;

    int2* bonds_tag = nullptr;      int2* bonds = nullptr;      int* bond_type = nullptr;
    int4* angles_tag = nullptr;     int4* angles = nullptr;     // .w = angle type
    int4* dihedrals_tag = nullptr;  int4* dihedrals = nullptr;  int* dihedral_type = nullptr;
    int2* pairs14_tag = nullptr;    int2* pairs14 = nullptr;

    // Neighbor-list exclusions (tag space, CSR rows indexed by tag)
    int* excl_offsets = nullptr;   int* excl_list = nullptr;   int nexcl = 0;
    // 1-2/1-3 pairs for the Ewald real-space correction (tag space)
    int* excl13_offsets = nullptr; int* excl13_list = nullptr; int nexcl13 = 0;

    // Per-atom bonded-term membership (tag space): row t lists
    // {term index, kind * 4 + role} for every bond/angle/dihedral that
    // contains the particle with tag t (see force/bonded_gather.cuh).
    int*  term_offsets = nullptr;  int2* terms = nullptr;  int nterms = 0;

    void upload(int natoms,
                const std::vector<int2>& bonds_h, const std::vector<int>& bond_types_h,
                const std::vector<int4>& angles_h,
                const std::vector<int4>& dihedrals_h, const std::vector<int>& dihedral_types_h,
                const std::vector<int2>& pairs14_h,
                const std::vector<int>& excl_offsets_h, const std::vector<int>& excl_list_h,
                const std::vector<int>& excl13_offsets_h, const std::vector<int>& excl13_list_h);
    void remap(const int* d_rank, cudaStream_t stream = 0);
    void free();
};
