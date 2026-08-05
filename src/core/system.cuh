#pragma once
#include "types.cuh"
#include "io/table_parser.hpp"
#include <vector>

static constexpr int MAX_TYPES = 16;
extern __constant__ float c_masses[MAX_TYPES];

struct System {
    float4* pos;
    float4* vel;
    float4* force;
    float4* pos_ref;
    float*  charges;

    int2*   bonds;
    int*    bond_param_idx;
    int4*   angles;
    int4*   dihedrals;
    int*    dihedral_param_idx;
    int     nbonds;
    int     nangles;
    int     ndihedrals;

    int*    exclusion_offsets;
    int*    exclusion_list;
    int     nexclusions;

    // 1-2/1-3-only CSR for Ewald real-space subtraction
    int*    excl13_offsets;
    int*    excl13_list;
    int     nexcl13;

    // unique 1-4 pairs (each stored once)
    int2*   pairs14;
    int     npairs14;

    int*    d_mol_id;
    int*    d_image;

    float2* lj_params;
    float*  virial;

    // Ewald reciprocal-space buffers
    int3*   ewald_kvecs;   // [kcount] integer triplets (nx, ny, nz)
    float*  ewald_sfac;    // [2*kcount] structure factor (rl, im)
    float*  ewald_ek;      // [natoms_padded*3] per-atom K-space field
    int     ewald_kcount;

    int*         d_table_idx;
    TableParams* d_table_params;
    float4*      d_table_data;

    int*    d_max_dr2_int;
    int*    h_rebuild_flag;

    int     natoms;
    int     natoms_padded;
    int     ntiles;
    int     ntypes;

    void allocate(const SimParams& params, const float* h_masses);
    void free();
    void zero_forces();
    void zero_virial();
    void allocate_rg_buffers(const std::vector<int>& mol_ids,
                             const std::vector<int>& images,
                             int natoms_padded);
};
