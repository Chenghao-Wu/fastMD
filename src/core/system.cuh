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

    int2*   bonds;
    int*    bond_param_idx;
    int4*   angles;
    int     nbonds;
    int     nangles;

    int*    exclusion_offsets;
    int*    exclusion_list;
    int     nexclusions;

    int*    d_mol_id;
    int*    d_image;

    float2* lj_params;
    float*  virial;

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
