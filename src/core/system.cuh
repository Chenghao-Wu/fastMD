#pragma once
#include "types.cuh"
#include <vector>

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

    int*    d_max_dr2_int;
    int*    h_rebuild_flag;

    int     natoms;
    int     natoms_padded;
    int     ntiles;
    int     ntypes;

    void allocate(const SimParams& params);
    void free();
    void zero_forces();
    void zero_virial();
    void allocate_rg_buffers(const std::vector<int>& mol_ids,
                             const std::vector<int>& images,
                             int natoms_padded);
};
