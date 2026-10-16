#pragma once
#include "../core/types.cuh"
#include "../force/fene.cuh"
#include "../force/angle.cuh"
#include "../force/bond_harmonic.cuh"
#include "../force/dihedral_opls.cuh"
#include "table_parser.hpp"
#include <string>
#include <vector>

struct TopologyData {
    std::vector<float4> positions;
    std::vector<float4> velocities;
    std::vector<float>  charges;
    std::vector<int2>   bonds;
    std::vector<int>    bond_types;
    std::vector<int4>   angles;
    std::vector<int4>   dihedrals;
    std::vector<int>    dihedral_types;
    std::vector<float2> lj_params;
    std::vector<FENEParams> bond_params;
    std::vector<AngleParams> angle_params;
    std::vector<BondHarmonicParams> hbond_params;
    std::vector<OPLSDihedralParams> dihedral_params;
    std::string data_file;

    std::vector<int> exclusion_offsets;
    std::vector<int> exclusion_list;

    // 1-2/1-3-only exclusion CSR (for Ewald real-space correction)
    std::vector<int> excl13_offsets;
    std::vector<int> excl13_list;

    // unique 1-4 pair list (scaled, not fully excluded)
    std::vector<int2> pairs14;

    std::vector<int> mol_ids;
    std::vector<int> images;
    std::vector<int> atom_ids;   // atom id from the data file, per tag
    float box_L = 0.0f;
    float3 box_lo = {0.0f, 0.0f, 0.0f};  // box origin in the input file
    // type counts declared by the data file (0 when absent)
    int n_atom_types = 0, n_bond_types = 0, n_angle_types = 0, n_dihedral_types = 0;

    std::vector<float> masses;

    std::vector<int>         table_idx;
    std::vector<TableParams> table_params;
    std::vector<float4>      table_data;

    std::vector<int>         bond_table_idx;
    std::vector<TableParams> bond_table_params;
    std::vector<float4>      bond_table_data;
    std::vector<int>         angle_table_idx;
    std::vector<TableParams> angle_table_params;
    std::vector<float4>      angle_table_data;
};

void finalize_params(SimParams& params);

SimParams parse_config(const std::string& filename, TopologyData& topo);
