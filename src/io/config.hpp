#pragma once
#include "../core/types.cuh"
#include "../force/fene.cuh"
#include "../force/angle.cuh"
#include <string>
#include <vector>

struct TopologyData {
    std::vector<float4> positions;
    std::vector<float4> velocities;
    std::vector<int2>   bonds;
    std::vector<int>    bond_types;
    std::vector<int4>   angles;
    std::vector<float2> lj_params;
    std::vector<FENEParams> bond_params;
    std::vector<AngleParams> angle_params;
    std::string data_file;

    std::vector<int> exclusion_offsets;
    std::vector<int> exclusion_list;
};

SimParams parse_config(const std::string& filename, TopologyData& topo);
