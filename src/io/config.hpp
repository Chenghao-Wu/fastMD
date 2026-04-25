#pragma once
#include "../core/types.cuh"
#include <string>
#include <vector>

struct TopologyData {
    std::vector<float4> positions;
    std::vector<float4> velocities;
    std::vector<int2>   bonds;
    std::vector<int>    bond_types;
    std::vector<int4>   angles;
    std::vector<float2> lj_params;
};

SimParams parse_config(const std::string& filename, TopologyData& topo);
