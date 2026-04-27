#pragma once
#include "config.hpp"
#include <string>

void parse_lammps_data(const std::string& path, TopologyData& topo);
void build_exclusions(const TopologyData& in_topo, TopologyData& out_topo);
