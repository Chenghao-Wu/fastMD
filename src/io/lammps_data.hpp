#pragma once
#include "config.hpp"
#include <string>

void parse_lammps_data(const std::string& path, TopologyData& topo);
void build_exclusions(const TopologyData& in_topo, TopologyData& out_topo);

void write_lammps_data(const std::string& path,
                       const float4* d_pos,
                       const float4* d_vel,
                       const int*    d_image,
                       const int*    d_mol_id,
                       const int2*   d_bonds,
                       const int*    d_bond_types,
                       const int4*   d_angles,
                       int natoms, int nbonds, int nangles,
                       int ntypes, int nbond_types, int nangle_types,
                       float box_L,
                       int64_t step,
                       cudaStream_t stream = 0);

std::string build_restart_filename(const std::string& base, int64_t step);
std::string build_restart_final_filename(const std::string& base);
