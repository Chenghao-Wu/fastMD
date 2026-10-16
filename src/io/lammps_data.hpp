#pragma once
#include "config.hpp"
#include <string>

void parse_lammps_data(const std::string& path, TopologyData& topo);
void build_exclusions(const TopologyData& in_topo, TopologyData& out_topo);
void build_exclusions_opls(const TopologyData& in_topo, TopologyData& out_topo,
                           bool from_dihedrals = false);

// Per-particle state for a restart, in tag (input-file) order.
struct RestartFrame {
    std::vector<float4> pos;     // wrapped positions + packed type
    std::vector<float4> vel;
    std::vector<int>    image;   // 3 per particle
    std::vector<float>  charge;
    std::vector<int>    mol_id;
};

// Write a LAMMPS data file (Atoms # full + Velocities + topology) that
// parse_lammps_data() reads back to the same state. Atom ids are taken from
// topo.atom_ids (input-file ids), topology from topo's tag-space lists.
void write_lammps_data(const std::string& path, const RestartFrame& frame,
                       const TopologyData& topo,
                       int ntypes, int nbond_types, int nangle_types, int ndihedral_types,
                       const Box& box, int64_t step);

std::string build_restart_filename(const std::string& base, int64_t step);
std::string build_restart_final_filename(const std::string& base);
