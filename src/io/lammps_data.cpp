#include "lammps_data.hpp"
#include "../core/types.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>

static bool is_section_header(const std::string& line) {
    if (line.empty()) return false;
    return std::isupper(static_cast<unsigned char>(line[0]));
}

void parse_lammps_data(const std::string& path, TopologyData& topo) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open LAMMPS data file: " + path);
    }

    enum class Section { None, Atoms, Velocities, Bonds, Angles };
    Section current = Section::None;

    std::unordered_map<int, size_t> id_to_idx;
    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        std::string trimmed = line.substr(start);

        if (trimmed[0] == '#') continue;

        if (is_section_header(trimmed)) {
            if (trimmed.find("Atoms") == 0) {
                current = Section::Atoms;
                continue;
            } else if (trimmed.find("Velocities") == 0) {
                current = Section::Velocities;
                continue;
            } else if (trimmed.find("Bonds") == 0) {
                current = Section::Bonds;
                continue;
            } else if (trimmed.find("Angles") == 0) {
                current = Section::Angles;
                continue;
            } else if (
                trimmed.find("Dihedrals") == 0 ||
                trimmed.find("Impropers") == 0 ||
                trimmed.find("Masses") == 0 ||
                trimmed.find("Pair Coeffs") == 0 ||
                trimmed.find("Bond Coeffs") == 0 ||
                trimmed.find("Angle Coeffs") == 0) {
                current = Section::None;
                continue;
            } else {
                current = Section::None;
                continue;
            }
        }

        if (current == Section::None) continue;

        std::istringstream iss(trimmed);
        if (current == Section::Atoms) {
            int id, mol, type;
            float x, y, z;
            iss >> id >> mol >> type >> x >> y >> z;
            if (iss.fail()) continue;
            size_t idx = topo.positions.size();
            id_to_idx[id] = idx;
            topo.positions.push_back(make_float4(x, y, z, pack_type_id(type - 1)));
        } else if (current == Section::Velocities) {
            int id;
            float vx, vy, vz;
            iss >> id >> vx >> vy >> vz;
            if (iss.fail()) continue;
            auto it = id_to_idx.find(id);
            if (it == id_to_idx.end()) continue;
            if (topo.velocities.size() < topo.positions.size())
                topo.velocities.resize(topo.positions.size(), make_float4(0,0,0,0));
            topo.velocities[it->second] = make_float4(vx, vy, vz, 0);
        } else if (current == Section::Bonds) {
            int id, type, atom1, atom2;
            iss >> id >> type >> atom1 >> atom2;
            if (iss.fail()) continue;
            auto it1 = id_to_idx.find(atom1);
            auto it2 = id_to_idx.find(atom2);
            if (it1 == id_to_idx.end() || it2 == id_to_idx.end()) continue;
            topo.bonds.push_back(make_int2(static_cast<int>(it1->second),
                                           static_cast<int>(it2->second)));
            topo.bond_types.push_back(type - 1);
        } else if (current == Section::Angles) {
            int id, type, atom1, atom2, atom3;
            iss >> id >> type >> atom1 >> atom2 >> atom3;
            if (iss.fail()) continue;
            auto it1 = id_to_idx.find(atom1);
            auto it2 = id_to_idx.find(atom2);
            auto it3 = id_to_idx.find(atom3);
            if (it1 == id_to_idx.end() || it2 == id_to_idx.end() || it3 == id_to_idx.end()) continue;
            topo.angles.push_back(make_int4(static_cast<int>(it1->second),
                                            static_cast<int>(it2->second),
                                            static_cast<int>(it3->second),
                                            type - 1));
        }
    }
}

void build_exclusions(const TopologyData& in_topo, TopologyData& out_topo) {
    int natoms = static_cast<int>(in_topo.positions.size());
    std::vector<std::vector<int>> per_atom(natoms);

    for (const auto& b : in_topo.bonds) {
        int i = b.x;
        int j = b.y;
        per_atom[i].push_back(j);
        per_atom[j].push_back(i);
    }

    out_topo.exclusion_offsets.resize(natoms + 1, 0);
    out_topo.exclusion_list.clear();

    for (int i = 0; i < natoms; i++) {
        auto& list = per_atom[i];
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        out_topo.exclusion_offsets[i] = static_cast<int>(out_topo.exclusion_list.size());
        out_topo.exclusion_list.insert(out_topo.exclusion_list.end(), list.begin(), list.end());
    }
    out_topo.exclusion_offsets[natoms] = static_cast<int>(out_topo.exclusion_list.size());
}
