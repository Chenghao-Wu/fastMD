#include "lammps_data.hpp"
#include "../core/types.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <string>

static bool is_section_header(const std::string& line) {
    if (line.empty()) return false;
    // A section header starts with a capital letter
    return std::isupper(static_cast<unsigned char>(line[0]));
}

void parse_lammps_data(const std::string& path, TopologyData& topo) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open LAMMPS data file: " + path);
    }

    enum class Section { None, Atoms, Bonds, Angles };
    Section current = Section::None;

    std::string line;
    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue; // empty line
        std::string trimmed = line.substr(start);

        // Skip comment lines
        if (trimmed[0] == '#') continue;

        if (is_section_header(trimmed)) {
            if (trimmed.find("Atoms") == 0) {
                current = Section::Atoms;
                continue;
            } else if (trimmed.find("Bonds") == 0) {
                current = Section::Bonds;
                continue;
            } else if (trimmed.find("Angles") == 0) {
                current = Section::Angles;
                continue;
            } else if (
                trimmed.find("Velocities") == 0 ||
                trimmed.find("Dihedrals") == 0 ||
                trimmed.find("Impropers") == 0 ||
                trimmed.find("Masses") == 0 ||
                trimmed.find("Pair Coeffs") == 0 ||
                trimmed.find("Bond Coeffs") == 0 ||
                trimmed.find("Angle Coeffs") == 0) {
                current = Section::None;
                continue;
            } else {
                // Unknown header: reset current section
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
            topo.positions.push_back(make_float4(x, y, z, pack_type_id(type - 1)));
        } else if (current == Section::Bonds) {
            int id, type, atom1, atom2;
            iss >> id >> type >> atom1 >> atom2;
            if (iss.fail()) continue;
            topo.bonds.push_back(make_int2(atom1 - 1, atom2 - 1));
            topo.bond_types.push_back(type - 1);
        } else if (current == Section::Angles) {
            int id, type, atom1, atom2, atom3;
            iss >> id >> type >> atom1 >> atom2 >> atom3;
            if (iss.fail()) continue;
            topo.angles.push_back(make_int4(atom1 - 1, atom2 - 1, atom3 - 1, type - 1));
        }
    }
}
