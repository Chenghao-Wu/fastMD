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
#include <cmath>

static bool is_section_header(const std::string& line) {
    if (line.empty()) return false;
    return std::isupper(static_cast<unsigned char>(line[0]));
}

void parse_lammps_data(const std::string& path, TopologyData& topo) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open LAMMPS data file: " + path);
    }

    // Parse box dimensions from header
    float box_Lx = 0, box_Ly = 0, box_Lz = 0;
    int box_count = 0;
    {
        std::string line;
        while (std::getline(file, line) && box_count < 3) {
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            if (line[start] == '#') continue;

            float lo, hi;
            char dim_label[4];
            if (sscanf(line.c_str() + start, "%f %f %3s", &lo, &hi, dim_label) == 3) {
                std::string label(dim_label);
                if (label == "xlo") { box_Lx = hi - lo; box_count++; }
                else if (label == "ylo") { box_Ly = hi - lo; box_count++; }
                else if (label == "zlo") { box_Lz = hi - lo; box_count++; }
            }
        }
        if (box_count != 3) {
            throw std::runtime_error("LAMMPS data file missing box dimensions");
        }
        if (std::fabs(box_Lx - box_Ly) > 1e-6f || std::fabs(box_Lx - box_Lz) > 1e-6f) {
            throw std::runtime_error("Non-cubic box not supported");
        }
        topo.box_L = box_Lx;
        file.clear();
        file.seekg(0);
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
            topo.mol_ids.push_back(mol);
            topo.positions.push_back(make_float4(x, y, z, pack_type_id(type - 1)));

            int ix = 0, iy = 0, iz = 0;
            iss >> ix >> iy >> iz;
            topo.images.push_back(ix);
            topo.images.push_back(iy);
            topo.images.push_back(iz);
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

std::string build_restart_filename(const std::string& base, int64_t step) {
    size_t dot = base.rfind('.');
    if (dot != std::string::npos)
        return base.substr(0, dot) + "_" + std::to_string(step) + base.substr(dot);
    return base + "_" + std::to_string(step);
}

std::string build_restart_final_filename(const std::string& base) {
    size_t dot = base.rfind('.');
    if (dot != std::string::npos)
        return base.substr(0, dot) + "_final" + base.substr(dot);
    return base + "_final";
}

void write_lammps_data(const std::string& path,
                       const float4* d_pos,
                       const float4* d_vel,
                       const int*    d_image,
                       const int*    d_mol_id,
                       const int2*   d_bonds,
                       const int*    d_bond_types,
                       const int4*   d_angles,
                       int natoms, int nbonds, int nangles,
                       int ntypes, float box_L,
                       int64_t step,
                       cudaStream_t stream) {

    std::vector<float4> h_pos(natoms);
    std::vector<float4> h_vel(natoms);
    std::vector<int> h_image(d_image ? natoms * 3 : 0);
    std::vector<int> h_mol_id(d_mol_id ? natoms : 0);
    std::vector<int2> h_bonds(nbonds);
    std::vector<int> h_bond_types(nbonds);
    std::vector<int4> h_angles(nangles);

    CUDA_CHECK(cudaMemcpyAsync(h_pos.data(), d_pos,
                                natoms * sizeof(float4),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_vel.data(), d_vel,
                                natoms * sizeof(float4),
                                cudaMemcpyDeviceToHost, stream));
    if (d_image)
        CUDA_CHECK(cudaMemcpyAsync(h_image.data(), d_image,
                                    natoms * 3 * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    if (d_mol_id)
        CUDA_CHECK(cudaMemcpyAsync(h_mol_id.data(), d_mol_id,
                                    natoms * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    if (nbonds > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_bonds.data(), d_bonds,
                                    nbonds * sizeof(int2),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_bond_types.data(), d_bond_types,
                                    nbonds * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    }
    if (nangles > 0)
        CUDA_CHECK(cudaMemcpyAsync(h_angles.data(), d_angles,
                                    nangles * sizeof(int4),
                                    cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open restart file %s\n", path.c_str());
        return;
    }

    fprintf(fp, "# Restart from step %ld\n", step);
    fprintf(fp, "%d atoms\n", natoms);
    fprintf(fp, "%d atom types\n", ntypes);
    if (nbonds > 0) fprintf(fp, "%d bonds\n", nbonds);
    if (nangles > 0) fprintf(fp, "%d angles\n", nangles);
    fprintf(fp, "\n");
    fprintf(fp, "0.000000 %.6f xlo xhi\n", box_L);
    fprintf(fp, "0.000000 %.6f ylo yhi\n", box_L);
    fprintf(fp, "0.000000 %.6f zlo zhi\n");
    fprintf(fp, "\n");

    fprintf(fp, "Atoms\n\n");
    for (int i = 0; i < natoms; i++) {
        int mol = d_mol_id ? (h_mol_id[i] + 1) : 1;
        int type = unpack_type_id(h_pos[i].w) + 1;
        int ix = d_image ? h_image[i * 3] : 0;
        int iy = d_image ? h_image[i * 3 + 1] : 0;
        int iz = d_image ? h_image[i * 3 + 2] : 0;
        fprintf(fp, "%d %d %d %.6f %.6f %.6f %d %d %d\n",
                i + 1, mol, type,
                h_pos[i].x, h_pos[i].y, h_pos[i].z,
                ix, iy, iz);
    }

    fprintf(fp, "\nVelocities\n\n");
    for (int i = 0; i < natoms; i++) {
        fprintf(fp, "%d %.6f %.6f %.6f\n",
                i + 1, h_vel[i].x, h_vel[i].y, h_vel[i].z);
    }

    if (nbonds > 0) {
        fprintf(fp, "\nBonds\n\n");
        for (int i = 0; i < nbonds; i++) {
            fprintf(fp, "%d %d %d %d\n",
                    i + 1, h_bond_types[i] + 1,
                    h_bonds[i].x + 1, h_bonds[i].y + 1);
        }
    }

    if (nangles > 0) {
        fprintf(fp, "\nAngles\n\n");
        for (int i = 0; i < nangles; i++) {
            fprintf(fp, "%d %d %d %d %d\n",
                    i + 1, h_angles[i].w + 1,
                    h_angles[i].x + 1, h_angles[i].y + 1,
                    h_angles[i].z + 1);
        }
    }

    fclose(fp);
    printf("Restart written to %s (step %ld)\n", path.c_str(), step);
}
