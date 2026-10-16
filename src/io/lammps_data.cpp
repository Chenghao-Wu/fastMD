#include "lammps_data.hpp"
#include "../core/types.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdio>

namespace {

std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tok;
    std::istringstream iss(line);
    std::string t;
    while (iss >> t) {
        if (t[0] == '#') break;
        tok.push_back(t);
    }
    return tok;
}

[[noreturn]] void data_error(const std::string& path, int lineno, const std::string& msg) {
    throw std::runtime_error(path + ":" + std::to_string(lineno) + ": " + msg);
}

int to_int(const std::string& s, const std::string& path, int lineno) {
    size_t pos = 0;
    long v = 0;
    try { v = std::stol(s, &pos); } catch (...) { pos = 0; }
    if (pos != s.size()) data_error(path, lineno, "expected an integer, got '" + s + "'");
    return (int)v;
}

float to_float(const std::string& s, const std::string& path, int lineno) {
    size_t pos = 0;
    float v = 0;
    try { v = std::stof(s, &pos); } catch (...) { pos = 0; }
    if (pos != s.size() || !std::isfinite(v))
        data_error(path, lineno, "expected a finite number, got '" + s + "'");
    return v;
}

// Column layout of supported atom styles: (has_mol, has_q, ncols without images)
struct StyleLayout { bool mol; bool q; int ncols; };
bool style_layout(const std::string& style, StyleLayout& out) {
    if (style == "atomic")    { out = {false, false, 5}; return true; }
    if (style == "charge")    { out = {false, true,  6}; return true; }
    if (style == "molecular" || style == "bond" || style == "angle")
                              { out = {true,  false, 6}; return true; }
    if (style == "full")      { out = {true,  true,  7}; return true; }
    return false;
}

}  // namespace

void parse_lammps_data(const std::string& path, TopologyData& topo) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open LAMMPS data file: " + path);

    long n_atoms = -1, n_bonds = 0, n_angles = 0, n_dihedrals = 0, n_impropers = 0;
    int n_atom_types = -1, n_bond_types = -1, n_angle_types = -1, n_dihedral_types = -1;
    float lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    bool have_box[3] = {false, false, false};

    enum class Section { Header, None, Atoms, Velocities, Bonds, Angles, Masses, Dihedrals };
    Section current = Section::Header;
    std::string atom_style;  // empty until known (hint or inferred)
    StyleLayout layout{};
    bool seen_velocities = false;
    size_t n_velocity_lines = 0;

    std::unordered_map<int, size_t> id_to_idx;
    auto lookup = [&](int id, int lineno) -> int {
        auto it = id_to_idx.find(id);
        if (it == id_to_idx.end())
            data_error(path, lineno, "reference to unknown atom id " + std::to_string(id));
        return (int)it->second;
    };
    auto check_type = [&](int type, int ntypes, const char* what, int lineno) {
        if (type < 1 || (ntypes > 0 && type > ntypes))
            data_error(path, lineno, std::string(what) + " type " + std::to_string(type) +
                       " out of range [1, " + std::to_string(ntypes) + "]");
    };

    std::string line;
    int lineno = 0;
    bool first_line = true;
    while (std::getline(file, line)) {
        ++lineno;
        if (first_line) { first_line = false; continue; }  // title line
        auto tok = split_ws(line);
        if (tok.empty()) continue;

        // Section keywords start with an upper-case letter.
        if (std::isupper(static_cast<unsigned char>(tok[0][0]))) {
            const std::string& k = tok[0];
            std::string k2 = tok.size() > 1 ? k + " " + tok[1] : k;
            if (k == "Atoms") {
                current = Section::Atoms;
                size_t hash = line.find('#');
                if (hash != std::string::npos) {
                    std::istringstream ss(line.substr(hash + 1));
                    ss >> atom_style;
                    // LAMMPS appends accelerator suffixes, e.g. "molecular/kk"
                    atom_style = atom_style.substr(0, atom_style.find('/'));
                    if (!style_layout(atom_style, layout))
                        data_error(path, lineno, "unsupported atom style '" + atom_style +
                                   "' (supported: atomic, charge, molecular, bond, angle, full)");
                }
            } else if (k == "Velocities") { current = Section::Velocities; seen_velocities = true; }
            else if (k == "Masses") current = Section::Masses;
            else if (k == "Bonds") current = Section::Bonds;
            else if (k == "Angles") current = Section::Angles;
            else if (k == "Dihedrals") current = Section::Dihedrals;
            else if (k == "Impropers") {
                if (n_impropers > 0)
                    fprintf(stderr, "Warning: %s: %ld impropers are ignored (no improper styles)\n",
                            path.c_str(), n_impropers);
                current = Section::None;
            } else if (k2 == "Pair Coeffs" || k2 == "PairIJ Coeffs" || k2 == "Bond Coeffs" ||
                       k2 == "Angle Coeffs" || k2 == "Dihedral Coeffs" || k2 == "Improper Coeffs" ||
                       k2 == "BondBond Coeffs" || k2 == "BondAngle Coeffs" ||
                       k2 == "MiddleBondTorsion Coeffs" || k2 == "EndBondTorsion Coeffs" ||
                       k2 == "AngleTorsion Coeffs" || k2 == "AngleAngleTorsion Coeffs" ||
                       k2 == "BondBond13 Coeffs" || k2 == "AngleAngle Coeffs") {
                current = Section::None;  // coefficients come from the fastMD config
            } else {
                data_error(path, lineno, "unknown section '" + line + "'");
            }
            continue;
        }

        if (current == Section::Header) {
            if (tok.size() == 2 && tok[1] == "atoms") n_atoms = to_int(tok[0], path, lineno);
            else if (tok.size() == 2 && tok[1] == "bonds") n_bonds = to_int(tok[0], path, lineno);
            else if (tok.size() == 2 && tok[1] == "angles") n_angles = to_int(tok[0], path, lineno);
            else if (tok.size() == 2 && tok[1] == "dihedrals") n_dihedrals = to_int(tok[0], path, lineno);
            else if (tok.size() == 2 && tok[1] == "impropers") n_impropers = to_int(tok[0], path, lineno);
            else if (tok.size() == 3 && tok[2] == "types") {
                int n = to_int(tok[0], path, lineno);
                if (tok[1] == "atom") n_atom_types = n;
                else if (tok[1] == "bond") n_bond_types = n;
                else if (tok[1] == "angle") n_angle_types = n;
                else if (tok[1] == "dihedral") n_dihedral_types = n;
            } else if (tok.size() == 4 && (tok[2] == "xlo" || tok[2] == "ylo" || tok[2] == "zlo")) {
                int d = tok[2][0] - 'x';
                lo[d] = to_float(tok[0], path, lineno);
                hi[d] = to_float(tok[1], path, lineno);
                have_box[d] = true;
            } else if (tok.size() >= 6 && tok[3] == "xy") {
                float xy = to_float(tok[0], path, lineno), xz = to_float(tok[1], path, lineno),
                      yz = to_float(tok[2], path, lineno);
                if (xz != 0.0f || yz != 0.0f)
                    data_error(path, lineno, "only an xy tilt is supported (xz = yz = 0)");
                topo.tilt_xy = xy;
            }
            // other header lines (extra/per-atom, ellipsoids, ...) are irrelevant here
            continue;
        }
        if (current == Section::None) continue;

        if (current == Section::Masses) {
            if (tok.size() < 2) data_error(path, lineno, "Masses line needs: type mass");
            int type = to_int(tok[0], path, lineno);
            check_type(type, n_atom_types, "atom", lineno);
            float mass = to_float(tok[1], path, lineno);
            if (!(mass > 0.0f)) data_error(path, lineno, "mass must be > 0");
            if ((size_t)type > topo.masses.size()) topo.masses.resize(type, 1.0f);
            topo.masses[type - 1] = mass;
        } else if (current == Section::Atoms) {
            if (atom_style.empty()) {
                // No style hint: infer from the column count of the first line.
                switch (tok.size()) {
                    case 5: case 8:  atom_style = "atomic"; break;
                    case 6: case 9:  atom_style = "molecular"; break;
                    case 7: case 10: atom_style = "full"; break;
                    default:
                        data_error(path, lineno, "cannot infer the atom style from " +
                                   std::to_string(tok.size()) + " columns; add e.g. 'Atoms # full'");
                }
                style_layout(atom_style, layout);
                fprintf(stderr, "Note: %s has no atom-style hint; inferred '%s' from the column count\n",
                        path.c_str(), atom_style.c_str());
            }
            int nc = layout.ncols;
            if ((int)tok.size() != nc && (int)tok.size() != nc + 3)
                data_error(path, lineno, "atom style '" + atom_style + "' expects " +
                           std::to_string(nc) + " or " + std::to_string(nc + 3) + " columns");
            int c = 0;
            int id = to_int(tok[c++], path, lineno);
            int mol = layout.mol ? to_int(tok[c++], path, lineno) : 1;
            int type = to_int(tok[c++], path, lineno);
            float q = layout.q ? to_float(tok[c++], path, lineno) : 0.0f;
            float x = to_float(tok[c++], path, lineno);
            float y = to_float(tok[c++], path, lineno);
            float z = to_float(tok[c++], path, lineno);
            int im[3] = {0, 0, 0};
            if ((int)tok.size() == nc + 3)
                for (int d = 0; d < 3; d++) im[d] = to_int(tok[c++], path, lineno);
            check_type(type, n_atom_types, "atom", lineno);
            if (!id_to_idx.emplace(id, topo.positions.size()).second)
                data_error(path, lineno, "duplicate atom id " + std::to_string(id));
            topo.atom_ids.push_back(id);
            topo.mol_ids.push_back(mol);
            if (layout.q) topo.charges.push_back(q);
            topo.positions.push_back(make_float4(x, y, z, pack_type_id(type - 1)));
            topo.images.insert(topo.images.end(), im, im + 3);
        } else if (current == Section::Velocities) {
            if (tok.size() < 4) data_error(path, lineno, "Velocities line needs: id vx vy vz");
            int i = lookup(to_int(tok[0], path, lineno), lineno);
            if (topo.velocities.size() < topo.positions.size())
                topo.velocities.resize(topo.positions.size(), make_float4(0, 0, 0, 0));
            n_velocity_lines++;
            topo.velocities[i] = make_float4(to_float(tok[1], path, lineno),
                                             to_float(tok[2], path, lineno),
                                             to_float(tok[3], path, lineno), 0.0f);
        } else if (current == Section::Bonds) {
            if (tok.size() < 4) data_error(path, lineno, "Bonds line needs: id type atom1 atom2");
            int type = to_int(tok[1], path, lineno);
            check_type(type, n_bond_types, "bond", lineno);
            int a = lookup(to_int(tok[2], path, lineno), lineno);
            int b = lookup(to_int(tok[3], path, lineno), lineno);
            if (a == b) data_error(path, lineno, "bond between an atom and itself");
            topo.bonds.push_back(make_int2(a, b));
            topo.bond_types.push_back(type - 1);
        } else if (current == Section::Angles) {
            if (tok.size() < 5) data_error(path, lineno, "Angles line needs: id type atom1 atom2 atom3");
            int type = to_int(tok[1], path, lineno);
            check_type(type, n_angle_types, "angle", lineno);
            topo.angles.push_back(make_int4(lookup(to_int(tok[2], path, lineno), lineno),
                                            lookup(to_int(tok[3], path, lineno), lineno),
                                            lookup(to_int(tok[4], path, lineno), lineno), type - 1));
        } else if (current == Section::Dihedrals) {
            if (tok.size() < 6) data_error(path, lineno, "Dihedrals line needs: id type a1 a2 a3 a4");
            int type = to_int(tok[1], path, lineno);
            check_type(type, n_dihedral_types, "dihedral", lineno);
            topo.dihedrals.push_back(make_int4(lookup(to_int(tok[2], path, lineno), lineno),
                                               lookup(to_int(tok[3], path, lineno), lineno),
                                               lookup(to_int(tok[4], path, lineno), lineno),
                                               lookup(to_int(tok[5], path, lineno), lineno)));
            topo.dihedral_types.push_back(type - 1);
        }
    }

    auto require_count = [&](const char* what, long declared, size_t got) {
        if (declared >= 0 && (size_t)declared != got)
            throw std::runtime_error(path + ": header declares " + std::to_string(declared) + " " +
                                     what + " but " + std::to_string(got) + " were read");
    };
    if (n_atoms < 0) throw std::runtime_error(path + ": missing 'N atoms' header line");
    require_count("atoms", n_atoms, topo.positions.size());
    require_count("bonds", n_bonds, topo.bonds.size());
    require_count("angles", n_angles, topo.angles.size());
    require_count("dihedrals", n_dihedrals, topo.dihedrals.size());
    if (seen_velocities && n_velocity_lines != topo.positions.size())
        throw std::runtime_error(path + ": Velocities section has " +
                                 std::to_string(n_velocity_lines) + " lines for " +
                                 std::to_string(topo.positions.size()) + " atoms");

    // Tags follow ascending atom id regardless of the order atoms appear in
    // the file (LAMMPS write_data emits storage order).
    {
        const size_t n = topo.positions.size();
        std::vector<int> order(n);
        for (size_t i = 0; i < n; i++) order[i] = (int)i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return topo.atom_ids[a] < topo.atom_ids[b]; });
        bool identity = true;
        for (size_t i = 0; i < n && identity; i++) identity = (order[i] == (int)i);
        if (!identity) {
            std::vector<int> new_of_old(n);
            for (size_t k = 0; k < n; k++) new_of_old[order[k]] = (int)k;
            auto permute = [&](auto& v, size_t width) {
                if (v.empty()) return;
                auto old = v;
                for (size_t k = 0; k < n; k++)
                    for (size_t c = 0; c < width; c++) v[k * width + c] = old[order[k] * width + c];
            };
            permute(topo.positions, 1);
            permute(topo.velocities, 1);
            permute(topo.charges, 1);
            permute(topo.mol_ids, 1);
            permute(topo.atom_ids, 1);
            permute(topo.images, 3);
            for (auto& b : topo.bonds) { b.x = new_of_old[b.x]; b.y = new_of_old[b.y]; }
            for (auto& a : topo.angles) { a.x = new_of_old[a.x]; a.y = new_of_old[a.y]; a.z = new_of_old[a.z]; }
            for (auto& d : topo.dihedrals) {
                d.x = new_of_old[d.x]; d.y = new_of_old[d.y]; d.z = new_of_old[d.z]; d.w = new_of_old[d.w];
            }
        }
    }

    if (!(have_box[0] && have_box[1] && have_box[2]))
        throw std::runtime_error(path + ": missing box dimensions (xlo xhi / ylo yhi / zlo zhi)");
    float L[3];
    for (int d = 0; d < 3; d++) {
        L[d] = hi[d] - lo[d];
        if (!(L[d] > 0.0f)) throw std::runtime_error(path + ": box length must be positive");
    }
    topo.box = make_float3(L[0], L[1], L[2]);
    if (std::fabs(topo.tilt_xy) > 0.5f * L[0])
        throw std::runtime_error(path + ": |xy| tilt exceeds Lx / 2");
    Box wrap_box = make_box(topo.box);
    wrap_box.xy = topo.tilt_xy;
    topo.box_lo = make_float3(lo[0], lo[1], lo[2]);

    // Shift to a box spanning [0, L) and wrap, keeping unwrapped coordinates
    // (position + image * L) unchanged.
    for (size_t i = 0; i < topo.positions.size(); i++) {
        float4& p = topo.positions[i];
        p.x -= lo[0];
        p.y -= lo[1];
        p.z -= lo[2];
        wrap_position_host(p.x, p.y, p.z, &topo.images[3 * i], wrap_box);
    }

    int ntypes = n_atom_types > 0 ? n_atom_types : 1;
    for (const auto& pos : topo.positions) ntypes = std::max(ntypes, unpack_type_id(pos.w) + 1);
    topo.masses.resize(ntypes, 1.0f);
    topo.n_atom_types = ntypes;
    topo.n_bond_types = std::max(n_bond_types, 0);
    topo.n_angle_types = std::max(n_angle_types, 0);
    topo.n_dihedral_types = std::max(n_dihedral_types, 0);
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

// OPLS mode (special_bonds lj/coul 0 0 f14):
//  - exclusion CSR gets 1-2, 1-3 and 1-4 (the pair14 kernel re-adds scaled 1-4)
//  - excl13 CSR gets 1-2 and 1-3 only (Ewald real-space erf subtraction)
//  - pairs14: unique (i < j) pairs at bond-graph distance exactly 3, as in
//    LAMMPS' default special_bonds; with from_dihedrals only the end atoms of
//    listed dihedrals qualify ('special_bonds dihedral yes'). A pair that is
//    also 1-2 or 1-3 (small rings) keeps the closer classification.
void build_exclusions_opls(const TopologyData& in_topo, TopologyData& out_topo,
                           bool from_dihedrals) {
    int natoms = static_cast<int>(in_topo.positions.size());
    std::vector<std::vector<int>> bond_adj(natoms);
    for (const auto& b : in_topo.bonds) {
        bond_adj[b.x].push_back(b.y);
        bond_adj[b.y].push_back(b.x);
    }
    for (auto& adj : bond_adj) {
        std::sort(adj.begin(), adj.end());
        adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
    }

    // 1-2 and 1-3 set per atom
    std::vector<std::vector<int>> ex13(natoms);
    for (int i = 0; i < natoms; i++) {
        ex13[i] = bond_adj[i];  // 1-2
        for (int j : bond_adj[i])
            for (int k : bond_adj[j])
                if (k != i) ex13[i].push_back(k);  // 1-3
        std::sort(ex13[i].begin(), ex13[i].end());
        ex13[i].erase(std::unique(ex13[i].begin(), ex13[i].end()), ex13[i].end());
    }

    std::set<std::pair<int,int>> p14set;
    auto add14 = [&](int i, int l) {
        if (i == l) return;
        if (i > l) std::swap(i, l);
        if (std::binary_search(ex13[i].begin(), ex13[i].end(), l)) return;
        p14set.insert({i, l});
    };
    if (from_dihedrals) {
        for (const auto& d : in_topo.dihedrals) add14(d.x, d.w);
    } else {
        for (int i = 0; i < natoms; i++)
            for (int j : bond_adj[i])
                for (int k : bond_adj[j]) {
                    if (k == i) continue;
                    for (int l : bond_adj[k])
                        if (l != j && l > i) add14(i, l);
                }
    }
    out_topo.pairs14.clear();
    for (auto& pr : p14set)
        out_topo.pairs14.push_back(make_int2(pr.first, pr.second));

    // full exclusion CSR = 1-2,1-3,1-4
    std::vector<std::vector<int>> adj14(natoms);
    for (auto& pr : p14set) {
        adj14[pr.first].push_back(pr.second);
        adj14[pr.second].push_back(pr.first);
    }
    out_topo.exclusion_offsets.assign(natoms + 1, 0);
    out_topo.exclusion_list.clear();
    for (int i = 0; i < natoms; i++) {
        std::vector<int> all = ex13[i];
        all.insert(all.end(), adj14[i].begin(), adj14[i].end());
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        out_topo.exclusion_offsets[i] = static_cast<int>(out_topo.exclusion_list.size());
        out_topo.exclusion_list.insert(out_topo.exclusion_list.end(),
                                       all.begin(), all.end());
    }
    out_topo.exclusion_offsets[natoms] = static_cast<int>(out_topo.exclusion_list.size());

    // 1-2/1-3-only CSR (Ewald real-space correction)
    out_topo.excl13_offsets.assign(natoms + 1, 0);
    out_topo.excl13_list.clear();
    for (int i = 0; i < natoms; i++) {
        out_topo.excl13_offsets[i] = static_cast<int>(out_topo.excl13_list.size());
        out_topo.excl13_list.insert(out_topo.excl13_list.end(),
                                    ex13[i].begin(), ex13[i].end());
    }
    out_topo.excl13_offsets[natoms] = static_cast<int>(out_topo.excl13_list.size());
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

void write_lammps_data(const std::string& path, const RestartFrame& f,
                       const TopologyData& topo,
                       int ntypes, int nbond_types, int nangle_types, int ndihedral_types,
                       const Box& box, int64_t step) {
    const int natoms = (int)f.pos.size();
    const int nbonds = (int)topo.bonds.size();
    const int nangles = (int)topo.angles.size();
    const int ndihedrals = (int)topo.dihedrals.size();
    auto id_of = [&](int tag) {
        return tag < (int)topo.atom_ids.size() ? topo.atom_ids[tag] : tag + 1;
    };

    // Write to a temporary file and rename, so an interrupted run never
    // leaves a truncated restart behind.
    const std::string tmp = path + ".tmp";
    FILE* fp = fopen(tmp.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot open restart file " + tmp);

    // %.9g round-trips IEEE single precision exactly.
    fprintf(fp, "LAMMPS data file written by fastmd (step %lld)\n\n", (long long)step);
    fprintf(fp, "%d atoms\n%d atom types\n", natoms, ntypes);
    if (nbonds > 0) fprintf(fp, "%d bonds\n%d bond types\n", nbonds, nbond_types);
    if (nangles > 0) fprintf(fp, "%d angles\n%d angle types\n", nangles, nangle_types);
    if (ndihedrals > 0) fprintf(fp, "%d dihedrals\n%d dihedral types\n", ndihedrals, ndihedral_types);
    const float lo[3] = {topo.box_lo.x, topo.box_lo.y, topo.box_lo.z};
    fprintf(fp, "\n%.9g %.9g xlo xhi\n%.9g %.9g ylo yhi\n%.9g %.9g zlo zhi\n",
            (double)lo[0], (double)(lo[0] + box.L.x), (double)lo[1], (double)(lo[1] + box.L.y),
            (double)lo[2], (double)(lo[2] + box.L.z));
    if (box.tilted())
        fprintf(fp, "%.9g %.9g %.9g xy xz yz\n", (double)box.xy, (double)box.xz, (double)box.yz);

    fprintf(fp, "\nMasses\n\n");
    for (int t = 0; t < ntypes; t++) {
        float m = t < (int)topo.masses.size() ? topo.masses[t] : 1.0f;
        fprintf(fp, "%d %.9g\n", t + 1, (double)m);
    }

    fprintf(fp, "\nAtoms # full\n\n");
    for (int t = 0; t < natoms; t++) {
        int mol = t < (int)f.mol_id.size() ? f.mol_id[t] : 1;
        float q = t < (int)f.charge.size() ? f.charge[t] : 0.0f;
        int ix = 0, iy = 0, iz = 0;
        if (3 * t + 2 < (int)f.image.size()) {
            ix = f.image[3 * t]; iy = f.image[3 * t + 1]; iz = f.image[3 * t + 2];
        }
        fprintf(fp, "%d %d %d %.9g %.9g %.9g %.9g %d %d %d\n",
                id_of(t), mol, unpack_type_id(f.pos[t].w) + 1, (double)q,
                (double)(f.pos[t].x + lo[0]), (double)(f.pos[t].y + lo[1]),
                (double)(f.pos[t].z + lo[2]), ix, iy, iz);
    }

    fprintf(fp, "\nVelocities\n\n");
    for (int t = 0; t < natoms; t++)
        fprintf(fp, "%d %.9g %.9g %.9g\n", id_of(t),
                (double)f.vel[t].x, (double)f.vel[t].y, (double)f.vel[t].z);

    if (nbonds > 0) {
        fprintf(fp, "\nBonds\n\n");
        for (int b = 0; b < nbonds; b++)
            fprintf(fp, "%d %d %d %d\n", b + 1, topo.bond_types[b] + 1,
                    id_of(topo.bonds[b].x), id_of(topo.bonds[b].y));
    }
    if (nangles > 0) {
        fprintf(fp, "\nAngles\n\n");
        for (int a = 0; a < nangles; a++)
            fprintf(fp, "%d %d %d %d %d\n", a + 1, topo.angles[a].w + 1,
                    id_of(topo.angles[a].x), id_of(topo.angles[a].y), id_of(topo.angles[a].z));
    }
    if (ndihedrals > 0) {
        fprintf(fp, "\nDihedrals\n\n");
        for (int d = 0; d < ndihedrals; d++)
            fprintf(fp, "%d %d %d %d %d %d\n", d + 1, topo.dihedral_types[d] + 1,
                    id_of(topo.dihedrals[d].x), id_of(topo.dihedrals[d].y),
                    id_of(topo.dihedrals[d].z), id_of(topo.dihedrals[d].w));
    }

    if (fclose(fp) != 0) throw std::runtime_error("error writing restart file " + tmp);
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot rename " + tmp + " to " + path);
}
