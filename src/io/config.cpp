#include "config.hpp"
#include "table_parser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <tuple>
#include <map>

SimParams parse_config(const std::string& filename, TopologyData& topo) {
    SimParams params = {};
    params.restart_freq = -1;
    std::string coords_file;
    std::vector<std::tuple<int,int,float,float>> lj_entries;
    std::vector<std::tuple<int,int,std::string,std::string>> table_entries;

    struct BondType { float k, R0, eps, sig; };
    struct AngleType { float k_theta, theta0; };
    std::vector<BondType> bond_types_params;
    std::vector<AngleType> angle_types_params;

    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open config: " + filename);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "natoms")      iss >> params.natoms;
        else if (key == "ntypes") iss >> params.ntypes;
        else if (key == "rc")     iss >> params.rc;
        else if (key == "skin")   iss >> params.skin;
        else if (key == "dt")     iss >> params.dt;
        else if (key == "temperature") {
            throw std::runtime_error(
                "'temperature' is deprecated. Use 'nvt_langevin T_start T_stop Tdamp [seed]' instead.");
        }
        else if (key == "gamma") {
            throw std::runtime_error(
                "'gamma' is deprecated. Use 'nvt_langevin T_start T_stop Tdamp [seed]' instead.");
        }
        else if (key == "nsteps") iss >> params.nsteps;
        else if (key == "dump_freq")   iss >> params.dump_freq;
        else if (key == "thermo") {
            iss >> params.thermo_on >> params.thermo_freq;
            std::string f; iss >> f;
            strncpy(params.thermo_file, f.c_str(), 255);
            params.thermo_file[255] = '\0';
        }
        else if (key == "stress") {
            iss >> params.stress_on >> params.stress_freq;
            std::string f; iss >> f;
            strncpy(params.stress_file, f.c_str(), 255);
            params.stress_file[255] = '\0';
        }
        else if (key == "rg") {
            iss >> params.rg_on >> params.rg_freq;
            std::string f; iss >> f;
            strncpy(params.rg_file, f.c_str(), 255);
            params.rg_file[255] = '\0';
        }
        else if (key == "restart") {
            iss >> params.restart_freq;
            std::string f; iss >> f;
            strncpy(params.restart_file, f.c_str(), 255);
            params.restart_file[255] = '\0';
        }
        else if (key == "seed") {
            throw std::runtime_error(
                "'seed' as a standalone key is deprecated. "
                "Specify seed in 'nvt_langevin T_start T_stop Tdamp [seed]'.");
        }
        else if (key == "nvt_langevin") {
            params.ensemble = Ensemble::Langevin;
            iss >> params.T_start >> params.T_stop >> params.Tdamp;
            if (!(iss >> params.seed)) {
                params.seed = 42;
            }
        }
        else if (key == "nvt_nh") {
            params.ensemble = Ensemble::NVT_NH;
            iss >> params.T_start >> params.T_stop >> params.Tdamp;
            if (!(iss >> params.nh_chain_length)) {
                params.nh_chain_length = 3;
            }
        }
        else if (key == "npt_nh") {
            params.ensemble = Ensemble::NPT_NH;
            iss >> params.T_start >> params.T_stop >> params.Tdamp
                >> params.P_start >> params.P_stop >> params.Pdamp;
            if (!(iss >> params.nh_chain_length)) {
                params.nh_chain_length = 3;
            }
        }
        else if (key == "coords_file") iss >> coords_file;
        else if (key == "lammps_data_file") iss >> topo.data_file;
        else if (key == "lj") {
            int ti, tj; float eps, sig;
            iss >> ti >> tj >> eps >> sig;
            lj_entries.push_back({ti, tj, eps, sig});
        }
        else if (key == "table") {
            int ti, tj; std::string filename, keyword;
            iss >> ti >> tj >> filename >> keyword;
            table_entries.push_back({ti, tj, filename, keyword});
        }
        else if (key == "bond_type") {
            int t; float k, R0, eps, sig;
            iss >> t >> k >> R0 >> eps >> sig;
            if ((int)bond_types_params.size() <= t) bond_types_params.resize(t+1);
            bond_types_params[t] = {k, R0, eps, sig};
        }
        else if (key == "angle_type") {
            int t; float k_theta, theta0;
            iss >> t >> k_theta >> theta0;
            if ((int)angle_types_params.size() <= t) angle_types_params.resize(t+1);
            angle_types_params[t] = {k_theta, theta0};
        }
        else if (key == "forces_dump_file") {
            std::string val;
            iss >> val;
            strncpy(params.forces_dump_file, val.c_str(), 255);
            params.forces_dump_file[255] = '\0';
        }
    }

    topo.lj_params.resize(params.ntypes * params.ntypes, make_float2(0,0));
    for (auto& [ti, tj, eps, sig] : lj_entries) {
        topo.lj_params[ti * params.ntypes + tj] = make_float2(eps, sig);
        topo.lj_params[tj * params.ntypes + ti] = make_float2(eps, sig);
    }

    topo.table_idx.resize(params.ntypes * params.ntypes, -1);

    std::map<std::pair<std::string,std::string>, int> file_keyword_to_idx;
    for (auto& [ti, tj, filename, keyword] : table_entries) {
        if (ti < 0 || ti >= params.ntypes || tj < 0 || tj >= params.ntypes)
            throw std::runtime_error("table type index out of range");

        float2 lj_p = topo.lj_params[ti * params.ntypes + tj];
        if (lj_p.x != 0.0f || lj_p.y != 0.0f)
            throw std::runtime_error("Both lj and table defined for pair (" +
                                      std::to_string(ti) + "," + std::to_string(tj) + ")");

        auto fk = std::make_pair(filename, keyword);
        int idx;
        auto it = file_keyword_to_idx.find(fk);
        if (it == file_keyword_to_idx.end()) {
            TableFileData tfd = parse_table_file(filename, keyword);
            idx = (int)topo.table_params.size();
            file_keyword_to_idx[fk] = idx;
            topo.table_params.push_back(tfd.params[0]);
            topo.table_data.insert(topo.table_data.end(),
                                    tfd.data.begin(), tfd.data.end());
        } else {
            idx = it->second;
        }
        topo.table_idx[ti * params.ntypes + tj] = idx;
        topo.table_idx[tj * params.ntypes + ti] = idx;
    }

    topo.bond_params.resize(bond_types_params.size());
    for (size_t i = 0; i < bond_types_params.size(); ++i) {
        topo.bond_params[i] = {bond_types_params[i].k,
                               bond_types_params[i].R0,
                               bond_types_params[i].eps,
                               bond_types_params[i].sig};
    }

    topo.angle_params.resize(angle_types_params.size());
    for (size_t i = 0; i < angle_types_params.size(); ++i) {
        topo.angle_params[i] = {angle_types_params[i].k_theta,
                                angle_types_params[i].theta0 * 3.14159265358979323846f / 180.0f};
    }

    if (!coords_file.empty()) {
        std::ifstream cf(coords_file);
        int n; cf >> n;
        std::string comment; std::getline(cf, comment); std::getline(cf, comment);
        topo.positions.resize(n);
        topo.velocities.resize(n, make_float4(0,0,0,0));
        for (int i = 0; i < n; i++) {
            std::string elem;
            float x, y, z;
            cf >> elem >> x >> y >> z;
            int type = 0;
            topo.positions[i] = make_float4(x, y, z, pack_type_id(type));
        }
    }

    // Validate ensemble parameters
    if (params.ensemble == Ensemble::Langevin) {
        if (params.Tdamp <= 0.0f) {
            throw std::runtime_error("Tdamp must be > 0 for nvt_langevin");
        }
    }
    if (params.ensemble == Ensemble::NVT_NH || params.ensemble == Ensemble::NPT_NH) {
        if (params.Tdamp <= 0.0f) {
            throw std::runtime_error("Tdamp must be > 0 for nvt_nh/npt_nh");
        }
        if (params.nh_chain_length < 1) {
            throw std::runtime_error("nh_chain_length must be >= 1");
        }
    }
    if (params.ensemble == Ensemble::NPT_NH && params.Pdamp <= 0.0f) {
        throw std::runtime_error("Pdamp must be > 0 for npt_nh");
    }

    return params;
}

void finalize_params(SimParams& params) {
    params.inv_L = 1.0f / params.box_L;
    params.half_L = params.box_L / 2.0f;
    params.rc2 = params.rc * params.rc;
    params.ntiles = div_ceil(params.natoms, TILE_SIZE);
}
