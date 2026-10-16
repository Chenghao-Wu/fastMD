#include "config.hpp"
#include "table_parser.hpp"
#include "lammps_data.hpp"
#include "../core/system.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

// Tokens of one config line with typed, validated accessors. Every error
// names the file, line and key.
class Args {
public:
    Args(std::string file, int line, std::vector<std::string> tok)
        : file_(std::move(file)), line_(line), tok_(std::move(tok)) {}

    const std::string& key() const { return tok_[0]; }
    bool more() const { return pos_ < tok_.size(); }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error(file_ + ":" + std::to_string(line_) + ": '" + key() + "': " + msg);
    }
    const std::string& peek() const { return tok_[pos_]; }
    std::string str(const char* what) {
        if (!more()) fail(std::string("missing argument <") + what + ">");
        return tok_[pos_++];
    }
    long integer(const char* what) {
        std::string s = str(what);
        size_t n = 0;
        long v = 0;
        try { v = std::stol(s, &n); } catch (...) { n = 0; }
        if (n != s.size()) fail(std::string("<") + what + "> must be an integer, got '" + s + "'");
        return v;
    }
    double real(const char* what) {
        std::string s = str(what);
        size_t n = 0;
        double v = 0;
        try { v = std::stod(s, &n); } catch (...) { n = 0; }
        if (n != s.size() || !std::isfinite(v))
            fail(std::string("<") + what + "> must be a finite number, got '" + s + "'");
        return v;
    }
    double positive(const char* what) {
        double v = real(what);
        if (!(v > 0)) fail(std::string("<") + what + "> must be > 0");
        return v;
    }
    double nonneg(const char* what) {
        double v = real(what);
        if (v < 0) fail(std::string("<") + what + "> must be >= 0");
        return v;
    }
    int flag(const char* what) {
        long v = integer(what);
        if (v != 0 && v != 1) fail(std::string("<") + what + "> must be 0 or 1");
        return (int)v;
    }
    void end() const {
        if (more()) fail("unexpected extra argument '" + tok_[pos_] + "'");
    }
    int line() const { return line_; }
    const std::string& file() const { return file_; }

private:
    std::string file_;
    int line_;
    std::vector<std::string> tok_;
    size_t pos_ = 1;
};

void copy_path(char* dst, const std::string& src, Args& a) {
    if (src.size() >= 256) a.fail("path longer than 255 characters");
    std::strncpy(dst, src.c_str(), 255);
    dst[255] = '\0';
}

}  // namespace

SimParams parse_config(const std::string& filename, TopologyData& topo) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open config: " + filename);
    return parse_config(in, filename, topo);
}

SimParams parse_config(std::istream& in, const std::string& filename, TopologyData& topo) {
    SimParams params = {};
    params.restart_freq = -1;
    std::string coords_file;

    std::set<std::string> seen;          // keys that may appear only once
    std::string ensemble_key;

    struct LJ { int line, ti, tj; float eps, sig; };
    struct LJType { int line, t; float eps, sig; };
    struct Table { int line, ti, tj; std::string file, keyword; TableForceConvention conv; };
    struct BondedTable { int line, t; std::string file, keyword; TableForceConvention conv; };
    struct Fene { int line, t; FENEParams p; };
    struct Angle { int line, t; float k, theta0_deg; };
    struct HBond { int line, t; float k, r0; };
    struct Dih { int line, t; OPLSDihedralParams p; };
    std::vector<LJ> lj_entries;
    std::vector<LJType> lj_type_entries;
    std::vector<Table> table_entries;
    std::vector<BondedTable> bond_table_entries, angle_table_entries;
    std::vector<Fene> fene_entries;
    std::vector<Angle> angle_entries, cosine_entries;
    std::vector<HBond> hbond_entries;
    std::vector<Dih> dih_entries;

    std::string raw;
    int lineno = 0;
    while (std::getline(in, raw)) {
        ++lineno;
        std::string text = raw.substr(0, raw.find('#'));
        std::istringstream iss(text);
        std::vector<std::string> tok;
        for (std::string t; iss >> t;) tok.push_back(t);
        if (tok.empty()) continue;
        Args a(filename, lineno, tok);
        const std::string& key = a.key();

        static const std::set<std::string> repeatable = {
            "lj", "lj_type", "table", "bond_type", "angle_type", "angle_cosine", "bond_harmonic",
            "dihedral_opls", "bond_table", "angle_table"};
        if (!repeatable.count(key) && !seen.insert(key).second)
            a.fail("specified more than once");

        if (key == "natoms") {
            long n = a.integer("N");
            if (n < 1) a.fail("<N> must be >= 1");
            params.natoms = (int)n;
        } else if (key == "ntypes") {
            long n = a.integer("N");
            if (n < 1 || n > MAX_TYPES)
                a.fail("<N> must be in [1, " + std::to_string(MAX_TYPES) + "]");
            params.ntypes = (int)n;
        } else if (key == "rc") {
            params.rc = (float)a.positive("rc");
        } else if (key == "skin") {
            params.skin = (float)a.nonneg("skin");
        } else if (key == "dt") {
            params.dt = (float)a.positive("dt");
        } else if (key == "nsteps") {
            long n = a.integer("N");
            if (n < 0) a.fail("<N> must be >= 0");
            params.nsteps = (int)n;
        } else if (key == "dump_freq") {
            long n = a.integer("N");
            if (n < 0) a.fail("<N> must be >= 0 (0 disables)");
            params.dump_freq = (int)n;
        } else if (key == "thermo" || key == "stress" || key == "rg") {
            int on = a.flag("on");
            long freq = a.integer("freq");
            std::string f = a.more() ? a.str("file") : "";
            if (on) {
                if (freq < 1) a.fail("<freq> must be >= 1 when enabled");
                if (f.empty()) a.fail("missing argument <file>");
            }
            if (key == "thermo") { params.thermo_on = on; params.thermo_freq = (int)freq; copy_path(params.thermo_file, f, a); }
            if (key == "stress") {
                params.stress_on = on; params.stress_freq = (int)freq; copy_path(params.stress_file, f, a);
                if (a.more()) {   // stress <on> <freq> <file> [p m]
                    long p = a.integer("p"), m = a.integer("m");
                    if (m < 2 || p < 2 || p > 1024 || p % m != 0)
                        a.fail("correlator needs m >= 2, 2 <= p <= 1024 and p a multiple of m");
                    params.stress_p = (int)p;
                    params.stress_m = (int)m;
                }
            }
            if (key == "rg")     { params.rg_on = on;     params.rg_freq = (int)freq;     copy_path(params.rg_file, f, a); }
        } else if (key == "restart") {
            long freq = a.integer("freq");
            if (freq < 0) a.fail("<freq> must be >= 0 (0 = final restart only)");
            params.restart_freq = (int)freq;
            copy_path(params.restart_file, a.str("file"), a);
        } else if (key == "nve") {
            if (!ensemble_key.empty())
                a.fail("only one ensemble may be given (already '" + ensemble_key + "')");
            ensemble_key = key;
            params.ensemble = Ensemble::NVE;
        } else if (key == "nvt_langevin" || key == "nvt_nh" || key == "npt_nh") {
            if (!ensemble_key.empty())
                a.fail("only one ensemble may be given (already '" + ensemble_key + "')");
            ensemble_key = key;
            params.T_start = (float)a.nonneg("T_start");
            params.T_stop = (float)a.nonneg("T_stop");
            params.Tdamp = (float)a.positive("Tdamp");
            if (key == "nvt_langevin") {
                params.ensemble = Ensemble::Langevin;
                long seed = a.more() ? a.integer("seed") : 42;
                if (seed < 0) a.fail("<seed> must be >= 0");
                params.seed = (uint64_t)seed;
            } else {
                params.ensemble = key == "nvt_nh" ? Ensemble::NVT_NH : Ensemble::NPT_NH;
                if (key == "npt_nh") {
                    params.P_start = (float)a.real("P_start");
                    params.P_stop = (float)a.real("P_stop");
                    params.Pdamp = (float)a.positive("Pdamp");
                }
                long chain = a.more() ? a.integer("chain_length") : 3;
                if (chain < 1 || chain > kMaxChainLength)
                    a.fail("<chain_length> must be in [1, " + std::to_string(kMaxChainLength) + "]");
                params.nh_chain_length = (int)chain;
                if (params.T_start <= 0.0f || params.T_stop <= 0.0f)
                    a.fail("Nose-Hoover temperatures must be > 0");
            }
        } else if (key == "coords_file") {
            coords_file = a.str("path");
        } else if (key == "lammps_data_file") {
            topo.data_file = a.str("path");
        } else if (key == "lj") {
            LJ e{a.line(), (int)a.integer("ti"), (int)a.integer("tj"), 0, 0};
            e.eps = (float)a.nonneg("eps");
            e.sig = (float)a.nonneg("sig");
            if (e.eps > 0 && e.sig <= 0) a.fail("<sig> must be > 0 when eps > 0");
            lj_entries.push_back(e);
        } else if (key == "lj_type") {
            LJType e{a.line(), (int)a.integer("t"), 0, 0};
            e.eps = (float)a.nonneg("eps");
            e.sig = (float)a.nonneg("sig");   // OPLS hydroxyl H: eps = sig = 0
            if (e.eps > 0 && e.sig <= 0) a.fail("<sig> must be > 0 when eps > 0");
            lj_type_entries.push_back(e);
        } else if (key == "lj_force_cap") {
            params.lj_force_cap = (float)a.positive("F_max");
        } else if (key == "coulomb") {
            // coulomb ewald <alpha> <kmax> | coulomb ewald auto <tol>
            // coulomb pme <alpha> <grid> [order] | coulomb pme auto <tol> [order]
            std::string style = a.str("style");
            if (style != "ewald" && style != "pme")
                a.fail("unknown style '" + style + "' (supported: ewald, pme)");
            params.ewald_on = 1;
            params.kspace = style == "pme" ? 1 : 0;
            if (a.more() && a.peek() == "auto") {
                a.str("auto");
                params.ewald_tol = (float)a.positive("tolerance");
                if (params.ewald_tol >= 0.1f) a.fail("<tolerance> must be < 0.1");
            } else {
                params.ewald_alpha = (float)a.positive("alpha");
                long k = a.integer(style == "pme" ? "grid" : "kmax");
                if (style == "ewald" && (k < 1 || k > 64)) a.fail("<kmax> must be in [1, 64]");
                if (style == "pme" && (k < 8 || k > 1024)) a.fail("<grid> must be in [8, 1024]");
                if (style == "pme") params.pme_grid = (int)k; else params.ewald_kmax = (int)k;
            }
            if (style == "pme" && a.more()) {
                long order = a.integer("order");
                if (order < 3 || order > 8) a.fail("<order> must be in [3, 8]");
                params.pme_order = (int)order;
            }
        } else if (key == "coulomb_const") {
            params.coul_const = (float)a.positive("C");
        } else if (key == "special_bonds_14") {
            params.lj14 = (float)a.real("lj14");
            params.ewald14_coul = (float)a.real("coul14");
            if (params.lj14 < 0 || params.lj14 > 1 || params.ewald14_coul < 0 || params.ewald14_coul > 1)
                a.fail("scaling factors must be in [0, 1]");
        } else if (key == "pair_soft") {
            params.pair_soft = 1;
            params.soft_A_start = (float)a.nonneg("A_start");
            params.soft_A_stop = (float)a.nonneg("A_stop");
        } else if (key == "pair_shift") {
            std::string v = a.str("yes|no");
            if (v != "yes" && v != "no") a.fail("expected 'yes' or 'no'");
            params.pair_shift = (v == "yes");
        } else if (key == "special_dihedral") {
            std::string v = a.str("yes|no");
            if (v != "yes" && v != "no") a.fail("expected 'yes' or 'no'");
            params.special_dihedral = (v == "yes");
        } else if (key == "bond_harmonic") {
            HBond e{a.line(), (int)a.integer("t"), 0, 0};
            e.k = (float)a.nonneg("K");
            e.r0 = (float)a.nonneg("r0");
            hbond_entries.push_back(e);
        } else if (key == "dihedral_opls") {
            Dih e{a.line(), (int)a.integer("t"), {}};
            e.p.k1 = (float)a.real("V1");
            e.p.k2 = (float)a.real("V2");
            e.p.k3 = (float)a.real("V3");
            e.p.k4 = (float)a.real("V4");
            dih_entries.push_back(e);
        } else if (key == "table") {
            Table e{a.line(), (int)a.integer("ti"), (int)a.integer("tj"), "", "", TableForceConvention::Force};
            e.file = a.str("file");
            e.keyword = a.str("keyword");
            if (a.more()) e.conv = parse_table_convention(a.str("force|force_over_r"), false);
            table_entries.push_back(e);
        } else if (key == "bond_type") {
            Fene e{a.line(), (int)a.integer("t"), {}};
            e.p.k = (float)a.nonneg("k");
            e.p.R0 = (float)a.positive("R0");
            e.p.eps = (float)a.nonneg("eps");
            e.p.sig = (float)a.positive("sig");
            fene_entries.push_back(e);
        } else if (key == "angle_type") {
            Angle e{a.line(), (int)a.integer("t"), 0, 0};
            e.k = (float)a.nonneg("k");
            e.theta0_deg = (float)a.real("theta0");
            if (e.theta0_deg < 0 || e.theta0_deg > 180) a.fail("<theta0> must be in [0, 180] degrees");
            angle_entries.push_back(e);
        } else if (key == "angle_cosine") {
            Angle e{a.line(), (int)a.integer("t"), 0, 0};
            e.k = (float)a.real("K");
            cosine_entries.push_back(e);
        } else if (key == "bond_table" || key == "angle_table") {
            bool angle = key == "angle_table";
            BondedTable e{a.line(), (int)a.integer("t"), "", "",
                          angle ? TableForceConvention::DegreesPerDegree : TableForceConvention::Force};
            e.file = a.str("file");
            e.keyword = a.str("keyword");
            if (a.more()) e.conv = parse_table_convention(a.str("convention"), angle);
            (angle ? angle_table_entries : bond_table_entries).push_back(e);
        } else if (key == "bonded_mode") {
            std::string v = a.str("gather|scatter");
            if (v != "gather" && v != "scatter") a.fail("expected 'gather' or 'scatter'");
            params.bonded_gather = (v == "gather");
        } else if (key == "pair_tpa") {
            long n = a.integer("N");
            if (n != 0 && n != 1 && n != 2 && n != 4 && n != 8)
                a.fail("<N> must be 0 (auto), 1, 2, 4 or 8");
            params.pair_tpa = (int)n;
        } else if (key == "sort_every") {
            long n = a.integer("N");
            if (n < 0) a.fail("<N> must be >= 0 (0 = never sort)");
            params.sort_every = (int)n;
        } else if (key == "cuda_graphs") {
            params.cuda_graphs = a.flag("0|1");
        } else if (key == "check_every") {
            long n = a.integer("N");
            if (n < 1) a.fail("<N> must be >= 1");
            params.check_every = (int)n;
        } else if (key == "neigh_capacity") {
            long n = a.integer("N");
            if (n < 0) a.fail("<N> must be >= 0 (0 = automatic)");
            params.neigh_capacity = (int)n;
        } else if (key == "neigh_max_age") {
            long n = a.integer("N");
            if (n < 0) a.fail("<N> must be >= 0 (0 = rebuild only when needed)");
            params.neigh_max_age = (int)n;
        } else if (key == "traj") {
            // traj <file> linear <N> | traj <file> log <block> [base]
            copy_path(params.traj_file, a.str("file"), a);
            std::string mode = a.str("linear|log");
            if (mode == "linear") {
                long n = a.integer("N");
                if (n < 1) a.fail("<N> must be >= 1");
                params.traj_mode = 1;
                params.traj_every = (int)n;
            } else if (mode == "log") {
                long b = a.integer("block");
                if (b < 1) a.fail("<block> must be >= 1");
                params.traj_mode = 2;
                params.traj_every = (int)b;
                if (a.more()) {
                    double base = a.real("base");
                    if (!(base > 1.0)) a.fail("<base> must be > 1");
                    params.traj_base = (float)base;
                }
            } else {
                a.fail("expected 'linear' or 'log'");
            }
        } else if (key == "dump_file") {
            copy_path(params.dump_file, a.str("path"), a);
        } else if (key == "forces_dump_file") {
            copy_path(params.forces_dump_file, a.str("path"), a);
        } else if (key == "temperature" || key == "gamma" || key == "seed") {
            a.fail("deprecated; use 'nvt_langevin T_start T_stop Tdamp [seed]'");
        } else {
            a.fail("unknown keyword");
        }
        a.end();
    }

    auto require = [&](const char* key) {
        if (!seen.count(key))
            throw std::runtime_error(filename + ": required keyword '" + key + "' is missing");
    };
    require("ntypes");
    require("rc");
    require("skin");
    require("dt");
    require("nsteps");
    if (ensemble_key.empty())
        throw std::runtime_error(filename + ": no ensemble given (nve, nvt_langevin, nvt_nh or npt_nh)");
    if (topo.data_file.empty() && coords_file.empty())
        throw std::runtime_error(filename + ": give lammps_data_file or coords_file");
    if (!topo.data_file.empty() && !coords_file.empty())
        throw std::runtime_error(filename + ": lammps_data_file and coords_file are mutually exclusive");
    if (coords_file.empty() == false && !seen.count("natoms"))
        throw std::runtime_error(filename + ": coords_file requires natoms");

    const int nt = params.ntypes;
    auto bad = [&](int line, const std::string& msg) -> std::runtime_error {
        return std::runtime_error(filename + ":" + std::to_string(line) + ": " + msg);
    };
    auto check_atom_type = [&](int line, int t, const char* what) {
        if (t < 0 || t >= nt)
            throw bad(line, std::string(what) + " type index " + std::to_string(t) +
                      " out of range [0, " + std::to_string(nt - 1) + "] (ntypes " + std::to_string(nt) + ")");
    };
    auto check_bonded_type = [&](int line, int t, const char* what) {
        if (t < 0 || t > 100000)
            throw bad(line, std::string(what) + " type index " + std::to_string(t) + " out of range");
    };

    // --- nonbonded parameters ---------------------------------------------
    topo.lj_params.assign((size_t)nt * nt, make_float2(0, 0));
    std::vector<char> lj_set((size_t)nt * nt, 0);
    if (!lj_type_entries.empty() && !lj_entries.empty())
        throw std::runtime_error(filename + ": lj_type and lj cannot both be used");
    if (!lj_type_entries.empty()) {
        std::vector<float2> lp(nt, make_float2(0, 0));
        std::vector<char> set(nt, 0);
        for (auto& e : lj_type_entries) {
            check_atom_type(e.line, e.t, "lj_type");
            if (set[e.t]) throw bad(e.line, "lj_type for type " + std::to_string(e.t) + " given twice");
            set[e.t] = 1;
            lp[e.t] = make_float2(e.eps, e.sig);
        }
        for (int t = 0; t < nt; t++)
            if (!set[t]) throw std::runtime_error(filename + ": lj_type missing for type " + std::to_string(t));
        // geometric mixing (OPLS)
        for (int ti = 0; ti < nt; ti++)
            for (int tj = 0; tj < nt; tj++) {
                topo.lj_params[ti * nt + tj] = make_float2(sqrtf(lp[ti].x * lp[tj].x),
                                                           sqrtf(lp[ti].y * lp[tj].y));
                lj_set[ti * nt + tj] = 1;
            }
    }
    for (auto& e : lj_entries) {
        check_atom_type(e.line, e.ti, "lj");
        check_atom_type(e.line, e.tj, "lj");
        if (lj_set[e.ti * nt + e.tj])
            throw bad(e.line, "lj parameters for pair (" + std::to_string(e.ti) + "," +
                      std::to_string(e.tj) + ") given twice");
        topo.lj_params[e.ti * nt + e.tj] = make_float2(e.eps, e.sig);
        topo.lj_params[e.tj * nt + e.ti] = make_float2(e.eps, e.sig);
        lj_set[e.ti * nt + e.tj] = lj_set[e.tj * nt + e.ti] = 1;
    }

    topo.table_idx.assign((size_t)nt * nt, -1);
    topo.table_params.clear();
    topo.table_data.clear();
    std::map<std::tuple<std::string, std::string, int>, int> table_cache;
    for (auto& e : table_entries) {
        check_atom_type(e.line, e.ti, "table");
        check_atom_type(e.line, e.tj, "table");
        if (lj_set[e.ti * nt + e.tj])
            throw bad(e.line, "both lj and table defined for pair (" + std::to_string(e.ti) + "," +
                      std::to_string(e.tj) + ")");
        if (topo.table_idx[e.ti * nt + e.tj] >= 0)
            throw bad(e.line, "table for pair (" + std::to_string(e.ti) + "," + std::to_string(e.tj) +
                      ") given twice");
        auto key = std::make_tuple(e.file, e.keyword, (int)e.conv);
        int idx;
        auto it = table_cache.find(key);
        if (it == table_cache.end()) {
            TableFileData tfd = parse_table_file(e.file, e.keyword, TableKind::Pair, e.conv);
            idx = (int)topo.table_params.size();
            table_cache[key] = idx;
            TableParams tp = tfd.params[0];
            tp.data_offset = (int)topo.table_data.size();
            if (tp.rmax < params.rc)
                throw bad(e.line, "table " + e.keyword + " ends at r=" + std::to_string(tp.rmax) +
                          " < rc=" + std::to_string(params.rc));
            topo.table_params.push_back(tp);
            topo.table_data.insert(topo.table_data.end(), tfd.data.begin(), tfd.data.end());
        } else {
            idx = it->second;
        }
        topo.table_idx[e.ti * nt + e.tj] = topo.table_idx[e.tj * nt + e.ti] = idx;
    }
    if (params.pair_soft) {
        if (!lj_entries.empty() || !lj_type_entries.empty() || !table_entries.empty() || params.ewald_on)
            throw std::runtime_error(filename + ": pair_soft replaces the pair interaction: "
                                     "do not combine it with lj, lj_type, table or coulomb");
    } else for (int ti = 0; ti < nt; ti++)
        for (int tj = 0; tj < nt; tj++)
            if (!lj_set[ti * nt + tj] && topo.table_idx[ti * nt + tj] < 0)
                throw std::runtime_error(filename + ": no lj or table interaction for type pair (" +
                                         std::to_string(ti) + "," + std::to_string(tj) + ")");
    if (params.lj_force_cap > 0.0f && params.ewald_on)
        throw std::runtime_error(filename + ": lj_force_cap applies to the plain LJ kernel and "
                                 "cannot be combined with coulomb ewald");

    // --- bonded parameters (types are 0-based; data-file type minus 1) ------
    auto place = [&](auto& vec, int t, auto value, int line, const char* what, auto zero) {
        check_bonded_type(line, t, what);
        if ((int)vec.size() <= t) vec.resize(t + 1, zero);
        vec[t] = value;
    };
    if (!angle_entries.empty() + !cosine_entries.empty() + !angle_table_entries.empty() > 1)
        throw std::runtime_error(filename + ": use only one angle style (angle_type, angle_cosine "
                                 "or angle_table)");
    std::vector<char> fene_set, hbond_set, angle_set, dih_set;
    auto mark = [&](std::vector<char>& set, int t, int line, const char* what) {
        if ((int)set.size() <= t) set.resize(t + 1, 0);
        if (set[t]) throw bad(line, std::string(what) + " type " + std::to_string(t) + " given twice");
        set[t] = 1;
    };
    for (auto& e : fene_entries) {
        place(topo.bond_params, e.t, e.p, e.line, "bond_type", FENEParams{0, 0, 0, 0});
        mark(fene_set, e.t, e.line, "bond_type");
    }
    for (auto& e : hbond_entries) {
        place(topo.hbond_params, e.t, BondHarmonicParams{e.k, e.r0}, e.line, "bond_harmonic",
              BondHarmonicParams{0, 0});
        mark(hbond_set, e.t, e.line, "bond_harmonic");
    }
    for (auto& e : angle_entries) {
        place(topo.angle_params, e.t,
              AngleParams{e.k, e.theta0_deg * 3.14159265358979323846f / 180.0f},
              e.line, "angle_type", AngleParams{0, 0});
        mark(angle_set, e.t, e.line, "angle_type");
    }
    for (auto& e : cosine_entries) {
        place(topo.angle_params, e.t, AngleParams{e.k, 0.0f}, e.line, "angle_cosine", AngleParams{0, 0});
        mark(angle_set, e.t, e.line, "angle_cosine");
    }
    topo.angle_cosine = !cosine_entries.empty();
    for (auto& e : dih_entries) {
        place(topo.dihedral_params, e.t, e.p, e.line, "dihedral_opls", OPLSDihedralParams{0, 0, 0, 0});
        mark(dih_set, e.t, e.line, "dihedral_opls");
    }
    topo.bond_params_set = fene_set;
    topo.hbond_params_set = hbond_set;
    topo.angle_params_set = angle_set;
    topo.dihedral_params_set = dih_set;

    int nbond_styles = !fene_entries.empty() + !hbond_entries.empty() + !bond_table_entries.empty();
    if (nbond_styles > 1)
        throw std::runtime_error(filename + ": use only one bond style (bond_type FENE, "
                                 "bond_harmonic or bond_table)");

    auto load_bonded_tables = [&](std::vector<BondedTable>& entries, TableKind kind,
                                  std::vector<int>& idx_vec, std::vector<TableParams>& params_vec,
                                  std::vector<float4>& data_vec, const char* what) {
        for (auto& e : entries) {
            check_bonded_type(e.line, e.t, what);
            if ((int)idx_vec.size() > e.t && idx_vec[e.t] >= 0)
                throw bad(e.line, std::string(what) + " type " + std::to_string(e.t) + " given twice");
            TableFileData tfd = parse_table_file(e.file, e.keyword, kind, e.conv);
            TableParams tp = tfd.params[0];
            tp.data_offset = (int)data_vec.size();
            int idx = (int)params_vec.size();
            params_vec.push_back(tp);
            data_vec.insert(data_vec.end(), tfd.data.begin(), tfd.data.end());
            if ((int)idx_vec.size() <= e.t) idx_vec.resize(e.t + 1, -1);
            idx_vec[e.t] = idx;
        }
    };
    load_bonded_tables(bond_table_entries, TableKind::Bond, topo.bond_table_idx,
                       topo.bond_table_params, topo.bond_table_data, "bond_table");
    load_bonded_tables(angle_table_entries, TableKind::Angle, topo.angle_table_idx,
                       topo.angle_table_params, topo.angle_table_data, "angle_table");

    // --- coordinates without a data file (xyz, single type) -----------------
    if (!coords_file.empty()) {
        std::ifstream cf(coords_file);
        if (!cf.is_open()) throw std::runtime_error("Cannot open coords_file: " + coords_file);
        int n = 0;
        if (!(cf >> n) || n != params.natoms)
            throw std::runtime_error(coords_file + ": atom count does not match natoms");
        std::string comment;
        std::getline(cf, comment);
        std::getline(cf, comment);
        topo.positions.resize(n);
        for (int i = 0; i < n; i++) {
            std::string elem;
            float x, y, z;
            if (!(cf >> elem >> x >> y >> z))
                throw std::runtime_error(coords_file + ": truncated at atom " + std::to_string(i + 1));
            topo.positions[i] = make_float4(x, y, z, pack_type_id(0));
            topo.atom_ids.push_back(i + 1);
        }
    }

    if (!dih_entries.empty() && !params.ewald_on)
        fprintf(stderr, "Note: dihedral_opls without coulomb ewald — 1-4 LJ pairs are scaled "
                        "by lj14; Coulomb is not computed\n");
    return params;
}

// Checks that need both the config and the system (data file) contents.
void validate_system(const SimParams& params, const TopologyData& topo) {
    const int n = (int)topo.positions.size();
    if (n == 0) throw std::runtime_error("system has no atoms");
    if (params.natoms != n)
        throw std::runtime_error("natoms " + std::to_string(params.natoms) + " does not match the " +
                                 std::to_string(n) + " atoms in the data file");
    for (const auto& p : topo.positions) {
        int t = unpack_type_id(p.w);
        if (t < 0 || t >= params.ntypes)
            throw std::runtime_error("atom type " + std::to_string(t + 1) + " in the data file exceeds ntypes " +
                                     std::to_string(params.ntypes));
    }
    if (topo.n_atom_types > params.ntypes)
        throw std::runtime_error("data file declares " + std::to_string(topo.n_atom_types) +
                                 " atom types but ntypes is " + std::to_string(params.ntypes));
    const float Lmin = std::min({topo.box.x, topo.box.y, topo.box.z});
    if (!(Lmin > 0.0f)) throw std::runtime_error("box lengths must be > 0");
    if (params.rc > 0.5f * Lmin)
        throw std::runtime_error("rc " + std::to_string(params.rc) + " exceeds half the shortest box edge " +
                                 std::to_string(0.5f * Lmin) + " (minimum image)");
    const bool cubic = topo.box.x == topo.box.y && topo.box.y == topo.box.z;
    if (params.ewald_on && !cubic)
        throw std::runtime_error("coulomb ewald/pme requires a cubic box (box is " +
                                 std::to_string(topo.box.x) + " x " + std::to_string(topo.box.y) +
                                 " x " + std::to_string(topo.box.z) + ")");
    if (params.dump_freq > 0 && !cubic)
        throw std::runtime_error("dump_freq (legacy trajectory format) requires a cubic box; use traj");

    auto covered = [](const std::vector<char>& set, int t) { return t < (int)set.size() && set[t]; };
    if (!topo.bonds.empty()) {
        bool fene = !topo.bond_params.empty(), harm = !topo.hbond_params.empty(),
             table = !topo.bond_table_idx.empty();
        if (!fene && !harm && !table)
            throw std::runtime_error("the data file has bonds but no bond parameters are given");
        for (int bt : topo.bond_types) {
            bool ok = fene ? covered(topo.bond_params_set, bt)
                    : harm ? covered(topo.hbond_params_set, bt)
                           : (bt < (int)topo.bond_table_idx.size() && topo.bond_table_idx[bt] >= 0);
            if (!ok) throw std::runtime_error("no parameters for bond type " + std::to_string(bt + 1) +
                                              " (config index " + std::to_string(bt) + ")");
        }
    }
    if (!topo.angles.empty()) {
        bool harm = !topo.angle_params.empty(), table = !topo.angle_table_idx.empty();
        if (!harm && !table)
            throw std::runtime_error("the data file has angles but no angle parameters are given");
        for (const auto& a : topo.angles) {
            bool ok = harm ? covered(topo.angle_params_set, a.w)
                           : (a.w < (int)topo.angle_table_idx.size() && topo.angle_table_idx[a.w] >= 0);
            if (!ok) throw std::runtime_error("no parameters for angle type " + std::to_string(a.w + 1) +
                                              " (config index " + std::to_string(a.w) + ")");
        }
    }
    if (!topo.dihedrals.empty()) {
        if (topo.dihedral_params.empty())
            throw std::runtime_error("the data file has dihedrals but no dihedral_opls parameters");
        for (int dt : topo.dihedral_types)
            if (!covered(topo.dihedral_params_set, dt))
                throw std::runtime_error("no parameters for dihedral type " + std::to_string(dt + 1) +
                                         " (config index " + std::to_string(dt) + ")");
    }
    if (params.ewald_on && topo.charges.empty())
        throw std::runtime_error("coulomb ewald requires per-atom charges (Atoms # full or charge)");
    if (params.rg_on && topo.mol_ids.empty())
        throw std::runtime_error("rg requires molecule ids in the data file");
}

void finalize_params(SimParams& params) {
    params.rc2 = params.rc * params.rc;
    params.ntiles = div_ceil(params.natoms, TILE_SIZE);
}

void load_config(std::istream& in, const std::string& source, SimParams& params,
                 TopologyData& topo) {
    params = parse_config(in, source, topo);
    if (!topo.data_file.empty()) parse_lammps_data(topo.data_file, topo);
    if (params.natoms == 0) params.natoms = (int)topo.positions.size();
    params.box_L3 = topo.box;
    validate_system(params, topo);
    finalize_params(params);
}

void load_config(const std::string& path, SimParams& params, TopologyData& topo) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("Cannot open config: " + path);
    load_config(in, path, params, topo);
}
