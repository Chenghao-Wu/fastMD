#include <gtest/gtest.h>
#include "io/config.hpp"
#include "io/lammps_data.hpp"
#include <fstream>

namespace {
const char* kBase =
    "ntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\n"
    "nvt_langevin 1.0 1.0 1.0\nlammps_data_file unused.data\n";

std::string write_cfg(const std::string& name, const std::string& body) {
    std::string path = "test_config_" + name + ".conf";
    std::ofstream(path) << body;
    return path;
}

std::string table_file() {
    std::ofstream("test_table.txt") << "LJ_TABLE\nN 2 R 0.5 3.0\n\n1 0.5 0 0\n2 3.0 0 0\n";
    return "test_table.txt";
}

// Expect parse_config to throw with a message containing `needle`.
void expect_reject(const std::string& name, const std::string& body, const std::string& needle) {
    TopologyData topo;
    try {
        parse_config(write_cfg(name, body), topo);
        FAIL() << name << ": accepted, expected an error containing '" << needle << "'";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
            << name << ": message was: " << e.what();
    }
}
}  // namespace

TEST(Config, ParsesTableLine) {
    std::string cfg = write_cfg("table", std::string(kBase) + "table 0 0 " + table_file() + " LJ_TABLE\n");
    TopologyData topo;
    SimParams params = parse_config(cfg, topo);
    EXPECT_EQ(params.ntypes, 1);
    EXPECT_EQ(topo.table_idx.size(), 1u);
    EXPECT_EQ(topo.table_idx[0], 0);
    EXPECT_EQ(topo.table_params.size(), 1u);
    EXPECT_EQ(topo.table_data.size(), 2u);
}

TEST(Config, RejectsLjAndTableConflict) {
    expect_reject("conflict",
                  std::string(kBase) + "lj 0 0 1.0 1.0\ntable 0 0 " + table_file() + " LJ_TABLE\n",
                  "both lj and table");
}

// Reproduces the heap-buffer overflow reported under AddressSanitizer:
// with ntypes 1, 'lj 3 0 1 1' indexed lj_params[3].
TEST(Config, RejectsOutOfRangeTypeIndices) {
    expect_reject("lj_oob", std::string(kBase) + "lj 3 0 1.0 1.0\n", "out of range");
    expect_reject("lj_neg", std::string(kBase) + "lj 0 -1 1.0 1.0\n", "out of range");
    expect_reject("lj_type_oob", std::string(kBase) + "lj_type 1 1.0 1.0\n", "out of range");
    expect_reject("table_oob", std::string(kBase) + "table 0 2 " + table_file() + " LJ_TABLE\n",
                  "out of range");
    expect_reject("bond_neg", std::string(kBase) + "lj 0 0 1 1\nbond_type -1 30 1.5 1 1\n", "out of range");
}

TEST(Config, RejectsMalformedArguments) {
    expect_reject("missing_arg", std::string(kBase) + "lj 0 0 1.0\n", "missing argument");
    expect_reject("extra_arg", std::string(kBase) + "lj 0 0 1.0 1.0 7\n", "unexpected extra argument");
    expect_reject("not_number", std::string(kBase) + "lj 0 0 abc 1.0\n", "must be a finite number");
    expect_reject("bad_int", std::string(kBase) + "lj 0.5 0 1 1\n", "must be an integer");
    expect_reject("neg_rc", "ntypes 1\nrc -2\n", "must be > 0");
    expect_reject("zero_sigma", std::string(kBase) + "lj 0 0 1.0 0\n", "must be > 0 when eps > 0");
    expect_reject("thermo_freq0", std::string(kBase) + "lj 0 0 1 1\nthermo 1 0 t.dat\n", ">= 1");
    expect_reject("ntypes_big", "ntypes 99\n", "ntypes");
}

TEST(Config, RejectsUnknownAndDuplicateKeys) {
    expect_reject("unknown", std::string(kBase) + "lj 0 0 1 1\nthermo_freq 100\n", "unknown keyword");
    expect_reject("dup_rc", std::string(kBase) + "lj 0 0 1 1\nrc 3.0\n", "more than once");
    expect_reject("two_ensembles", std::string(kBase) + "lj 0 0 1 1\nnvt_nh 1 1 1\n", "only one ensemble");
    expect_reject("nve_and_langevin", std::string(kBase) + "lj 0 0 1 1\nnve\n", "only one ensemble");
    expect_reject("nve_args", "ntypes 1\nnve 1.0\n", "unexpected extra argument");
    expect_reject("pair_shift_bad", std::string(kBase) + "pair_shift maybe\n", "yes");
    expect_reject("two_angle_styles", std::string(kBase) + "lj 0 0 1 1\nangle_type 0 3 180\n"
                  "angle_cosine 0 1.5\n", "only one angle style");
    expect_reject("wall_face", std::string(kBase) + "wall top lj93 1 1 2.5 0\n", "face must be");
    expect_reject("wall_twice", std::string(kBase) + "wall zlo lj93 1 1 2.5 0\nwall zlo harmonic 1 1 1 0\n",
                  "given twice");
    expect_reject("wall_style", std::string(kBase) + "wall zlo soft 1 1 2.5 0\n", "lj93 or harmonic");
    expect_reject("film_axis", std::string(kBase) + "film w\n", "film normal");
    expect_reject("andersen_nu", "ntypes 1\nnvt_andersen 1 1 0\n", "must be > 0");
    expect_reject("friction_kind", std::string(kBase) + "friction_profile cubic 1 1 1\n", "analytic");
    expect_reject("layers_n", std::string(kBase) + "layer_stress 40 2 10 l.dat\n", "[1, 32]");
    expect_reject("minimize_iter", std::string(kBase) + "minimize 1e-4 0\n", ">= 1");
    expect_reject("stress_pm", std::string(kBase) + "stress 1 10 s.dat 16 3\n", "multiple of m");
    expect_reject("dup_lj", std::string(kBase) + "lj 0 0 1 1\nlj 0 0 2 1\n", "given twice");
}

TEST(Config, RequiresCoreKeysEnsembleAndCoverage) {
    expect_reject("no_ensemble", "ntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\nlj 0 0 1 1\n"
                                 "lammps_data_file x.data\n", "no ensemble");
    expect_reject("no_dt", "ntypes 1\nrc 2.5\nskin 0.3\nnsteps 1\nnvt_nh 1 1 1\nlj 0 0 1 1\n"
                           "lammps_data_file x.data\n", "'dt'");
    expect_reject("missing_pair", "ntypes 2\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\nnvt_nh 1 1 1\n"
                                  "lj 0 0 1 1\nlj 1 1 1 1\nlammps_data_file x.data\n",
                  "no lj or table interaction for type pair (0,1)");
    expect_reject("two_bond_styles", std::string(kBase) + "lj 0 0 1 1\nbond_type 0 30 1.5 1 1\n"
                                                          "bond_harmonic 0 100 1\n", "one bond style");
}

TEST(Config, TrailingCommentsAndOptionalArgs) {
    TopologyData topo;
    SimParams p = parse_config(write_cfg("ok", std::string(kBase) +
                                         "lj 0 0 1.0 1.0  # epsilon sigma\nspecial_dihedral yes\n"), topo);
    EXPECT_EQ(p.seed, 42u);
    EXPECT_EQ(p.special_dihedral, 1);
    EXPECT_FLOAT_EQ(topo.lj_params[0].x, 1.0f);
}

// validate_system: parameters must cover every bonded type in the data file.
TEST(Config, ValidateSystemChecksCoverage) {
    std::ofstream("cov.data") << "t\n\n3 atoms\n2 bonds\n1 atom types\n2 bond types\n\n"
                                 "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nAtoms # molecular\n\n"
                                 "1 1 1 1 1 1\n2 1 1 2 1 1\n3 1 1 3 1 1\n\n"
                                 "Bonds\n\n1 1 1 2\n2 2 2 3\n";
    TopologyData topo;
    SimParams p = parse_config(write_cfg("cov", "ntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\n"
                                         "nvt_nh 1 1 1\nlj 0 0 1 1\nbond_type 0 30 1.5 1 1\n"
                                         "lammps_data_file cov.data\n"), topo);
    parse_lammps_data(topo.data_file, topo);
    p.natoms = (int)topo.positions.size();
    try {
        validate_system(p, topo);
        FAIL() << "missing bond type 2 accepted";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("bond type 2"), std::string::npos) << e.what();
    }
}
