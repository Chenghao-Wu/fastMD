#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "io/config.hpp"
#include "io/lammps_data.hpp"
#include "sim/simulation.cuh"
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <array>
#include <memory>

namespace {

// Bead-spring melt on a jittered lattice traced by a snake path (bonds and
// angles along chains), written as a LAMMPS data file.
std::string write_melt(const std::string& path, int nchains, int len, float density, unsigned seed) {
    const int n = nchains * len;
    const float L = std::cbrt(n / density);
    int side = (int)std::ceil(std::cbrt((double)n));
    const float a = L / side;
    std::vector<std::array<float, 3>> sites;
    int row = 0;
    for (int z = 0; z < side; z++)
        for (int yy = 0; yy < side; yy++) {
            int y = z % 2 ? side - 1 - yy : yy;
            for (int xx = 0; xx < side; xx++) {
                int x = row % 2 ? side - 1 - xx : xx;
                sites.push_back({(x + 0.5f) * a, (y + 0.5f) * a, (z + 0.5f) * a});
            }
            row++;
        }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jit(-0.04f, 0.04f);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::ostringstream os;
    os << "melt\n\n" << n << " atoms\n" << nchains * (len - 1) << " bonds\n"
       << nchains * (len - 2) << " angles\n1 atom types\n1 bond types\n1 angle types\n\n"
       << "0 " << L << " xlo xhi\n0 " << L << " ylo yhi\n0 " << L << " zlo zhi\n\n"
       << "Masses\n\n1 1\n\nAtoms # molecular\n\n";
    for (int c = 0; c < nchains; c++) {
        int start = (int)((long)c * sites.size() / nchains);
        for (int k = 0; k < len; k++) {
            auto s = sites[start + k];
            os << c * len + k + 1 << " " << c + 1 << " 1 " << s[0] + jit(rng) << " "
               << s[1] + jit(rng) << " " << s[2] + jit(rng) << "\n";
        }
    }
    os << "\nVelocities\n\n";
    for (int i = 0; i < n; i++) os << i + 1 << " " << g(rng) << " " << g(rng) << " " << g(rng) << "\n";
    os << "\nBonds\n\n";
    int b = 1;
    for (int c = 0; c < nchains; c++)
        for (int k = 0; k + 1 < len; k++) os << b++ << " 1 " << c * len + k + 1 << " " << c * len + k + 2 << "\n";
    os << "\nAngles\n\n";
    b = 1;
    for (int c = 0; c < nchains; c++)
        for (int k = 0; k + 2 < len; k++)
            os << b++ << " 1 " << c * len + k + 1 << " " << c * len + k + 2 << " " << c * len + k + 3 << "\n";
    std::ofstream(path) << os.str();
    return path;
}

std::unique_ptr<Simulation> make_sim(const std::string& data, const std::string& extra, int nsteps,
                                     const std::string& ensemble = "nvt_nh 1.0 1.0 0.5 3") {
    std::string conf = "sim_test_" + std::to_string(std::hash<std::string>{}(extra + ensemble)) + ".conf";
    std::ofstream(conf) << "ntypes 1\nrc 2.5\nskin 0.3\ndt 0.004\nnsteps " << nsteps << "\n"
                        << ensemble << "\nlj 0 0 1.0 1.0\nbond_type 0 30.0 1.5 1.0 1.0\n"
                        << "angle_type 0 3.0 180.0\nlammps_data_file " << data << "\n" << extra;
    TopologyData topo;
    SimParams p = parse_config(conf, topo);
    parse_lammps_data(topo.data_file, topo);
    p.natoms = (int)topo.positions.size();
    p.box_L = topo.box_L;
    validate_system(p, topo);
    finalize_params(p);
    return std::make_unique<Simulation>(p, std::move(topo));
}

double max_pos_diff(const std::vector<float4>& a, const std::vector<float4>& b, float L) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d[3] = {a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        for (double& x : d) { x -= L * std::nearbyint(x / L); m = std::max(m, std::fabs(x)); }
    }
    return m;
}

}  // namespace

// Reordering all particles (positions, velocities, images, tags, topology)
// every few steps must not change the trajectory beyond float summation-
// order noise.
TEST(Simulation, ParticleSortingDoesNotChangeTrajectory) {
    std::string data = write_melt("sim_melt.data", 30, 50, 0.85f, 1);
    auto ref = make_sim(data, "sort_every 0\ncheck_every 7\n", 60);
    auto sorted = make_sim(data, "sort_every 5\ncheck_every 5\n", 60);
    ref->run();
    sorted->run();
    double d = max_pos_diff(ref->particles().download_positions(),
                            sorted->particles().download_positions(), ref->box_length());
    EXPECT_LT(d, 2e-3) << "max position difference after 60 steps";
}

// A neighbor capacity that is too small for the first device-side build must
// be detected at the next check point, rolled back and re-run with a grown
// list, giving the same trajectory as a run that never overflowed.
TEST(Simulation, DeviceNeighborOverflowRollsBack) {
    std::string data = write_melt("sim_melt.data", 30, 50, 0.85f, 1);
    auto ref = make_sim(data, "check_every 10\n", 40);
    auto small = make_sim(data, "check_every 10\nneigh_capacity 8\n", 40);
    ref->run();
    small->run();
    EXPECT_GE(small->rollbacks(), 1);
    EXPECT_EQ(ref->rollbacks(), 0);
    EXPECT_GT(small->neighbor_capacity(), 8);
    double d = max_pos_diff(ref->particles().download_positions(),
                            small->particles().download_positions(), ref->box_length());
    EXPECT_LT(d, 2e-3);
}

// The list maintained on the device must be complete: forces from it match
// forces after a fresh synchronous rebuild.
TEST(Simulation, DeviceMaintainedListIsComplete) {
    std::string data = write_melt("sim_melt.data", 30, 50, 0.85f, 1);
    auto sim = make_sim(data, "check_every 1000\nsort_every 0\n", 150);
    sim->run();
    sim->compute_forces();
    auto f_dev = sim->particles().download_forces();
    sim->rebuild_neighbor_list();
    sim->compute_forces();
    auto f_new = sim->particles().download_forces();
    double fmax = 0, dmax = 0;
    for (size_t i = 0; i < f_dev.size(); i++) {
        fmax = std::max({fmax, (double)std::fabs(f_new[i].x), (double)std::fabs(f_new[i].y)});
        dmax = std::max({dmax, (double)std::fabs(f_dev[i].x - f_new[i].x),
                         (double)std::fabs(f_dev[i].y - f_new[i].y),
                         (double)std::fabs(f_dev[i].z - f_new[i].z)});
    }
    EXPECT_LT(dmax, 1e-4 * fmax);
}

// NVE (velocity Verlet, no thermostat) with energy-shifted LJ conserves the
// total energy; the lattice start relaxes violently, so this also exercises
// large KE/PE exchange.
TEST(Simulation, NVEConservesEnergy) {
    std::string data = write_melt("sim_nve.data", 20, 25, 0.85f, 5);
    auto sim = make_sim(data, "pair_shift yes\n", 4000, "nve");
    auto total = [&] {
        ThermoOutput t = sim->thermo();
        return (double)t.kinetic_energy + t.potential_energy;   // per atom
    };
    sim->run(200);   // let the lattice start relax a little
    const double e0 = total();
    double worst = 0;
    for (int k = 0; k < 20; k++) {
        sim->run(200);
        worst = std::max(worst, std::fabs(total() - e0));
    }
    EXPECT_LT(worst, 2e-3 * std::fabs(e0)) << "E0 per atom " << e0;
    printf("NVE: max |E - E0| per atom %.3e (E0 %.4f)\n", worst, e0);
}
