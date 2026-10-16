#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "io/config.hpp"
#include "io/lammps_data.hpp"
#include "sim/simulation.cuh"
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
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

// Copies of the system side by side along x, y, z (box nx Lx x ny Ly x nz Lz).
void replicate(TopologyData& t, int nx, int ny, int nz) {
    const int n = (int)t.positions.size(), ncopy = nx * ny * nz;
    const float3 L = t.box;
    int nmol = 0;
    for (int m : t.mol_ids) nmol = std::max(nmol, m);
    auto pos = t.positions; auto vel = t.velocities; auto img = t.images; auto mol = t.mol_ids;
    auto ids = t.atom_ids; auto bonds = t.bonds; auto btypes = t.bond_types; auto angles = t.angles;
    t.positions.clear(); t.velocities.clear(); t.images.clear(); t.mol_ids.clear(); t.atom_ids.clear();
    t.bonds.clear(); t.bond_types.clear(); t.angles.clear();
    for (int c = 0; c < ncopy; c++) {
        const int cx = c % nx, cy = (c / nx) % ny, cz = c / (nx * ny), off = c * n;
        for (int i = 0; i < n; i++) {
            float4 p = pos[i];
            p.x += cx * L.x; p.y += cy * L.y; p.z += cz * L.z;
            t.positions.push_back(p);
            if (!vel.empty()) t.velocities.push_back(vel[i]);
            for (int d = 0; d < 3; d++) t.images.push_back(img[3 * i + d]);
            if (!mol.empty()) t.mol_ids.push_back(mol[i] + c * nmol);
            t.atom_ids.push_back(off + i + 1);
        }
        for (size_t b = 0; b < bonds.size(); b++) {
            t.bonds.push_back(make_int2(bonds[b].x + off, bonds[b].y + off));
            t.bond_types.push_back(btypes[b]);
        }
        for (const int4& a : angles) t.angles.push_back(make_int4(a.x + off, a.y + off, a.z + off, a.w));
    }
    t.box = make_float3(nx * L.x, ny * L.y, nz * L.z);
}

std::unique_ptr<Simulation> make_sim(const std::string& data, const std::string& extra, int nsteps,
                                     const std::string& ensemble = "nvt_nh 1.0 1.0 0.5 3",
                                     int3 copies = make_int3(1, 1, 1),
                                     std::function<void(TopologyData&)> edit = nullptr) {
    std::string conf = "sim_test_" + std::to_string(std::hash<std::string>{}(extra + ensemble)) + ".conf";
    std::ofstream(conf) << "ntypes 1\nrc 2.5\nskin 0.3\ndt 0.004\nnsteps " << nsteps << "\n"
                        << ensemble << "\nlj 0 0 1.0 1.0\nbond_type 0 30.0 1.5 1.0 1.0\n"
                        << "angle_type 0 3.0 180.0\nlammps_data_file " << data << "\n" << extra;
    TopologyData topo;
    SimParams p = parse_config(conf, topo);
    parse_lammps_data(topo.data_file, topo);
    if (copies.x * copies.y * copies.z > 1) replicate(topo, copies.x, copies.y, copies.z);
    if (edit) edit(topo);
    p.natoms = (int)topo.positions.size();
    p.box_L3 = topo.box;
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
                            sorted->particles().download_positions(), ref->box().L.x);
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
                            small->particles().download_positions(), ref->box().L.x);
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

// traj log-periodic frames at block offsets 0,1,2,4,...; each frame holds
// the state of its step (checked for the final frame) in tag order.
TEST(Simulation, TrajectoryFramesFollowScheduleAndState) {
    std::string data = write_melt("sim_traj.data", 8, 20, 0.85f, 9);
    auto sim = make_sim(data, "traj sim_traj.fmd log 50\n", 100);
    sim->run(100);
    std::vector<float4> pos = sim->positions();
    std::vector<int> img = sim->images();
    const double L = sim->box().L.x;
    sim.reset();   // closes the file
    std::ifstream in("sim_traj.fmd", std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t n = pos.size(), head = 24 + 8 * n, frame = 8 + 48 + 24 * n;
    ASSERT_EQ((b.size() - head) % frame, 0u);
    std::vector<long long> steps;
    for (size_t off = head; off < b.size(); off += frame) {
        int64_t s;
        std::memcpy(&s, b.data() + off, 8);
        steps.push_back(s);
    }
    EXPECT_EQ(steps, (std::vector<long long>{0, 1, 2, 4, 8, 16, 32, 50, 51, 52, 54, 58, 66, 82, 100}));
    const char* last = b.data() + b.size() - frame;
    double box[6];
    std::memcpy(box, last + 8, 48);
    EXPECT_EQ(box[0], L);
    EXPECT_EQ(box[3], 0.0);
    std::vector<float> xyz(3 * n);
    std::vector<int32_t> im(3 * n);
    std::memcpy(xyz.data(), last + 56, 12 * n);
    std::memcpy(im.data(), last + 56 + 12 * n, 12 * n);
    for (size_t t = 0; t < n; t++) {
        EXPECT_EQ(xyz[3 * t], pos[t].x);
        EXPECT_EQ(xyz[3 * t + 2], pos[t].z);
        EXPECT_EQ(im[3 * t + 1], img[3 * t + 1]);
    }
}

// Orthorhombic periodic boxes: copies of a melt placed side by side along
// one axis feel exactly the forces of the original in its cubic box.
TEST(Simulation, ReplicatedOrthorhombicBoxGivesSameForces) {
    std::string data = write_melt("sim_rep.data", 12, 20, 0.85f, 21);
    auto ref = make_sim(data, "pair_tpa 1\n", 10);
    std::vector<float4> f1 = ref->forces();
    const size_t n = f1.size();
    for (int3 c : {make_int3(2, 1, 1), make_int3(1, 1, 3)}) {
        auto rep = make_sim(data, "pair_tpa 1\n", 10, "nvt_nh 1.0 1.0 0.5 3", c);
        const Box& b = rep->box();
        EXPECT_FLOAT_EQ(b.L.x, c.x * ref->box().L.x);
        EXPECT_FLOAT_EQ(b.L.z, c.z * ref->box().L.z);
        std::vector<float4> f2 = rep->forces();
        ASSERT_EQ(f2.size(), n * c.x * c.y * c.z);
        double worst = 0, fmax = 0;
        for (size_t t = 0; t < f2.size(); t++) {
            const float4 a = f2[t], r = f1[t % n];
            worst = std::max({worst, (double)std::fabs(a.x - r.x), (double)std::fabs(a.y - r.y),
                              (double)std::fabs(a.z - r.z)});
            fmax = std::max(fmax, (double)std::fabs(r.x));
        }
        // float32 coordinates of the shifted copies (up to 3 L) are a few
        // times coarser than in the original box; steep LJ contacts turn
        // that into ~1e-5 relative force differences
        EXPECT_LT(worst, 1e-4 * fmax) << "copies " << c.x << "x" << c.y << "x" << c.z;
        ThermoOutput t1 = ref->thermo(), t2 = rep->thermo();
        EXPECT_NEAR(t2.potential_energy, t1.potential_energy, 1e-4 * std::fabs(t1.potential_energy));
        for (int k = 0; k < 6; k++)   // intensive pressure tensor is unchanged
            EXPECT_NEAR(t2.stress[k], t1.stress[k], 2e-3 * std::fabs(t1.stress[VXX]) + 1e-4) << k;
    }
}

// The isotropic barostat scales every edge of an orthorhombic box by the
// same factor.
TEST(Simulation, NPTKeepsOrthorhombicAspectRatio) {
    std::string data = write_melt("sim_npt_rect.data", 12, 20, 0.80f, 33);
    auto sim = make_sim(data, "", 400, "npt_nh 1.0 1.0 0.5 2.0 2.0 2.0 3", make_int3(2, 1, 1));
    const Box b0 = sim->box();
    sim->run(400);
    const Box b1 = sim->box();
    EXPECT_NE(b1.L.x, b0.L.x) << "barostat did not act";
    EXPECT_NEAR(b1.L.x / b1.L.y, 2.0, 1e-5);
    EXPECT_NEAR(b1.L.z / b1.L.y, 1.0, 1e-5);
}

// Walls along z confine a slab placed in a taller box; a moving wall
// compresses it; a wall plane through the system is an error.
TEST(Simulation, WallsConfineAndCompress) {
    std::string data = write_melt("sim_wall.data", 12, 20, 0.85f, 41);
    const float margin = 3.0f;
    auto lift = [&](TopologyData& t) {   // snake-lattice chains do not cross the z boundary
        for (auto& p : t.positions) p.z += margin;
        t.box.z += 2.0f * margin;
    };
    TopologyData probe;
    parse_lammps_data(data, probe);
    const float L = probe.box.z, top = L + 2.0f * margin;
    std::ostringstream walls;
    walls << "wall zlo lj93 1.0 1.0 2.5 " << margin - 1.0f << "\n"
          << "wall zhi lj93 1.0 1.0 2.5 " << top - margin + 1.0f << " " << top - margin - 1.5f << "\n";
    auto sim = make_sim(data, walls.str(), 2000, "nvt_langevin 1.0 1.0 1.0 3", make_int3(1, 1, 1), lift);
    sim->run(2000);
    std::vector<float4> pos = sim->positions();
    float zmin = 1e9f, zmax = -1e9f;
    for (auto& p : pos) { zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z); }
    EXPECT_GT(zmin, margin - 1.0f);
    EXPECT_LT(zmax, top - margin - 1.5f) << "the moving wall compressed the slab";
    EXPECT_LT(zmax - zmin, L + 0.5f);
    // remove the walls: the slab may now expand into the vacuum
    sim->set_walls(WallSet{});
    sim->run(200);
    EXPECT_EQ(sim->walls().n, 0);

    std::ostringstream bad;
    bad << "wall zhi harmonic 1.0 1.0 1.0 " << margin + 0.5f * L << "\n";
    auto broken = make_sim(data, bad.str(), 10, "nvt_langevin 1.0 1.0 1.0 3", make_int3(1, 1, 1), lift);
    EXPECT_THROW(broken->run(10), std::runtime_error);
}

// The film reference plane is the mass-weighted mean unwrapped z.
TEST(Simulation, FilmCenterTracksSlab) {
    std::string data = write_melt("sim_film.data", 12, 20, 0.85f, 47);
    auto lift = [](TopologyData& t) {
        for (auto& p : t.positions) p.z += 8.0f;
        t.box.z += 16.0f;
    };
    auto sim = make_sim(data, "film z\n", 1000, "nvt_langevin 1.0 1.0 1.0 9", make_int3(1, 1, 1), lift);
    sim->run(1000);
    std::vector<float4> pos = sim->positions();
    std::vector<int> img = sim->images();
    const double Lz = sim->box().L.z;
    double sum = 0;
    for (size_t t = 0; t < pos.size(); t++) sum += pos[t].z + img[3 * t + 2] * Lz;   // unit masses
    EXPECT_NEAR(sim->film_center(), sum / pos.size(), 1e-4 * Lz);
    auto plain = make_sim(data, "", 10);
    EXPECT_THROW(plain->film_center(), std::logic_error);
}

// Sheared boxes. Tilts +Lx/2 and -Lx/2 are the same lattice: re-tilting
// without moving particles leaves every force unchanged.
TEST(Simulation, TiltedLatticeEquivalence) {
    std::string data = write_melt("sim_tilt.data", 12, 20, 0.85f, 51);
    auto sim = make_sim(data, "pair_tpa 1\n", 10, "nvt_langevin 1.0 1.0 1.0 3", make_int3(2, 1, 1));
    const float Lx = sim->box().L.x;
    sim->set_tilt(0.5f * Lx, true);   // a large affine shear
    EXPECT_FLOAT_EQ(sim->box().xy, 0.5f * Lx);
    std::vector<float4> f1 = sim->forces();
    sim->set_tilt(-0.5f * Lx, false);  // same lattice, positions re-wrapped only
    std::vector<float4> f2 = sim->forces();
    double worst = 0, fmax = 0;
    for (size_t t = 0; t < f1.size(); t++) {
        worst = std::max({worst, (double)std::fabs(f1[t].x - f2[t].x), (double)std::fabs(f1[t].y - f2[t].y),
                          (double)std::fabs(f1[t].z - f2[t].z)});
        fmax = std::max(fmax, (double)std::fabs(f1[t].x));
    }
    EXPECT_LT(worst, 1e-4 * fmax);
    sim->run(50);   // dynamics in the tilted box
    for (const float4& p : sim->positions()) EXPECT_TRUE(std::isfinite(p.x) && std::isfinite(p.y));
}

// Stress-strain consistency in a sheared box: for an affine shear
// x += gamma y, dU/dgamma = -W_xy (W_xy = sum r_y F_x over pairs), i.e.
// at zero velocities dU/dgamma = -V P_xy.
TEST(Simulation, ShearVirialIsStrainDerivative) {
    std::string data = write_melt("sim_strain.data", 12, 20, 0.85f, 53);
    // energy-shifted LJ: the energy must be continuous where pairs cross rc
    auto sim = make_sim(data, "pair_tpa 1\npair_shift yes\n", 10, "nvt_langevin 1.0 1.0 1.0 3");
    sim->run(300);   // leave the lattice
    sim->set_velocities(std::vector<float3>(sim->positions().size(), make_float3(0, 0, 0)));
    for (float xy0 : {0.0f, 0.3f * sim->box().L.y}) {
        sim->set_tilt(xy0, true);
        const double V = sim->box().volume(), N = (double)sim->positions().size(), Ly = sim->box().L.y;
        const double Pxy = sim->thermo().stress[VXY];
        const float dg = 2e-3f;
        auto U_at = [&](float g) {
            sim->set_tilt(xy0 + g * (float)Ly, true);
            double U = sim->thermo().potential_energy * N;
            sim->set_tilt(xy0, true);
            return U;
        };
        // strains measured relative to the current (sheared) state
        const double dUdg = (U_at(dg) - U_at(-dg)) / (2.0 * dg);
        EXPECT_NEAR(dUdg, -V * Pxy, 0.01 * std::fabs(V * Pxy) + 0.5) << "tilt " << xy0;
        printf("tilt %.3f: dU/dgamma %.4f  -V Pxy %.4f\n", xy0, dUdg, -V * Pxy);
    }
}

// A positive step strain makes P_xy jump negative (shear stress
// sigma_xy = -P_xy > 0), then relax.
TEST(Simulation, StepShearStressResponse) {
    std::string data = write_melt("sim_step.data", 16, 25, 0.85f, 55);
    auto sim = make_sim(data, "", 10, "nvt_langevin 1.0 1.0 1.0 3");
    sim->run(2000);
    double before = 0;
    for (int k = 0; k < 10; k++) { sim->run(20); before += sim->thermo().stress[VXY]; }
    before /= 10;
    sim->apply_step_shear(0.05f);
    EXPECT_NEAR(sim->box().xy, 0.05f * sim->box().L.y, 1e-5);
    const double after = sim->thermo().stress[VXY];
    EXPECT_LT(after - before, -0.3) << "before " << before << " after " << after;
    sim->run(2000);
    EXPECT_GT(sim->thermo().stress[VXY] - before, 0.5 * (after - before));   // relaxing
}

namespace {
// Slab of the snake-lattice melt in a box taller by 2 x 8 along z.
std::function<void(TopologyData&)> slab_lift() {
    return [](TopologyData& t) {
        for (auto& p : t.positions) p.z += 8.0f;
        t.box.z += 16.0f;
    };
}

// Kinetic temperature in bins of z - z_c (unit masses), accumulated over calls.
struct TempProfile {
    std::vector<double> ke, n;
    double width = 2.0, half = 12.0;
    void add(Simulation& sim) {
        if (ke.empty()) { ke.assign((int)(2 * half / width), 0.0); n = ke; }
        std::vector<float4> v = sim.velocities(), x = sim.positions();
        std::vector<int> im = sim.images();
        const double zc = sim.film_center(), Lz = sim.box().L.z;
        for (size_t t = 0; t < x.size(); t++) {
            double d = x[t].z + im[3 * t + 2] * Lz - zc;
            int k = (int)std::floor((d + half) / width);
            if (k < 0 || k >= (int)ke.size()) continue;
            ke[k] += v[t].x * v[t].x + v[t].y * v[t].y + v[t].z * v[t].z;
            n[k] += 3;
        }
    }
};
}  // namespace

// Depth-dependent friction changes the dynamics, not the equilibrium:
// after equilibrating with uniform friction, switching on strongly varying
// profiles leaves the temperature flat.
TEST(Simulation, FrictionProfileKeepsTemperatureFlat) {
    std::string data = write_melt("sim_fric.data", 16, 25, 0.85f, 61);
    std::vector<std::pair<FrictionProfile, std::vector<float>>> profiles;
    FrictionProfile an;
    an.mode = 1; an.ell = 1.5f; an.h = 12.0f;
    an.a = 2.0f;
    profiles.push_back({an, {}});
    an.a = -2.0f;
    profiles.push_back({an, {}});
    FrictionProfile tab;
    tab.mode = 2;
    std::vector<float> step(64);
    for (int k = 0; k < 64; k++) step[k] = k < 32 ? 30.0f : 0.1f;   // d in [-30, 30]
    profiles.push_back({tab, step});
    for (auto& pr : profiles) {
        auto sim = make_sim(data, "film z\npair_tpa 1\n", 10000, "nvt_langevin 1.0 1.0 1.0 5",
                            make_int3(1, 1, 1), slab_lift());
        sim->run(3000);   // relax the lattice start with uniform friction
        sim->set_friction_profile(pr.first, pr.second, -30.0f, 60.0f / 63.0f);
        sim->run(3000);
        TempProfile tp;
        // bins hold ~70 particles: per-snapshot T scatter ~10%, 60 correlated snapshots
        for (int k = 0; k < 60; k++) { sim->run(100); tp.add(*sim); }
        for (size_t k = 0; k < tp.ke.size(); k++) {
            if (tp.n[k] < 6000) continue;   // vacuum / sparse surface bins
            double T = tp.ke[k] / tp.n[k];
            EXPECT_NEAR(T, 1.0, 0.07) << "profile mode " << pr.first.mode << " a " << pr.first.a
                                      << " bin " << k << " (d = " << (k + 0.5) * tp.width - tp.half
                                      << ", " << tp.n[k] / 3 << " samples)";
        }
    }
}

TEST(Simulation, FrictionProfileSlowsDynamicsWhereLarge) {
    std::string data = write_melt("sim_fric2.data", 16, 25, 0.85f, 63);
    std::ofstream("fric_half.txt") << "-30 100\n-0.01 100\n0.01 1\n30 1\n";
    auto sim = make_sim(data, "film z\nfriction_profile table fric_half.txt\n", 4000,
                        "nvt_langevin 1.0 1.0 2.0 7", make_int3(1, 1, 1), slab_lift());
    sim->run(500);
    auto unwrapped = [&] {
        std::vector<float4> x = sim->positions();
        std::vector<int> im = sim->images();
        const Box b = sim->box();
        std::vector<double> u(3 * x.size());
        for (size_t t = 0; t < x.size(); t++) {
            u[3 * t] = x[t].x + im[3 * t] * b.L.x + im[3 * t + 1] * b.xy;
            u[3 * t + 1] = x[t].y + im[3 * t + 1] * b.L.y;
            u[3 * t + 2] = x[t].z + im[3 * t + 2] * b.L.z;
        }
        return u;
    };
    std::vector<double> u0 = unwrapped();
    const double zc = sim->film_center();
    sim->run(1000);
    std::vector<double> u1 = unwrapped();
    double msd[2] = {0, 0}, cnt[2] = {0, 0};
    for (size_t t = 0; t < u0.size() / 3; t++) {
        double d = u0[3 * t + 2] - zc;
        if (std::fabs(d) < 1.0 || std::fabs(d) > 4.0) continue;   // away from the step and surfaces
        int side = d < 0 ? 0 : 1;
        for (int c = 0; c < 2; c++) msd[side] += std::pow(u1[3 * t + c] - u0[3 * t + c], 2);   // in-plane
        cnt[side] += 1;
    }
    ASSERT_GT(cnt[0], 50);
    ASSERT_GT(cnt[1], 50);
    EXPECT_LT(msd[0] / cnt[0], 0.5 * msd[1] / cnt[1]) << "high-friction side should move less";
}

// A unit table reproduces uniform Langevin friction.
TEST(Simulation, UnitFrictionProfileMatchesUniform) {
    std::string data = write_melt("sim_fric3.data", 8, 20, 0.85f, 65);
    std::ofstream("fric_one.txt") << "-50 1\n50 1\n";
    auto a = make_sim(data, "film z\npair_tpa 1\n", 100, "nvt_langevin 1.0 1.0 1.0 9");
    auto b = make_sim(data, "film z\npair_tpa 1\nfriction_profile table fric_one.txt\n", 100,
                      "nvt_langevin 1.0 1.0 1.0 9");
    a->run(100);
    b->run(100);
    EXPECT_LT(max_pos_diff(a->positions(), b->positions(), a->box().L.x), 1e-3);
}

// Andersen collisions thermostat a system started at rest, also with a
// collision-rate profile that is zero in half of the film.
TEST(Simulation, AndersenThermostat) {
    std::string data = write_melt("sim_and.data", 16, 25, 0.85f, 67);
    auto sim = make_sim(data, "", 3000, "nvt_andersen 1.2 1.2 5.0 3");
    sim->set_velocities(std::vector<float3>(sim->positions().size(), make_float3(0, 0, 0)));
    sim->run(2000);
    double T = 0;
    for (int k = 0; k < 10; k++) { sim->run(50); T += sim->thermo().temperature; }
    EXPECT_NEAR(T / 10, 1.2, 0.04);

    auto film = make_sim(data, "film z\npair_tpa 1\n", 10000, "nvt_andersen 1.0 1.0 5.0 4",
                         make_int3(1, 1, 1), slab_lift());
    film->run(3000);
    FrictionProfile half;
    half.mode = 2;
    std::vector<float> rate(64);
    for (int k = 0; k < 64; k++) rate[k] = k < 32 ? 0.0f : 2.0f;   // no collisions for d < 0
    film->set_friction_profile(half, rate, -30.0f, 60.0f / 63.0f);
    film->run(3000);
    TempProfile tp;
    for (int k = 0; k < 60; k++) { film->run(100); tp.add(*film); }
    for (size_t k = 0; k < tp.ke.size(); k++) {
        if (tp.n[k] < 6000) continue;
        EXPECT_NEAR(tp.ke[k] / tp.n[k], 1.0, 0.07) << "bin " << k;
    }
}
