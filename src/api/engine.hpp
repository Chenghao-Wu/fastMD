#pragma once
// Stateful C++ interface to one fastMD simulation, using only standard types
// (no CUDA headers), for embedding and for the Python binding.
//
// Arrays are flat, in tag order (the atom order of the input data file):
// per-particle vectors have 3 * natoms entries (x0 y0 z0 x1 ...). Positions
// are wrapped into the box (the data-file box shifted to the origin);
// unwrapped positions are pos + image * (Lx, Ly, Lz) for orthorhombic boxes.
#include <array>
#include <memory>
#include <string>
#include <vector>

class Simulation;

namespace fastmd {

struct Thermo {
    double kinetic_energy;     // total (not per atom)
    double potential_energy;   // total, including the Ewald self energy
    double temperature;        // 2 KE / (3 N k_B)
    double pressure[6];   // xx xy xz yy yz zz
};

class Engine {
public:
    // Load a config file (and the LAMMPS data file it names), as the
    // command-line driver does, and set up the simulation on the GPU.
    static std::unique_ptr<Engine> from_file(const std::string& config_path);
    // Same with the config given as text.
    static std::unique_ptr<Engine> from_text(const std::string& config_text);
    ~Engine();

    int natoms() const;
    int ntypes() const;
    int step() const;              // steps taken so far
    double dt() const;
    std::array<double, 6> box() const;   // Lx Ly Lz xy xz yz

    // Advance nsteps time steps; periodic outputs configured in the config
    // (thermo, Rg, dump, restart, stress samples) are written when due.
    void run(int nsteps);
    // The command-line run: config nsteps with progress output, then the
    // final restart and the stress autocorrelation file.
    void run_config();

    std::vector<float> positions() const;
    std::vector<int>   images() const;
    std::vector<float> velocities();          // applies a pending thermostat scale
    std::vector<float> forces() const;
    std::vector<float> per_atom_energy() const;
    std::vector<int>   types() const;
    std::vector<float> charges() const;

    // Replace positions (wrapped into the box; image counts default to 0)
    // and recompute the neighbor list and forces.
    void set_positions(const std::vector<float>& xyz, const std::vector<int>* images = nullptr);
    void set_velocities(const std::vector<float>& v);

    Thermo thermo();   // energies, temperature and pressure tensor now

private:
    explicit Engine(std::unique_ptr<Simulation> sim);
    std::unique_ptr<Simulation> sim_;
};

}  // namespace fastmd
