#pragma once
#include "../core/types.cuh"
#include "../core/particles.cuh"
#include "../core/topology.cuh"
#include "../core/morton.cuh"
#include "../neighbor/verlet_list.cuh"
#include "../force/fene.cuh"
#include "../force/angle.cuh"
#include "../force/bond_harmonic.cuh"
#include "../force/dihedral_opls.cuh"
#include "../integrate/langevin.cuh"
#include "../integrate/nose_hoover.cuh"
#include "../analysis/thermo.cuh"
#include "../analysis/correlator.cuh"
#include "../analysis/rg.cuh"
#include "../io/dump.cuh"
#include "../io/config.hpp"
#include <chrono>
#include <string>
#include <vector>

// One MD simulation: owns all device state and drives the time loop.
//
// Construction uploads the system, builds the neighbor list and evaluates the
// initial forces. step() advances one time step; run() advances
// params.nsteps steps and writes the configured outputs.
class Simulation {
public:
    Simulation(const SimParams& params, TopologyData topo);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void print_setup() const;
    void run();
    void step();

    // Evaluate forces/energies/virial at the current positions (rebuilding
    // the neighbor list first if particles moved beyond the skin).
    void compute_forces();

    // Write per-atom forces (tag order) to a text file.
    void write_forces(const std::string& path) const;

    int  current_step() const { return step_; }
    const SimParams& params() const { return params_; }
    float box_length() const { return box_L_; }
    const ParticleData& particles() const { return p_; }

private:
    void upload_force_field();
    void setup_integrator();
    void setup_analysis();
    void build_neighbor_list();
    bool need_rebuild();
    void spatial_sort();
    void launch_forces();
    void integrate_pre_force();
    void integrate_post_force();
    void write_outputs();
    void write_restart(const std::string& fname, int64_t step);
    void print_progress();
    void finish();

    SimParams    params_;
    TopologyData topo_;      // host copy (tag order); never permuted
    ParticleData p_;
    DeviceTopology t_;
    bool opls_mode_ = false;
    bool sort_enabled_ = false;

    // box (cubic). box_L_ is the host mirror of the current box length.
    float box_L_ = 0.0f, inv_L_ = 0.0f;
    float box_L_at_build_ = 0.0f;

    // force field (device)
    float2*             d_lj_params_ = nullptr;
    FENEParams*         d_fene_params_ = nullptr;
    AngleParams*        d_angle_params_ = nullptr;
    BondHarmonicParams* d_hbond_params_ = nullptr;
    OPLSDihedralParams* d_dihedral_params_ = nullptr;
    int*         d_table_idx_ = nullptr;
    TableParams* d_table_params_ = nullptr;
    float4*      d_table_data_ = nullptr;
    int*         d_bond_table_idx_ = nullptr;
    TableParams* d_bond_table_params_ = nullptr;
    float4*      d_bond_table_data_ = nullptr;
    int*         d_angle_table_idx_ = nullptr;
    TableParams* d_angle_table_params_ = nullptr;
    float4*      d_angle_table_data_ = nullptr;
    int3*  d_ewald_kvecs_ = nullptr;
    float* d_ewald_sfac_ = nullptr;
    float* d_virial_ = nullptr;          // [6] xx xy xz yy yz zz

    VerletList   verlet_;
    MortonSorter morton_;
    int* d_max_dr2_int_ = nullptr;
    int  steps_since_rebuild_ = 0;
    int  n_rebuilds_ = 0;

    // integrators
    LangevinState langevin_;
    NoseHooverState nose_hoover_;
    NoseHooverDeviceState* d_nh_state_ = nullptr;
    NoseHooverNPTScratch npt_scratch_ = {};
    float* d_ke_buf_ = nullptr;

    // analysis / output
    MultipleTauCorrelator correlator_;
    ThermoBuffers thermo_bufs_;
    RgBuffers rg_bufs_;
    BinaryDumper dumper_;
    float4* d_out4_ = nullptr;           // tag-ordered scratch for output

    cudaStream_t stream_lj_ = nullptr, stream_bonded_ = nullptr, stream_io_ = nullptr;
    cudaEvent_t force_done_lj_ = nullptr, force_done_bonded_ = nullptr, pos_ready_ = nullptr;

    int step_ = 0;
    std::chrono::steady_clock::time_point t_start_, last_progress_time_;
    int last_progress_step_ = 0, progress_interval_ = 1000;
    double progress_speed_ = 0.0;
};
