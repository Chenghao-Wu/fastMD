#pragma once
#include "../core/types.cuh"
#include "../core/particles.cuh"
#include "../core/topology.cuh"
#include "../neighbor/verlet_list.cuh"
#include "../force/fene.cuh"
#include "../force/angle.cuh"
#include "../force/bond_harmonic.cuh"
#include "../force/dihedral_opls.cuh"
#include "../force/bonded_gather.cuh"
#include "../force/pme.cuh"
#include "../integrate/langevin.cuh"
#include "../integrate/nose_hoover.cuh"
#include "../analysis/thermo.cuh"
#include "../analysis/correlator.cuh"
#include "../analysis/rg.cuh"
#include "../io/dump.cuh"
#include "../io/config.hpp"
#include "../io/trajectory.hpp"
#include <chrono>
#include <string>
#include <vector>

// One MD simulation: owns all device state and drives the time loop.
//
// Construction uploads the system, builds the neighbor list and evaluates the
// initial forces. run() is the command-line driver: it advances
// params.nsteps steps with progress output, then writes the final restart and
// the stress correlation. run(n) advances n steps (writing the configured
// periodic outputs) and can be called repeatedly, interleaved with reading or
// replacing the state (the Python binding).
//
// State accessors work in tag order (the order of the input data file), with
// positions wrapped into [0, L) (the data-file box shifted to the origin);
// unwrapped positions are pos + image * L.
class Simulation {
public:
    Simulation(const SimParams& params, TopologyData topo);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void print_setup() const;
    void run();
    void run(int nsteps);
    void step() { advance_one(true); }   // one step; energies and virial valid afterwards

    // Evaluate forces/energies/virial at the current positions (rebuilding
    // the neighbor list first if particles moved beyond the skin).
    void compute_forces();
    // Synchronous full neighbor-list rebuild (optionally re-sorting particles).
    void rebuild_neighbor_list(bool sort = false) { rebuild_neighbors(sort); }

    // Write per-atom forces (tag order) to a text file.
    void write_forces(const std::string& path) const;

    // Current state (synchronous). velocities() first applies a pending
    // thermostat scale, so it returns the full-step velocities.
    std::vector<float4> positions() const { return p_.download_positions(); }   // w: type bits
    std::vector<int>    images() const { return p_.download_images(); }         // [3 N]
    std::vector<float4> velocities();
    std::vector<float4> forces() const { return p_.download_forces(); }         // w: per-atom PE
    std::vector<float>  charges() const { return p_.download_charges(); }
    // Replace positions (xyz in tag order; wrapped into the box; image counts
    // reset to 0 unless given, [3 N]) and recompute the neighbor list and forces.
    void set_positions(const std::vector<float3>& xyz, const std::vector<int>* images = nullptr);
    void set_velocities(const std::vector<float3>& v);
    // Sheared box: set the xy tilt (affine: particles follow the box, a step
    // strain; otherwise unwrapped positions are kept) / apply a step shear
    // strain gamma (x += gamma y, tilt += gamma Ly).
    void set_tilt(float xy, bool affine);
    void apply_step_shear(float gamma);
    // Depth-dependent Langevin friction / Andersen collision rate factor
    // f(z - z_c) (see integrate/langevin.cuh); mode 0 removes the profile.
    // table: f on the uniform grid zmin + k dz (mode 2).
    void set_friction_profile(const FrictionProfile& spec, const std::vector<float>& table = {},
                              float zmin = 0.0f, float dz = 0.0f);
    // Replace the walls (e.g. remove them after confined preparation).
    void set_walls(const WallSet& walls);
    const WallSet& walls() const { return params_.walls; }
    // Film reference plane (mass-weighted mean unwrapped coordinate along
    // the film normal); requires `film <axis>`.
    float film_center();
    // Kinetic/potential energy, temperature and pressure tensor (xx xy xz yy
    // yz zz) of the current state.
    ThermoOutput thermo();

    int  current_step() const { return step_; }
    const SimParams& params() const { return params_; }
    const Box& box() const { return box_; }
    const ParticleData& particles() const { return p_; }
    int rollbacks() const { return n_rollbacks_; }
    int neighbor_capacity() const { return verlet_.max_neighbors; }

private:
    void upload_force_field();
    void setup_integrator();
    void setup_analysis();
    void rebuild_neighbors(bool sort);   // synchronous, host-driven
    bool output_due(int step) const;
    void advance(int nsteps, bool progress);
    void save_checkpoint();
    void restore_checkpoint();
    bool validate();                     // neighbor/grid flags since the checkpoint
    // ev: also per-atom energies (force.w) and the virial; without it only
    // forces are valid (the pair and bonded kernels skip the rest)
    void launch_forces(bool ev);
    void launch_pair_kernel(int tpa, bool ev);   // the overwriting nonbonded kernel
    void tune_pair_tpa();
    void launch_bonded_kernel(int tpa, bool ev);
    void tune_bonded_tpa();
    void integrate_pre_force();
    void integrate_post_force();
    void launch_step(bool with_stress, bool ev);   // all device work of one step
    void advance_one(bool want_ev);
    void invalidate_graphs();
    void write_outputs();
    void flush_velocities();     // apply a pending thermostat scale (fused NVT)
    void write_restart(const std::string& fname, int64_t step);
    void print_progress();
    void finish();

    SimParams    params_;
    TopologyData topo_;      // host copy (tag order); never permuted
    ParticleData p_;
    DeviceTopology t_;
    bool opls_mode_ = false;

    // host mirror of the current box (the NPT barostat changes it on the device)
    Box box_ = cubic_box(1.0f);

    // force field (device)
    float2*             d_lj_params_ = nullptr;
    float2*             d_lj_cap_ = nullptr;     // (r_cap, U(r_cap)) when lj_force_cap > 0
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
    PME    pme_;
    int3*  d_ewald_kvecs_ = nullptr;
    float* d_ewald_sfac_ = nullptr;
    double* d_virial_ = nullptr;         // [6] xx xy xz yy yz zz
    BondedParams bonded_params_;

    VerletList   verlet_;
    int  n_sorts_ = 0;
    int  pair_tpa_ = 1;
    int  bonded_tpa_ = 1;

    // Small device state read back once per step (one 16-byte copy):
    // maximum squared displacement since the last build (float bits, written
    // with atomicMax) and the current box. The NPT barostat writes the box
    // here; integrators read it on the device.
    struct DeviceStatus {
        int   max_dr2_bits = 0;
        float V = 0.0f;
        Box   box = cubic_box(1.0f);
        int   wall_error = 0;   // k + 1: a particle reached wall k's plane
        float film_center = 0.0f;         // film reference plane (unwrapped)
        unsigned int film_ticket = 0;     // scratch of the film-center reduction
        double film_accum = 0.0;
    };
    // device counter of the current step (ramps), owned by the active integrator
    const int* current_step_ptr() const {
        return is_nose_hoover(params_.ensemble) ? &d_nh_state_->step : d_step_;
    }
    DeviceStatus* d_status_ = nullptr;
    DeviceStatus* h_status_ = nullptr;   // pinned
    int* d_step_ = nullptr;              // device step counter (Langevin; NH keeps its own)

    // integrators
    LangevinState langevin_;
    NoseHooverState nose_hoover_;
    NoseHooverDeviceState* d_nh_state_ = nullptr;
    NoseHooverNPTScratch npt_scratch_ = {};
    float* d_ke_buf_ = nullptr;
    bool velocities_pending_ = false;    // fused NVT: s_post not yet applied

    // checkpoint for rolling back device-side neighbor-list overflow/grid errors
    ParticleData ckpt_p_;
    NoseHooverDeviceState* d_ckpt_nh_ = nullptr;
    void* d_ckpt_status_ = nullptr;
    int*  d_ckpt_step_ = nullptr;
    int   ckpt_step_ = 0;
    bool  ckpt_vel_pending_ = false;
    int   check_every_ = 100;
    int   last_sort_step_ = 0;
    int   n_rollbacks_ = 0;

    // analysis / output
    StressSampler stress_sampler_;
    MultipleTauCorrelator correlator_;
    ThermoBuffers thermo_bufs_;
    RgBuffers rg_bufs_;
    BinaryDumper dumper_;
    TrajectorySchedule traj_sched_;
    TrajectoryWriter traj_;
    long long last_traj_step_ = -1;
    void write_traj_frame_if_due();
    double inv_total_mass_ = 0.0;
    float* d_friction_table_ = nullptr;
    void launch_film_center();
    void check_tilt_allowed(float xy) const;
    float4* d_out4_ = nullptr;           // tag-ordered scratch for output

    // All device work is ordered on one stream; host waits only for the
    // per-step status read-back and for output.
    cudaStream_t stream_ = nullptr;
    // captured single steps: [0] forces only, [1] with energies/virial,
    // [2] with energies/virial and a stress sample
    cudaGraphExec_t graph_exec_[3] = {nullptr, nullptr, nullptr};
    int n_graph_captures_ = 0;
    void read_status();

    int step_ = 0;
    std::chrono::steady_clock::time_point t_start_, last_progress_time_;
    int last_progress_step_ = 0, progress_interval_ = 1000;
    double progress_speed_ = 0.0;
};
