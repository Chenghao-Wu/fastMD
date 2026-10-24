#pragma once
#include "../core/types.cuh"

static constexpr int kMaxChainLength = 10;

struct NoseHooverState {
    // Thermostat chain — host-side, single system-wide chain
    float xi[kMaxChainLength] = {};
    float v_xi[kMaxChainLength] = {};

    // Barostat — host
    float eps = 0.0f;     // log strain: V = V0 * exp(3*eps)
    float v_eps = 0.0f;   // barostat momentum

    // Mass parameters (host)
    float Q1 = 1.0f;
    float Q_rest = 1.0f;
    float W = 1.0f;

    // Target values (updated each step for ramping)
    float T_target = 1.0f;
    float P_target = 0.0f;
    float T_start = 1.0f;
    float T_stop = 1.0f;
    float P_start = 0.0f;
    float P_stop = 0.0f;

    // Geometry
    float V0 = 1.0f;
    float V = 1.0f;
    Box   box0 = cubic_box(1.0f);   // box at the start (the barostat scales it isotropically)
    int   M = 3;

    int   natoms = 0;
    int   natoms_padded = 0;
    float dt = 0.001f;
    int   nsteps = 0;
    bool  is_npt = false;

    void init(const SimParams& params);
    void free();
};

// Device-resident NH chain state — eliminates D2H/H2D round-trips in the inner loop.
// Persists across time steps. Host State retains geometry/mass params only.
struct NoseHooverDeviceState {
    float xi[kMaxChainLength];
    float v_xi[kMaxChainLength];
    float eps;               // log strain (NPT)
    float v_eps;             // barostat momentum (NPT)
    float chain_KE_carry;    // KE * scale^2, carried to next step's pre-force
    float nh_scale;          // latest thermostat scale, consumed by next kernel
    float T_target;
    float P_target;
    // Linear ramps evaluated on the device (no per-step host transfer):
    // target(step) = start + (stop - start) * (step - 1) / nsteps. The
    // pre-force chain kernel updates the targets, the post-force one
    // advances step.
    float T_start, T_stop, P_start, P_stop;
    int   nsteps;
    int   step;
    // Fused NVT path (nh_nvt_post_force_chain): pending factors and the
    // grid-wide kinetic-energy reduction state.
    float s_post, s_pre;
    double ke_accum;
    unsigned int blocks_done;
};

void allocate_nh_device_state(NoseHooverDeviceState*& d_state);
void init_nh_device_state(const NoseHooverState& host,
                          NoseHooverDeviceState* d_state);
void free_nh_device_state(NoseHooverDeviceState* d_state);
void set_nh_scale_device(NoseHooverDeviceState* d_state, float scale);

// Propagate system-wide NH chain on GPU.
// If use_carry: reads KE from d_state->chain_KE_carry (pre-force call).
// If !use_carry: reads KE from d_ke_buf (post-force call), and writes
//   d_state->chain_KE_carry = KE * scale^2 for the next step's pre-force.
// Always writes d_state->nh_scale, updates xi/v_xi.
// Launch with <<<1, 32>>>.
__global__ void nh_propagate_chain_kernel(
    NoseHooverDeviceState* __restrict__ d_state,
    const float* __restrict__ d_ke_buf,
    bool use_carry,
    int M, float Q1, float Q_rest,
    float half_dt, int natoms);

// Fused NVT post-force step (kick + KE reduction + both chain half-steps in
// the last block); see nose_hoover.cu. Leaves s_post * s_pre in nh_scale.
void launch_nh_nvt_post_force_chain(float4* vel, const float4* force, const float4* pos,
                                    NoseHooverDeviceState* d_state, int natoms, float half_dt,
                                    int M, float Q1, float Q_rest, cudaStream_t stream = 0);

// Apply the pending post-force scale to the velocities (before output).
void launch_nh_flush_post_scale(float4* vel, NoseHooverDeviceState* d_state, int natoms,
                                cudaStream_t stream = 0);

// --- Kernel launches ---

// Propagate system-wide NH chain for half_dt, return global velocity scale factor.
// total_KE is the total kinetic energy of the system.
void nh_propagate_chain(NoseHooverState& nh, float total_KE,
                         float half_dt, float& scale_out);

// --- Fused pre-force kernels ---
// NVT: thermostat scale + velocity half-step + position update + PBC + displacement
void launch_nh_nvt_pre_force_fused(float4* pos, float4* vel, const float4* force,
                                    const float4* pos_ref,
                                    int* d_max_dr2_int, int* d_image,
                                    const NoseHooverDeviceState* d_state,
                                    float half_dt, float dt,
                                    int natoms, const Box& box,
                                    cudaStream_t stream = 0);

// NPT: barostat+thermostat scale + velocity half-step + position update + PBC + displacement.
// The box is read on the device.
void launch_nh_npt_pre_force_fused(float4* pos, float4* vel, const float4* force,
                                    float4* pos_ref,
                                    int* d_max_dr2_int, int* d_image,
                                    const NoseHooverDeviceState* d_state,
                                    const float* d_baro_scale,
                                    const float* d_exp_vW,
                                    const float* d_v_eps_W_dt,
                                    float half_dt, float dt,
                                    int natoms, const Box* d_box,
                                    cudaStream_t stream = 0);

// --- NPT scratch buffers ---
// Holds per-step outputs from the NPT chain+barostat kernel that are
// consumed by subsequent velocity/position kernels in the same step.
// d_V / d_box are owned by the caller (the device box state).
struct NoseHooverNPTScratch {
    float* d_V;
    Box*   d_box;
    float* d_baro_scale;
    float* d_exp_vW;
    float* d_v_eps_W;
    float* d_v_eps_W_dt;
    float* d_virial_trace;
};

void allocate_npt_scratch(NoseHooverNPTScratch& s, float* d_box_V, Box* d_box);
void free_npt_scratch(NoseHooverNPTScratch& s);

// Lightweight kernel: compute KE (trace of v^2 tensor) into d_ke_out
// and sum virial trace from d_virial (6-component) into d_virial_trace_out.
// Replaces the heavy compute_thermo in the NPT inner loop.
void launch_nh_npt_ke_and_virial_trace(
    const float4* vel, const double* virial,
    const float4* pos,
    float* d_ke_out, float* d_virial_trace_out,
    int natoms, cudaStream_t stream = 0);

// NPT barostat half kick + NH chain half step on GPU (symmetric MTK splitting, LAMMPS fix_nh order).
// Reads KE from d_ke_buf and the virial trace from d_virial_trace; one thread updates the state.
// If use_carry: pre-force call (thermostat half step, barostat half kick, box drift over the full step;
//   writes the box and the strain-rate factors for the fused pre-force kernel).
// If !use_carry: post-force call (KE after the kick and the barostat velocity scaling; barostat half kick,
//   thermostat half step).
__global__ void nh_npt_chain_baro_kernel(
    NoseHooverDeviceState* __restrict__ d_state,
    const float* __restrict__ d_ke_buf,
    const float* __restrict__ d_virial_trace,
    bool use_carry,
    int M, float Q1, float Q_rest, float W,
    float half_dt, float dt,
    float V0, Box box0, int natoms,
    float* d_V, Box* d_box,
    float* d_baro_scale, float* d_exp_vW, float* d_v_eps_W, float* d_v_eps_W_dt);

// --- Fused post-force kernel ---
// NVT: velocity half-step + KE reduction in a single pass
void launch_nh_nvt_v_half_ke_reduce(float4* vel, const float4* force,
                                     const float4* pos,
                                     float* ke_out, int natoms, float half_dt,
                                     cudaStream_t stream = 0);

// --- Original (non-fused) kernels, still used by NPT post-force ---
void launch_nh_global_scale_vel(float4* vel,
                                 const NoseHooverDeviceState* d_state,
                                 int natoms, cudaStream_t stream = 0);

void launch_nh_barostat_vel_half(float4* vel, const float* d_v_eps_W,
                                  float N_f_inv, int natoms,
                                  float half_dt,
                                  cudaStream_t stream = 0);

void launch_nh_v_verlet_half(float4* vel, const float4* force,
                              const float4* pos,
                              int natoms, float half_dt,
                              cudaStream_t stream = 0);

void launch_nh_update_pos(float4* pos, float4* vel, float4* pos_ref,
                           int* d_image, int* d_max_dr2_int,
                           int natoms, const Box& box,
                           float exp_vW_dt, float v_eps_W_dt, float dt,
                           cudaStream_t stream = 0);

// Lightweight KE-only reduction: single kernel + single D2H copy + sync.
float compute_ke_only(const float4* vel, const float4* pos, int natoms,
                      float* d_ke_buf, cudaStream_t stream = 0);
