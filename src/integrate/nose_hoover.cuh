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
    float L = 1.0f;
    float inv_L = 1.0f;
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
};

void allocate_nh_device_state(NoseHooverDeviceState*& d_state);
void init_nh_device_state(const NoseHooverState& host,
                          NoseHooverDeviceState* d_state);
void free_nh_device_state(NoseHooverDeviceState* d_state);
void set_nh_scale_device(NoseHooverDeviceState* d_state, float scale);
void set_nh_targets_device(NoseHooverDeviceState* d_state,
                           float T_target, float P_target);

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
                                    int natoms, float L, float inv_L,
                                    cudaStream_t stream = 0);

// NPT: barostat+thermostat scale + velocity half-step + position update + PBC + displacement
void launch_nh_npt_pre_force_fused(float4* pos, float4* vel, const float4* force,
                                    float4* pos_ref,
                                    int* d_max_dr2_int, int* d_image,
                                    float nh_scale, float baro_scale,
                                    float half_dt, float dt,
                                    float exp_vW, float v_eps_W_dt,
                                    int natoms, float L, float inv_L,
                                    cudaStream_t stream = 0);

// --- Fused post-force kernel ---
// NVT: velocity half-step + KE reduction in a single pass
void launch_nh_nvt_v_half_ke_reduce(float4* vel, const float4* force,
                                     float* ke_out, int natoms, float half_dt,
                                     cudaStream_t stream = 0);

// --- Original (non-fused) kernels, still used by NPT post-force ---
void launch_nh_global_scale_vel(float4* vel,
                                 const NoseHooverDeviceState* d_state,
                                 int natoms, cudaStream_t stream = 0);

void launch_nh_barostat_vel_half(float4* vel, float v_eps_W,
                                  float N_f_inv, int natoms,
                                  float half_dt,
                                  cudaStream_t stream = 0);

void launch_nh_v_verlet_half(float4* vel, const float4* force,
                              int natoms, float half_dt,
                              cudaStream_t stream = 0);

void launch_nh_update_pos(float4* pos, float4* vel, float4* pos_ref,
                           int* d_image, int* d_max_dr2_int,
                           int natoms, float L, float inv_L,
                           float exp_vW_dt, float v_eps_W_dt, float dt,
                           cudaStream_t stream = 0);

// Lightweight KE-only reduction: single kernel + single D2H copy + sync.
float compute_ke_only(const float4* vel, int natoms,
                      float* d_ke_buf, cudaStream_t stream = 0);
