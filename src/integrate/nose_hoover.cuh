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
                                    float nh_scale, float half_dt, float dt,
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
void launch_nh_global_scale_vel(float4* vel, float scale,
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
