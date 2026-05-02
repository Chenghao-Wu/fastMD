#pragma once
#include "../core/types.cuh"

struct NoseHooverState {
    // Thermostat chain — device: natoms_padded * M floats each
    float* d_xi = nullptr;
    float* d_v_xi = nullptr;

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

void launch_nh_thermostat_half(float4* vel, float* d_xi, float* d_v_xi,
                                int natoms, int M, float half_dt,
                                float Q1_inv, float Q_rest_inv,
                                float kT_target,
                                cudaStream_t stream = 0);

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
