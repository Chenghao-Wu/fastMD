#include "nose_hoover.cuh"
#include "../neighbor/skin_trigger.cuh"
#include "../core/pbc.cuh"

// --- Init / Free ---

void NoseHooverState::init(const SimParams& params) {
    natoms = params.natoms;
    natoms_padded = div_ceil(natoms, TILE_SIZE) * TILE_SIZE;
    M = params.nh_chain_length;
    if (M > kMaxChainLength) M = kMaxChainLength;
    dt = params.dt;
    nsteps = params.nsteps;
    is_npt = (params.ensemble == Ensemble::NPT_NH);

    T_start = params.T_start;
    T_stop  = params.T_stop;
    T_target = T_start;
    P_start = params.P_start;
    P_stop  = params.P_stop;
    P_target = P_start;

    V0 = params.box_L * params.box_L * params.box_L;
    V = V0;
    L = params.box_L;
    inv_L = params.inv_L;

    eps = 0.0f;
    v_eps = 0.0f;

    for (int k = 0; k < kMaxChainLength; k++) {
        xi[k] = 0.0f;
        v_xi[k] = 0.0f;
    }

    float N_f = 3.0f * static_cast<float>(natoms);
    float kT = T_target;
    Q1 = N_f * kT * params.Tdamp * params.Tdamp;
    Q_rest = kT * params.Tdamp * params.Tdamp;

    if (is_npt) {
        W = (N_f + 3.0f) * kT * params.Pdamp * params.Pdamp;
    }
}

void NoseHooverState::free() {
    // No device allocations for the chain (host-side)
}

// --- System-wide NH chain propagation (host-side) ---
// Matches the LAMMPS fix_nh.cpp algorithm:
// backward-recursion (top-down) → velocity scaling → forward-recursion (bottom-up)
// with analytic temperature update after velocity scaling.

void nh_propagate_chain(NoseHooverState& nh, float total_KE,
                         float half_dt, float& scale_out) {
    int M = nh.M;
    float Q1 = nh.Q1;
    float kT_target = nh.T_target;
    float N_f = 3.0f * static_cast<float>(nh.natoms);

    // LAMMPS uses dt8/dt4/dthalf; we simplify to half_dt based factors
    float dt2 = half_dt;
    float dt4 = half_dt * 0.5f;
    float dt8 = half_dt * 0.25f;

    // v_xi_dotdot[k] = acceleration of chain element k (Gk in our old notation)
    float v_xi_dotdot[10];

    // Current temperature from KE: T = 2*KE / (N_f)
    float T_current = (2.0f * total_KE) / N_f;

    // ke = N_f * kT, ke_target = N_f * kT_target
    // dotdot[0] = (ke_current - ke_target) / Q1
    //           = (N_f*T_current - N_f*T_target) / (N_f*kT_target*Tdamp^2)
    //           = (T_current - T_target) / (kT_target * Tdamp^2)
    if (Q1 > 0.0f) {
        float kecurrent = N_f * T_current;
        float ke_target = N_f * kT_target;
        v_xi_dotdot[0] = (kecurrent - ke_target) / Q1;
    } else {
        v_xi_dotdot[0] = 0.0f;
    }

    int nc_tchain = 1;    // number of SY loops (LAMMPS default)
    float ncfac = 1.0f / static_cast<float>(nc_tchain);

    for (int iloop = 0; iloop < nc_tchain; iloop++) {

        // --- Backward recursion: ich = M-1 down to 1 ---
        for (int ich = M - 1; ich > 0; ich--) {
            // expfac = exp(-dt8 * v_xi[ich+1]) — coupling from element above
            // v_xi[ich+1] = 0 for ich+1 = M (out of bounds), so expfac ≈ 1 for top element
            float expfac = (ich + 1 < M)
                ? expf(-ncfac * dt8 * nh.v_xi[ich + 1])
                : 1.0f;

            nh.v_xi[ich] *= expfac;
            nh.v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            nh.v_xi[ich] *= expfac;
        }

        // --- k = 0 ---
        float expfac0 = (M > 1)
            ? expf(-ncfac * dt8 * nh.v_xi[1])
            : 1.0f;

        nh.v_xi[0] *= expfac0;
        nh.v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        nh.v_xi[0] *= expfac0;

        // --- Velocity scaling ---
        float factor_eta = expf(-ncfac * dt2 * nh.v_xi[0]);
        scale_out = factor_eta;  // applied to GPU velocities separately

        // --- Analytic temperature update ---
        // T_new = T_old * factor_eta^2 (since KE ∝ v^2)
        T_current *= factor_eta * factor_eta;

        // Recompute dotdot[0] with updated temperature
        if (Q1 > 0.0f) {
            float kecurrent = N_f * T_current;
            float ke_target = N_f * kT_target;
            v_xi_dotdot[0] = (kecurrent - ke_target) / Q1;
        }

        // --- Update chain positions ---
        for (int ich = 0; ich < M; ich++) {
            nh.xi[ich] += ncfac * dt2 * nh.v_xi[ich];
        }

        // --- Forward recursion: k=0 ---
        nh.v_xi[0] *= expfac0;
        nh.v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        nh.v_xi[0] *= expfac0;

        // --- Forward recursion: ich = 1 to M-1 ---
        for (int ich = 1; ich < M; ich++) {
            float expfac = (ich + 1 < M)
                ? expf(-ncfac * dt8 * nh.v_xi[ich + 1])
                : 1.0f;

            nh.v_xi[ich] *= expfac;

            // Recompute dotdot[ich] from element below: Q_{ich-1} * v_{ich-1}^2 - kT
            float Q_prev = (ich == 1) ? Q1 : nh.Q_rest;
            v_xi_dotdot[ich] = (Q_prev * nh.v_xi[ich - 1] * nh.v_xi[ich - 1]
                               - kT_target) / nh.Q_rest;

            nh.v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            nh.v_xi[ich] *= expfac;
        }
    }
}

// --- Global velocity scaling kernel ---

__global__ void nh_global_scale_vel_kernel(
    float4* __restrict__ vel,
    float scale, int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 v = vel[i];
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    vel[i] = v;
}

void launch_nh_global_scale_vel(float4* vel, float scale,
                                 int natoms, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_global_scale_vel_kernel<<<blocks, 256, 0, stream>>>(
        vel, scale, natoms);
}

// --- Barostat velocity rescale kernel ---

__global__ void nh_barostat_vel_half_kernel(
    float4* __restrict__ vel,
    float v_eps_W, float N_f_inv, int natoms, float half_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float factor = (1.0f + 3.0f * N_f_inv) * v_eps_W * half_dt;
    float scale = expf(-factor);

    float4 v = vel[i];
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    vel[i] = v;
}

void launch_nh_barostat_vel_half(float4* vel, float v_eps_W,
                                  float N_f_inv, int natoms,
                                  float half_dt, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_barostat_vel_half_kernel<<<blocks, 256, 0, stream>>>(
        vel, v_eps_W, N_f_inv, natoms, half_dt);
}

// --- Velocity Verlet half-step ---

__global__ void nh_v_verlet_half_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    int natoms, float half_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 v = vel[i];
    float4 f = force[i];
    v.x += half_dt * f.x;
    v.y += half_dt * f.y;
    v.z += half_dt * f.z;
    vel[i] = v;
}

void launch_nh_v_verlet_half(float4* vel, const float4* force,
                              int natoms, float half_dt,
                              cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_v_verlet_half_kernel<<<blocks, 256, 0, stream>>>(
        vel, force, natoms, half_dt);
}

// --- Position update with barostat scaling ---

__device__ inline float sinchf(float x) {
    if (fabsf(x) < 1e-6f) return 1.0f + x * x / 6.0f;
    return sinhf(x) / x;
}

__global__ void nh_update_pos_kernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    float4* __restrict__ pos_ref,
    int* __restrict__ d_image,
    int* __restrict__ d_max_dr2_int,
    int natoms, float L, float inv_L,
    float exp_vW_dt, float v_eps_W_dt, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 r = pos[i];
    float4 v = vel[i];
    float4 r_ref = pos_ref[i];

    float half_vW_dt = 0.5f * v_eps_W_dt;
    float f = dt * expf(half_vW_dt) * sinchf(half_vW_dt);

    r.x = r.x * exp_vW_dt + v.x * f;
    r.y = r.y * exp_vW_dt + v.y * f;
    r.z = r.z * exp_vW_dt + v.z * f;

    r_ref.x *= exp_vW_dt;
    r_ref.y *= exp_vW_dt;
    r_ref.z *= exp_vW_dt;

    // PBC wrapping
    int ix = (int)floorf(r.x * inv_L);
    int iy = (int)floorf(r.y * inv_L);
    int iz = (int)floorf(r.z * inv_L);
    r.x -= ix * L;
    r.y -= iy * L;
    r.z -= iz * L;
    if (d_image != nullptr) {
        int i3 = i * 3;
        d_image[i3 + 0] += ix;
        d_image[i3 + 1] += iy;
        d_image[i3 + 2] += iz;
    }

    update_max_displacement(r, r_ref, d_max_dr2_int, L, inv_L);

    pos[i]     = r;
    pos_ref[i] = r_ref;
}

void launch_nh_update_pos(float4* pos, float4* vel, float4* pos_ref,
                           int* d_image, int* d_max_dr2_int,
                           int natoms, float L, float inv_L,
                           float exp_vW_dt, float v_eps_W_dt, float dt,
                           cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_update_pos_kernel<<<blocks, 256, 0, stream>>>(
        pos, vel, pos_ref, d_image, d_max_dr2_int,
        natoms, L, inv_L, exp_vW_dt, v_eps_W_dt, dt);
}
