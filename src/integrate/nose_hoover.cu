#include "nose_hoover.cuh"
#include "../neighbor/skin_trigger.cuh"
#include "../core/pbc.cuh"

// --- Init / Free ---

void NoseHooverState::init(const SimParams& params) {
    natoms = params.natoms;
    natoms_padded = div_ceil(natoms, TILE_SIZE) * TILE_SIZE;
    M = params.nh_chain_length;
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

    int n_chain = natoms_padded * M;
    CUDA_CHECK(cudaMalloc(&d_xi, n_chain * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_xi, 0, n_chain * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_xi, n_chain * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_xi, 0, n_chain * sizeof(float)));

    float N_f = 3.0f * static_cast<float>(natoms);
    float kT = T_target;
    Q1 = N_f * kT * params.Tdamp * params.Tdamp;
    Q_rest = kT * params.Tdamp * params.Tdamp;

    if (is_npt) {
        W = (N_f + 3.0f) * kT * params.Pdamp * params.Pdamp;
    }
}

void NoseHooverState::free() {
    if (d_xi)   { CUDA_CHECK(cudaFree(d_xi));   d_xi   = nullptr; }
    if (d_v_xi) { CUDA_CHECK(cudaFree(d_v_xi)); d_v_xi = nullptr; }
}

// --- Thermostat chain kernel ---

__global__ void nh_thermostat_half_kernel(
    float4* __restrict__ vel,
    float* __restrict__ d_xi,
    float* __restrict__ d_v_xi,
    int natoms, int M, float half_dt,
    float Q1_inv, float Q_rest_inv, float kT_target)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 v = vel[i];
    float KE_i = 0.5f * (v.x*v.x + v.y*v.y + v.z*v.z);

    float* xi   = d_xi   + i * M;
    float* v_xi = d_v_xi + i * M;

    // Chain element 0: coupled to particle KE
    float G0 = (2.0f * KE_i - kT_target) * Q1_inv;
    v_xi[0] += 0.5f * half_dt * G0;
    xi[0]   += half_dt * v_xi[0];

    // Chain elements 1..M-1
    for (int k = 1; k < M; k++) {
        float prev_Q_inv = (k == 1) ? Q1_inv : Q_rest_inv;
        float Gk = (v_xi[k-1] * v_xi[k-1] * prev_Q_inv - kT_target) * Q_rest_inv;
        v_xi[k] += 0.5f * half_dt * Gk;
        xi[k]   += half_dt * v_xi[k];
    }

    // Recompute forces with updated xi
    G0 = (2.0f * KE_i - kT_target) * Q1_inv;
    v_xi[0] += 0.5f * half_dt * G0;
    for (int k = 1; k < M; k++) {
        float prev_Q_inv = (k == 1) ? Q1_inv : Q_rest_inv;
        float Gk = (v_xi[k-1] * v_xi[k-1] * prev_Q_inv - kT_target) * Q_rest_inv;
        v_xi[k] += 0.5f * half_dt * Gk;
    }

    // Scale velocity by exp(-v_xi_0 * half_dt)
    float scale = expf(-v_xi[0] * half_dt);
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    vel[i] = v;
}

void launch_nh_thermostat_half(float4* vel, float* d_xi, float* d_v_xi,
                                int natoms, int M, float half_dt,
                                float Q1_inv, float Q_rest_inv,
                                float kT_target, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_thermostat_half_kernel<<<blocks, 256, 0, stream>>>(
        vel, d_xi, d_v_xi, natoms, M, half_dt,
        Q1_inv, Q_rest_inv, kT_target);
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
