#include "nose_hoover.cuh"
#include "../neighbor/skin_trigger.cuh"
#include "../core/pbc.cuh"
#include "../core/system.cuh"
#include "../core/reduce.cuh"

void allocate_nh_device_state(NoseHooverDeviceState*& d_state) {
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(NoseHooverDeviceState)));
}

void init_nh_device_state(const NoseHooverState& host,
                          NoseHooverDeviceState* d_state) {
    NoseHooverDeviceState h;
    for (int k = 0; k < kMaxChainLength; k++) {
        h.xi[k] = host.xi[k];
        h.v_xi[k] = host.v_xi[k];
    }
    h.eps = host.eps;
    h.v_eps = host.v_eps;
    h.chain_KE_carry = 0.0f;
    h.nh_scale = 1.0f;
    h.T_target = host.T_target;
    h.P_target = host.P_target;
    h.T_start = host.T_start;
    h.T_stop = host.T_stop;
    h.P_start = host.P_start;
    h.P_stop = host.P_stop;
    h.nsteps = host.nsteps;
    h.step = 1;
    h.s_post = 1.0f;
    h.s_pre = 1.0f;
    h.ke_accum = 0.0;
    h.blocks_done = 0u;
    CUDA_CHECK(cudaMemcpy(d_state, &h, sizeof(NoseHooverDeviceState),
                          cudaMemcpyHostToDevice));
}

// Ramp targets for the current step (called by the pre-force chain kernels).
__device__ inline void nh_update_targets(NoseHooverDeviceState* s) {
    float frac = s->nsteps > 0 ? (float)(s->step - 1) / (float)s->nsteps : 0.0f;
    s->T_target = s->T_start + (s->T_stop - s->T_start) * frac;
    s->P_target = s->P_start + (s->P_stop - s->P_start) * frac;
}

void free_nh_device_state(NoseHooverDeviceState* d_state) {
    CUDA_CHECK(cudaFree(d_state));
}

void set_nh_scale_device(NoseHooverDeviceState* d_state, float scale) {
    CUDA_CHECK(cudaMemcpy(&d_state->nh_scale, &scale, sizeof(float),
                          cudaMemcpyHostToDevice));
}

void allocate_npt_scratch(NoseHooverNPTScratch& s, float* d_box_V, float* d_box_L,
                          float* d_box_inv_L) {
    s.d_V = d_box_V;
    s.d_L = d_box_L;
    s.d_inv_L = d_box_inv_L;
    CUDA_CHECK(cudaMalloc(&s.d_baro_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_exp_vW, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_v_eps_W, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_v_eps_W_dt, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_virial_trace, sizeof(float)));
}

void free_npt_scratch(NoseHooverNPTScratch& s) {
    CUDA_CHECK(cudaFree(s.d_baro_scale));
    CUDA_CHECK(cudaFree(s.d_exp_vW));
    CUDA_CHECK(cudaFree(s.d_v_eps_W));
    CUDA_CHECK(cudaFree(s.d_v_eps_W_dt));
    CUDA_CHECK(cudaFree(s.d_virial_trace));
    s = NoseHooverNPTScratch{};
}

// Bit-identical exp approximation for NH chain.
// Arguments are always small negated products (~ -1e-6 to -1e-2), so a 6th-order
// Taylor series around 0 gives ~1e-7 relative error — more than adequate.
__host__ __device__ inline float nh_expf(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x3 * x;
    float x5 = x4 * x;
    float x6 = x5 * x;
    return 1.0f + x
         + x2 * 0.5f
         + x3 * (1.0f / 6.0f)
         + x4 * (1.0f / 24.0f)
         + x5 * (1.0f / 120.0f)
         + x6 * (1.0f / 720.0f);
}

// One half-step of the system-wide NH chain (LAMMPS fix_nh ordering).
// use_carry: pre-force call, KE taken from chain_KE_carry and the ramp
// targets refreshed for the current step; otherwise post-force call with the
// given kinetic energy, which also carries KE*scale^2 forward and ends the
// step. Returns the velocity scale factor (also stored in nh_scale).
__device__ float nh_chain_update(NoseHooverDeviceState* __restrict__ d_state,
                                 float ke_now, bool use_carry,
                                 int M, float Q1, float Q_rest,
                                 float half_dt, int natoms)
{
    if (use_carry) nh_update_targets(d_state);
    float total_KE = use_carry ? d_state->chain_KE_carry : ke_now;
    float kT_target = d_state->T_target;
    float N_f = 3.0f * static_cast<float>(natoms);

    float dt2 = half_dt;
    float dt4 = half_dt * 0.5f;
    float dt8 = half_dt * 0.25f;

    float v_xi_dotdot[10] = {};

    float T_current = (2.0f * total_KE) / N_f;

    if (Q1 > 0.0f) {
        float kecurrent = N_f * T_current;
        float ke_target = N_f * kT_target;
        v_xi_dotdot[0] = (kecurrent - ke_target) / Q1;
    }

    int nc_tchain = 1;
    float ncfac = 1.0f / static_cast<float>(nc_tchain);
    float scale = 1.0f;

    for (int iloop = 0; iloop < nc_tchain; iloop++) {

        // Backward recursion: ich = M-1 down to 1
        for (int ich = M - 1; ich > 0; ich--) {
            float expfac = (ich + 1 < M)
                ? nh_expf(-ncfac * dt8 * d_state->v_xi[ich + 1])
                : 1.0f;
            d_state->v_xi[ich] *= expfac;
            d_state->v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            d_state->v_xi[ich] *= expfac;
        }

        // k = 0
        float expfac0 = (M > 1)
            ? nh_expf(-ncfac * dt8 * d_state->v_xi[1])
            : 1.0f;
        d_state->v_xi[0] *= expfac0;
        d_state->v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        d_state->v_xi[0] *= expfac0;

        // Velocity scaling factor
        float factor_eta = nh_expf(-ncfac * dt2 * d_state->v_xi[0]);
        scale = factor_eta;

        // Analytic temperature update
        T_current *= factor_eta * factor_eta;

        if (Q1 > 0.0f) {
            float kecurrent = N_f * T_current;
            float ke_target = N_f * kT_target;
            v_xi_dotdot[0] = (kecurrent - ke_target) / Q1;
        }

        // Update chain positions
        for (int ich = 0; ich < M; ich++) {
            d_state->xi[ich] += ncfac * dt2 * d_state->v_xi[ich];
        }

        // Forward recursion: k = 0
        d_state->v_xi[0] *= expfac0;
        d_state->v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        d_state->v_xi[0] *= expfac0;

        // Forward recursion: ich = 1 to M-1
        for (int ich = 1; ich < M; ich++) {
            float expfac = (ich + 1 < M)
                ? nh_expf(-ncfac * dt8 * d_state->v_xi[ich + 1])
                : 1.0f;
            d_state->v_xi[ich] *= expfac;
            float Q_prev = (ich == 1) ? Q1 : Q_rest;
            v_xi_dotdot[ich] = (Q_prev * d_state->v_xi[ich - 1] * d_state->v_xi[ich - 1]
                                - kT_target) / Q_rest;
            d_state->v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            d_state->v_xi[ich] *= expfac;
        }
    }

    d_state->nh_scale = scale;

    // Carry KE forward only for post-force calls (use_carry=false); the
    // post-force call ends the step.
    if (!use_carry) {
        d_state->chain_KE_carry = total_KE * scale * scale;
        d_state->step++;
    }
    return scale;
}

__global__ void nh_propagate_chain_kernel(
    NoseHooverDeviceState* __restrict__ d_state,
    const float* __restrict__ d_ke_buf,
    bool use_carry,
    int M, float Q1, float Q_rest,
    float half_dt, int natoms)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        nh_chain_update(d_state, use_carry ? 0.0f : (*d_ke_buf * 0.5f), use_carry,
                        M, Q1, Q_rest, half_dt, natoms);
}

// Fused NVT post-force: velocity half-kick + kinetic-energy reduction; the
// last block to finish runs the post-force chain half-step of this step and
// the pre-force half-step of the next one. The combined scale
// s_post * s_pre is left in nh_scale for the next pre-force kernel, which
// applies it; s_post alone is applied by nh_flush_post_scale when velocities
// are needed at the end of the step (output).
__global__ void nh_nvt_post_force_chain_kernel(
    float4* __restrict__ vel, const float4* __restrict__ force,
    const float4* __restrict__ pos,
    NoseHooverDeviceState* __restrict__ d_state,
    int natoms, float half_dt, int M, float Q1, float Q_rest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v2[1] = {0.0};
    if (i < natoms) {
        float4 v = vel[i];
        float4 f = force[i];
        float m_i = c_masses[unpack_type_id(pos[i].w)];
        float h = half_dt / m_i;
        v.x += h * f.x; v.y += h * f.y; v.z += h * f.z;
        vel[i] = v;
        v2[0] = (double)m_i * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    block_reduce_atomic_add<1, double>(v2, &d_state->ke_accum);

    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        __threadfence();   // make this block's contribution visible first
        unsigned int done = atomicAdd(&d_state->blocks_done, 1u);
        is_last = (done == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        __threadfence();
        double sum = atomicAdd(&d_state->ke_accum, 0.0);   // coherent read
        float s_post = nh_chain_update(d_state, (float)(0.5 * sum), false,
                                       M, Q1, Q_rest, half_dt, natoms);
        float s_pre = nh_chain_update(d_state, 0.0f, true, M, Q1, Q_rest, half_dt, natoms);
        d_state->s_post = s_post;
        d_state->s_pre = s_pre;
        d_state->nh_scale = s_post * s_pre;
        d_state->ke_accum = 0.0;
        d_state->blocks_done = 0u;
    }
}

// Apply the pending post-force thermostat scale (end-of-step velocities) and
// leave only the next step's pre-force factor in nh_scale.
__global__ void nh_flush_post_scale_kernel(float4* __restrict__ vel,
                                           NoseHooverDeviceState* __restrict__ d_state,
                                           int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = d_state->s_post;
    if (i < natoms) {
        float4 v = vel[i];
        v.x *= s; v.y *= s; v.z *= s;
        vel[i] = v;
    }
    if (i == 0) d_state->nh_scale = d_state->s_pre;   // nobody else reads nh_scale here
}

void launch_nh_nvt_post_force_chain(float4* vel, const float4* force, const float4* pos,
                                    NoseHooverDeviceState* d_state, int natoms, float half_dt,
                                    int M, float Q1, float Q_rest, cudaStream_t stream) {
    nh_nvt_post_force_chain_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(
        vel, force, pos, d_state, natoms, half_dt, M, Q1, Q_rest);
}

void launch_nh_flush_post_scale(float4* vel, NoseHooverDeviceState* d_state, int natoms,
                                cudaStream_t stream) {
    nh_flush_post_scale_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(vel, d_state, natoms);
}


// --- NPT chain + barostat GPU kernel ---
// Combined barostat Suzuki-Yoshida half-step + NH chain propagation.
// Pre-force (use_carry=true): reads chain_KE_carry, updates eps/v_eps/V/L/inv_L.
// Post-force (use_carry=false): reads raw KE from d_ke_buf, second barostat half-step.

__global__ void nh_npt_chain_baro_kernel(
    NoseHooverDeviceState* __restrict__ d_state,
    const float* __restrict__ d_ke_buf,
    const float* __restrict__ d_virial_trace,
    bool use_carry,
    int M, float Q1, float Q_rest, float W,
    float half_dt, float dt,
    float V0, int natoms,
    float* d_V, float* d_L, float* d_inv_L,
    float* d_baro_scale, float* d_exp_vW, float* d_v_eps_W, float* d_v_eps_W_dt)
{
    if (use_carry) nh_update_targets(d_state);
    float total_KE = use_carry ? d_state->chain_KE_carry : (*d_ke_buf * 0.5f);
    float virial_trace_raw = *d_virial_trace;
    float N_f = 3.0f * static_cast<float>(natoms);
    float N_f_inv = 1.0f / N_f;
    float V = V0 * expf(3.0f * d_state->eps);
    float KE = total_KE;

    // Barostat variables are only advanced in post-force (use_carry=false).
    // Pre-force re-uses eps/v_eps from the previous post-force update,
    // matching the LAMMPS operator splitting: barostat advance happens
    // after force evaluation, not before.
    if (!use_carry) {
        // P_inst = (2*KE + virial_trace) / (3*V) — LAMMPS convention
        float P_inst = (2.0f * total_KE + virial_trace_raw) / (3.0f * V);

        float sy_w[3];
        sy_w[0] = 1.0f / (2.0f - cbrtf(2.0f));
        sy_w[1] = -cbrtf(2.0f) / (2.0f - cbrtf(2.0f));
        sy_w[2] = 1.0f / (2.0f - cbrtf(2.0f));

        for (int sy = 0; sy < 3; sy++) {
            float w = sy_w[sy] * dt;
            float dv_eps = 3.0f * V * (P_inst - d_state->P_target) * w;
            float v_eps_half = d_state->v_eps + 0.5f * dv_eps;
            d_state->eps += v_eps_half * w / W;
            dv_eps = 3.0f * V0 * expf(3.0f * d_state->eps)
                     * (P_inst - d_state->P_target) * w;
            d_state->v_eps += dv_eps;
        }

        V = V0 * expf(3.0f * d_state->eps);
    }
    float L = cbrtf(V);
    float inv_L = 1.0f / L;
    *d_V = V;
    *d_L = L;
    *d_inv_L = inv_L;

    // Barostat velocity scale (applied in pre-force fused kernel)
    float v_eps_W = d_state->v_eps / W;
    *d_v_eps_W = v_eps_W;
    *d_baro_scale = expf(-(1.0f + 3.0f * N_f_inv) * v_eps_W * half_dt);
    float baro_scale = *d_baro_scale;
    KE *= baro_scale * baro_scale;

    // Position update barostat parameters
    *d_exp_vW = expf(v_eps_W * dt);
    *d_v_eps_W_dt = v_eps_W * dt;

    // --- NH chain (same as NVT, but uses barostat-scaled KE) ---
    float kT_target = d_state->T_target;
    float dt2 = half_dt;
    float dt4 = half_dt * 0.5f;
    float dt8 = half_dt * 0.25f;

    float v_xi_dotdot[10] = {};
    float T_current = (2.0f * KE) / N_f;

    if (Q1 > 0.0f) {
        v_xi_dotdot[0] = (N_f * T_current - N_f * kT_target) / Q1;
    }

    int nc_tchain = 1;
    float ncfac = 1.0f / static_cast<float>(nc_tchain);
    float scale = 1.0f;

    for (int iloop = 0; iloop < nc_tchain; iloop++) {
        for (int ich = M - 1; ich > 0; ich--) {
            float expfac = (ich + 1 < M)
                ? nh_expf(-ncfac * dt8 * d_state->v_xi[ich + 1]) : 1.0f;
            d_state->v_xi[ich] *= expfac;
            d_state->v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            d_state->v_xi[ich] *= expfac;
        }
        float expfac0 = (M > 1)
            ? nh_expf(-ncfac * dt8 * d_state->v_xi[1]) : 1.0f;
        d_state->v_xi[0] *= expfac0;
        d_state->v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        d_state->v_xi[0] *= expfac0;

        float factor_eta = nh_expf(-ncfac * dt2 * d_state->v_xi[0]);
        scale = factor_eta;
        T_current *= factor_eta * factor_eta;

        if (Q1 > 0.0f) {
            v_xi_dotdot[0] = (N_f * T_current - N_f * kT_target) / Q1;
        }
        for (int ich = 0; ich < M; ich++) {
            d_state->xi[ich] += ncfac * dt2 * d_state->v_xi[ich];
        }
        d_state->v_xi[0] *= expfac0;
        d_state->v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        d_state->v_xi[0] *= expfac0;
        for (int ich = 1; ich < M; ich++) {
            float expfac = (ich + 1 < M)
                ? nh_expf(-ncfac * dt8 * d_state->v_xi[ich + 1]) : 1.0f;
            d_state->v_xi[ich] *= expfac;
            float Q_prev = (ich == 1) ? Q1 : Q_rest;
            v_xi_dotdot[ich] = (Q_prev * d_state->v_xi[ich - 1] * d_state->v_xi[ich - 1]
                                - kT_target) / Q_rest;
            d_state->v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            d_state->v_xi[ich] *= expfac;
        }
    }

    d_state->nh_scale = scale;
    if (!use_carry) {
        d_state->chain_KE_carry = KE * scale * scale;
        d_state->step++;
    }
}

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
    float v_xi_dotdot[10] = {};

    // Current temperature from KE: T = 2*KE / (N_f)
    float T_current = (2.0f * total_KE) / N_f;

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
            float expfac = (ich + 1 < M)
                ? nh_expf(-ncfac * dt8 * nh.v_xi[ich + 1])
                : 1.0f;

            nh.v_xi[ich] *= expfac;
            nh.v_xi[ich] += v_xi_dotdot[ich] * ncfac * dt4;
            nh.v_xi[ich] *= expfac;
        }

        // --- k = 0 ---
        float expfac0 = (M > 1)
            ? nh_expf(-ncfac * dt8 * nh.v_xi[1])
            : 1.0f;

        nh.v_xi[0] *= expfac0;
        nh.v_xi[0] += v_xi_dotdot[0] * ncfac * dt4;
        nh.v_xi[0] *= expfac0;

        // --- Velocity scaling ---
        float factor_eta = nh_expf(-ncfac * dt2 * nh.v_xi[0]);
        scale_out = factor_eta;

        // --- Analytic temperature update ---
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
                ? nh_expf(-ncfac * dt8 * nh.v_xi[ich + 1])
                : 1.0f;

            nh.v_xi[ich] *= expfac;

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
    const NoseHooverDeviceState* __restrict__ d_state,
    int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float scale = d_state->nh_scale;
    float4 v = vel[i];
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    vel[i] = v;
}

void launch_nh_global_scale_vel(float4* vel,
                                 const NoseHooverDeviceState* d_state,
                                 int natoms, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_global_scale_vel_kernel<<<blocks, 256, 0, stream>>>(
        vel, d_state, natoms);
}

// --- Barostat velocity rescale kernel ---

__global__ void nh_barostat_vel_half_kernel(
    float4* __restrict__ vel,
    const float* __restrict__ d_v_eps_W,
    float N_f_inv, int natoms, float half_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float v_eps_W = *d_v_eps_W;
    float factor = (1.0f + 3.0f * N_f_inv) * v_eps_W * half_dt;
    float scale = expf(-factor);

    float4 v = vel[i];
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    vel[i] = v;
}

void launch_nh_barostat_vel_half(float4* vel, const float* d_v_eps_W,
                                  float N_f_inv, int natoms,
                                  float half_dt, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_barostat_vel_half_kernel<<<blocks, 256, 0, stream>>>(
        vel, d_v_eps_W, N_f_inv, natoms, half_dt);
}

// --- Velocity Verlet half-step ---

__global__ void nh_v_verlet_half_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ pos,
    int natoms, float half_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float4 v = vel[i];
    float4 f = force[i];
    float4 r = pos[i];

    int type_i = unpack_type_id(r.w);
    float m_i = c_masses[type_i];
    float half_dt_over_m = half_dt / m_i;

    v.x += half_dt_over_m * f.x;
    v.y += half_dt_over_m * f.y;
    v.z += half_dt_over_m * f.z;
    vel[i] = v;
}

void launch_nh_v_verlet_half(float4* vel, const float4* force,
                              const float4* pos,
                              int natoms, float half_dt,
                              cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_v_verlet_half_kernel<<<blocks, 256, 0, stream>>>(
        vel, force, pos, natoms, half_dt);
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

// --- Fused NVT pre-force kernel ---
// Combines: thermostat velocity scaling + velocity half-step + position update + PBC + displacement

__global__ void nh_nvt_pre_force_fused_kernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ pos_ref,
    int* __restrict__ d_max_dr2_int,
    int* __restrict__ d_image,
    const NoseHooverDeviceState* __restrict__ d_state,
    float half_dt, float dt,
    float L, float inv_L, int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float nh_scale = d_state->nh_scale;

    float4 r = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    int type_i = unpack_type_id(r.w);
    float m_i = c_masses[type_i];
    float half_dt_over_m = half_dt / m_i;

    // Thermostat velocity scaling
    v.x *= nh_scale;
    v.y *= nh_scale;
    v.z *= nh_scale;

    // Velocity Verlet half-kick
    v.x += half_dt_over_m * f.x;
    v.y += half_dt_over_m * f.y;
    v.z += half_dt_over_m * f.z;

    // Position update (NVT: no barostat)
    r.x += dt * v.x;
    r.y += dt * v.y;
    r.z += dt * v.z;

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

    // Displacement tracking (pos_ref unchanged for NVT)
    update_max_displacement(r, pos_ref[i], d_max_dr2_int, L, inv_L);

    pos[i] = r;
    vel[i] = v;
}

void launch_nh_nvt_pre_force_fused(float4* pos, float4* vel, const float4* force,
                                    const float4* pos_ref,
                                    int* d_max_dr2_int, int* d_image,
                                    const NoseHooverDeviceState* d_state,
                                    float half_dt, float dt,
                                    int natoms, float L, float inv_L,
                                    cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_nvt_pre_force_fused_kernel<<<blocks, 256, 0, stream>>>(
        pos, vel, force, pos_ref, d_max_dr2_int, d_image,
        d_state, half_dt, dt, L, inv_L, natoms);
}

// --- Fused NPT pre-force kernel ---
// Combines: barostat + thermostat velocity scaling + velocity half-step +
//           position update with barostat + PBC + displacement (pos_ref updated inline)

__global__ void nh_npt_pre_force_fused_kernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float4* __restrict__ pos_ref,
    int* __restrict__ d_max_dr2_int,
    int* __restrict__ d_image,
    const NoseHooverDeviceState* __restrict__ d_state,
    const float* __restrict__ d_baro_scale,
    const float* __restrict__ d_exp_vW,
    const float* __restrict__ d_v_eps_W_dt,
    float half_dt, float dt,
    const float* __restrict__ d_L, const float* __restrict__ d_inv_L, int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    const float L = *d_L, inv_L = *d_inv_L;
    float nh_scale = d_state->nh_scale;
    float baro_scale = *d_baro_scale;
    float exp_vW = *d_exp_vW;
    float v_eps_W_dt = *d_v_eps_W_dt;

    float4 r = pos[i];
    float4 v = vel[i];
    float4 f = force[i];
    float4 r_ref = pos_ref[i];

    int type_i = unpack_type_id(r.w);
    float m_i = c_masses[type_i];
    float half_dt_over_m = half_dt / m_i;

    // Combined thermostat + barostat velocity scaling
    float scale = nh_scale * baro_scale;
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;

    // Velocity Verlet half-kick
    v.x += half_dt_over_m * f.x;
    v.y += half_dt_over_m * f.y;
    v.z += half_dt_over_m * f.z;

    // Position update with barostat
    float half_vW_dt = 0.5f * v_eps_W_dt;
    float f_pos = dt * expf(half_vW_dt) * sinchf(half_vW_dt);

    r.x = r.x * exp_vW + v.x * f_pos;
    r.y = r.y * exp_vW + v.y * f_pos;
    r.z = r.z * exp_vW + v.z * f_pos;

    r_ref.x *= exp_vW;
    r_ref.y *= exp_vW;
    r_ref.z *= exp_vW;

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

    // Displacement tracking
    update_max_displacement(r, r_ref, d_max_dr2_int, L, inv_L);

    pos[i] = r;
    vel[i] = v;
    pos_ref[i] = r_ref;
}

void launch_nh_npt_pre_force_fused(float4* pos, float4* vel, const float4* force,
                                    float4* pos_ref,
                                    int* d_max_dr2_int, int* d_image,
                                    const NoseHooverDeviceState* d_state,
                                    const float* d_baro_scale,
                                    const float* d_exp_vW,
                                    const float* d_v_eps_W_dt,
                                    float half_dt, float dt,
                                    int natoms, const float* d_L, const float* d_inv_L,
                                    cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_npt_pre_force_fused_kernel<<<blocks, 256, 0, stream>>>(
        pos, vel, force, pos_ref, d_max_dr2_int, d_image,
        d_state, d_baro_scale, d_exp_vW, d_v_eps_W_dt,
        half_dt, dt, d_L, d_inv_L, natoms);
}

// --- Lightweight KE + virial-trace kernel for NPT ---
// Computes KE (sum of v^2) into d_ke_out. Only block 0 writes the
// virial trace (sum of 3 diagonal virial components) to d_virial_trace_out.

__global__ void nh_npt_ke_virial_trace_kernel(
    const float4* __restrict__ vel,
    const double* __restrict__ virial,
    const float4* __restrict__ pos,
    float* __restrict__ d_ke_out,
    float* __restrict__ d_virial_trace_out,
    int natoms)
{
    __shared__ float sdata[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v2 = 0.0f;
    if (i < natoms) {
        float4 v = vel[i];
        float4 r = pos[i];
        int type_i = unpack_type_id(r.w);
        float m_i = c_masses[type_i];
        v2 = m_i * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
    }

    if (tid % 32 == 0) {
        sdata[tid / 32] = v2;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            atomicAdd(d_ke_out, val);
        }
    }

    // Virial trace: only block 0 writes it (once)
    if (blockIdx.x == 0 && tid == 0) {
        float vt = (float)(virial[VXX] + virial[VYY] + virial[VZZ]);
        *d_virial_trace_out = vt;
    }
}

void launch_nh_npt_ke_and_virial_trace(
    const float4* vel, const double* virial,
    const float4* pos,
    float* d_ke_out, float* d_virial_trace_out,
    int natoms, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(d_ke_out, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_virial_trace_out, 0, sizeof(float), stream));
    int blocks = div_ceil(natoms, 256);
    nh_npt_ke_virial_trace_kernel<<<blocks, 256, 0, stream>>>(
        vel, virial, pos, d_ke_out, d_virial_trace_out, natoms);
}

// --- Fused NVT post-force v-half + KE reduction kernel ---
// Updates velocity AND accumulates KE in shared memory in one pass.

__global__ void nh_nvt_v_half_ke_reduce_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ pos,
    float* __restrict__ ke_out,
    int natoms, float half_dt)
{
    __shared__ float sdata[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v2 = 0.0f;
    if (i < natoms) {
        float4 v = vel[i];
        float4 f = force[i];
        float4 r = pos[i];

        int type_i = unpack_type_id(r.w);
        float m_i = c_masses[type_i];
        float half_dt_over_m = half_dt / m_i;

        v.x += half_dt_over_m * f.x;
        v.y += half_dt_over_m * f.y;
        v.z += half_dt_over_m * f.z;
        vel[i] = v;
        v2 = m_i * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
    }

    if (tid % 32 == 0) {
        sdata[tid / 32] = v2;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            atomicAdd(ke_out, val);
        }
    }
}

void launch_nh_nvt_v_half_ke_reduce(float4* vel, const float4* force,
                                     const float4* pos,
                                     float* ke_out, int natoms, float half_dt,
                                     cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    nh_nvt_v_half_ke_reduce_kernel<<<blocks, 256, 0, stream>>>(
        vel, force, pos, ke_out, natoms, half_dt);
}

// --- Lightweight KE-only reduction ---
// Much cheaper than compute_thermo: single kernel + single D2H copy + sync.
// Used for NH chain thermostat where only total KE is needed.

__global__ void ke_reduce_kernel(const float4* __restrict__ vel,
                                  const float4* __restrict__ pos,
                                  float* __restrict__ ke_out,
                                  int natoms) {
    __shared__ float sdata[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v2 = 0.0f;
    if (i < natoms) {
        float4 v = vel[i];
        float4 r = pos[i];
        int type_i = unpack_type_id(r.w);
        float m_i = c_masses[type_i];
        v2 = m_i * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
    }

    if (tid % 32 == 0) {
        sdata[tid / 32] = v2;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            atomicAdd(ke_out, val);
        }
    }
}

float compute_ke_only(const float4* vel, const float4* pos, int natoms,
                      float* d_ke_buf, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_ke_buf, 0, sizeof(float), stream));
    int blocks = div_ceil(natoms, 256);
    ke_reduce_kernel<<<blocks, 256, 0, stream>>>(vel, pos, d_ke_buf, natoms);
    float ke_total;
    CUDA_CHECK(cudaMemcpyAsync(&ke_total, d_ke_buf, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0.5f * ke_total;
}
