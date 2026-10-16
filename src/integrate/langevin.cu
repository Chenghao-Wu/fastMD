#include "langevin.cuh"
#include "../core/pbc.cuh"
#include "../core/system.cuh"
#include "../neighbor/skin_trigger.cuh"
#include <curand_kernel.h>

void LangevinState::init(int /*natoms_padded*/, float Tdamp, float dt,
                         float T_start_in, uint64_t seed_in, float T_stop_in, int nsteps_in) {
    float gamma = Tdamp > 0.0f ? 1.0f / Tdamp : 0.0f;
    gamma_dt = gamma * dt;
    c1 = expf(-gamma_dt);
    c2 = sqrtf(1.0f - c1 * c1);
    half_dt = 0.5f * dt;
    T_start = T_start_in;
    T_stop = T_stop_in < 0.0f ? T_start_in : T_stop_in;
    nsteps = nsteps_in;
    seed = seed_in;
}

template <bool TILT>
__global__ void integrator_pre_force_kernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    float4* __restrict__ pos_ref,
    int* __restrict__ d_max_dr2_int,
    int* __restrict__ image,
    const int* __restrict__ tag,
    const int* __restrict__ d_step,
    int natoms, Box box, LangevinState lang)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    const int step = *d_step;
    const float half_dt = lang.half_dt;
    const float kT = lang.T_start + (lang.T_stop - lang.T_start) *
                     (lang.nsteps > 0 ? (float)(step - 1) / (float)lang.nsteps : 0.0f);

    float4 r = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    int type_i = unpack_type_id(r.w);
    float m_i = c_masses[type_i];
    float half_dt_over_m = half_dt / m_i;

    v.x += half_dt_over_m * f.x;
    v.y += half_dt_over_m * f.y;
    v.z += half_dt_over_m * f.z;

    r.x += half_dt * v.x;
    r.y += half_dt * v.y;
    r.z += half_dt * v.z;

    float c1 = lang.c1, c2 = lang.c2, factor = 1.0f;
    if (lang.profile.mode != 0) {   // depth below/above the film reference plane
        const int* im = image ? image + 3 * i : nullptr;
        const float ix = im ? (float)im[0] : 0.0f, iy = im ? (float)im[1] : 0.0f,
                    iz = im ? (float)im[2] : 0.0f;
        const float u = lang.profile.axis == 0 ? r.x + ix * box.L.x + iy * box.xy
                      : lang.profile.axis == 1 ? r.y + iy * box.L.y : r.z + iz * box.L.z;
        factor = friction_factor(lang.profile, u - *lang.d_film_center);
        if (!lang.andersen) {
            c1 = __expf(-lang.gamma_dt * factor);
            c2 = sqrtf(fmaxf(1.0f - c1 * c1, 0.0f));
        }
    }
    if (c2 > 0.0f || lang.andersen) {   // NVE (c2 = 0): no random numbers drawn
        // Philox stream: subsequence = particle tag, one 4-word block per step
        curandStatePhilox4_32_10_t rs;
        curand_init(lang.seed, (unsigned long long)(tag ? tag[i] : i), (unsigned long long)step * 4ull, &rs);
        float4 rand4 = curand_normal4(&rs);
        const float sd = sqrtf(kT / m_i);
        if (lang.andersen) {
            // collision with probability nu f dt; the fourth normal gives the
            // uniform variate through its CDF
            if (normcdff(rand4.w) < lang.nu_dt * factor) {
                v.x = sd * rand4.x;
                v.y = sd * rand4.y;
                v.z = sd * rand4.z;
            }
        } else {
            v.x = c1 * v.x + c2 * sd * rand4.x;
            v.y = c1 * v.y + c2 * sd * rand4.y;
            v.z = c1 * v.z + c2 * sd * rand4.z;
        }
    }

    r.x += half_dt * v.x;
    r.y += half_dt * v.y;
    r.z += half_dt * v.z;

    wrap_into_box_t<TILT>(r, image ? image + 3 * i : nullptr, box);

    update_max_displacement<TILT>(r, pos_ref[i], d_max_dr2_int, box);

    pos[i] = r;
    vel[i] = v;
}

void launch_integrator_pre_force(float4* pos, float4* vel, const float4* force,
                                  float4* pos_ref, int* d_max_dr2_int,
                                  int* image, const int* tag, const int* d_step,
                                  const LangevinState& lang,
                                  int natoms, const Box& box,
                                  float half_dt, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    LangevinState l = lang;
    l.half_dt = half_dt;
    if (box.xy != 0.0f)   // the tilt only changes at host events
        integrator_pre_force_kernel<true><<<blocks, 256, 0, stream>>>(
            pos, vel, force, pos_ref, d_max_dr2_int, image, tag, d_step, natoms, box, l);
    else
        integrator_pre_force_kernel<false><<<blocks, 256, 0, stream>>>(
            pos, vel, force, pos_ref, d_max_dr2_int, image, tag, d_step, natoms, box, l);
}

__global__ void integrator_post_force_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ pos,
    int natoms, float half_dt, int* __restrict__ d_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && d_step) (*d_step)++;   // no other thread reads the counter here
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

void launch_integrator_post_force(float4* vel, const float4* force,
                                   const float4* pos,
                                   int natoms, float half_dt,
                                   cudaStream_t stream, int* d_step) {
    int blocks = div_ceil(natoms, 256);
    integrator_post_force_kernel<<<blocks, 256, 0, stream>>>(
        vel, force, pos, natoms, half_dt, d_step);
}
