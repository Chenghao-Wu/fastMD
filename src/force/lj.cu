#include "lj.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"
#include "../core/launch.cuh"
#include <cmath>

// EV = false skips the per-atom energy (force.w is left 0) and the virial:
// most steps need neither (see Simulation::step).
//
// TPA threads cooperate on one particle: lane `sub` of the group takes
// neighbors sub, sub+TPA, ... (rows of the transposed list stay contiguous
// across the consecutive particles of a warp), then the group sums its
// partial forces with shuffles. TPA > 1 raises occupancy when the system is
// small relative to the GPU.
template <bool CAP, int TPA, bool EV, bool SHIFT, bool TILT>
__global__ void lj_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const float2* __restrict__ lj_params,
    const float2* __restrict__ lj_cap,
    float force_cap,
    const int* __restrict__ neighbors,
    const int* __restrict__ num_neighbors,
    int natoms, int ntypes,
    float rc2, Box box, const Box* __restrict__ d_box, float* __restrict__ atom_vir)
{
    if (d_box) box = *d_box;   // box on the device (NPT)
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = t / TPA;
    const int sub = t & (TPA - 1);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    if (i < natoms) {
        float4 pos_i = pos[i];
        int type_i = __float_as_int(pos_i.w);

        int nneigh = __ldg(&num_neighbors[i]);
        #pragma unroll 4
        for (int k = sub; k < nneigh; k += TPA) {
            int j = neighbors[k * natoms + i];
            float4 pos_j = __ldg(&pos[j]);
            int type_j = __float_as_int(pos_j.w);

            const float3 dd_ = min_image_dr_t<TILT>(pos_i, pos_j, box);
            float dx = dd_.x, dy = dd_.y, dz = dd_.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2 && r2 > 1e-10f) {
                float2 params = __ldg(&lj_params[type_i * ntypes + type_j]);
                float eps = params.x, sig = params.y;
                float sig2 = sig * sig;
                float r2inv = __fdividef(1.0f, r2);
                float sr2 = sig2 * r2inv;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                // force_r = F(r) / r, so that F_vec = force_r * dr
                float force_r = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);
                float pe_pair = 4.0f * eps * (sr12 - sr6);
                if (CAP) {
                    float2 cap = __ldg(&lj_cap[type_i * ntypes + type_j]);  // (r_cap, U(r_cap))
                    if (r2 < cap.x * cap.x) {
                        float r = sqrtf(r2);
                        force_r = force_cap / r;
                        pe_pair = cap.y + force_cap * (cap.x - r);
                    }
                }

                float fxij = force_r * dx;
                float fyij = force_r * dy;
                float fzij = force_r * dz;

                fx += fxij; fy += fyij; fz += fzij;
                if (EV) {
                    if (SHIFT) {   // energy-shifted LJ: U(rc) = 0
                        float sc6 = sig2 / rc2; sc6 = sc6 * sc6 * sc6;
                        pe_pair -= 4.0f * eps * (sc6 * sc6 - sc6);
                    }
                    pe_i += 0.5f * pe_pair;
                    vir_xx += 0.5f * fxij * dx;
                    vir_xy += 0.5f * fxij * dy;
                    vir_xz += 0.5f * fxij * dz;
                    vir_yy += 0.5f * fyij * dy;
                    vir_yz += 0.5f * fyij * dz;
                    vir_zz += 0.5f * fzij * dz;
                }
            }
        }
    }
    if (TPA > 1) {
#pragma unroll
        for (int o = TPA / 2; o > 0; o >>= 1) {
            fx += __shfl_down_sync(0xffffffffu, fx, o, TPA);
            fy += __shfl_down_sync(0xffffffffu, fy, o, TPA);
            fz += __shfl_down_sync(0xffffffffu, fz, o, TPA);
            if (EV) pe_i += __shfl_down_sync(0xffffffffu, pe_i, o, TPA);
        }
    }
    if (i < natoms && sub == 0) force[i] = make_float4(fx, fy, fz, pe_i);

    if (EV) {
        float vir[6] = {vir_xx, vir_xy, vir_xz, vir_yy, vir_yz, vir_zz};
        if (atom_vir) {   // per-atom virial (half of every pair) for layer stresses
            float av[6] = {vir_xx, vir_xy, vir_xz, vir_yy, vir_yz, vir_zz};
            if (TPA > 1) {
#pragma unroll
                for (int o = TPA / 2; o > 0; o >>= 1)
#pragma unroll
                    for (int c = 0; c < 6; c++) av[c] += __shfl_down_sync(0xffffffffu, av[c], o, TPA);
            }
            if (i < natoms && sub == 0)
#pragma unroll
                for (int c = 0; c < 6; c++) atom_vir[c * natoms + i] = av[c];
        }
        block_reduce_virial(vir, virial);
    }
}

void launch_lj_kernel(const float4* pos, float4* force, double* virial,
                       const float2* lj_params,
                       const int* neighbors, const int* num_neighbors,
                       int natoms, int ntypes,
                       float rc2, Box box,
                       cudaStream_t stream,
                       const float2* lj_cap, float force_cap, int tpa, const Box* d_box,
                       bool ev, bool shift, float* atom_vir)
{
    if (tpa <= 0) tpa = pair_threads_per_atom(natoms);
    const bool cap = lj_cap && force_cap > 0.0f;
    const bool tilted = box.xy != 0.0f;   // the tilt only changes at host events
    const int blocks = div_ceil(natoms * tpa, 256);
#define FASTMD_LJ_K2(C, T, E, S, X)                                                \
    lj_verlet_kernel<C, T, E, S, X><<<blocks, 256, 0, stream>>>(                  \
        pos, force, virial, lj_params, lj_cap, force_cap,                         \
        neighbors, num_neighbors, natoms, ntypes, rc2, box, d_box, atom_vir)
#define FASTMD_LJ_K(C, T, E, S)                                                    \
    do { if (tilted) FASTMD_LJ_K2(C, T, E, S, true); else FASTMD_LJ_K2(C, T, E, S, false); } while (0)
#define FASTMD_LJ_LAUNCH(C, T)                                                     \
    do {                                                                           \
        if (!ev) FASTMD_LJ_K(C, T, false, false);                                  \
        else if (shift) FASTMD_LJ_K(C, T, true, true);                             \
        else FASTMD_LJ_K(C, T, true, false);                                       \
    } while (0)
    switch ((cap ? 16 : 0) + tpa) {
        case 1:  FASTMD_LJ_LAUNCH(false, 1); break;
        case 2:  FASTMD_LJ_LAUNCH(false, 2); break;
        case 4:  FASTMD_LJ_LAUNCH(false, 4); break;
        case 8:  FASTMD_LJ_LAUNCH(false, 8); break;
        case 17: FASTMD_LJ_LAUNCH(true, 1); break;
        case 18: FASTMD_LJ_LAUNCH(true, 2); break;
        case 20: FASTMD_LJ_LAUNCH(true, 4); break;
        default: FASTMD_LJ_LAUNCH(true, 8); break;
    }
#undef FASTMD_LJ_LAUNCH
#undef FASTMD_LJ_K
#undef FASTMD_LJ_K2
}

std::vector<float2> lj_force_cap_table(const std::vector<float2>& lj_params,
                                       int ntypes, float force_cap) {
    std::vector<float2> cap(lj_params.size(), make_float2(0.0f, 0.0f));
    for (int k = 0; k < ntypes * ntypes; k++) {
        double eps = lj_params[k].x, sig = lj_params[k].y;
        if (eps <= 0.0 || sig <= 0.0) continue;
        auto F = [&](double r) {  // radial LJ force, positive = repulsive
            double sr6 = std::pow(sig / r, 6);
            return 24.0 * eps / r * (2.0 * sr6 * sr6 - sr6);
        };
        // F decreases monotonically on (0, 2^(1/6) sig], where it reaches 0.
        double lo = 1e-3 * sig, hi = std::pow(2.0, 1.0 / 6.0) * sig;
        if (F(lo) <= force_cap) continue;   // cap never reached
        for (int it = 0; it < 200; it++) {
            double mid = 0.5 * (lo + hi);
            (F(mid) > force_cap ? lo : hi) = mid;
        }
        double rcap = 0.5 * (lo + hi);
        double sr6 = std::pow(sig / rcap, 6);
        cap[k] = make_float2((float)rcap, (float)(4.0 * eps * (sr6 * sr6 - sr6)));
    }
    return cap;
}
