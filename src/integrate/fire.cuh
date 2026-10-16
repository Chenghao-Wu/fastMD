#pragma once
#include "../core/types.cuh"

// FIRE energy minimisation (Bitzek et al., PRL 97, 170201 (2006)) with the
// FIRE 2.0 half-step backtracking on uphill steps (Guenole et al., Comput.
// Mater. Sci. 175, 109584 (2020)). Each iteration runs on the device:
//   fire_reduce:  P = F.v, |v|^2, |F|^2, max |F|^2; its last block updates
//                 the FIRE scalars (dt, alpha, downhill count, mixing).
//   fire_update:  v <- (1 - a) v + a |v| F/|F| (downhill) or v <- 0 after
//                 x <- x - dt v / 2 (uphill); then v += dt F / m,
//                 x += dt v (semi-implicit Euler), wrap, displacement
//                 tracking for the neighbor list.
struct FireParams {
    float dt_start = 0.005f, dt_max = 0.02f, dt_min = 1e-5f;
    float f_inc = 1.1f, f_dec = 0.5f, alpha_start = 0.1f, f_alpha = 0.99f;
    int   n_delay = 5;
};

struct FireState {
    float dt, alpha;
    int   n_pos;           // consecutive downhill steps
    int   uphill;          // 1: this iteration resets the velocities
    float mix_v, mix_f;    // v <- mix_v v + mix_f F
    float fmax2;           // max |F|^2 of the last reduction
    double p, vv, ff;      // accumulators (zero between launches)
    unsigned int ticket;
    int   iterations;
};

void fire_init(FireState* d_state, const FireParams& p, cudaStream_t stream);

void launch_fire_reduce(const float4* vel, const float4* force, int natoms, FireState* d_state,
                        const FireParams& p, cudaStream_t stream);

void launch_fire_update(float4* pos, float4* vel, const float4* force, float4* pos_ref,
                        int* d_max_dr2_bits, int* image, int natoms, const Box* d_box,
                        const FireState* d_state, cudaStream_t stream);
