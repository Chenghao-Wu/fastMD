#include "wall.cuh"

namespace {

__host__ __device__ inline float ramp_position(const WallSpec& w, int step, int nsteps) {
    float f = nsteps > 0 ? fminf(fmaxf((float)(step - 1) / (float)nsteps, 0.0f), 1.0f) : 0.0f;
    return w.pos_start + (w.pos_stop - w.pos_start) * f;
}

template <bool EV>
__global__ void wall_kernel(const float4* __restrict__ pos, float4* __restrict__ force, int natoms,
                            WallSet walls, const int* __restrict__ d_step, int nsteps,
                            int* __restrict__ d_error) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    const int step = d_step ? *d_step : 1;
    const float4 r = pos[i];
    const float x[3] = {r.x, r.y, r.z};
    float f[3] = {0.0f, 0.0f, 0.0f};
    float e = 0.0f;
    for (int k = 0; k < walls.n; k++) {
        const WallSpec w = walls.w[k];
        const float plane = ramp_position(w, step, nsteps);
        // distance from the plane on the side the wall acts on
        const float d = w.side == 0 ? x[w.axis] - plane : plane - x[w.axis];
        if (d >= w.cutoff) continue;
        if (d <= 0.0f) {
            atomicMax(d_error, k + 1);
            continue;
        }
        float fr, er;   // fr: force along the wall normal (away from the wall)
        if (w.style == 0) {
            float s = w.sigma / d, s3 = s * s * s, s9 = s3 * s3 * s3;
            fr = w.eps * (1.2f * s9 - 3.0f * s3) / d;
            er = w.eps * (2.0f / 15.0f * s9 - s3) - w.e_shift;
        } else {
            float dr = d - w.cutoff;
            fr = -2.0f * w.eps * dr;
            er = w.eps * dr * dr;
        }
        f[w.axis] += w.side == 0 ? fr : -fr;
        if (EV) e += er;
    }
    float4 fi = force[i];
    force[i] = make_float4(fi.x + f[0], fi.y + f[1], fi.z + f[2], EV ? fi.w + e : fi.w);
}

}  // namespace

float wall_position(const WallSpec& w, int step, int nsteps) { return ramp_position(w, step, nsteps); }

void launch_wall_kernel(const float4* pos, float4* force, int natoms, const WallSet& walls,
                        const int* d_step, int nsteps, int* d_error, cudaStream_t stream, bool ev) {
    if (walls.n == 0 || natoms == 0) return;
    const int blocks = div_ceil(natoms, 256);
    if (ev) wall_kernel<true><<<blocks, 256, 0, stream>>>(pos, force, natoms, walls, d_step, nsteps, d_error);
    else wall_kernel<false><<<blocks, 256, 0, stream>>>(pos, force, natoms, walls, d_step, nsteps, d_error);
    CUDA_CHECK(cudaGetLastError());
}
