#include "layer_stress.cuh"
#include "../core/system.cuh"

namespace {

__global__ void layer_stress_kernel(const float4* __restrict__ pos, const float4* __restrict__ vel,
                                    const int* __restrict__ image, const float* __restrict__ atom_vir,
                                    int natoms, const Box* __restrict__ d_box,
                                    const float* __restrict__ d_center, LayerSpec spec,
                                    const float* __restrict__ d_vscale, double* __restrict__ accum,
                                    unsigned int* __restrict__ ticket, double* __restrict__ out,
                                    double* __restrict__ shear) {
    __shared__ float s_acc[kMaxLayers * 6];
    for (int k = threadIdx.x; k < spec.n * 6; k += blockDim.x) s_acc[k] = 0.0f;
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < natoms) {
        const Box b = *d_box;
        const float4 r = pos[i];
        const int* im = image + 3 * i;
        const float u = spec.axis == 0 ? r.x + im[0] * b.L.x + im[1] * b.xy
                      : spec.axis == 1 ? r.y + im[1] * b.L.y : r.z + im[2] * b.L.z;
        int k = (int)floorf((u - *d_center) / spec.width + 0.5f * spec.n);
        k = min(max(k, 0), spec.n - 1);
        const float m = c_masses[unpack_type_id(r.w)];
        const float s = d_vscale ? *d_vscale : 1.0f;
        const float4 v = vel[i];
        const float vx = v.x * s, vy = v.y * s, vz = v.z * s;
        const float kin[6] = {m * vx * vx, m * vx * vy, m * vx * vz, m * vy * vy, m * vy * vz, m * vz * vz};
        for (int c = 0; c < 6; c++) atomicAdd(&s_acc[k * 6 + c], kin[c] + atom_vir[c * natoms + i]);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < spec.n * 6; k += blockDim.x)
        if (s_acc[k] != 0.0f) atomicAdd(&accum[k], (double)s_acc[k]);
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = (atomicAdd(ticket, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int k = threadIdx.x; k < spec.n * 6; k += blockDim.x) {
        double val = atomicAdd(&accum[k], 0.0);
        out[k] = val;
        if (k % 6 == spec.shear) shear[k / 6] = val;
        accum[k] = 0.0;
    }
    if (threadIdx.x == 0) *ticket = 0u;
}

}  // namespace

void launch_layer_stress(const float4* pos, const float4* vel, const int* image,
                         const float* atom_vir, int natoms, const Box* d_box,
                         const float* d_center, const LayerSpec& spec, const float* d_vscale,
                         double* d_accum, unsigned int* d_ticket, double* d_out, double* d_shear,
                         cudaStream_t stream) {
    layer_stress_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(
        pos, vel, image, atom_vir, natoms, d_box, d_center, spec, d_vscale, d_accum, d_ticket,
        d_out, d_shear);
    CUDA_CHECK(cudaGetLastError());
}
