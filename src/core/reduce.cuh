#pragma once
#include "types.cuh"

// Block-wide reductions used by the force and thermo kernels.
//
// Requirements: blockDim.x is a multiple of 32 and <= 1024, and every thread
// of the block calls the function (it contains __syncthreads). Threads with
// nothing to contribute pass zeros.

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Sum N per-thread components over the block and atomically add the block
// total to out[0..N-1] (double accumulators: the global sum over all blocks
// is order-dependent only at double round-off). Warp shuffle per component,
// one shared slot per warp, the first warp combines the warp partials.
template <int N, typename T>
__device__ __forceinline__ void block_reduce_atomic_add(const T (&v)[N], double* out) {
    __shared__ T s_part[N][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    __syncthreads();   // the static scratch may be reused by a previous call
#pragma unroll
    for (int c = 0; c < N; c++) {
        T w = warp_sum(v[c]);
        if (lane == 0) s_part[c][warp] = w;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < N; c++) {
            T t = lane < nwarps ? s_part[c][lane] : T(0);
            t = warp_sum(t);
            if (lane == 0) atomicAdd(&out[c], (double)t);
        }
    }
}

// Convenience for the 6-component virial (xx xy xz yy yz zz).
__device__ __forceinline__ void block_reduce_virial(const float (&v)[6], double* virial) {
    block_reduce_atomic_add<6, float>(v, virial);
}
