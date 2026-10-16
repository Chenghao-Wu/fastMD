#include "film.cuh"
#include "../core/reduce.cuh"
#include "../core/system.cuh"

namespace {

__global__ void film_center_kernel(const float4* __restrict__ pos, const int* __restrict__ image,
                                   int natoms, int axis, const Box* __restrict__ d_box,
                                   double inv_total_mass, double* __restrict__ accum,
                                   unsigned int* __restrict__ ticket, float* __restrict__ center) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float L = axis == 0 ? d_box->L.x : axis == 1 ? d_box->L.y : d_box->L.z;
    double v[1] = {0.0};
    if (i < natoms) {
        const float4 r = pos[i];
        const float x = axis == 0 ? r.x : axis == 1 ? r.y : r.z;
        v[0] = (double)c_masses[unpack_type_id(r.w)] * ((double)x + (double)image[3 * i + axis] * L);
    }
    block_reduce_atomic_add<1, double>(v, accum);
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = (atomicAdd(ticket, 1u) == gridDim.x - 1);
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        *center = (float)(atomicAdd(accum, 0.0) * inv_total_mass);
        *accum = 0.0;
        *ticket = 0u;
    }
}

}  // namespace

void launch_film_center(const float4* pos, const int* image, int natoms, int axis,
                        const Box* d_box, double inv_total_mass, double* d_accum,
                        unsigned int* d_ticket, float* d_center, cudaStream_t stream) {
    film_center_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(
        pos, image, natoms, axis, d_box, inv_total_mass, d_accum, d_ticket, d_center);
    CUDA_CHECK(cudaGetLastError());
}
