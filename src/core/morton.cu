#include "morton.cuh"
#include <cub/cub.cuh>

__device__ uint32_t expand_bits_gpu(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__global__ void compute_morton_keys(const float4* __restrict__ pos,
                                     uint32_t* __restrict__ keys,
                                     int* __restrict__ indices,
                                     int natoms, float inv_L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < natoms) {
        float4 p = pos[i];
        uint32_t ix = min(max(__float2uint_rz(p.x * inv_L * 1024.0f), 0u), 1023u);
        uint32_t iy = min(max(__float2uint_rz(p.y * inv_L * 1024.0f), 0u), 1023u);
        uint32_t iz = min(max(__float2uint_rz(p.z * inv_L * 1024.0f), 0u), 1023u);
        keys[i] = (expand_bits_gpu(iz) << 2) | (expand_bits_gpu(iy) << 1) | expand_bits_gpu(ix);
        indices[i] = i;
    } else if (i < ((natoms + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE) {
        keys[i] = 0xFFFFFFFF;
        indices[i] = i;
    }
}

__global__ void gather_float4(float4* __restrict__ dst,
                               const float4* __restrict__ src,
                               const int* __restrict__ perm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[perm[i]];
    }
}

void MortonSorter::allocate(int natoms_padded) {
    capacity = natoms_padded;
    CUDA_CHECK(cudaMalloc(&d_keys,           capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_sorted,    capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_indices,        capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices_sorted, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp,           capacity * sizeof(float4)));

    d_cub_temp = nullptr;
    cub_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
                                     d_keys, d_keys_sorted,
                                     d_indices, d_indices_sorted,
                                     capacity);
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
}

void MortonSorter::free() {
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_keys_sorted));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_indices_sorted));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_cub_temp));
}

void MortonSorter::sort_and_permute(float4* pos, float4* vel,
                                     int natoms, float inv_L,
                                     cudaStream_t stream) {
    int np = div_ceil(natoms, TILE_SIZE) * TILE_SIZE;
    int blocks = div_ceil(np, 256);

    compute_morton_keys<<<blocks, 256, 0, stream>>>(pos, d_keys, d_indices, natoms, inv_L);

    cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes,
                                     d_keys, d_keys_sorted,
                                     d_indices, d_indices_sorted,
                                     np, 0, 30, stream);

    gather_float4<<<blocks, 256, 0, stream>>>(d_temp, pos, d_indices_sorted, np);
    CUDA_CHECK(cudaMemcpyAsync(pos, d_temp, np * sizeof(float4),
                                cudaMemcpyDeviceToDevice, stream));

    gather_float4<<<blocks, 256, 0, stream>>>(d_temp, vel, d_indices_sorted, np);
    CUDA_CHECK(cudaMemcpyAsync(vel, d_temp, np * sizeof(float4),
                                cudaMemcpyDeviceToDevice, stream));
}
