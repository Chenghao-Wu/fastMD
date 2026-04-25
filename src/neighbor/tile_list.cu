#include "tile_list.cuh"
#include "../core/pbc.cuh"
#include <cub/cub.cuh>

__global__ void compute_tile_bbox(const float4* __restrict__ pos,
                                    float3* __restrict__ tile_min_out,
                                    float3* __restrict__ tile_max_out,
                                    int natoms, int ntiles) {
    int tile_id = blockIdx.x;
    int lane = threadIdx.x;
    int atom_id = tile_id * TILE_SIZE + lane;

    float3 p;
    if (atom_id < natoms) {
        float4 r = pos[atom_id];
        p = make_float3(r.x, r.y, r.z);
    } else {
        p = make_float3(1e30f, 1e30f, 1e30f);
    }

    float3 bmin = p, bmax = p;
    if (atom_id >= natoms) {
        bmax = make_float3(-1e30f, -1e30f, -1e30f);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        bmin.x = fminf(bmin.x, __shfl_down_sync(0xFFFFFFFF, bmin.x, offset));
        bmin.y = fminf(bmin.y, __shfl_down_sync(0xFFFFFFFF, bmin.y, offset));
        bmin.z = fminf(bmin.z, __shfl_down_sync(0xFFFFFFFF, bmin.z, offset));
        bmax.x = fmaxf(bmax.x, __shfl_down_sync(0xFFFFFFFF, bmax.x, offset));
        bmax.y = fmaxf(bmax.y, __shfl_down_sync(0xFFFFFFFF, bmax.y, offset));
        bmax.z = fmaxf(bmax.z, __shfl_down_sync(0xFFFFFFFF, bmax.z, offset));
    }

    if (lane == 0) {
        tile_min_out[tile_id] = bmin;
        tile_max_out[tile_id] = bmax;
    }
}

__global__ void check_tile_pairs(const float3* __restrict__ tile_min,
                                  const float3* __restrict__ tile_max,
                                  int* __restrict__ flags,
                                  int ntiles, float rc_skin,
                                  float L, float inv_L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ta = idx / ntiles;
    int tb = idx % ntiles;

    if (ta >= ntiles) {
        return;
    }

    float3 cen_a = make_float3(
        0.5f * (tile_min[ta].x + tile_max[ta].x),
        0.5f * (tile_min[ta].y + tile_max[ta].y),
        0.5f * (tile_min[ta].z + tile_max[ta].z)
    );
    float3 cen_b = make_float3(
        0.5f * (tile_min[tb].x + tile_max[tb].x),
        0.5f * (tile_min[tb].y + tile_max[tb].y),
        0.5f * (tile_min[tb].z + tile_max[tb].z)
    );

    float3 ext_a = make_float3(
        0.5f * (tile_max[ta].x - tile_min[ta].x),
        0.5f * (tile_max[ta].y - tile_min[ta].y),
        0.5f * (tile_max[ta].z - tile_min[ta].z)
    );
    float3 ext_b = make_float3(
        0.5f * (tile_max[tb].x - tile_min[tb].x),
        0.5f * (tile_max[tb].y - tile_min[tb].y),
        0.5f * (tile_max[tb].z - tile_min[tb].z)
    );

    float dx = fabsf(min_image(cen_a.x - cen_b.x, L, inv_L)) - ext_a.x - ext_b.x;
    float dy = fabsf(min_image(cen_a.y - cen_b.y, L, inv_L)) - ext_a.y - ext_b.y;
    float dz = fabsf(min_image(cen_a.z - cen_b.z, L, inv_L)) - ext_a.z - ext_b.z;

    dx = fmaxf(dx, 0.0f);
    dy = fmaxf(dy, 0.0f);
    dz = fmaxf(dz, 0.0f);

    float surf_dist2 = dx*dx + dy*dy + dz*dz;

    flags[idx] = (surf_dist2 < rc_skin * rc_skin) ? 1 : 0;
}

__global__ void count_per_tile(const int* __restrict__ flags,
                                int* __restrict__ counts,
                                int ntiles) {
    int ta = blockIdx.x * blockDim.x + threadIdx.x;
    if (ta >= ntiles) return;
    int c = 0;
    for (int tb = 0; tb < ntiles; tb++) {
        c += flags[ta * ntiles + tb];
    }
    counts[ta] = c;
}

__global__ void scatter_neighbors(const int* __restrict__ flags,
                                    const int* __restrict__ offsets,
                                    int* __restrict__ neighbors,
                                    int ntiles) {
    int ta = blockIdx.x * blockDim.x + threadIdx.x;
    if (ta >= ntiles) return;
    int base = offsets[ta];
    int idx = 0;
    for (int tb = 0; tb < ntiles; tb++) {
        if (flags[ta * ntiles + tb]) {
            neighbors[base + idx] = tb;
            idx++;
        }
    }
}

void TileList::allocate(int ntiles_in, int max_pairs_estimate) {
    ntiles = ntiles_in;
    capacity = max_pairs_estimate;

    CUDA_CHECK(cudaMalloc(&offsets,        (ntiles + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tile_neighbors, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tile_min,       ntiles * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&tile_max,       ntiles * sizeof(float3)));

    int total_pairs = ntiles * ntiles;
    CUDA_CHECK(cudaMalloc(&d_flags,  total_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_npairs, sizeof(int)));

    d_cub_temp = nullptr;
    cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes,
                                   (int*)nullptr, (int*)nullptr, ntiles + 1);
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
}

void TileList::free() {
    CUDA_CHECK(cudaFree(offsets));
    CUDA_CHECK(cudaFree(tile_neighbors));
    CUDA_CHECK(cudaFree(tile_min));
    CUDA_CHECK(cudaFree(tile_max));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_npairs));
    CUDA_CHECK(cudaFree(d_cub_temp));
}

void TileList::build(const float4* pos, int natoms, int ntiles_in,
                      float rc_skin, float L, float inv_L,
                      cudaStream_t stream) {
    ntiles = ntiles_in;
    int total_candidate = ntiles * ntiles;

    compute_tile_bbox<<<ntiles, TILE_SIZE, 0, stream>>>(
        pos, tile_min, tile_max, natoms, ntiles);

    int pair_blocks = div_ceil(total_candidate, 256);
    check_tile_pairs<<<pair_blocks, 256, 0, stream>>>(
        tile_min, tile_max, d_flags, ntiles, rc_skin, L, inv_L);

    int tile_blocks = div_ceil(ntiles, 256);
    count_per_tile<<<tile_blocks, 256, 0, stream>>>(d_flags, offsets, ntiles);

    cub::DeviceScan::ExclusiveSum(d_cub_temp, cub_temp_bytes,
                                   offsets, offsets, ntiles + 1, stream);

    scatter_neighbors<<<tile_blocks, 256, 0, stream>>>(
        d_flags, offsets, tile_neighbors, ntiles);

    CUDA_CHECK(cudaMemcpyAsync(&npairs, offsets + ntiles, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
