#include "rg.cuh"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {

constexpr int kRgBlock = 128;

// Block-wide sum. Every thread receives the result. The leading
// __syncthreads() makes the scratch reusable across consecutive calls.
__device__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    float total = 0.0f;
    for (int w = 0; w < nwarps; w++) total += scratch[w];
    return total;
}

__global__ void rg_kernel(const float4* __restrict__ pos,
                          const int* __restrict__ image,
                          const int* __restrict__ rank,
                          const int* __restrict__ mol_offsets,
                          const int* __restrict__ mol_members,
                          Box box,
                          float* __restrict__ d_ensemble_sum)
{
    __shared__ float scratch[kRgBlock / 32];
    int m = blockIdx.x;
    int start = mol_offsets[m];
    int len = mol_offsets[m + 1] - start;

    // Pass 1: centre of mass of the unwrapped coordinates
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (int k = threadIdx.x; k < len; k += blockDim.x) {
        int t = mol_members[start + k];
        int i = rank ? rank[t] : t;
        float4 r = pos[i];
        cx += r.x + image[3 * i + 0] * box.L.x + image[3 * i + 1] * box.xy;
        cy += r.y + image[3 * i + 1] * box.L.y;
        cz += r.z + image[3 * i + 2] * box.L.z;
    }
    float inv_len = 1.0f / len;
    cx = block_sum(cx, scratch) * inv_len;
    cy = block_sum(cy, scratch) * inv_len;
    cz = block_sum(cz, scratch) * inv_len;

    // Pass 2: mean squared distance from the centre of mass
    float rg2 = 0.0f;
    for (int k = threadIdx.x; k < len; k += blockDim.x) {
        int t = mol_members[start + k];
        int i = rank ? rank[t] : t;
        float4 r = pos[i];
        float dx = r.x + image[3 * i + 0] * box.L.x + image[3 * i + 1] * box.xy - cx;
        float dy = r.y + image[3 * i + 1] * box.L.y - cy;
        float dz = r.z + image[3 * i + 2] * box.L.z - cz;
        rg2 += dx * dx + dy * dy + dz * dz;
    }
    rg2 = block_sum(rg2, scratch);
    if (threadIdx.x == 0) atomicAdd(d_ensemble_sum, sqrtf(rg2 * inv_len));
}

}  // namespace

void RgBuffers::allocate(const std::vector<int>& mol_ids, const char* output_file) {
    const int n = (int)mol_ids.size();
    if (n == 0) throw std::runtime_error("Rg requires molecule ids");
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return mol_ids[a] < mol_ids[b]; });
    std::vector<int> offsets{0};
    for (int k = 1; k < n; k++)
        if (mol_ids[order[k]] != mol_ids[order[k - 1]]) offsets.push_back(k);
    offsets.push_back(n);
    nmols = (int)offsets.size() - 1;

    CUDA_CHECK(cudaMalloc(&d_mol_offsets, offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mol_offsets, offsets.data(), offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_mol_members, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mol_members, order.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ensemble_sum, sizeof(float)));

    fp = nullptr;
    if (output_file && output_file[0]) {
        fp = fopen(output_file, "w");
        if (!fp) throw std::runtime_error(std::string("cannot open Rg output file ") + output_file);
        fprintf(fp, "# step  Rg\n");
        fflush(fp);
    }
}

void RgBuffers::free() {
    if (d_mol_offsets) CUDA_CHECK(cudaFree(d_mol_offsets));
    if (d_mol_members) CUDA_CHECK(cudaFree(d_mol_members));
    if (d_ensemble_sum) CUDA_CHECK(cudaFree(d_ensemble_sum));
    if (fp) fclose(fp);
    *this = RgBuffers{};
}

void compute_rg(const float4* pos, const int* image, const int* rank,
                const RgBuffers& bufs, const Box& box,
                int step, float* h_rg, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(bufs.d_ensemble_sum, 0, sizeof(float), stream));
    rg_kernel<<<bufs.nmols, kRgBlock, 0, stream>>>(
        pos, image, rank, bufs.d_mol_offsets, bufs.d_mol_members, box, bufs.d_ensemble_sum);
    CUDA_CHECK(cudaGetLastError());

    float ensemble_sum;
    CUDA_CHECK(cudaMemcpyAsync(&ensemble_sum, bufs.d_ensemble_sum, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *h_rg = ensemble_sum / bufs.nmols;

    if (bufs.fp) {
        fprintf(bufs.fp, "%d  %.6f\n", step, *h_rg);
        fflush(bufs.fp);
    }
}
