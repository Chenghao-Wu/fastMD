#include "rg.cuh"

__global__ void rg_kernel(
    const float4* __restrict__ pos,
    const int* __restrict__ image,
    const int* __restrict__ chain_offsets,
    const int* __restrict__ chain_lengths,
    float box_L,
    float* __restrict__ d_ensemble_sum)
{
    extern __shared__ float sdata[];
    int chain_id = blockIdx.x;
    int start = chain_offsets[chain_id];
    int len = chain_lengths[chain_id];
    int tid = threadIdx.x;
    int bd = blockDim.x;

    // Phase 1: strided accumulation for COM
    float3 com = make_float3(0, 0, 0);
    for (int i = tid; i < len; i += bd) {
        float4 r = pos[start + i];
        int i3 = (start + i) * 3;
        com.x += r.x + image[i3 + 0] * box_L;
        com.y += r.y + image[i3 + 1] * box_L;
        com.z += r.z + image[i3 + 2] * box_L;
    }
    sdata[tid * 3 + 0] = com.x;
    sdata[tid * 3 + 1] = com.y;
    sdata[tid * 3 + 2] = com.z;
    __syncthreads();

    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
            sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
            sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
        }
        __syncthreads();
    }

    com.x = sdata[0] / len;
    com.y = sdata[1] / len;
    com.z = sdata[2] / len;

    // Phase 2: strided accumulation for Rg²
    float rg2 = 0;
    for (int i = tid; i < len; i += bd) {
        float4 r = pos[start + i];
        int i3 = (start + i) * 3;
        float dx = r.x + image[i3 + 0] * box_L - com.x;
        float dy = r.y + image[i3 + 1] * box_L - com.y;
        float dz = r.z + image[i3 + 2] * box_L - com.z;
        rg2 += dx*dx + dy*dy + dz*dz;
    }
    sdata[tid] = rg2;
    __syncthreads();

    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float chain_rg = sqrtf(sdata[0] / len);
        atomicAdd(d_ensemble_sum, chain_rg);
    }
}

void RgBuffers::allocate(const std::vector<int>& chain_offsets,
                          const std::vector<int>& chain_lengths,
                          int nchains_in, int max_chain_len_in,
                          const char* output_file) {
    nchains = nchains_in;
    max_chain_len = max_chain_len_in;
    block_size = next_pow2(max_chain_len);
    if (block_size > 1024) block_size = 1024;

    CUDA_CHECK(cudaMalloc(&d_chain_offsets,
                          (nchains + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_chain_offsets, chain_offsets.data(),
                          (nchains + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_chain_lengths,
                          nchains * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_chain_lengths, chain_lengths.data(),
                          nchains * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_ensemble_sum, sizeof(float)));

    fp = fopen(output_file, "w");
    fprintf(fp, "# step  Rg\n");
    fflush(fp);
}

void RgBuffers::free() {
    if (d_chain_offsets) CUDA_CHECK(cudaFree(d_chain_offsets));
    if (d_chain_lengths) CUDA_CHECK(cudaFree(d_chain_lengths));
    if (d_ensemble_sum) CUDA_CHECK(cudaFree(d_ensemble_sum));
    if (fp) fclose(fp);
}

void compute_rg(const float4* pos, const int* image,
                const RgBuffers& bufs, float box_L,
                int step, float* h_rg,
                cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(bufs.d_ensemble_sum, 0, sizeof(float), stream));

    int smem_bytes = bufs.block_size * 3 * sizeof(float);
    int smem_bytes_p2 = bufs.block_size * sizeof(float);
    int smem = smem_bytes > smem_bytes_p2 ? smem_bytes : smem_bytes_p2;

    rg_kernel<<<bufs.nchains, bufs.block_size, smem, stream>>>(
        pos, image, bufs.d_chain_offsets, bufs.d_chain_lengths,
        box_L, bufs.d_ensemble_sum);

    float ensemble_sum;
    CUDA_CHECK(cudaMemcpyAsync(&ensemble_sum, bufs.d_ensemble_sum,
                                sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    *h_rg = ensemble_sum / bufs.nchains;

    if (bufs.fp) {
        fprintf(bufs.fp, "%d  %.4f\n", step, *h_rg);
        fflush(bufs.fp);
    }
}
