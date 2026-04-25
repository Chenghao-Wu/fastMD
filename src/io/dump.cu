#include "dump.cuh"

__global__ void pack_float3(float3* __restrict__ dst,
                             const float4* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 r = src[i];
        dst[i] = make_float3(r.x, r.y, r.z);
    }
}

void BinaryDumper::open(const char* filename, int natoms_in, int ntypes) {
    natoms = natoms_in;
    current_buf = 0;

    CUDA_CHECK(cudaHostAlloc(&h_buf[0], natoms * sizeof(float3),
                              cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_buf[1], natoms * sizeof(float3),
                              cudaHostAllocDefault));

    fp = fopen(filename, "wb");
    int32_t magic = 0x4D444247;
    fwrite(&magic, sizeof(int32_t), 1, fp);
    fwrite(&natoms, sizeof(int32_t), 1, fp);
    int32_t nt = ntypes;
    fwrite(&nt, sizeof(int32_t), 1, fp);
}

void BinaryDumper::close() {
    if (fp) fclose(fp);
    CUDA_CHECK(cudaFreeHost(h_buf[0]));
    CUDA_CHECK(cudaFreeHost(h_buf[1]));
}

void BinaryDumper::dump_frame(const float4* d_pos, int64_t step, float box_L,
                                cudaStream_t stream) {
    float3* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, natoms * sizeof(float3)));

    int blocks = div_ceil(natoms, 256);
    pack_float3<<<blocks, 256, 0, stream>>>(d_temp, d_pos, natoms);

    int buf_idx = current_buf;
    CUDA_CHECK(cudaMemcpyAsync(h_buf[buf_idx], d_temp,
                                natoms * sizeof(float3),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_temp));

    fwrite(&step, sizeof(int64_t), 1, fp);
    int32_t n = natoms;
    fwrite(&n, sizeof(int32_t), 1, fp);
    fwrite(&box_L, sizeof(float), 1, fp);
    fwrite(h_buf[buf_idx], sizeof(float3), natoms, fp);

    current_buf = 1 - current_buf;
}
