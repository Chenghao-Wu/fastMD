#pragma once
#include "../core/types.cuh"
#include <cstdio>
#include <vector>

inline int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}

struct RgBuffers {
    int* d_chain_offsets;
    int* d_chain_lengths;
    float* d_ensemble_sum;
    int nchains;
    int max_chain_len;
    int block_size;
    FILE* fp;

    void allocate(const std::vector<int>& chain_offsets,
                  const std::vector<int>& chain_lengths,
                  int nchains, int max_chain_len,
                  const char* output_file);
    void free();
};

void compute_rg(const float4* pos, const int* image,
                const RgBuffers& bufs, float box_L,
                int step, float* h_rg,
                cudaStream_t stream = 0);
