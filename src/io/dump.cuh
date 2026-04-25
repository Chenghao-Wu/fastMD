#pragma once
#include "../core/types.cuh"
#include <cstdio>

struct BinaryDumper {
    float3* h_buf[2];
    int current_buf;
    int natoms;
    FILE* fp;

    void open(const char* filename, int natoms, int ntypes);
    void close();

    void dump_frame(const float4* d_pos, int64_t step, float box_L,
                     cudaStream_t stream = 0);
};
