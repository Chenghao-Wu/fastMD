#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static constexpr int TILE_SIZE = 32;

__host__ __device__ inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

__device__ inline float pack_type_id(int type_id) {
    return __int_as_float(type_id);
}

__device__ inline int unpack_type_id(float w) {
    return __float_as_int(w);
}

struct SimParams {
    float box_L;
    float inv_L;
    float half_L;
    float rc;
    float rc2;
    float skin;
    float dt;
    float temperature;
    float gamma;
    int   natoms;
    int   ntiles;
    int   ntypes;
    int   nsteps;
    int   dump_freq;
    int   thermo_freq;
    uint64_t seed;
};
