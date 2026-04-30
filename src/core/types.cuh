#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

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

__host__ __device__ inline float pack_type_id(int type_id) {
#ifdef __CUDA_ARCH__
    return __int_as_float(type_id);
#else
    float f;
    std::memcpy(&f, &type_id, sizeof(float));
    return f;
#endif
}

__host__ __device__ inline int unpack_type_id(float w) {
#ifdef __CUDA_ARCH__
    return __float_as_int(w);
#else
    int i;
    std::memcpy(&i, &w, sizeof(int));
    return i;
#endif
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
    uint64_t seed;

    int   thermo_on;
    int   thermo_freq;
    char  thermo_file[256];

    int   stress_on;
    int   stress_freq;
    char  stress_file[256];

    int   restart_freq;
    char  restart_file[256];

    int   rg_on;
    int   rg_freq;
    char  rg_file[256];
};
