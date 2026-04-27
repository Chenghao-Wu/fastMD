#pragma once
#include "../core/types.cuh"
#include <cstring>

__device__ __forceinline__
void update_max_displacement(float4 pos_new, float4 pos_ref,
                              int* d_max_dr2_int, float L, float inv_L) {
    float dx = pos_new.x - pos_ref.x;
    float dy = pos_new.y - pos_ref.y;
    float dz = pos_new.z - pos_ref.z;
    dx -= L * roundf(dx * inv_L);
    dy -= L * roundf(dy * inv_L);
    dz -= L * roundf(dz * inv_L);
    float dr2 = dx*dx + dy*dy + dz*dz;
    atomicMax(d_max_dr2_int, __float_as_int(dr2));
}

inline bool check_and_reset_trigger(int* d_max_dr2_int, float skin) {
    int max_int;
    CUDA_CHECK(cudaMemcpy(&max_int, d_max_dr2_int, sizeof(int),
                           cudaMemcpyDeviceToHost));
    float max_dr2;
    std::memcpy(&max_dr2, &max_int, sizeof(float));
    float half_skin = skin * 0.5f;

    if (max_dr2 > half_skin * half_skin) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_max_dr2_int, &zero, sizeof(int),
                               cudaMemcpyHostToDevice));
        return true;
    }
    return false;
}
