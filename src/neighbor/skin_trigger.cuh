#pragma once
#include "../core/types.cuh"
#include <cstring>

template <bool TILT = true>
__device__ __forceinline__
void update_max_displacement(float4 pos_new, float4 pos_ref,
                              int* d_max_dr2_int, const Box& box) {
    float3 d = min_image_dr_t<TILT>(pos_new, pos_ref, box);
    float dr2 = d.x * d.x + d.y * d.y + d.z * d.z;
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
