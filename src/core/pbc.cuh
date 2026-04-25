#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float min_image(float dr, float L, float inv_L) {
    return dr - L * rintf(dr * inv_L);
}

__device__ __forceinline__ float3 min_image_dr(float4 r1, float4 r2,
                                                float L, float inv_L) {
    return make_float3(
        min_image(r1.x - r2.x, L, inv_L),
        min_image(r1.y - r2.y, L, inv_L),
        min_image(r1.z - r2.z, L, inv_L)
    );
}

__device__ __forceinline__ float wrap(float x, float L, float inv_L) {
    return x - L * floorf(x * inv_L);
}

__device__ __forceinline__ float4 wrap_position(float4 r, float L, float inv_L) {
    return make_float4(
        wrap(r.x, L, inv_L),
        wrap(r.y, L, inv_L),
        wrap(r.z, L, inv_L),
        r.w
    );
}
