#pragma once
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "../src/core/types.cuh"

template<typename T>
T* to_device(const std::vector<T>& host_data) {
    T* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, host_data.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_ptr, host_data.data(),
                          host_data.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

template<typename T>
std::vector<T> to_host(const T* d_ptr, size_t count) {
    std::vector<T> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), d_ptr,
                          count * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return result;
}

template<typename T>
void free_device(T* d_ptr) {
    CUDA_CHECK(cudaFree(d_ptr));
}

inline void assert_float3_near(float3 a, float3 b, float rel_tol = 1e-4f) {
    auto rel_err = [](float x, float y) -> float {
        float denom = fmaxf(fabsf(x), fabsf(y));
        if (denom < 1e-12f) return fabsf(x - y);
        return fabsf(x - y) / denom;
    };
    EXPECT_LT(rel_err(a.x, b.x), rel_tol) << "x mismatch";
    EXPECT_LT(rel_err(a.y, b.y), rel_tol) << "y mismatch";
    EXPECT_LT(rel_err(a.z, b.z), rel_tol) << "z mismatch";
}
