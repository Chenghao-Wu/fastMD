#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "analysis/thermo.cuh"

TEST(Thermo, KineticEnergyAndTemperature) {
    const int N = 96;
    const float L = 10.0f;
    const int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;

    std::vector<float4> h_vel(np, make_float4(0,0,0,0));
    for (int i = 0; i < N; i++) {
        h_vel[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    }

    float4* d_vel = to_device(h_vel);

    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));

    ThermoBuffers bufs;
    bufs.allocate();
    ThermoOutput output;
    compute_thermo(d_vel, d_force, d_virial, N, L, &output, bufs, 0, nullptr);

    EXPECT_NEAR(output.kinetic_energy, 0.5f, 0.01f);
    EXPECT_NEAR(output.temperature, 1.0f/3.0f, 0.001f);

    bufs.free();
    free_device(d_vel);
    free_device(d_force);
    free_device(d_virial);
}
