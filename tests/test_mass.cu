#include <gtest/gtest.h>
#include <cmath>
#include "test_utils.cuh"
#include "core/system.cuh"
#include "core/types.cuh"
#include "integrate/langevin.cuh"
#include "analysis/thermo.cuh"

extern __global__ void integrator_post_force_kernel(
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ pos,
    int natoms, float half_dt);

extern __global__ void kinetic_stress_kernel(
    const float4* __restrict__ vel,
    const float4* __restrict__ pos,
    float* __restrict__ kin_stress,
    int natoms);

TEST(MassTest, VelocityHalfStepMassScaling) {
    int natoms = 2;
    int ntypes = 2;

    std::vector<float4> h_pos = {
        make_float4(0.0f, 0.0f, 0.0f, pack_type_id(0)),
        make_float4(1.0f, 0.0f, 0.0f, pack_type_id(1)),
    };
    std::vector<float4> h_vel = {
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
    };
    std::vector<float4> h_force = {
        make_float4(2.0f, 0.0f, 0.0f, 0.0f),
        make_float4(2.0f, 0.0f, 0.0f, 0.0f),
    };

    float4* d_pos   = to_device(h_pos);
    float4* d_vel   = to_device(h_vel);
    float4* d_force = to_device(h_force);

    float h_masses[2] = {1.0f, 2.0f};
    CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_masses, 2 * sizeof(float)));

    float half_dt = 0.001f;
    int blocks = div_ceil(natoms, 256);
    integrator_post_force_kernel<<<blocks, 256>>>(d_vel, d_force, d_pos,
                                                   natoms, half_dt);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float4> result_vel = to_host(d_vel, natoms);

    EXPECT_NEAR(result_vel[0].x, 1.0f + half_dt * 2.0f / 1.0f, 1e-6f);
    EXPECT_NEAR(result_vel[1].x, 1.0f + half_dt * 2.0f / 2.0f, 1e-6f);

    float dv0 = result_vel[0].x - 1.0f;
    float dv1 = result_vel[1].x - 1.0f;
    EXPECT_NEAR(dv0, 2.0f * dv1, 1e-6f);

    free_device(d_pos);
    free_device(d_vel);
    free_device(d_force);
}

TEST(MassTest, KineticEnergyMassWeighted) {
    int natoms = 2;
    int ntypes = 2;

    std::vector<float4> h_pos = {
        make_float4(0.0f, 0.0f, 0.0f, pack_type_id(0)),
        make_float4(1.0f, 0.0f, 0.0f, pack_type_id(1)),
    };
    std::vector<float4> h_vel = {
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
    };

    float4* d_pos = to_device(h_pos);
    float4* d_vel = to_device(h_vel);

    float h_masses[2] = {1.0f, 2.0f};
    CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_masses, 2 * sizeof(float)));

    float* d_kin_stress;
    CUDA_CHECK(cudaMalloc(&d_kin_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kin_stress, 0, 6 * sizeof(float)));

    int blocks = div_ceil(natoms, 256);
    kinetic_stress_kernel<<<blocks, 256>>>(d_vel, d_pos, d_kin_stress, natoms);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_stress = to_host(d_kin_stress, 6);

    float expected_xx = 1.0f * 1.0f * 1.0f + 2.0f * 1.0f * 1.0f;
    EXPECT_NEAR(h_stress[0], expected_xx, 1e-5f);

    float ke = 0.5f * (h_stress[0] + h_stress[3] + h_stress[5]);
    float expected_ke = 0.5f * (1.0f + 2.0f);
    EXPECT_NEAR(ke, expected_ke, 1e-5f);

    free_device(d_pos);
    free_device(d_vel);
    CUDA_CHECK(cudaFree(d_kin_stress));
}

TEST(MassTest, DefaultMassesAreUnity) {
    int natoms = 3;
    int ntypes = 1;

    std::vector<float4> h_pos = {
        make_float4(0.0f, 0.0f, 0.0f, pack_type_id(0)),
        make_float4(1.0f, 0.0f, 0.0f, pack_type_id(0)),
        make_float4(2.0f, 0.0f, 0.0f, pack_type_id(0)),
    };
    std::vector<float4> h_vel = {
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        make_float4(2.0f, 0.0f, 0.0f, 0.0f),
        make_float4(3.0f, 0.0f, 0.0f, 0.0f),
    };

    float4* d_pos = to_device(h_pos);
    float4* d_vel = to_device(h_vel);

    float h_masses[16];
    for (int i = 0; i < 16; i++) h_masses[i] = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_masses, 16 * sizeof(float)));

    float* d_kin_stress;
    CUDA_CHECK(cudaMalloc(&d_kin_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kin_stress, 0, 6 * sizeof(float)));

    int blocks = div_ceil(natoms, 256);
    kinetic_stress_kernel<<<blocks, 256>>>(d_vel, d_pos, d_kin_stress, natoms);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_stress = to_host(d_kin_stress, 6);

    float expected_xx = 1.0f + 4.0f + 9.0f;
    EXPECT_NEAR(h_stress[0], expected_xx, 1e-5f);

    free_device(d_pos);
    free_device(d_vel);
    CUDA_CHECK(cudaFree(d_kin_stress));
}
