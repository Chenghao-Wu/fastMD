#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/dihedral_opls.cuh"
#include "../reference/cpu_reference.hpp"
#include <cstdlib>

// butane-like zigzag: C1-C2-C3-C4, with an out-of-plane kink so the
// dihedral is well away from 0 and 180 (avoids the sin(phi)->0 singular).
static std::vector<float4> make_zigzag() {
    return {
        {0.000f, 0.000f, 0.000f, pack_type_id(0)},
        {1.530f, 0.000f, 0.000f, pack_type_id(0)},
        {2.400f, 1.250f, 0.300f, pack_type_id(0)},
        {3.300f, 0.400f, 1.350f, pack_type_id(0)},
    };
}

TEST(DihedralOPLS, ForceMatchesCPU) {
    const float L = 20.0f;
    const float inv_L = 1.0f / L;
    auto h_pos = make_zigzag();
    const int N = h_pos.size();

    std::vector<int4> h_dih = {{0, 1, 2, 3}};
    std::vector<int> h_dt = {0};
    std::vector<float> K1 = {3.1f}, K2 = {-0.6f}, K3 = {5.9f}, K4 = {0.2f};

    auto ref = ref::brute_force_opls_dihedral(h_pos, h_dih, h_dt,
                                               K1, K2, K3, K4, L, inv_L);

    int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> h_pos_pad(np, make_float4(0,0,0,pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());

    float4* d_pos = to_device(h_pos_pad);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    double* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(double)));

    int4* d_dih = to_device(h_dih);
    int* d_dt = to_device(h_dt);
    OPLSDihedralParams hp = {3.1f, -0.6f, 5.9f, 0.2f};
    OPLSDihedralParams* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(OPLSDihedralParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &hp, sizeof(OPLSDihedralParams),
                           cudaMemcpyHostToDevice));

    launch_dihedral_opls_kernel(d_pos, d_force, d_virial, d_dih, d_dt,
                                 d_params, 1, cubic_box(L));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_force_out = to_host(d_force, N);
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            float g = (d == 0) ? h_force_out[i].x : (d == 1) ? h_force_out[i].y : h_force_out[i].z;
            float r = (d == 0) ? ref.fx[i] : (d == 1) ? ref.fy[i] : ref.fz[i];
            float denom = fmaxf(fabsf(g), fabsf(r));
            float err = (denom < 1e-3f) ? fabsf(g - r) : fabsf(g - r) / denom;
            EXPECT_LT(err, 1e-3f) << "atom " << i << " dim " << d
                                  << " gpu=" << g << " ref=" << r;
        }
    }

    // total force must vanish
    float tfx = 0, tfy = 0, tfz = 0;
    for (int i = 0; i < N; i++) {
        tfx += h_force_out[i].x; tfy += h_force_out[i].y; tfz += h_force_out[i].z;
    }
    EXPECT_NEAR(tfx, 0.0f, 1e-3f);
    EXPECT_NEAR(tfy, 0.0f, 1e-3f);
    EXPECT_NEAR(tfz, 0.0f, 1e-3f);

    free_device(d_pos); free_device(d_force); free_device(d_virial);
    free_device(d_dih); free_device(d_dt); free_device(d_params);
}

// Finite-difference check of the energy/force consistency (catches sign
// errors the CPU reference might share with the kernel).
TEST(DihedralOPLS, ForceIsNegativeEnergyGradient) {
    const float L = 20.0f;
    const float inv_L = 1.0f / L;
    auto h_pos = make_zigzag();
    const int N = h_pos.size();
    std::vector<int4> h_dih = {{0, 1, 2, 3}};
    std::vector<int> h_dt = {0};
    std::vector<float> K1 = {3.1f}, K2 = {-0.6f}, K3 = {5.9f}, K4 = {0.2f};

    int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> h_pos_pad(np, make_float4(0,0,0,pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());
    float4* d_pos = to_device(h_pos_pad);
    int4* d_dih = to_device(h_dih);
    int* d_dt = to_device(h_dt);
    OPLSDihedralParams hp = {3.1f, -0.6f, 5.9f, 0.2f};
    OPLSDihedralParams* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(OPLSDihedralParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &hp, sizeof(OPLSDihedralParams),
                           cudaMemcpyHostToDevice));

    auto run_forces = [&](const std::vector<float4>& pos,
                          std::vector<float4>& fout, float& pe_tot) {
        std::vector<float4> pad(np, make_float4(0,0,0,pack_type_id(-1)));
        std::copy(pos.begin(), pos.end(), pad.begin());
        CUDA_CHECK(cudaMemcpy(d_pos, pad.data(), np * sizeof(float4),
                               cudaMemcpyHostToDevice));
        float4* d_force;
        CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
        CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
        double* d_virial;
        CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(double)));
        launch_dihedral_opls_kernel(d_pos, d_force, d_virial, d_dih, d_dt,
                                     d_params, 1, cubic_box(L));
        CUDA_CHECK(cudaDeviceSynchronize());
        fout = to_host(d_force, N);
        pe_tot = 0;
        for (int i = 0; i < N; i++) pe_tot += fout[i].w;
        free_device(d_force); free_device(d_virial);
    };

    std::vector<float4> f0;
    float pe0;
    run_forces(h_pos, f0, pe0);

    const float eps_fd = 1e-3f;
    srand(1234);
    for (int trial = 0; trial < 12; trial++) {
        int atom = rand() % N;
        int dim = rand() % 3;
        auto pos_p = h_pos, pos_m = h_pos;
        float* cp = (dim == 0) ? &pos_p[atom].x : (dim == 1) ? &pos_p[atom].y : &pos_p[atom].z;
        float* cm = (dim == 0) ? &pos_m[atom].x : (dim == 1) ? &pos_m[atom].y : &pos_m[atom].z;
        *cp += eps_fd;
        *cm -= eps_fd;
        std::vector<float4> fp, fm;
        float pep, pem;
        run_forces(pos_p, fp, pep);
        run_forces(pos_m, fm, pem);
        float fd = -(pep - pem) / (2.0f * eps_fd);
        float f_gpu = (dim == 0) ? f0[atom].x : (dim == 1) ? f0[atom].y : f0[atom].z;
        float denom = fmaxf(fabsf(fd), fabsf(f_gpu));
        float rel = (denom < 1e-6f) ? fabsf(fd - f_gpu) : fabsf(fd - f_gpu) / denom;
        EXPECT_LT(rel, 5e-2f) << "atom " << atom << " dim " << dim
                              << " fd=" << fd << " gpu=" << f_gpu;
    }

    free_device(d_pos); free_device(d_dih); free_device(d_dt); free_device(d_params);
}
