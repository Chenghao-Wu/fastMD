#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/bond_harmonic.cuh"
#include "../reference/cpu_reference.hpp"

TEST(BondHarmonic, TwoAtomMatchesCPU) {
    const int N = 2;
    const float L = 10.0f;
    const float inv_L = 1.0f / L;

    std::vector<float4> h_pos = {
        {4.0f, 5.0f, 5.0f, pack_type_id(0)},
        {5.3f, 5.0f, 5.0f, pack_type_id(0)},
    };
    std::vector<int2> h_bonds = {{0, 1}};
    std::vector<int> h_btypes = {0};
    std::vector<float> h_k = {300.0f};
    std::vector<float> h_r0 = {1.0f};

    auto ref = ref::brute_force_harmonic_bond(h_pos, h_bonds, h_btypes,
                                               h_k, h_r0, L, inv_L);

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

    int2* d_bonds = to_device(h_bonds);
    int* d_btypes = to_device(h_btypes);
    BondHarmonicParams hp = {300.0f, 1.0f};
    BondHarmonicParams* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(BondHarmonicParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &hp, sizeof(BondHarmonicParams),
                           cudaMemcpyHostToDevice));

    launch_bond_harmonic_kernel(d_pos, d_force, d_virial, d_bonds, d_btypes,
                                 d_params, 1, cubic_box(L));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_force_out = to_host(d_force, N);
    for (int i = 0; i < N; i++) {
        float3 gpu_f = make_float3(h_force_out[i].x, h_force_out[i].y, h_force_out[i].z);
        float3 ref_f = make_float3(ref.fx[i], ref.fy[i], ref.fz[i]);
        assert_float3_near(gpu_f, ref_f, 1e-3f);
    }

    // Sign check (catches shared-convention sign errors): bond is stretched
    // (r = 1.3 > r0 = 1.0), so the force must be ATTRACTIVE — atom 0 pulled
    // toward +x, atom 1 toward -x (LAMMPS bond_harmonic: fbond = -2K dr/r).
    EXPECT_GT(h_force_out[0].x, 0.0f);
    EXPECT_LT(h_force_out[1].x, 0.0f);
    EXPECT_NEAR(h_force_out[0].x, 2.0f * 300.0f * 0.3f, 1e-3f);

    // PE: K (r-r0)^2 = 300 * 0.3^2 = 27
    float pe_tot = h_force_out[0].w + h_force_out[1].w;
    EXPECT_NEAR(pe_tot, 27.0f, 1e-3f);

    // virial xx = fx * dx = (2*300*0.3/1.3*1.3) * 1.3 = 2*300*0.3*1.3 = 234
    auto h_vir = to_host(d_virial, 6);
    EXPECT_NEAR(h_vir[0], ref.virial[0], 1e-2f);

    free_device(d_pos); free_device(d_force); free_device(d_virial);
    free_device(d_bonds); free_device(d_btypes); free_device(d_params);
}
