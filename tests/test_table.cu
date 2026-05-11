#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/lj.cuh"
#include "force/table.cuh"
#include "neighbor/verlet_list.cuh"
#include "io/table_parser.hpp"

TEST(Table, MatchesLJWithinInterpolationTolerance) {
    const int N = 64;
    const float L = 8.0f;
    const float inv_L = 1.0f / L;
    const float rc = 2.5f;
    const float rc2 = rc * rc;
    const float skin = 0.5f;
    const int ntypes = 1;
    const int ntiles = div_ceil(N, TILE_SIZE);
    const int np = ntiles * TILE_SIZE;

    std::vector<float4> h_pos(N);
    srand(123);
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            pack_type_id(0));
    }

    // Build LJ-equivalent table with clamped forces to match LJ kernel
    int ntable = 10000;
    float rlo = 0.01f, rhi = rc;
    float dr = (rhi - rlo) / (ntable - 1);
    std::vector<float4> h_table_data(ntable);
    for (int k = 0; k < ntable; k++) {
        float r = rlo + k * dr;
        float r2 = r * r;
        float r2inv = 1.0f / r2;
        float sr2 = r2inv;
        float sr6 = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6;
        float force = 24.0f * r2inv * (2.0f * sr12 - sr6);
        force = fminf(fmaxf(force, -1000.0f), 1000.0f);
        float energy = 4.0f * (sr12 - sr6);
        h_table_data[k] = make_float4(r, force, energy, 0.0f);
    }
    TableParams tp;
    tp.rmin = rlo; tp.rmax = rhi; tp.dr = dr;
    tp.inv_dr = 1.0f / dr; tp.npoints = ntable; tp.data_offset = 0;

    std::vector<int> h_table_idx(ntypes * ntypes, 0);
    std::vector<float2> h_lj = {{1.0f, 1.0f}};

    std::vector<float4> h_pos_pad(np, make_float4(0,0,0, pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());

    float4* d_pos = to_device(h_pos_pad);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));
    float2* d_lj = to_device(h_lj);
    int* d_table_idx = to_device(h_table_idx);
    TableParams* d_table_params = to_device(std::vector<TableParams>{tp});
    float4* d_table_data = to_device(h_table_data);

    VerletList verlet;
    verlet.allocate(N, rc + skin, L);
    verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);

    // Run LJ
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                     verlet.neighbors, verlet.num_neighbors,
                     N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto h_force_lj = to_host(d_force, N);
    float h_virial_lj[6];
    CUDA_CHECK(cudaMemcpy(h_virial_lj, d_virial, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    // Run table
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_table_kernel(d_pos, d_force, d_virial,
                        d_table_idx, d_table_params, d_table_data,
                        verlet.neighbors, verlet.num_neighbors,
                        N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto h_force_table = to_host(d_force, N);
    float h_virial_table[6];
    CUDA_CHECK(cudaMemcpy(h_virial_table, d_virial, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float3 lj_f = make_float3(h_force_lj[i].x, h_force_lj[i].y, h_force_lj[i].z);
        float3 t_f  = make_float3(h_force_table[i].x, h_force_table[i].y, h_force_table[i].z);
        assert_float3_near(lj_f, t_f, 1e-3f);
        float rel_pe = fabsf(h_force_lj[i].w - h_force_table[i].w)
                       / fmaxf(fabsf(h_force_lj[i].w), 1e-6f);
        EXPECT_LT(rel_pe, 1e-3f) << "PE mismatch at atom " << i;
    }
    for (int k = 0; k < 6; k++) {
        float denom = fmaxf(fabsf(h_virial_lj[k]), 1e-6f);
        float rel = fabsf(h_virial_table[k] - h_virial_lj[k]) / denom;
        EXPECT_LT(rel, 1e-2f) << "virial component " << k;
    }

    verlet.free();
    free_device(d_pos); free_device(d_force); free_device(d_virial);
    free_device(d_lj); free_device(d_table_idx); free_device(d_table_params);
    free_device(d_table_data);
}
