#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "integrate/langevin.cuh"
#include "force/lj.cuh"
#include "neighbor/verlet_list.cuh"
#include "neighbor/skin_trigger.cuh"

TEST(Integrator, NVEEnergyConservation) {
    const int N = 64;
    const float L = 8.0f;
    const float inv_L = 1.0f / L;
    const float rc = 2.5f;
    const float rc2 = rc * rc;
    const float skin = 0.5f;
    const float dt = 0.001f;
    const int ntypes = 1;
    const int ntiles = div_ceil(N, TILE_SIZE);
    const int np = ntiles * TILE_SIZE;
    const int nsteps = 1000;

    std::vector<float4> h_pos(np, make_float4(0,0,0, pack_type_id(-1)));
    int idx = 0;
    float spacing = L / 4.0f;
    for (int ix = 0; ix < 4 && idx < N; ix++)
        for (int iy = 0; iy < 4 && idx < N; iy++)
            for (int iz = 0; iz < 4 && idx < N; iz++) {
                h_pos[idx] = make_float4(ix * spacing + 0.1f,
                                          iy * spacing + 0.1f,
                                          iz * spacing + 0.1f,
                                          pack_type_id(0));
                idx++;
            }

    std::vector<float4> h_vel(np, make_float4(0,0,0,0));
    srand(42);
    for (int i = 0; i < N; i++) {
        h_vel[i] = make_float4(0.1f * (rand()/(float)RAND_MAX - 0.5f),
                                0.1f * (rand()/(float)RAND_MAX - 0.5f),
                                0.1f * (rand()/(float)RAND_MAX - 0.5f), 0.0f);
    }

    float4* d_pos = to_device(h_pos);
    float4* d_vel = to_device(h_vel);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    float4* d_pos_ref;
    CUDA_CHECK(cudaMalloc(&d_pos_ref, np * sizeof(float4)));
    int* d_max_dr2;
    CUDA_CHECK(cudaMalloc(&d_max_dr2, sizeof(int)));
    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));

    std::vector<float2> h_lj = {{1.0f, 1.0f}};
    float2* d_lj = to_device(h_lj);

    VerletList verlet;
    verlet.allocate(N, rc + skin, L);

    LangevinState lang;
    lang.init(np, 1e10f, dt, 1.0f, 12345);  // Tdamp≈∞ → gamma≈0 → NVE limit

    CUDA_CHECK(cudaMemcpy(d_pos_ref, d_pos, np * sizeof(float4),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_max_dr2, 0, sizeof(int)));
    verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);

    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                      verlet.neighbors, verlet.num_neighbors,
                      N, ntypes, rc2, L, inv_L);

    auto compute_total_energy = [&]() -> float {
        auto h_f = to_host(d_force, N);
        auto h_v = to_host(d_vel, N);
        float ke = 0, pe = 0;
        for (int i = 0; i < N; i++) {
            ke += 0.5f * (h_v[i].x*h_v[i].x + h_v[i].y*h_v[i].y + h_v[i].z*h_v[i].z);
            pe += h_f[i].w;
        }
        return ke + pe;
    };

    float E0 = compute_total_energy();

    for (int step = 0; step < nsteps; step++) {
        launch_integrator_pre_force(d_pos, d_vel, d_force, d_pos_ref,
                                     d_max_dr2, nullptr, lang, N, L, inv_L,
                                     lang.half_dt);

        if (check_and_reset_trigger(d_max_dr2, skin)) {
            CUDA_CHECK(cudaMemcpy(d_pos_ref, d_pos, np * sizeof(float4),
                                   cudaMemcpyDeviceToDevice));
            verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);
        }

        CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
        CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
        launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                          verlet.neighbors, verlet.num_neighbors,
                          N, ntypes, rc2, L, inv_L);

        launch_integrator_post_force(d_vel, d_force, N, lang.half_dt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float E_final = compute_total_energy();
    float drift = fabsf(E_final - E0) / fabsf(E0);

    EXPECT_LT(drift, 0.01f) << "E0=" << E0 << " E_final=" << E_final;

    lang.free();
    verlet.free();
    free_device(d_pos);
    free_device(d_vel);
    free_device(d_force);
    free_device(d_pos_ref);
    free_device(d_max_dr2);
    free_device(d_virial);
    free_device(d_lj);
}
