#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "integrate/nose_hoover.cuh"
#include "force/lj.cuh"
#include "neighbor/verlet_list.cuh"
#include "neighbor/skin_trigger.cuh"

static float compute_total_ke(const std::vector<float4>& vel_h, int N) {
    float ke = 0.0f;
    for (int i = 0; i < N; i++) {
        ke += 0.5f * (vel_h[i].x * vel_h[i].x
                    + vel_h[i].y * vel_h[i].y
                    + vel_h[i].z * vel_h[i].z);
    }
    return ke;
}

static float compute_conserved(const std::vector<float4>& vel_h,
                                const std::vector<float4>& force_h,
                                const NoseHooverState& nh,
                                int natoms) {
    float ke = compute_total_ke(vel_h, natoms);
    float pe = 0.0f;
    for (int i = 0; i < natoms; i++) pe += force_h[i].w;

    // System-wide chain energy
    float N_f = 3.0f * natoms;
    float kT = nh.T_target;
    float Q1 = nh.Q1;
    float Q_rest = nh.Q_rest;

    float chain_energy = 0.5f * Q1 * nh.v_xi[0] * nh.v_xi[0]
                       + N_f * kT * nh.xi[0];
    for (int k = 1; k < nh.M; k++) {
        chain_energy += 0.5f * Q_rest * nh.v_xi[k] * nh.v_xi[k]
                      + kT * nh.xi[k];
    }
    return ke + pe + chain_energy;
}

// Test 1: NVE limit — NH NVT with very large Tdamp should conserve energy
TEST(NoseHoover, NVTLimitEnergyConservation) {
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
    const int nsteps = 500;
    const int M = 3;
    const float Tdamp_large = 1000.0f;

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

    float4* d_pos   = to_device(h_pos);
    float4* d_vel   = to_device(h_vel);
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

    SimParams params = {};
    params.natoms = N;
    params.box_L = L;
    params.inv_L = inv_L;
    params.half_L = L * 0.5f;
    params.dt = dt;
    params.nsteps = nsteps;
    params.ntypes = ntypes;
    params.ensemble = Ensemble::NVT_NH;
    params.T_start = 1.0f;
    params.T_stop  = 1.0f;
    params.Tdamp = Tdamp_large;
    params.nh_chain_length = M;

    NoseHooverState nh;
    nh.init(params);

    CUDA_CHECK(cudaMemcpy(d_pos_ref, d_pos, np * sizeof(float4),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_max_dr2, 0, sizeof(int)));
    verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                      verlet.neighbors, verlet.num_neighbors,
                      N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto vel_h   = to_host(d_vel, N);
    auto force_h = to_host(d_force, np);
    float E0 = compute_conserved(vel_h, force_h, nh, N);

    float hdt = 0.5f * dt;

    for (int step = 0; step < nsteps; step++) {
        float total_ke = compute_total_ke(to_host(d_vel, N), N);

        float nh_scale;
        nh_propagate_chain(nh, total_ke, hdt, nh_scale);
        launch_nh_global_scale_vel(d_vel, nh_scale, N);

        launch_nh_v_verlet_half(d_vel, d_force, N, hdt);
        launch_nh_update_pos(d_pos, d_vel, d_pos_ref,
                              nullptr, d_max_dr2,
                              N, L, inv_L, 1.0f, 0.0f, dt);

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

        launch_nh_v_verlet_half(d_vel, d_force, N, hdt);

        total_ke = compute_total_ke(to_host(d_vel, N), N);
        nh_propagate_chain(nh, total_ke, hdt, nh_scale);
        launch_nh_global_scale_vel(d_vel, nh_scale, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    vel_h   = to_host(d_vel, N);
    force_h = to_host(d_force, np);

    float E_final = compute_conserved(vel_h, force_h, nh, N);
    float drift = fabsf(E_final - E0) / fabsf(E0);

    EXPECT_LT(drift, 0.01f) << "Conserved quantity drift too large: "
                             << "E0=" << E0 << " E_final=" << E_final;

    nh.free();
    verlet.free();
    free_device(d_pos);
    free_device(d_vel);
    free_device(d_force);
    free_device(d_pos_ref);
    free_device(d_max_dr2);
    free_device(d_virial);
    free_device(d_lj);
}

// Test 2: NH NVT reaches target temperature
TEST(NoseHoover, NVTReachesTargetTemperature) {
    const int N = 100;
    const float L = 10.0f;
    const float inv_L = 1.0f / L;
    const float rc = 2.5f;
    const float rc2 = rc * rc;
    const float skin = 0.5f;
    const float dt = 0.002f;
    const int ntypes = 1;
    const int ntiles = div_ceil(N, TILE_SIZE);
    const int np = ntiles * TILE_SIZE;
    const int nsteps = 5000;
    const int M = 3;
    const float Tdamp = 0.5f;
    const float T_target = 1.5f;

    std::vector<float4> h_pos(np, make_float4(0,0,0, pack_type_id(-1)));
    srand(123);
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4((rand()/(float)RAND_MAX) * L,
                                (rand()/(float)RAND_MAX) * L,
                                (rand()/(float)RAND_MAX) * L,
                                pack_type_id(0));
    }
    std::vector<float4> h_vel(np, make_float4(0,0,0,0));
    for (int i = 0; i < N; i++) {
        h_vel[i] = make_float4(0.3f * (rand()/(float)RAND_MAX - 0.5f),
                                0.3f * (rand()/(float)RAND_MAX - 0.5f),
                                0.3f * (rand()/(float)RAND_MAX - 0.5f), 0.0f);
    }

    float4* d_pos   = to_device(h_pos);
    float4* d_vel   = to_device(h_vel);
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

    SimParams params = {};
    params.natoms = N;
    params.box_L = L;
    params.inv_L = inv_L;
    params.half_L = L * 0.5f;
    params.dt = dt;
    params.nsteps = nsteps;
    params.ntypes = ntypes;
    params.ensemble = Ensemble::NVT_NH;
    params.T_start = T_target;
    params.T_stop  = T_target;
    params.Tdamp = Tdamp;
    params.nh_chain_length = M;

    NoseHooverState nh;
    nh.init(params);

    CUDA_CHECK(cudaMemcpy(d_pos_ref, d_pos, np * sizeof(float4),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_max_dr2, 0, sizeof(int)));
    verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                      verlet.neighbors, verlet.num_neighbors,
                      N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());

    float hdt = 0.5f * dt;

    double ke_sum = 0.0;
    int ke_count = 0;

    for (int step = 0; step < nsteps; step++) {
        float total_ke = compute_total_ke(to_host(d_vel, N), N);

        float nh_scale;
        nh_propagate_chain(nh, total_ke, hdt, nh_scale);
        launch_nh_global_scale_vel(d_vel, nh_scale, N);

        launch_nh_v_verlet_half(d_vel, d_force, N, hdt);
        launch_nh_update_pos(d_pos, d_vel, d_pos_ref,
                              nullptr, d_max_dr2,
                              N, L, inv_L, 1.0f, 0.0f, dt);

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

        launch_nh_v_verlet_half(d_vel, d_force, N, hdt);

        total_ke = compute_total_ke(to_host(d_vel, N), N);
        nh_propagate_chain(nh, total_ke, hdt, nh_scale);
        launch_nh_global_scale_vel(d_vel, nh_scale, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (step >= nsteps / 2) {
            auto vel_h = to_host(d_vel, N);
            ke_sum += compute_total_ke(vel_h, N);
            ke_count++;
        }
    }

    double T_avg = (2.0 * ke_sum / ke_count) / (3.0 * N);
    EXPECT_NEAR(T_avg, T_target, 0.1f)
        << "Average temperature " << T_avg
        << " deviates too much from target " << T_target;

    nh.free();
    verlet.free();
    free_device(d_pos);
    free_device(d_vel);
    free_device(d_force);
    free_device(d_pos_ref);
    free_device(d_max_dr2);
    free_device(d_virial);
    free_device(d_lj);
}
