#include "core/types.cuh"
#include "core/system.cuh"
#include "core/morton.cuh"
#include "core/pbc.cuh"
#include "neighbor/verlet_list.cuh"
#include "neighbor/skin_trigger.cuh"
#include "force/lj.cuh"
#include "force/fene.cuh"
#include "force/angle.cuh"
#include "integrate/langevin.cuh"
#include "integrate/nose_hoover.cuh"
#include "analysis/thermo.cuh"
#include "analysis/correlator.cuh"
#include "analysis/rg.cuh"
#include "io/dump.cuh"
#include "io/config.hpp"
#include "io/lammps_data.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }

    TopologyData topo;
    SimParams params = parse_config(argv[1], topo);
    if (!topo.data_file.empty()) {
        parse_lammps_data(topo.data_file, topo);
    }
    params.box_L = topo.box_L;
    finalize_params(params);

    printf("[setup]  atoms=%d  box=%.2f  rc=%.2f  dt=%.4f  nsteps=%d",
           params.natoms, params.box_L, params.rc, params.dt, params.nsteps);
    if (topo.bonds.size() > 0) printf("  bonds=%zu", topo.bonds.size());
    if (topo.angles.size() > 0) printf("  angles=%zu", topo.angles.size());
    printf("\n");

    if (params.ensemble != Ensemble::Langevin) {
        printf("[setup]  ensemble=%s  T_start=%.2f  T_stop=%.2f  Tdamp=%.2f  chain=%d",
               params.ensemble == Ensemble::NPT_NH ? "npt_nh" : "nvt_nh",
               params.T_start, params.T_stop, params.Tdamp, params.nh_chain_length);
        if (params.ensemble == Ensemble::NPT_NH) {
            printf("  P_start=%.2f  P_stop=%.2f  Pdamp=%.2f",
                   params.P_start, params.P_stop, params.Pdamp);
        }
    } else {
        printf("[setup]  ensemble=nvt_langevin  T_start=%.2f  T_stop=%.2f  Tdamp=%.2f  seed=%lu",
               params.T_start, params.T_stop, params.Tdamp, (unsigned long)params.seed);
    }
    if (params.rg_on) printf("  rg_on");
    if (params.stress_on) printf("  stress_on");
    if (params.restart_freq >= 0) printf("  restart_on");
    printf("\n\n");

    System sys;
    sys.allocate(params);
    if (topo.bonds.size() > 0) {
        sys.nbonds = static_cast<int>(topo.bonds.size());
        CUDA_CHECK(cudaMalloc(&sys.bonds, sys.nbonds * sizeof(int2)));
        CUDA_CHECK(cudaMalloc(&sys.bond_param_idx, sys.nbonds * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(sys.bonds, topo.bonds.data(),
                              sys.nbonds * sizeof(int2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sys.bond_param_idx, topo.bond_types.data(),
                              sys.nbonds * sizeof(int), cudaMemcpyHostToDevice));

        build_exclusions(topo, topo);
        sys.nexclusions = static_cast<int>(topo.exclusion_list.size());
        if (sys.nexclusions > 0) {
            CUDA_CHECK(cudaMalloc(&sys.exclusion_offsets,
                                  (params.natoms + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&sys.exclusion_list,
                                  sys.nexclusions * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(sys.exclusion_offsets, topo.exclusion_offsets.data(),
                                  (params.natoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(sys.exclusion_list, topo.exclusion_list.data(),
                                  sys.nexclusions * sizeof(int), cudaMemcpyHostToDevice));
        }
    }
    if (topo.angles.size() > 0) {
        sys.nangles = static_cast<int>(topo.angles.size());
        CUDA_CHECK(cudaMalloc(&sys.angles, sys.nangles * sizeof(int4)));
        CUDA_CHECK(cudaMemcpy(sys.angles, topo.angles.data(),
                              sys.nangles * sizeof(int4), cudaMemcpyHostToDevice));
    }

    int np = sys.natoms_padded;
    std::vector<float4> pos_pad(np, make_float4(0,0,0, pack_type_id(-1)));
    std::vector<float4> vel_pad(np, make_float4(0,0,0,0));
    std::copy(topo.positions.begin(), topo.positions.end(), pos_pad.begin());
    std::copy(topo.velocities.begin(), topo.velocities.end(), vel_pad.begin());
    CUDA_CHECK(cudaMemcpy(sys.pos, pos_pad.data(), np * sizeof(float4),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sys.vel, vel_pad.data(), np * sizeof(float4),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sys.lj_params, topo.lj_params.data(),
                           params.ntypes * params.ntypes * sizeof(float2),
                           cudaMemcpyHostToDevice));

    FENEParams* d_fene_params = nullptr;
    AngleParams* d_angle_params = nullptr;
    if (topo.bond_params.size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_fene_params,
                              topo.bond_params.size() * sizeof(FENEParams)));
        CUDA_CHECK(cudaMemcpy(d_fene_params, topo.bond_params.data(),
                              topo.bond_params.size() * sizeof(FENEParams),
                              cudaMemcpyHostToDevice));
    }
    if (topo.angle_params.size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_angle_params,
                              topo.angle_params.size() * sizeof(AngleParams)));
        CUDA_CHECK(cudaMemcpy(d_angle_params, topo.angle_params.data(),
                              topo.angle_params.size() * sizeof(AngleParams),
                              cudaMemcpyHostToDevice));
    }

    MortonSorter morton;
    morton.allocate(np);

    VerletList verlet;
    verlet.allocate(params.natoms, params.rc + params.skin, params.box_L);

    LangevinState langevin;
    langevin.init(np, params.Tdamp, params.dt, params.T_start, params.seed);

    NoseHooverState nose_hoover;
    if (params.ensemble != Ensemble::Langevin) {
        nose_hoover.init(params);
    }

    MultipleTauCorrelator correlator;
    if (params.stress_on) {
        correlator.allocate();
    }

    ThermoBuffers thermo_bufs;
    thermo_bufs.allocate();
    if (params.thermo_on) {
        thermo_bufs.open_file(params.thermo_file);
    }

    RgBuffers rg_bufs;
    std::vector<int> chain_offsets;
    std::vector<int> chain_lengths;
    int nchains = 0;

    if (params.rg_on) {
        if (topo.mol_ids.empty()) {
            fprintf(stderr, "Warning: rg_on but no mol_ids in data file, disabling Rg\n");
            params.rg_on = 0;
        } else {
            // Sort atoms by mol_id so chains are contiguous
            int natoms = (int)topo.mol_ids.size();
            std::vector<int> perm(natoms);
            for (int i = 0; i < natoms; i++) perm[i] = i;
            std::sort(perm.begin(), perm.end(),
                      [&](int a, int b) { return topo.mol_ids[a] < topo.mol_ids[b]; });

            // Build inverse mapping: old_idx -> new_idx
            std::vector<int> inv_perm(natoms);
            for (int i = 0; i < natoms; i++) inv_perm[perm[i]] = i;

            // Permute positions, velocities, mol_ids, images
            std::vector<float4> sorted_pos(natoms);
            std::vector<float4> sorted_vel(natoms);
            std::vector<int> sorted_mol(natoms);
            std::vector<int> sorted_img(topo.images.size());
            for (int i = 0; i < natoms; i++) {
                sorted_pos[i] = topo.positions[perm[i]];
                sorted_mol[i] = topo.mol_ids[perm[i]];
            }
            for (int i = 0; i < natoms; i++)
                sorted_vel[i] = (perm[i] < (int)topo.velocities.size())
                    ? topo.velocities[perm[i]] : make_float4(0,0,0,0);
            if (!topo.images.empty()) {
                for (int i = 0; i < natoms; i++) {
                    int s = perm[i] * 3;
                    int d = i * 3;
                    sorted_img[d + 0] = topo.images[s + 0];
                    sorted_img[d + 1] = topo.images[s + 1];
                    sorted_img[d + 2] = topo.images[s + 2];
                }
            }
            topo.positions.swap(sorted_pos);
            topo.velocities.swap(sorted_vel);
            topo.mol_ids.swap(sorted_mol);
            topo.images.swap(sorted_img);

            // Remap bond and angle atom indices
            for (auto& b : topo.bonds) {
                b.x = inv_perm[b.x];
                b.y = inv_perm[b.y];
            }
            for (auto& a : topo.angles) {
                a.x = inv_perm[a.x];
                a.y = inv_perm[a.y];
                a.z = inv_perm[a.z];
            }

            // Re-upload sorted data to GPU
            std::copy(topo.positions.begin(), topo.positions.end(), pos_pad.begin());
            std::copy(topo.velocities.begin(), topo.velocities.end(), vel_pad.begin());
            CUDA_CHECK(cudaMemcpy(sys.pos, pos_pad.data(), np * sizeof(float4),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(sys.vel, vel_pad.data(), np * sizeof(float4),
                                   cudaMemcpyHostToDevice));
            if (sys.nbonds > 0) {
                CUDA_CHECK(cudaMemcpy(sys.bonds, topo.bonds.data(),
                                      sys.nbonds * sizeof(int2), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(sys.bond_param_idx, topo.bond_types.data(),
                                      sys.nbonds * sizeof(int), cudaMemcpyHostToDevice));
            }
            if (sys.nangles > 0) {
                CUDA_CHECK(cudaMemcpy(sys.angles, topo.angles.data(),
                                      sys.nangles * sizeof(int4), cudaMemcpyHostToDevice));
            }
            // Also rebuild exclusions since indices changed
            if (sys.nbonds > 0) {
                build_exclusions(topo, topo);
                sys.nexclusions = static_cast<int>(topo.exclusion_list.size());
                if (sys.nexclusions > 0) {
                    CUDA_CHECK(cudaFree(sys.exclusion_offsets));
                    CUDA_CHECK(cudaFree(sys.exclusion_list));
                    CUDA_CHECK(cudaMalloc(&sys.exclusion_offsets,
                                          (params.natoms + 1) * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&sys.exclusion_list,
                                          sys.nexclusions * sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(sys.exclusion_offsets, topo.exclusion_offsets.data(),
                                          (params.natoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(sys.exclusion_list, topo.exclusion_list.data(),
                                          sys.nexclusions * sizeof(int), cudaMemcpyHostToDevice));
                }
            }

            int cur_mol = topo.mol_ids[0];
            chain_offsets.push_back(0);
            for (size_t i = 1; i < topo.mol_ids.size(); i++) {
                if (topo.mol_ids[i] != cur_mol) {
                    chain_offsets.push_back((int)i);
                    cur_mol = topo.mol_ids[i];
                }
            }
            chain_offsets.push_back((int)topo.mol_ids.size());
            nchains = (int)chain_offsets.size() - 1;
            chain_lengths.resize(nchains);
            int max_len = 0;
            for (int c = 0; c < nchains; c++) {
                chain_lengths[c] = chain_offsets[c + 1] - chain_offsets[c];
                if (chain_lengths[c] > max_len) max_len = chain_lengths[c];
            }
            sys.allocate_rg_buffers(topo.mol_ids, topo.images, np);
            rg_bufs.allocate(chain_offsets, chain_lengths, nchains, max_len,
                             params.rg_file);
        }
    }

    BinaryDumper dumper;
    dumper.open("traj.bin", params.natoms, params.ntypes);

    cudaStream_t stream_lj, stream_bonded, stream_io;
    CUDA_CHECK(cudaStreamCreate(&stream_lj));
    CUDA_CHECK(cudaStreamCreate(&stream_bonded));
    CUDA_CHECK(cudaStreamCreate(&stream_io));

    cudaEvent_t force_done_lj, force_done_bonded, pos_ready;
    CUDA_CHECK(cudaEventCreate(&force_done_lj));
    CUDA_CHECK(cudaEventCreate(&force_done_bonded));
    CUDA_CHECK(cudaEventCreate(&pos_ready));

    if (sys.nbonds == 0 && sys.nangles == 0)
        morton.sort_and_permute(sys.pos, sys.vel, params.natoms, params.inv_L);
    CUDA_CHECK(cudaMemcpy(sys.pos_ref, sys.pos, np * sizeof(float4),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(sys.d_max_dr2_int, 0, sizeof(int)));
    verlet.build(sys.pos, params.natoms, params.box_L, params.inv_L,
                 sys.exclusion_offsets, sys.exclusion_list,
                 params.rc + params.skin);

    sys.zero_virial();
    launch_lj_kernel(sys.pos, sys.force, sys.virial, sys.lj_params,
                      verlet.neighbors, verlet.num_neighbors,
                      params.natoms, params.ntypes,
                      params.rc2, params.box_L, params.inv_L);
    if (sys.nbonds > 0) {
        launch_fene_kernel(sys.pos, sys.force, sys.virial,
                            sys.bonds, sys.bond_param_idx, d_fene_params,
                            sys.nbonds, 1, params.box_L, params.inv_L);
    }
    if (sys.nangles > 0) {
        launch_angle_kernel(sys.pos, sys.force, sys.virial,
                             sys.angles, d_angle_params,
                             sys.nangles, params.box_L, params.inv_L);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (params.ensemble == Ensemble::NPT_NH) {
        ThermoOutput init_thermo;
        compute_thermo(sys.vel, sys.force, sys.virial,
                       params.natoms, params.box_L, &init_thermo,
                       thermo_bufs, 0, nullptr);
        (void)init_thermo;
    }

    float half_dt = 0.5f * params.dt;

    auto t_start = std::chrono::steady_clock::now();

    int progress_interval = 1000;
    if (params.nsteps < 100) progress_interval = 1;
    else if (params.nsteps < 1000) progress_interval = params.nsteps / 2;
    int last_progress_step = 0;
    double progress_speed = 0.0;
    auto last_progress_time = t_start;
    int step_field_width = snprintf(nullptr, 0, "%d", params.nsteps);

    int steps_since_rebuild = 0;
    const int max_steps_between_rebuilds = 20;

    for (int step = 1; step <= params.nsteps; step++) {

        if (params.ensemble == Ensemble::Langevin) {
            // Linear temperature ramping from T_start to T_stop
            float frac = (float)(step - 1) / (float)params.nsteps;
            langevin.kT = params.T_start
                + (params.T_stop - params.T_start) * frac;

            launch_integrator_pre_force(sys.pos, sys.vel, sys.force,
                                         sys.pos_ref, sys.d_max_dr2_int,
                                         sys.d_image,
                                         langevin, params.natoms,
                                         params.box_L, params.inv_L, half_dt);
        } else {
            // --- NH path ---

            // Ramping: update T_target and P_target (linear from start to stop)
            float frac = (float)(step - 1) / (float)params.nsteps;
            nose_hoover.T_target = nose_hoover.T_start
                + (nose_hoover.T_stop - nose_hoover.T_start) * frac;
            if (nose_hoover.is_npt) {
                nose_hoover.P_target = nose_hoover.P_start
                    + (nose_hoover.P_stop - nose_hoover.P_start) * frac;
            }

            float hdt = 0.5f * params.dt;

            // Compute thermo for total KE and virial
            ThermoOutput nh_thermo;
            compute_thermo(sys.vel, sys.force, sys.virial,
                           params.natoms, nose_hoover.L, &nh_thermo,
                           thermo_bufs, step, nullptr);
            float chain_KE = nh_thermo.kinetic_energy;

            if (nose_hoover.is_npt) {
                // Barostat half-step on host (Suzuki-Yoshida)
                static const float sy_w[3] = {
                    1.0f / (2.0f - cbrtf(2.0f)),
                    -cbrtf(2.0f) / (2.0f - cbrtf(2.0f)),
                    1.0f / (2.0f - cbrtf(2.0f))
                };

                float N_f = 3.0f * params.natoms;
                float N_f_inv = 1.0f / N_f;
                float KE = nh_thermo.kinetic_energy;
                float vir_trace = nh_thermo.stress[0]
                                + nh_thermo.stress[1]
                                + nh_thermo.stress[2];
                float P_inst = (2.0f * KE - vir_trace)
                             / (3.0f * nose_hoover.V);

                for (int sy = 0; sy < 3; sy++) {
                    float w = sy_w[sy] * hdt;
                    float dv_eps = 3.0f * nose_hoover.V
                                 * (P_inst - nose_hoover.P_target) * w;
                    float v_eps_half = nose_hoover.v_eps + 0.5f * dv_eps;
                    nose_hoover.eps += v_eps_half * w / nose_hoover.W;
                    dv_eps = 3.0f * nose_hoover.V0
                           * expf(3.0f * nose_hoover.eps)
                           * (P_inst - nose_hoover.P_target) * w;
                    nose_hoover.v_eps += dv_eps;
                }

                // Update volume and box dimensions
                nose_hoover.V = nose_hoover.V0
                              * expf(3.0f * nose_hoover.eps);
                nose_hoover.L = cbrtf(nose_hoover.V);
                nose_hoover.inv_L = 1.0f / nose_hoover.L;

                // Barostat velocity rescale
                float v_eps_W = nose_hoover.v_eps / nose_hoover.W;
                launch_nh_barostat_vel_half(sys.vel, v_eps_W, N_f_inv,
                                             params.natoms, hdt);

                // Update KE analytically: KE_new = KE_old * baro_scale^2
                float baro_scale = expf(-(1.0f + 3.0f * N_f_inv)
                                          * v_eps_W * hdt);
                chain_KE *= baro_scale * baro_scale;
            }

            // System-wide NH chain thermostat with current KE
            float nh_scale;
            nh_propagate_chain(nose_hoover, chain_KE, hdt, nh_scale);
            launch_nh_global_scale_vel(sys.vel, nh_scale, params.natoms);

            // Half-step velocity Verlet
            launch_nh_v_verlet_half(sys.vel, sys.force, params.natoms, hdt);

            // Position update
            float pos_exp_vW = 1.0f;
            float pos_vW_dt = 0.0f;
            if (nose_hoover.is_npt) {
                float v_eps_W = nose_hoover.v_eps / nose_hoover.W;
                pos_vW_dt = v_eps_W * params.dt;
                pos_exp_vW = expf(pos_vW_dt);
            }
            launch_nh_update_pos(sys.pos, sys.vel, sys.pos_ref,
                                  sys.d_image, sys.d_max_dr2_int,
                                  params.natoms, nose_hoover.L,
                                  nose_hoover.inv_L,
                                  pos_exp_vW, pos_vW_dt, params.dt);
        }

        // --- Verlet rebuild check (common to both paths) ---
        steps_since_rebuild++;
        bool triggered = check_and_reset_trigger(sys.d_max_dr2_int, params.skin);
        bool force_rebuild = (steps_since_rebuild >= max_steps_between_rebuilds);

        if (triggered || force_rebuild) {
            if (!triggered) {
                int zero = 0;
                CUDA_CHECK(cudaMemcpy(sys.d_max_dr2_int, &zero, sizeof(int),
                                       cudaMemcpyHostToDevice));
            }
            steps_since_rebuild = 0;

            if (params.ensemble == Ensemble::Langevin) {
                if (sys.nbonds == 0 && sys.nangles == 0)
                    morton.sort_and_permute(sys.pos, sys.vel, params.natoms,
                                             params.inv_L);
                CUDA_CHECK(cudaMemcpy(sys.pos_ref, sys.pos,
                                       np * sizeof(float4),
                                       cudaMemcpyDeviceToDevice));
            } else if (params.ensemble == Ensemble::NVT_NH) {
                CUDA_CHECK(cudaMemcpy(sys.pos_ref, sys.pos,
                                       np * sizeof(float4),
                                       cudaMemcpyDeviceToDevice));
            }
            // NPT_NH: pos_ref already updated in nh_update_pos_kernel, skip copy

            float L_v = (params.ensemble != Ensemble::Langevin)
                        ? nose_hoover.L : params.box_L;
            float inv_L_v = (params.ensemble != Ensemble::Langevin)
                           ? nose_hoover.inv_L : params.inv_L;
            verlet.build(sys.pos, params.natoms, L_v, inv_L_v,
                         sys.exclusion_offsets, sys.exclusion_list,
                         params.rc + params.skin);
            printf("  [step %d] verlet rebuild: max_nneigh=%d max_cell=%d ncells=%d nx=%d\n",
                   step, verlet.max_nneigh, verlet.max_cell_atoms, verlet.ncells, verlet.nx);
        }

        // --- Force computation (common to both paths) ---
        CUDA_CHECK(cudaEventRecord(pos_ready, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream_lj, pos_ready, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream_bonded, pos_ready, 0));

        sys.zero_virial();

        float L_f = (params.ensemble != Ensemble::Langevin)
                    ? nose_hoover.L : params.box_L;
        float inv_L_f = (params.ensemble != Ensemble::Langevin)
                        ? nose_hoover.inv_L : params.inv_L;

        launch_lj_kernel(sys.pos, sys.force, sys.virial, sys.lj_params,
                          verlet.neighbors, verlet.num_neighbors,
                          params.natoms, params.ntypes,
                          params.rc2, L_f, inv_L_f, stream_lj);
        CUDA_CHECK(cudaEventRecord(force_done_lj, stream_lj));
        CUDA_CHECK(cudaStreamWaitEvent(stream_bonded, force_done_lj, 0));
        if (sys.nbonds > 0) {
            launch_fene_kernel(sys.pos, sys.force, sys.virial,
                                sys.bonds, sys.bond_param_idx, d_fene_params,
                                sys.nbonds, 1, L_f, inv_L_f,
                                stream_bonded);
        }
        if (sys.nangles > 0) {
            launch_angle_kernel(sys.pos, sys.force, sys.virial,
                                 sys.angles, d_angle_params,
                                 sys.nangles, L_f, inv_L_f,
                                 stream_bonded);
        }

        CUDA_CHECK(cudaEventRecord(force_done_bonded, stream_bonded));
        CUDA_CHECK(cudaStreamWaitEvent(0, force_done_lj, 0));
        CUDA_CHECK(cudaStreamWaitEvent(0, force_done_bonded, 0));

        if (params.ensemble == Ensemble::Langevin) {
            // --- Existing Langevin post-force (unchanged) ---
            launch_integrator_post_force(sys.vel, sys.force,
                                          params.natoms, half_dt);
        } else {
            // --- NH post-force ---
            float hdt = 0.5f * params.dt;

            // Velocity Verlet half-step
            launch_nh_v_verlet_half(sys.vel, sys.force, params.natoms, hdt);

            // Compute thermo for current KE
            ThermoOutput nh_thermo2;
            compute_thermo(sys.vel, sys.force, sys.virial,
                           params.natoms, nose_hoover.L, &nh_thermo2,
                           thermo_bufs, step, nullptr);
            float chain_KE2 = nh_thermo2.kinetic_energy;

            if (nose_hoover.is_npt) {
                float v_eps_W = nose_hoover.v_eps / nose_hoover.W;
                float N_f_inv = 1.0f / (3.0f * params.natoms);
                launch_nh_barostat_vel_half(sys.vel, v_eps_W, N_f_inv,
                                             params.natoms, hdt);

                // Update KE analytically after barostat rescale
                float baro_scale = expf(-(1.0f + 3.0f * N_f_inv)
                                          * v_eps_W * hdt);
                chain_KE2 *= baro_scale * baro_scale;

                // Barostat half-step (host)
                float vir_trace2 = nh_thermo2.stress[0]
                                 + nh_thermo2.stress[1] + nh_thermo2.stress[2];
                float P2 = (2.0f * chain_KE2 - vir_trace2)
                         / (3.0f * nose_hoover.V);

                static const float sy_w[3] = {
                    1.0f / (2.0f - cbrtf(2.0f)),
                    -cbrtf(2.0f) / (2.0f - cbrtf(2.0f)),
                    1.0f / (2.0f - cbrtf(2.0f))
                };
                for (int sy = 0; sy < 3; sy++) {
                    float w = sy_w[sy] * hdt;
                    float dv_eps = 3.0f * nose_hoover.V
                                 * (P2 - nose_hoover.P_target) * w;
                    float v_eps_half = nose_hoover.v_eps + 0.5f * dv_eps;
                    nose_hoover.eps += v_eps_half * w / nose_hoover.W;
                    dv_eps = 3.0f * nose_hoover.V0
                           * expf(3.0f * nose_hoover.eps)
                           * (P2 - nose_hoover.P_target) * w;
                    nose_hoover.v_eps += dv_eps;
                }
                nose_hoover.V = nose_hoover.V0
                              * expf(3.0f * nose_hoover.eps);
                nose_hoover.L = cbrtf(nose_hoover.V);
                nose_hoover.inv_L = 1.0f / nose_hoover.L;
            }

            // System-wide NH chain thermostat with current KE
            float nh_scale2;
            nh_propagate_chain(nose_hoover, chain_KE2, hdt, nh_scale2);
            launch_nh_global_scale_vel(sys.vel, nh_scale2, params.natoms);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        bool need_thermo = params.thermo_on && (step % params.thermo_freq == 0);
        bool need_stress = params.stress_on && (step % params.stress_freq == 0);

        if (need_thermo || need_stress) {
            float thermo_L = (params.ensemble != Ensemble::Langevin)
                            ? nose_hoover.L : params.box_L;
            ThermoOutput thermo;
            compute_thermo(sys.vel, sys.force, sys.virial,
                            params.natoms, thermo_L, &thermo,
                            thermo_bufs, step, thermo_bufs.fp);

            if (need_thermo) {
                float etot = thermo.kinetic_energy + thermo.potential_energy;
                printf("\nStep %6d  T=%10.4f  KE=%11.2f  PE=%11.2f  Etot=%11.2f  Pxx=%8.2f  maxV=%5.2f  maxF=%5.2f\n",
                       step, thermo.temperature, thermo.kinetic_energy,
                       thermo.potential_energy, etot, thermo.stress[0],
                       thermo.max_vel, thermo.max_force);
                if (thermo.max_vel > 10.0f) {
                    printf("  ** VELOCITY SPIKE: atom %d v=%.2f\n",
                           thermo.max_vel_idx, thermo.max_vel);
                }
                if (thermo.max_force > 100.0f) {
                    printf("  ** FORCE SPIKE: atom %d f=%.2f\n",
                           thermo.max_force_idx, thermo.max_force);
                }
            }
            if (need_stress) {
                float* d_stress_for_corr;
                CUDA_CHECK(cudaMalloc(&d_stress_for_corr, 6 * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_stress_for_corr, thermo.stress,
                                       6 * sizeof(float), cudaMemcpyHostToDevice));
                correlator.push_sample(d_stress_for_corr);
                CUDA_CHECK(cudaFree(d_stress_for_corr));
            }
        }

        if (params.rg_on && step % params.rg_freq == 0) {
            float rg;
            float rg_L = (params.ensemble != Ensemble::Langevin)
                        ? nose_hoover.L : params.box_L;
            compute_rg(sys.pos, sys.d_image, rg_bufs, rg_L, step, &rg);
        }

        if (params.dump_freq > 0 && step % params.dump_freq == 0) {
            float dump_L = (params.ensemble != Ensemble::Langevin)
                          ? nose_hoover.L : params.box_L;
            dumper.dump_frame(sys.pos, step, dump_L, stream_io);
        }

        if (params.restart_freq > 0 && step % params.restart_freq == 0 && step != params.nsteps) {
            float restart_L = (params.ensemble != Ensemble::Langevin)
                             ? nose_hoover.L : params.box_L;
            std::string fname = build_restart_filename(params.restart_file, step);
            write_lammps_data(fname, sys.pos, sys.vel, sys.d_image, sys.d_mol_id,
                              sys.bonds, sys.bond_param_idx, sys.angles,
                              params.natoms, sys.nbonds, sys.nangles,
                              params.ntypes, (int)topo.bond_params.size(), (int)topo.angle_params.size(),
                              restart_L, step);
        }

        bool at_progress = (step % progress_interval == 0);
        if (at_progress || step == params.nsteps) {
            auto now = std::chrono::steady_clock::now();
            double dt_since_last = std::chrono::duration<double>(now - last_progress_time).count();
            int steps_since_last = step - last_progress_step;
            if (steps_since_last > 0 && dt_since_last > 0.0)
                progress_speed = steps_since_last / dt_since_last;
            last_progress_step = step;
            last_progress_time = now;

            int pct = (int)((long long)step * 100 / params.nsteps);
            int bar_width = 20;
            int filled = (int)((long long)step * bar_width / params.nsteps);
            if (filled > bar_width) filled = bar_width;

            char bar[23];
            bar[0] = '[';
            for (int i = 0; i < bar_width; i++) {
                if (i < filled - 1) bar[i + 1] = '=';
                else if (i == filled - 1 && step < params.nsteps) bar[i + 1] = '>';
                else if (i < filled) bar[i + 1] = '=';
                else bar[i + 1] = ' ';
            }
            bar[bar_width + 1] = ']';
            bar[bar_width + 2] = '\0';

            if (step == params.nsteps) {
                printf("\r%s %3d%%  Step %*d/%d  done\n",
                       bar, pct, step_field_width, step, params.nsteps);
            } else {
                double elapsed_total = std::chrono::duration<double>(now - t_start).count();
                double eta = (step > 0) ? (elapsed_total / step) * (params.nsteps - step) : 0.0;
                double eta_s = eta;
                const char* eta_unit = "s";
                if (eta_s > 86400.0) { eta_s /= 86400.0; eta_unit = "d"; }
                else if (eta_s > 3600.0) { eta_s /= 3600.0; eta_unit = "h"; }
                else if (eta_s > 60.0) { eta_s /= 60.0; eta_unit = "m"; }
                printf("\r%s %3d%%  Step %*d/%d  speed=%.0f st/s  ETA=%.1f%s",
                       bar, pct, step_field_width, step, params.nsteps,
                       progress_speed, eta_s, eta_unit);
                fflush(stdout);
            }
        }
    }

    if (params.restart_freq >= 0) {
        float final_L = (params.ensemble != Ensemble::Langevin)
                       ? nose_hoover.L : params.box_L;
        std::string fname = build_restart_final_filename(params.restart_file);
        write_lammps_data(fname, sys.pos, sys.vel, sys.d_image, sys.d_mol_id,
                          sys.bonds, sys.bond_param_idx, sys.angles,
                          params.natoms, sys.nbonds, sys.nangles,
                          params.ntypes, (int)topo.bond_params.size(), (int)topo.angle_params.size(),
                          final_L, params.nsteps);
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    double steps_per_sec = params.nsteps / elapsed;
    double ns_per_day = (params.nsteps * params.dt) / (elapsed / 86400.0);

    printf("Wall time: %.3f s  |  Steps/s: %.1f  |  ns/day: %.3f\n",
           elapsed, steps_per_sec, ns_per_day);

    if (params.stress_on) {
        int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
        std::vector<float> h_corr(buf_size);
        std::vector<int> h_counts(CORR_LEVELS * CORR_POINTS);
        int total_levels;
        correlator.get_results(h_corr.data(), h_counts.data(), &total_levels);

        FILE* corr_fp = fopen(params.stress_file, "w");
        fprintf(corr_fp, "# lag_steps  C_xx  C_xy  C_xz  C_yy  C_yz  C_zz  count\n");
        int lag_step = params.stress_freq;
        for (int level = 0; level < total_levels; level++) {
            for (int lag = 0; lag < CORR_POINTS; lag++) {
                int count = h_counts[level * CORR_POINTS + lag];
                if (count == 0) continue;
                fprintf(corr_fp, "%d", lag_step * (lag + 1));
                for (int c = 0; c < STRESS_COMPONENTS; c++) {
                    float val = h_corr[(level * CORR_POINTS + lag) * STRESS_COMPONENTS + c] / count;
                    fprintf(corr_fp, " %.8e", val);
                }
                fprintf(corr_fp, " %d\n", count);
            }
            lag_step *= CORR_POINTS;
        }
        fclose(corr_fp);
        printf("Stress autocorrelation written to %s\n", params.stress_file);
    }

    dumper.close();
    thermo_bufs.close_file();
    thermo_bufs.free();
    if (params.rg_on) rg_bufs.free();
    if (params.stress_on) correlator.free();
    if (params.ensemble != Ensemble::Langevin) {
        nose_hoover.free();
    }
    langevin.free();
    verlet.free();
    morton.free();
    sys.free();
    CUDA_CHECK(cudaEventDestroy(force_done_lj));
    CUDA_CHECK(cudaEventDestroy(force_done_bonded));
    CUDA_CHECK(cudaEventDestroy(pos_ready));
    CUDA_CHECK(cudaStreamDestroy(stream_lj));
    CUDA_CHECK(cudaStreamDestroy(stream_bonded));
    CUDA_CHECK(cudaStreamDestroy(stream_io));

    printf("Simulation complete.\n");
    return 0;
}
