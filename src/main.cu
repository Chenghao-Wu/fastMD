#include "core/types.cuh"
#include "core/system.cuh"
#include "core/morton.cuh"
#include "core/pbc.cuh"
#include "neighbor/tile_list.cuh"
#include "neighbor/skin_trigger.cuh"
#include "force/lj.cuh"
#include "force/fene.cuh"
#include "force/angle.cuh"
#include "integrate/langevin.cuh"
#include "analysis/thermo.cuh"
#include "analysis/correlator.cuh"
#include "io/dump.cuh"
#include "io/config.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }

    TopologyData topo;
    SimParams params = parse_config(argv[1], topo);
    printf("Loaded %d atoms, box=%.2f, rc=%.2f, dt=%.4f, nsteps=%d\n",
           params.natoms, params.box_L, params.rc, params.dt, params.nsteps);

    System sys;
    sys.allocate(params);

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

    MortonSorter morton;
    morton.allocate(np);

    TileList tile_list;
    tile_list.allocate(sys.ntiles, sys.ntiles * sys.ntiles);

    LangevinState langevin;
    langevin.init(np, params.gamma, params.dt, params.temperature, params.seed);

    MultipleTauCorrelator correlator;
    correlator.allocate();

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

    morton.sort_and_permute(sys.pos, sys.vel, params.natoms, params.inv_L);
    CUDA_CHECK(cudaMemcpy(sys.pos_ref, sys.pos, np * sizeof(float4),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(sys.d_max_dr2_int, 0, sizeof(int)));
    tile_list.build(sys.pos, params.natoms, sys.ntiles,
                     params.rc + params.skin, params.box_L, params.inv_L);

    sys.zero_forces();
    sys.zero_virial();
    launch_lj_kernel(sys.pos, sys.force, sys.virial, sys.lj_params,
                      tile_list.offsets, tile_list.tile_neighbors,
                      sys.ntiles, params.natoms, params.ntypes,
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

    float half_dt = 0.5f * params.dt;

    for (int step = 1; step <= params.nsteps; step++) {
        launch_integrator_pre_force(sys.pos, sys.vel, sys.force,
                                     sys.pos_ref, sys.d_max_dr2_int,
                                     langevin, params.natoms,
                                     params.box_L, params.inv_L, half_dt);

        if (check_and_reset_trigger(sys.d_max_dr2_int, params.skin)) {
            morton.sort_and_permute(sys.pos, sys.vel, params.natoms, params.inv_L);
            CUDA_CHECK(cudaMemcpy(sys.pos_ref, sys.pos, np * sizeof(float4),
                                   cudaMemcpyDeviceToDevice));
            tile_list.build(sys.pos, params.natoms, sys.ntiles,
                             params.rc + params.skin, params.box_L, params.inv_L);
        }

        CUDA_CHECK(cudaEventRecord(pos_ready, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream_lj, pos_ready, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream_bonded, pos_ready, 0));

        sys.zero_forces();
        sys.zero_virial();

        launch_lj_kernel(sys.pos, sys.force, sys.virial, sys.lj_params,
                          tile_list.offsets, tile_list.tile_neighbors,
                          sys.ntiles, params.natoms, params.ntypes,
                          params.rc2, params.box_L, params.inv_L, stream_lj);
        if (sys.nbonds > 0) {
            launch_fene_kernel(sys.pos, sys.force, sys.virial,
                                sys.bonds, sys.bond_param_idx, d_fene_params,
                                sys.nbonds, 1, params.box_L, params.inv_L,
                                stream_bonded);
        }
        if (sys.nangles > 0) {
            launch_angle_kernel(sys.pos, sys.force, sys.virial,
                                 sys.angles, d_angle_params,
                                 sys.nangles, params.box_L, params.inv_L,
                                 stream_bonded);
        }

        CUDA_CHECK(cudaEventRecord(force_done_lj, stream_lj));
        CUDA_CHECK(cudaEventRecord(force_done_bonded, stream_bonded));
        CUDA_CHECK(cudaStreamWaitEvent(0, force_done_lj, 0));
        CUDA_CHECK(cudaStreamWaitEvent(0, force_done_bonded, 0));

        launch_integrator_post_force(sys.vel, sys.force, params.natoms, half_dt);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (step % params.thermo_freq == 0) {
            ThermoOutput thermo;
            compute_thermo(sys.vel, sys.force, sys.virial,
                            params.natoms, params.box_L, &thermo);
            printf("Step %d: T=%.4f KE=%.4f PE=%.4f Pxx=%.4f\n",
                   step, thermo.temperature, thermo.kinetic_energy,
                   thermo.potential_energy, thermo.stress[0]);

            float* d_stress_for_corr;
            CUDA_CHECK(cudaMalloc(&d_stress_for_corr, 6 * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_stress_for_corr, thermo.stress,
                                   6 * sizeof(float), cudaMemcpyHostToDevice));
            correlator.push_sample(d_stress_for_corr);
            CUDA_CHECK(cudaFree(d_stress_for_corr));
        }

        if (params.dump_freq > 0 && step % params.dump_freq == 0) {
            dumper.dump_frame(sys.pos, step, params.box_L, stream_io);
        }
    }

    {
        int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
        std::vector<float> h_corr(buf_size);
        std::vector<int> h_counts(CORR_LEVELS * CORR_POINTS);
        int total_levels;
        correlator.get_results(h_corr.data(), h_counts.data(), &total_levels);

        FILE* corr_fp = fopen("stress_acf.dat", "w");
        fprintf(corr_fp, "# lag_steps  C_xx  C_xy  C_xz  C_yy  C_yz  C_zz  count\n");
        int lag_step = params.thermo_freq;
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
        printf("Stress autocorrelation written to stress_acf.dat\n");
    }

    dumper.close();
    correlator.free();
    langevin.free();
    tile_list.free();
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
