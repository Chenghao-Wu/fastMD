#include "simulation.cuh"
#include "../core/system.cuh"
#include "../neighbor/skin_trigger.cuh"
#include "../force/lj.cuh"
#include "../force/lj_coul_long.cuh"
#include "../force/ewald_recip.cuh"
#include "../force/ewald_excl.cuh"
#include "../force/table.cuh"
#include "../force/bond_table.cuh"
#include "../force/angle_table.cuh"
#include "../io/lammps_data.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

template <typename T>
T* upload_vec(const std::vector<T>& h) {
    if (h.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
void free_dev(T*& p) {
    if (p) CUDA_CHECK(cudaFree(p));
    p = nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

Simulation::Simulation(const SimParams& params, TopologyData topo)
    : params_(params), topo_(std::move(topo)) {
    if ((int)topo_.positions.size() != params_.natoms)
        throw std::runtime_error("natoms (" + std::to_string(params_.natoms) +
                                 ") does not match the number of atoms read (" +
                                 std::to_string(topo_.positions.size()) + ")");
    box_L_ = params_.box_L;
    inv_L_ = 1.0f / box_L_;

    set_type_masses(topo_.masses, params_.ntypes);

    // OPLS exclusion scheme (1-2,1-3 fully excluded; 1-4 excluded from the
    // neighbor list and re-added scaled by the pair14 kernel) applies
    // whenever the system has dihedrals or uses Ewald.
    opls_mode_ = params_.ewald_on || !topo_.dihedrals.empty();
    if (opls_mode_) build_exclusions_opls(topo_, topo_);
    else if (!topo_.bonds.empty()) build_exclusions(topo_, topo_);

    p_.allocate(params_.natoms);
    p_.upload(topo_.positions, topo_.velocities, topo_.images, topo_.charges, topo_.mol_ids);
    t_.upload(params_.natoms, topo_.bonds, topo_.bond_types, topo_.angles,
              topo_.dihedrals, topo_.dihedral_types, topo_.pairs14,
              topo_.exclusion_offsets, topo_.exclusion_list,
              topo_.excl13_offsets, topo_.excl13_list);

    upload_force_field();

    morton_.allocate(p_.natoms_padded);
    verlet_.allocate(params_.natoms, params_.rc + params_.skin, box_L_);
    CUDA_CHECK(cudaMalloc(&d_max_dr2_int_, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max_dr2_int_, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_virial_, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out4_, p_.natoms_padded * sizeof(float4)));

    // Spatial sorting is kept to the previous policy (systems without
    // topology or Ewald) until it has been benchmarked for polymers; the
    // permutation itself is now complete for every per-particle field.
    sort_enabled_ = topo_.bonds.empty() && topo_.angles.empty() &&
                    topo_.dihedrals.empty() && !params_.ewald_on;

    CUDA_CHECK(cudaStreamCreate(&stream_lj_));
    CUDA_CHECK(cudaStreamCreate(&stream_bonded_));
    CUDA_CHECK(cudaStreamCreate(&stream_io_));
    CUDA_CHECK(cudaEventCreateWithFlags(&force_done_lj_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&force_done_bonded_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pos_ready_, cudaEventDisableTiming));

    setup_integrator();
    setup_analysis();

    if (sort_enabled_) spatial_sort();
    build_neighbor_list();
    launch_forces();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (params_.ensemble != Ensemble::Langevin) {
        // first pre-force thermostat half-step uses the raw kinetic energy
        float init_ke = compute_ke_only(p_.vel, p_.pos, params_.natoms, d_ke_buf_);
        CUDA_CHECK(cudaMemcpy(&d_nh_state_->chain_KE_carry, &init_ke, sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

void Simulation::upload_force_field() {
    const int nt2 = params_.ntypes * params_.ntypes;
    d_lj_params_ = upload_vec(topo_.lj_params);
    if ((int)topo_.lj_params.size() != nt2)
        throw std::runtime_error("LJ parameter table has wrong size");

    if (!topo_.table_params.empty()) {
        d_table_idx_ = upload_vec(topo_.table_idx);
        d_table_params_ = upload_vec(topo_.table_params);
        d_table_data_ = upload_vec(topo_.table_data);
    }
    d_fene_params_ = upload_vec(topo_.bond_params);
    d_hbond_params_ = upload_vec(topo_.hbond_params);
    d_dihedral_params_ = upload_vec(topo_.dihedral_params);
    d_angle_params_ = upload_vec(topo_.angle_params);
    if (!topo_.bond_table_data.empty()) {
        d_bond_table_idx_ = upload_vec(topo_.bond_table_idx);
        d_bond_table_params_ = upload_vec(topo_.bond_table_params);
        d_bond_table_data_ = upload_vec(topo_.bond_table_data);
    }
    if (!topo_.angle_table_data.empty()) {
        d_angle_table_idx_ = upload_vec(topo_.angle_table_idx);
        d_angle_table_params_ = upload_vec(topo_.angle_table_params);
        d_angle_table_data_ = upload_vec(topo_.angle_table_data);
    }

    if (params_.ewald_on) {
        if (topo_.charges.empty())
            throw std::runtime_error("coulomb ewald requires charges (Atoms # full)");
        std::vector<int3> kvecs = ewald_build_kvectors(params_.ewald_kmax, box_L_,
                                                       params_.ewald_alpha);
        params_.ewald_kcount = (int)kvecs.size();
        d_ewald_kvecs_ = upload_vec(kvecs);
        CUDA_CHECK(cudaMalloc(&d_ewald_sfac_, 2 * kvecs.size() * sizeof(float)));
        params_.ewald_self_offset = ewald_self_offset(topo_.charges, params_.coul_const,
                                                      params_.ewald_alpha, box_L_);
    }

    if (t_.nbonds > 0 && !d_bond_table_data_ && !d_hbond_params_ && !d_fene_params_)
        fprintf(stderr, "Warning: %d bonds in data file but no bond params "
                        "— skipping bond forces\n", t_.nbonds);
    if (t_.nangles > 0 && !d_angle_table_data_ && !d_angle_params_)
        fprintf(stderr, "Warning: %d angles in data file but no angle params "
                        "— skipping angle forces\n", t_.nangles);
    if (t_.ndihedrals > 0 && !d_dihedral_params_)
        fprintf(stderr, "Warning: %d dihedrals in data file but no "
                        "dihedral_opls params — skipping dihedral forces\n", t_.ndihedrals);
}

void Simulation::setup_integrator() {
    langevin_.init(p_.natoms_padded, params_.Tdamp, params_.dt, params_.T_start, params_.seed);
    if (params_.ensemble != Ensemble::Langevin) {
        nose_hoover_.init(params_);
        allocate_nh_device_state(d_nh_state_);
        init_nh_device_state(nose_hoover_, d_nh_state_);
        CUDA_CHECK(cudaMalloc(&d_ke_buf_, sizeof(float)));
    }
    if (params_.ensemble == Ensemble::NPT_NH) {
        allocate_npt_scratch(npt_scratch_);
        float V0 = nose_hoover_.V, L0 = nose_hoover_.L, inv_L0 = nose_hoover_.inv_L;
        CUDA_CHECK(cudaMemcpy(npt_scratch_.d_V, &V0, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(npt_scratch_.d_L, &L0, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(npt_scratch_.d_inv_L, &inv_L0, sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Simulation::setup_analysis() {
    if (params_.stress_on) correlator_.allocate();
    thermo_bufs_.allocate();
    if (params_.thermo_on) thermo_bufs_.open_file(params_.thermo_file);
    if (params_.rg_on) {
        if (topo_.mol_ids.empty()) {
            fprintf(stderr, "Warning: rg_on but no mol_ids in data file, disabling Rg\n");
            params_.rg_on = 0;
        } else {
            rg_bufs_.allocate(topo_.mol_ids, params_.rg_file);
        }
    }
    dumper_.open("traj.bin", params_.natoms, params_.ntypes);
}

void Simulation::print_setup() const {
    printf("[setup]  atoms=%d  box=%.2f  rc=%.2f  dt=%.4f  nsteps=%d",
           params_.natoms, box_L_, params_.rc, params_.dt, params_.nsteps);
    if (!topo_.bonds.empty()) printf("  bonds=%zu", topo_.bonds.size());
    if (!topo_.angles.empty()) printf("  angles=%zu", topo_.angles.size());
    if (!topo_.dihedrals.empty()) printf("  dihedrals=%zu", topo_.dihedrals.size());
    printf("\n");
    if (params_.ewald_on) {
        printf("[setup]  coulomb=ewald  alpha=%.4f  kmax=%d  kcount=%d  coul_const=%.4f  "
               "lj14=%.2f  coul14=%.2f  self-energy=%.4f\n",
               params_.ewald_alpha, params_.ewald_kmax, params_.ewald_kcount,
               params_.coul_const, params_.lj14, params_.ewald14_coul,
               params_.ewald_self_offset);
    }
    if (params_.ensemble != Ensemble::Langevin) {
        printf("[setup]  ensemble=%s  T_start=%.2f  T_stop=%.2f  Tdamp=%.2f  chain=%d",
               params_.ensemble == Ensemble::NPT_NH ? "npt_nh" : "nvt_nh",
               params_.T_start, params_.T_stop, params_.Tdamp, params_.nh_chain_length);
        if (params_.ensemble == Ensemble::NPT_NH)
            printf("  P_start=%.2f  P_stop=%.2f  Pdamp=%.2f",
                   params_.P_start, params_.P_stop, params_.Pdamp);
    } else {
        printf("[setup]  ensemble=nvt_langevin  T_start=%.2f  T_stop=%.2f  Tdamp=%.2f  seed=%lu",
               params_.T_start, params_.T_stop, params_.Tdamp, (unsigned long)params_.seed);
    }
    if (params_.rg_on) printf("  rg_on");
    if (params_.stress_on) printf("  stress_on");
    if (params_.restart_freq >= 0) printf("  restart_on");
    printf("\n[setup]  neighbor list: capacity=%d  max_count=%d  cells=%d^3\n\n",
           verlet_.max_neighbors, verlet_.max_count, verlet_.nx);
}

// ---------------------------------------------------------------------------
// Neighbor list / spatial sorting
// ---------------------------------------------------------------------------

void Simulation::spatial_sort() {
    const int* order = morton_.compute_order(p_.pos, p_.natoms, inv_L_);
    p_.apply_permutation(order);
    t_.remap(p_.rank);
}

void Simulation::build_neighbor_list() {
    if (box_L_ < 2.0f * params_.rc)
        throw std::runtime_error("box length " + std::to_string(box_L_) +
                                 " is smaller than 2*rc: minimum image is invalid");
    CUDA_CHECK(cudaMemcpy(p_.pos_ref, p_.pos, p_.natoms_padded * sizeof(float4),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_max_dr2_int_, 0, sizeof(int)));
    verlet_.build(p_.pos, params_.natoms, box_L_, inv_L_, p_.tag,
                  t_.excl_offsets, t_.excl_list, params_.rc + params_.skin);
    box_L_at_build_ = box_L_;
    steps_since_rebuild_ = 0;
    n_rebuilds_++;
}

// A pair outside the list (distance > rc + skin at build time) can come
// within rc once particles move. Positions are tracked relative to the
// reference configuration scaled affinely with the box (NPT scales pos_ref
// each step), so with s = L / L_build and maximum displacement d the list
// stays valid while  s (rc + skin) - 2 d > rc.
bool Simulation::need_rebuild() {
    steps_since_rebuild_++;
    int max_int;
    CUDA_CHECK(cudaMemcpy(&max_int, d_max_dr2_int_, sizeof(int), cudaMemcpyDeviceToHost));
    float max_dr2;
    std::memcpy(&max_dr2, &max_int, sizeof(float));
    float s = box_L_ / box_L_at_build_;
    float half_margin = 0.5f * (s * (params_.rc + params_.skin) - params_.rc);
    bool triggered = half_margin <= 0.0f || max_dr2 > half_margin * half_margin;
    const int max_steps_between_rebuilds = 20;
    return triggered || steps_since_rebuild_ >= max_steps_between_rebuilds;
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

void Simulation::launch_forces() {
    const float L = box_L_, inv_L = inv_L_;
    const int n = params_.natoms;

    CUDA_CHECK(cudaEventRecord(pos_ready_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream_lj_, pos_ready_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream_bonded_, pos_ready_, 0));
    CUDA_CHECK(cudaMemsetAsync(d_virial_, 0, 6 * sizeof(float), stream_lj_));

    // Nonbonded kernels overwrite force[i] (=), everything after accumulates.
    if (params_.ewald_on) {
        launch_lj_coul_long_kernel(p_.pos, p_.charge, p_.force, d_virial_, d_lj_params_,
                                   verlet_.neighbors, verlet_.num_neighbors,
                                   n, params_.ntypes, params_.coul_const, params_.ewald_alpha,
                                   params_.rc2, L, inv_L, stream_lj_);
        launch_ewald_sf_kernel(p_.pos, p_.charge, d_ewald_kvecs_, d_ewald_sfac_,
                               n, params_.ewald_kcount, L, stream_lj_);
        launch_ewald_recip_kernel(p_.pos, p_.charge, d_ewald_kvecs_, d_ewald_sfac_,
                                  p_.force, d_virial_, n, params_.ewald_kcount,
                                  params_.coul_const, params_.ewald_alpha, L, stream_lj_);
        launch_ewald_excl_kernel(p_.pos, p_.charge, p_.force, d_virial_,
                                 t_.excl13_offsets, t_.excl13_list, p_.rank,
                                 n, t_.nexcl13, params_.coul_const, params_.ewald_alpha,
                                 L, inv_L, stream_lj_);
    } else {
        launch_lj_kernel(p_.pos, p_.force, d_virial_, d_lj_params_,
                         verlet_.neighbors, verlet_.num_neighbors,
                         n, params_.ntypes, params_.rc2, L, inv_L, stream_lj_);
    }
    if (t_.npairs14 > 0) {
        launch_pair14_kernel(p_.pos, p_.charge, p_.force, d_virial_, t_.pairs14, d_lj_params_,
                             t_.npairs14, params_.ntypes, params_.coul_const,
                             params_.ewald_alpha, params_.lj14, params_.ewald14_coul,
                             L, inv_L, stream_lj_);
    }
    if (d_table_params_) {
        launch_table_kernel(p_.pos, p_.force, d_virial_, d_table_idx_, d_table_params_,
                            d_table_data_, verlet_.neighbors, verlet_.num_neighbors,
                            n, params_.ntypes, params_.rc2, L, inv_L, stream_lj_);
    }
    CUDA_CHECK(cudaEventRecord(force_done_lj_, stream_lj_));

    // Bonded terms accumulate into the same force buffer, so they must follow
    // the overwriting nonbonded kernel.
    CUDA_CHECK(cudaStreamWaitEvent(stream_bonded_, force_done_lj_, 0));
    if (t_.nbonds > 0 && d_bond_table_data_) {
        launch_bond_table_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                                 d_bond_table_idx_, d_bond_table_params_, d_bond_table_data_,
                                 t_.nbonds, L, inv_L, stream_bonded_);
    } else if (t_.nbonds > 0 && d_hbond_params_) {
        launch_bond_harmonic_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                                    d_hbond_params_, t_.nbonds, L, inv_L, stream_bonded_);
    } else if (t_.nbonds > 0 && d_fene_params_) {
        launch_fene_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                           d_fene_params_, t_.nbonds, 1, L, inv_L, stream_bonded_);
    }
    if (t_.nangles > 0 && d_angle_table_data_) {
        launch_angle_table_kernel(p_.pos, p_.force, d_virial_, t_.angles, d_angle_table_idx_,
                                  d_angle_table_params_, d_angle_table_data_,
                                  t_.nangles, L, inv_L, stream_bonded_);
    } else if (t_.nangles > 0 && d_angle_params_) {
        launch_angle_kernel(p_.pos, p_.force, d_virial_, t_.angles, d_angle_params_,
                            t_.nangles, L, inv_L, stream_bonded_);
    }
    if (t_.ndihedrals > 0 && d_dihedral_params_) {
        launch_dihedral_opls_kernel(p_.pos, p_.force, d_virial_, t_.dihedrals,
                                    t_.dihedral_type, d_dihedral_params_,
                                    t_.ndihedrals, L, inv_L, stream_bonded_);
    }
    CUDA_CHECK(cudaEventRecord(force_done_bonded_, stream_bonded_));
    CUDA_CHECK(cudaStreamWaitEvent(0, force_done_lj_, 0));
    CUDA_CHECK(cudaStreamWaitEvent(0, force_done_bonded_, 0));
    CUDA_CHECK(cudaGetLastError());
}

void Simulation::compute_forces() {
    if (need_rebuild()) build_neighbor_list();
    launch_forces();
    CUDA_CHECK(cudaStreamSynchronize(0));
}

void Simulation::write_forces(const std::string& path) const {
    std::vector<float4> f = p_.download_forces();
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot open forces_dump_file " + path);
    fprintf(fp, "# id fx fy fz pe\n");
    for (int t = 0; t < params_.natoms; ++t)
        fprintf(fp, "%d %.6e %.6e %.6e %.6e\n", t, f[t].x, f[t].y, f[t].z, f[t].w);
    fclose(fp);
}

// ---------------------------------------------------------------------------
// Time step
// ---------------------------------------------------------------------------

void Simulation::integrate_pre_force() {
    const float hdt = 0.5f * params_.dt;
    const float frac = (float)(step_ - 1) / (float)params_.nsteps;

    if (params_.ensemble == Ensemble::Langevin) {
        langevin_.kT = params_.T_start + (params_.T_stop - params_.T_start) * frac;
        launch_integrator_pre_force(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2_int_,
                                    p_.image, p_.tag, (uint64_t)step_, langevin_,
                                    params_.natoms, box_L_, inv_L_, hdt);
        return;
    }

    nose_hoover_.T_target = nose_hoover_.T_start + (nose_hoover_.T_stop - nose_hoover_.T_start) * frac;
    if (nose_hoover_.is_npt) {
        nose_hoover_.P_target = nose_hoover_.P_start + (nose_hoover_.P_stop - nose_hoover_.P_start) * frac;
        set_nh_targets_device(d_nh_state_, nose_hoover_.T_target, nose_hoover_.P_target);
        launch_nh_npt_ke_and_virial_trace(p_.vel, d_virial_, p_.pos, d_ke_buf_,
                                          npt_scratch_.d_virial_trace, params_.natoms);
        nh_npt_chain_baro_kernel<<<1, 32>>>(
            d_nh_state_, d_ke_buf_, npt_scratch_.d_virial_trace, true,
            nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest, nose_hoover_.W,
            hdt, params_.dt, nose_hoover_.V0, params_.natoms,
            npt_scratch_.d_V, npt_scratch_.d_L, npt_scratch_.d_inv_L,
            npt_scratch_.d_baro_scale, npt_scratch_.d_exp_vW,
            npt_scratch_.d_v_eps_W, npt_scratch_.d_v_eps_W_dt);
        CUDA_CHECK(cudaMemcpy(&nose_hoover_.L, npt_scratch_.d_L, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&nose_hoover_.inv_L, npt_scratch_.d_inv_L, sizeof(float), cudaMemcpyDeviceToHost));
        box_L_ = nose_hoover_.L;
        inv_L_ = nose_hoover_.inv_L;
        launch_nh_npt_pre_force_fused(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2_int_,
                                      p_.image, d_nh_state_, npt_scratch_.d_baro_scale,
                                      npt_scratch_.d_exp_vW, npt_scratch_.d_v_eps_W_dt,
                                      hdt, params_.dt, params_.natoms, box_L_, inv_L_);
    } else {
        set_nh_targets_device(d_nh_state_, nose_hoover_.T_target, 0.0f);
        nh_propagate_chain_kernel<<<1, 32>>>(d_nh_state_, d_ke_buf_, true,
                                             nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest,
                                             hdt, params_.natoms);
        launch_nh_nvt_pre_force_fused(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2_int_,
                                      p_.image, d_nh_state_, hdt, params_.dt,
                                      params_.natoms, box_L_, inv_L_);
    }
}

void Simulation::integrate_post_force() {
    const float hdt = 0.5f * params_.dt;
    if (params_.ensemble == Ensemble::Langevin) {
        launch_integrator_post_force(p_.vel, p_.force, p_.pos, params_.natoms, hdt);
        return;
    }
    if (nose_hoover_.is_npt) {
        launch_nh_v_verlet_half(p_.vel, p_.force, p_.pos, params_.natoms, hdt);
        launch_nh_npt_ke_and_virial_trace(p_.vel, d_virial_, p_.pos, d_ke_buf_,
                                          npt_scratch_.d_virial_trace, params_.natoms);
        nh_npt_chain_baro_kernel<<<1, 32>>>(
            d_nh_state_, d_ke_buf_, npt_scratch_.d_virial_trace, false,
            nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest, nose_hoover_.W,
            hdt, params_.dt, nose_hoover_.V0, params_.natoms,
            npt_scratch_.d_V, npt_scratch_.d_L, npt_scratch_.d_inv_L,
            npt_scratch_.d_baro_scale, npt_scratch_.d_exp_vW,
            npt_scratch_.d_v_eps_W, npt_scratch_.d_v_eps_W_dt);
        launch_nh_barostat_vel_half(p_.vel, npt_scratch_.d_v_eps_W,
                                    1.0f / (3.0f * params_.natoms), params_.natoms, hdt);
        launch_nh_global_scale_vel(p_.vel, d_nh_state_, params_.natoms);
        CUDA_CHECK(cudaMemcpy(&nose_hoover_.L, npt_scratch_.d_L, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&nose_hoover_.inv_L, npt_scratch_.d_inv_L, sizeof(float), cudaMemcpyDeviceToHost));
        box_L_ = nose_hoover_.L;
        inv_L_ = nose_hoover_.inv_L;
    } else {
        CUDA_CHECK(cudaMemsetAsync(d_ke_buf_, 0, sizeof(float), 0));
        launch_nh_nvt_v_half_ke_reduce(p_.vel, p_.force, p_.pos, d_ke_buf_, params_.natoms, hdt);
        nh_propagate_chain_kernel<<<1, 32>>>(d_nh_state_, d_ke_buf_, false,
                                             nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest,
                                             hdt, params_.natoms);
        launch_nh_global_scale_vel(p_.vel, d_nh_state_, params_.natoms);
    }
}

void Simulation::step() {
    ++step_;
    integrate_pre_force();
    if (need_rebuild()) {
        if (sort_enabled_ && params_.ensemble == Ensemble::Langevin) spatial_sort();
        build_neighbor_list();
    }
    launch_forces();
    integrate_post_force();
    CUDA_CHECK(cudaStreamSynchronize(0));
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void Simulation::write_restart(const std::string& fname, int64_t step) {
    RestartFrame f;
    f.pos = p_.download_positions();
    f.vel = p_.download_velocities();
    f.image = p_.download_images();
    f.charge = p_.download_charges();
    f.mol_id = p_.download_mol_ids();
    int nbond_types = std::max<int>({(int)topo_.bond_params.size(), (int)topo_.hbond_params.size(),
                                     (int)topo_.bond_table_idx.size()});
    int nangle_types = std::max<int>((int)topo_.angle_params.size(), (int)topo_.angle_table_idx.size());
    for (int b : topo_.bond_types) nbond_types = std::max(nbond_types, b + 1);
    for (const auto& a : topo_.angles) nangle_types = std::max(nangle_types, a.w + 1);
    int ndih_types = (int)topo_.dihedral_params.size();
    for (int d : topo_.dihedral_types) ndih_types = std::max(ndih_types, d + 1);
    write_lammps_data(fname, f, topo_, params_.ntypes, nbond_types, nangle_types, ndih_types,
                      box_L_, step);
    printf("Restart written to %s (step %lld)\n", fname.c_str(), (long long)step);
}

void Simulation::write_outputs() {
    const int s = step_;
    bool need_thermo = params_.thermo_on && (s % params_.thermo_freq == 0);
    bool need_stress = params_.stress_on && (s % params_.stress_freq == 0);

    if (need_thermo || need_stress) {
        ThermoOutput thermo;
        compute_thermo(p_.vel, p_.pos, p_.force, d_virial_, params_.natoms, box_L_, &thermo,
                       thermo_bufs_, s, need_thermo ? thermo_bufs_.fp : nullptr, 0,
                       params_.ewald_on ? params_.ewald_self_offset / params_.natoms : 0.0f);
        if (need_thermo) {
            float etot = thermo.kinetic_energy + thermo.potential_energy;
            printf("\nStep %6d  T=%10.4f  KE=%11.2f  PE=%11.2f  Etot=%11.2f  Pxx=%8.2f  maxV=%5.2f  maxF=%5.2f\n",
                   s, thermo.temperature, thermo.kinetic_energy, thermo.potential_energy, etot,
                   thermo.stress[0], thermo.max_vel, thermo.max_force);
        }
        if (need_stress) {
            float* d_stress_for_corr;
            CUDA_CHECK(cudaMalloc(&d_stress_for_corr, 6 * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_stress_for_corr, thermo.stress, 6 * sizeof(float),
                                  cudaMemcpyHostToDevice));
            correlator_.push_sample(d_stress_for_corr);
            CUDA_CHECK(cudaFree(d_stress_for_corr));
        }
    }

    if (params_.rg_on && s % params_.rg_freq == 0) {
        float rg;
        compute_rg(p_.pos, p_.image, p_.rank, rg_bufs_, box_L_, s, &rg);
    }

    if (params_.dump_freq > 0 && s % params_.dump_freq == 0) {
        p_.gather_float4_by_tag(d_out4_, p_.pos, stream_io_);
        dumper_.dump_frame(d_out4_, s, box_L_, stream_io_);
    }

    if (params_.restart_freq > 0 && s % params_.restart_freq == 0 && s != params_.nsteps)
        write_restart(build_restart_filename(params_.restart_file, s), s);
}

void Simulation::print_progress() {
    const int s = step_;
    if (s % progress_interval_ != 0 && s != params_.nsteps) return;
    auto now = std::chrono::steady_clock::now();
    double dt_since_last = std::chrono::duration<double>(now - last_progress_time_).count();
    int steps_since_last = s - last_progress_step_;
    if (steps_since_last > 0 && dt_since_last > 0.0)
        progress_speed_ = steps_since_last / dt_since_last;
    last_progress_step_ = s;
    last_progress_time_ = now;

    int pct = (int)((long long)s * 100 / params_.nsteps);
    const int bar_width = 20;
    int filled = std::min(bar_width, (int)((long long)s * bar_width / params_.nsteps));
    char bar[bar_width + 3];
    bar[0] = '[';
    for (int i = 0; i < bar_width; i++) {
        if (i < filled - 1) bar[i + 1] = '=';
        else if (i == filled - 1 && s < params_.nsteps) bar[i + 1] = '>';
        else if (i < filled) bar[i + 1] = '=';
        else bar[i + 1] = ' ';
    }
    bar[bar_width + 1] = ']';
    bar[bar_width + 2] = '\0';
    int width = snprintf(nullptr, 0, "%d", params_.nsteps);
    if (s == params_.nsteps) {
        printf("\r%s %3d%%  Step %*d/%d  done\n", bar, pct, width, s, params_.nsteps);
    } else {
        double elapsed = std::chrono::duration<double>(now - t_start_).count();
        double eta = (elapsed / s) * (params_.nsteps - s);
        const char* unit = "s";
        if (eta > 86400.0) { eta /= 86400.0; unit = "d"; }
        else if (eta > 3600.0) { eta /= 3600.0; unit = "h"; }
        else if (eta > 60.0) { eta /= 60.0; unit = "m"; }
        printf("\r%s %3d%%  Step %*d/%d  speed=%.0f st/s  ETA=%.1f%s",
               bar, pct, width, s, params_.nsteps, progress_speed_, eta, unit);
        fflush(stdout);
    }
}

void Simulation::run() {
    progress_interval_ = 1000;
    if (params_.nsteps < 100) progress_interval_ = 1;
    else if (params_.nsteps < 1000) progress_interval_ = params_.nsteps / 2;
    t_start_ = last_progress_time_ = std::chrono::steady_clock::now();
    last_progress_step_ = step_;

    const int first = step_;
    for (int k = 0; k < params_.nsteps; k++) {
        step();
        write_outputs();
        print_progress();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start_).count();
    int nrun = step_ - first;

    if (params_.restart_freq >= 0)
        write_restart(build_restart_final_filename(params_.restart_file), step_);

    printf("Wall time: %.3f s  |  Steps/s: %.1f  |  ns/day: %.3f  |  rebuilds: %d\n",
           elapsed, nrun / elapsed, (nrun * params_.dt / 1000.0) / (elapsed / 86400.0),
           n_rebuilds_);
    finish();
}

void Simulation::finish() {
    if (!params_.stress_on) return;
    int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
    std::vector<float> h_corr(buf_size);
    std::vector<int> h_counts(CORR_LEVELS * CORR_POINTS);
    int total_levels;
    correlator_.get_results(h_corr.data(), h_counts.data(), &total_levels);

    FILE* corr_fp = fopen(params_.stress_file, "w");
    if (!corr_fp) throw std::runtime_error(std::string("cannot open ") + params_.stress_file);
    fprintf(corr_fp, "# lag_steps  C_xx  C_xy  C_xz  C_yy  C_yz  C_zz  count\n");
    int lag_step = params_.stress_freq;
    for (int level = 0; level < total_levels; level++) {
        for (int lag = 0; lag < CORR_POINTS; lag++) {
            int count = h_counts[level * CORR_POINTS + lag];
            if (count == 0) continue;
            fprintf(corr_fp, "%d", lag_step * (lag + 1));
            for (int c = 0; c < STRESS_COMPONENTS; c++)
                fprintf(corr_fp, " %.8e",
                        h_corr[(level * CORR_POINTS + lag) * STRESS_COMPONENTS + c] / count);
            fprintf(corr_fp, " %d\n", count);
        }
        lag_step *= CORR_POINTS;
    }
    fclose(corr_fp);
    printf("Stress autocorrelation written to %s\n", params_.stress_file);
}

Simulation::~Simulation() {
    cudaDeviceSynchronize();
    dumper_.close();
    thermo_bufs_.close_file();
    thermo_bufs_.free();
    if (params_.rg_on) rg_bufs_.free();
    if (params_.stress_on) correlator_.free();
    if (d_nh_state_) free_nh_device_state(d_nh_state_);
    if (params_.ensemble == Ensemble::NPT_NH) free_npt_scratch(npt_scratch_);
    free_dev(d_ke_buf_);
    langevin_.free();
    verlet_.free();
    morton_.free();
    free_dev(d_lj_params_);
    free_dev(d_fene_params_);
    free_dev(d_angle_params_);
    free_dev(d_hbond_params_);
    free_dev(d_dihedral_params_);
    free_dev(d_table_idx_);
    free_dev(d_table_params_);
    free_dev(d_table_data_);
    free_dev(d_bond_table_idx_);
    free_dev(d_bond_table_params_);
    free_dev(d_bond_table_data_);
    free_dev(d_angle_table_idx_);
    free_dev(d_angle_table_params_);
    free_dev(d_angle_table_data_);
    free_dev(d_ewald_kvecs_);
    free_dev(d_ewald_sfac_);
    free_dev(d_virial_);
    free_dev(d_max_dr2_int_);
    free_dev(d_out4_);
    t_.free();
    p_.free();
    for (cudaEvent_t e : {force_done_lj_, force_done_bonded_, pos_ready_})
        if (e) cudaEventDestroy(e);
    for (cudaStream_t s : {stream_lj_, stream_bonded_, stream_io_})
        if (s) cudaStreamDestroy(s);
}
