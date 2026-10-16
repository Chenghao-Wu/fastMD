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
    if (opls_mode_) build_exclusions_opls(topo_, topo_, params_.special_dihedral != 0);
    else if (!topo_.bonds.empty()) build_exclusions(topo_, topo_);

    p_.allocate(params_.natoms);
    p_.upload(topo_.positions, topo_.velocities, topo_.images, topo_.charges, topo_.mol_ids);
    t_.upload(params_.natoms, topo_.bonds, topo_.bond_types, topo_.angles,
              topo_.dihedrals, topo_.dihedral_types, topo_.pairs14,
              topo_.exclusion_offsets, topo_.exclusion_list,
              topo_.excl13_offsets, topo_.excl13_list);

    upload_force_field();

    verlet_.allocate(params_.natoms, params_.rc + params_.skin, box_L_);
    CUDA_CHECK(cudaMalloc(&d_status_, sizeof(DeviceStatus)));
    CUDA_CHECK(cudaHostAlloc(&h_status_, sizeof(DeviceStatus), cudaHostAllocDefault));
    *h_status_ = DeviceStatus{0, box_L_, inv_L_, box_L_ * box_L_ * box_L_};
    CUDA_CHECK(cudaMemcpy(d_status_, h_status_, sizeof(DeviceStatus), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_step_, sizeof(int)));
    const int first_step = 1;
    CUDA_CHECK(cudaMemcpy(d_step_, &first_step, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_virial_, 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out4_, p_.natoms_padded * sizeof(float4)));


    // Blocking stream: stray legacy-default-stream operations (synchronous
    // copies in output code) still order after it.
    CUDA_CHECK(cudaStreamCreate(&stream_));

    setup_integrator();
    setup_analysis();

    build_neighbor_list();
    tune_pair_tpa();
    tune_bonded_tpa();
    launch_forces();
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (params_.ensemble != Ensemble::Langevin) {
        // first pre-force thermostat half-step uses the raw kinetic energy
        float init_ke = compute_ke_only(p_.vel, p_.pos, params_.natoms, d_ke_buf_, stream_);
        CUDA_CHECK(cudaMemcpy(&d_nh_state_->chain_KE_carry, &init_ke, sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

void Simulation::upload_force_field() {
    const int nt2 = params_.ntypes * params_.ntypes;
    d_lj_params_ = upload_vec(topo_.lj_params);
    if ((int)topo_.lj_params.size() != nt2)
        throw std::runtime_error("LJ parameter table has wrong size");
    if (params_.lj_force_cap > 0.0f)
        d_lj_cap_ = upload_vec(lj_force_cap_table(topo_.lj_params, params_.ntypes,
                                                  params_.lj_force_cap));

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

    BondedParams& bp = bonded_params_;
    bp.bond_type = t_.bond_type;
    bp.fene = d_fene_params_;
    bp.harmonic = d_hbond_params_;
    bp.bond_table_idx = d_bond_table_idx_;
    bp.bond_table_params = d_bond_table_params_;
    bp.bond_table_data = d_bond_table_data_;
    bp.bond_style = d_bond_table_data_ ? 3 : d_hbond_params_ ? 2 : d_fene_params_ ? 1 : 0;
    bp.angle = d_angle_params_;
    bp.angle_table_idx = d_angle_table_idx_;
    bp.angle_table_params = d_angle_table_params_;
    bp.angle_table_data = d_angle_table_data_;
    bp.angle_style = d_angle_table_data_ ? 2 : d_angle_params_ ? 1 : 0;
    bp.dihedral_type = t_.dihedral_type;
    bp.opls = d_dihedral_params_;
    bp.dihedral_style = d_dihedral_params_ ? 1 : 0;

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

}

void Simulation::setup_integrator() {
    langevin_.init(p_.natoms_padded, params_.Tdamp, params_.dt, params_.T_start, params_.seed,
                   params_.T_stop, params_.nsteps);
    if (params_.ensemble != Ensemble::Langevin) {
        nose_hoover_.init(params_);
        allocate_nh_device_state(d_nh_state_);
        init_nh_device_state(nose_hoover_, d_nh_state_);
        CUDA_CHECK(cudaMalloc(&d_ke_buf_, sizeof(float)));
    }
    if (params_.ensemble == Ensemble::NPT_NH)
        allocate_npt_scratch(npt_scratch_, &d_status_->V, &d_status_->L, &d_status_->inv_L);
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
    printf("\n[setup]  neighbor list: capacity=%d  max_count=%d  cells=%d^3  threads/atom: pair=%d bonded=%d\n\n",
           verlet_.max_neighbors, verlet_.max_count, verlet_.nx, pair_tpa_, bonded_tpa_);
}

// ---------------------------------------------------------------------------
// Neighbor list / spatial sorting
// ---------------------------------------------------------------------------

// Rebuild the neighbor list; every sort_every-th rebuild first reorders all
// particle data into cell order (ParticleData moves every per-particle field,
// DeviceTopology remaps the bonded lists).
void Simulation::build_neighbor_list() {
    if (box_L_ < 2.0f * params_.rc)
        throw std::runtime_error("box length " + std::to_string(box_L_) +
                                 " is smaller than 2*rc: minimum image is invalid");
    const float rc_skin = params_.rc + params_.skin;
    verlet_.bin(p_.pos, box_L_, rc_skin, stream_);
    bool sort_now = params_.sort_every > 0 && (n_rebuilds_ % params_.sort_every == 0);
    if (sort_now) {
        p_.apply_permutation(verlet_.order_padded(), stream_);
        t_.remap(p_.rank, stream_);
        n_sorts_++;
    }
    verlet_.build_from_bins(p_.pos, sort_now, box_L_, inv_L_, p_.tag,
                            t_.excl_offsets, t_.excl_list, rc_skin, stream_);
    CUDA_CHECK(cudaMemcpyAsync(p_.pos_ref, p_.pos, p_.natoms_padded * sizeof(float4),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(&d_status_->max_dr2_bits, 0, sizeof(int), stream_));
    box_L_at_build_ = box_L_;
    steps_since_rebuild_ = 0;
    n_rebuilds_++;
}

void Simulation::read_status() {
    CUDA_CHECK(cudaMemcpyAsync(h_status_, d_status_, sizeof(DeviceStatus),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (params_.ensemble == Ensemble::NPT_NH) {
        box_L_ = h_status_->L;
        inv_L_ = h_status_->inv_L;
    }
}

// A pair outside the list (distance > rc + skin at build time) can come
// within rc once particles move. Positions are tracked relative to the
// reference configuration scaled affinely with the box (NPT scales pos_ref
// each step), so with s = L / L_build and maximum displacement d the list
// stays valid while  s (rc + skin) - 2 d > rc.
bool Simulation::need_rebuild() {
    steps_since_rebuild_++;
    read_status();
    float max_dr2;
    std::memcpy(&max_dr2, &h_status_->max_dr2_bits, sizeof(float));
    float s = box_L_ / box_L_at_build_;
    float half_margin = 0.5f * (s * (params_.rc + params_.skin) - params_.rc);
    bool triggered = half_margin <= 0.0f || max_dr2 > half_margin * half_margin;
    bool too_old = params_.neigh_max_age > 0 && steps_since_rebuild_ >= params_.neigh_max_age;
    return triggered || too_old;
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

void Simulation::launch_forces() {
    const float L = box_L_, inv_L = inv_L_;
    const int n = params_.natoms;

    cudaStream_t st = stream_;
    CUDA_CHECK(cudaMemsetAsync(d_virial_, 0, 6 * sizeof(double), st));

    // Nonbonded kernels overwrite force[i] (=), everything after accumulates.
    launch_pair_kernel(pair_tpa_);
    if (params_.ewald_on) {
        launch_ewald_sf_kernel(p_.pos, p_.charge, d_ewald_kvecs_, d_ewald_sfac_,
                               n, params_.ewald_kcount, L, st);
        launch_ewald_recip_kernel(p_.pos, p_.charge, d_ewald_kvecs_, d_ewald_sfac_,
                                  p_.force, d_virial_, n, params_.ewald_kcount,
                                  params_.coul_const, params_.ewald_alpha, L, st);
        launch_ewald_excl_kernel(p_.pos, p_.charge, p_.force, d_virial_,
                                 t_.excl13_offsets, t_.excl13_list, p_.rank,
                                 n, t_.nexcl13, params_.coul_const, params_.ewald_alpha,
                                 L, inv_L, st);
    }
    if (t_.npairs14 > 0) {
        launch_pair14_kernel(p_.pos, p_.charge, p_.force, d_virial_, t_.pairs14, d_lj_params_,
                             t_.npairs14, params_.ntypes, params_.coul_const,
                             params_.ewald_alpha, params_.lj14, params_.ewald14_coul,
                             L, inv_L, st);
    }
    if (d_table_params_) {
        launch_table_kernel(p_.pos, p_.force, d_virial_, d_table_idx_, d_table_params_,
                            d_table_data_, verlet_.neighbors, verlet_.num_neighbors,
                            n, params_.ntypes, params_.rc2, L, inv_L, st, pair_tpa_);
    }

    // Bonded terms accumulate into the same force buffer, so they follow the
    // overwriting nonbonded kernel on the same stream.
    if (params_.bonded_gather) {
        launch_bonded_kernel(bonded_tpa_);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t_.nbonds > 0 && d_bond_table_data_) {
        launch_bond_table_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                                 d_bond_table_idx_, d_bond_table_params_, d_bond_table_data_,
                                 t_.nbonds, L, inv_L, st);
    } else if (t_.nbonds > 0 && d_hbond_params_) {
        launch_bond_harmonic_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                                    d_hbond_params_, t_.nbonds, L, inv_L, st);
    } else if (t_.nbonds > 0 && d_fene_params_) {
        launch_fene_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                           d_fene_params_, t_.nbonds, 1, L, inv_L, st);
    }
    if (t_.nangles > 0 && d_angle_table_data_) {
        launch_angle_table_kernel(p_.pos, p_.force, d_virial_, t_.angles, d_angle_table_idx_,
                                  d_angle_table_params_, d_angle_table_data_,
                                  t_.nangles, L, inv_L, st);
    } else if (t_.nangles > 0 && d_angle_params_) {
        launch_angle_kernel(p_.pos, p_.force, d_virial_, t_.angles, d_angle_params_,
                            t_.nangles, L, inv_L, st);
    }
    if (t_.ndihedrals > 0 && d_dihedral_params_) {
        launch_dihedral_opls_kernel(p_.pos, p_.force, d_virial_, t_.dihedrals,
                                    t_.dihedral_type, d_dihedral_params_,
                                    t_.ndihedrals, L, inv_L, st);
    }
    CUDA_CHECK(cudaGetLastError());
}

void Simulation::launch_pair_kernel(int tpa) {
    const int n = params_.natoms;
    if (params_.ewald_on)
        launch_lj_coul_long_kernel(p_.pos, p_.charge, p_.force, d_virial_, d_lj_params_,
                                   verlet_.neighbors, verlet_.num_neighbors,
                                   n, params_.ntypes, params_.coul_const, params_.ewald_alpha,
                                   params_.rc2, box_L_, inv_L_, stream_, tpa);
    else
        launch_lj_kernel(p_.pos, p_.force, d_virial_, d_lj_params_,
                         verlet_.neighbors, verlet_.num_neighbors,
                         n, params_.ntypes, params_.rc2, box_L_, inv_L_, stream_,
                         d_lj_cap_, params_.lj_force_cap, tpa);
}

// Time the overwriting pair kernel with 1, 2, 4 and 8 threads per particle on
// the actual system and keep the fastest (a few milliseconds at start-up).
// The optimum depends on system size, neighbor count and GPU.
void Simulation::tune_pair_tpa() {
    if (params_.pair_tpa > 0) { pair_tpa_ = params_.pair_tpa; return; }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float best_ms = 1e30f;
    for (int tpa : {1, 2, 4, 8}) {
        launch_pair_kernel(tpa);   // warm-up
        CUDA_CHECK(cudaEventRecord(e0, stream_));
        const int reps = 10;
        for (int r = 0; r < reps; r++) launch_pair_kernel(tpa);
        CUDA_CHECK(cudaEventRecord(e1, stream_));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < 0.97f * best_ms) {   // prefer fewer threads unless clearly faster
            best_ms = ms;
            pair_tpa_ = tpa;
        }
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}

void Simulation::launch_bonded_kernel(int tpa) {
    launch_bonded_gather_kernel(p_.pos, p_.force, d_virial_, p_.tag, t_.term_offsets,
                                t_.terms, t_.bonds, t_.angles, t_.dihedrals, bonded_params_,
                                params_.natoms, box_L_, inv_L_, stream_, tpa);
}

// Same timing-based choice as tune_pair_tpa for the bonded gather kernel
// (repeated launches accumulate into force[]; the first real force
// evaluation overwrites it).
void Simulation::tune_bonded_tpa() {
    if (!params_.bonded_gather || !t_.term_offsets) return;
    if (params_.pair_tpa > 0) { bonded_tpa_ = params_.pair_tpa; return; }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float best_ms = 1e30f;
    for (int tpa : {1, 2, 4, 8}) {
        launch_bonded_kernel(tpa);
        CUDA_CHECK(cudaEventRecord(e0, stream_));
        for (int r = 0; r < 10; r++) launch_bonded_kernel(tpa);
        CUDA_CHECK(cudaEventRecord(e1, stream_));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < 0.97f * best_ms) { best_ms = ms; bonded_tpa_ = tpa; }
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}

void Simulation::compute_forces() {
    if (need_rebuild()) build_neighbor_list();
    launch_forces();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Simulation::write_forces(const std::string& path) const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    const int n = params_.natoms;
    int* d_max_dr2 = &d_status_->max_dr2_bits;

    if (params_.ensemble == Ensemble::Langevin) {
        launch_integrator_pre_force(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2,
                                    p_.image, p_.tag, d_step_, langevin_,
                                    n, box_L_, inv_L_, hdt, stream_);
    } else if (nose_hoover_.is_npt) {
        launch_nh_npt_ke_and_virial_trace(p_.vel, d_virial_, p_.pos, d_ke_buf_,
                                          npt_scratch_.d_virial_trace, n, stream_);
        nh_npt_chain_baro_kernel<<<1, 32, 0, stream_>>>(
            d_nh_state_, d_ke_buf_, npt_scratch_.d_virial_trace, true,
            nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest, nose_hoover_.W,
            hdt, params_.dt, nose_hoover_.V0, n,
            npt_scratch_.d_V, npt_scratch_.d_L, npt_scratch_.d_inv_L,
            npt_scratch_.d_baro_scale, npt_scratch_.d_exp_vW,
            npt_scratch_.d_v_eps_W, npt_scratch_.d_v_eps_W_dt);
        launch_nh_npt_pre_force_fused(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2,
                                      p_.image, d_nh_state_, npt_scratch_.d_baro_scale,
                                      npt_scratch_.d_exp_vW, npt_scratch_.d_v_eps_W_dt,
                                      hdt, params_.dt, n, npt_scratch_.d_L,
                                      npt_scratch_.d_inv_L, stream_);
    } else {
        nh_propagate_chain_kernel<<<1, 32, 0, stream_>>>(
            d_nh_state_, d_ke_buf_, true, nose_hoover_.M, nose_hoover_.Q1,
            nose_hoover_.Q_rest, hdt, n);
        launch_nh_nvt_pre_force_fused(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2,
                                      p_.image, d_nh_state_, hdt, params_.dt,
                                      n, box_L_, inv_L_, stream_);
    }
}

void Simulation::integrate_post_force() {
    const float hdt = 0.5f * params_.dt;
    const int n = params_.natoms;
    if (params_.ensemble == Ensemble::Langevin) {
        launch_integrator_post_force(p_.vel, p_.force, p_.pos, n, hdt, stream_, d_step_);
    } else if (nose_hoover_.is_npt) {
        launch_nh_v_verlet_half(p_.vel, p_.force, p_.pos, n, hdt, stream_);
        launch_nh_npt_ke_and_virial_trace(p_.vel, d_virial_, p_.pos, d_ke_buf_,
                                          npt_scratch_.d_virial_trace, n, stream_);
        nh_npt_chain_baro_kernel<<<1, 32, 0, stream_>>>(
            d_nh_state_, d_ke_buf_, npt_scratch_.d_virial_trace, false,
            nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest, nose_hoover_.W,
            hdt, params_.dt, nose_hoover_.V0, n,
            npt_scratch_.d_V, npt_scratch_.d_L, npt_scratch_.d_inv_L,
            npt_scratch_.d_baro_scale, npt_scratch_.d_exp_vW,
            npt_scratch_.d_v_eps_W, npt_scratch_.d_v_eps_W_dt);
        launch_nh_barostat_vel_half(p_.vel, npt_scratch_.d_v_eps_W, 1.0f / (3.0f * n), n, hdt,
                                    stream_);
        launch_nh_global_scale_vel(p_.vel, d_nh_state_, n, stream_);
    } else {
        CUDA_CHECK(cudaMemsetAsync(d_ke_buf_, 0, sizeof(float), stream_));
        launch_nh_nvt_v_half_ke_reduce(p_.vel, p_.force, p_.pos, d_ke_buf_, n, hdt, stream_);
        nh_propagate_chain_kernel<<<1, 32, 0, stream_>>>(
            d_nh_state_, d_ke_buf_, false, nose_hoover_.M, nose_hoover_.Q1,
            nose_hoover_.Q_rest, hdt, n);
        launch_nh_global_scale_vel(p_.vel, d_nh_state_, n, stream_);
    }
    CUDA_CHECK(cudaGetLastError());
}

void Simulation::step() {
    ++step_;
    integrate_pre_force();
    if (need_rebuild()) build_neighbor_list();
    launch_forces();
    integrate_post_force();
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void Simulation::write_restart(const std::string& fname, int64_t step) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
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
    bool need_rg = params_.rg_on && s % params_.rg_freq == 0;
    bool need_dump = params_.dump_freq > 0 && s % params_.dump_freq == 0;
    bool need_restart = params_.restart_freq > 0 && s % params_.restart_freq == 0 &&
                        s != params_.nsteps;
    if (!(need_thermo || need_stress || need_rg || need_dump || need_restart)) return;
    // the barostat may have changed the box during the post-force half step
    read_status();

    if (need_thermo || need_stress) {
        ThermoOutput thermo;
        compute_thermo(p_.vel, p_.pos, p_.force, d_virial_, params_.natoms, box_L_, &thermo,
                       thermo_bufs_, s, need_thermo ? thermo_bufs_.fp : nullptr, stream_,
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

    if (need_rg) {
        float rg;
        compute_rg(p_.pos, p_.image, p_.rank, rg_bufs_, box_L_, s, &rg, stream_);
    }

    if (need_dump) {
        p_.gather_float4_by_tag(d_out4_, p_.pos, stream_);
        dumper_.dump_frame(d_out4_, s, box_L_, stream_);
    }

    if (need_restart) write_restart(build_restart_filename(params_.restart_file, s), s);
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
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    read_status();
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start_).count();
    int nrun = step_ - first;

    if (params_.restart_freq >= 0)
        write_restart(build_restart_final_filename(params_.restart_file), step_);

    printf("Wall time: %.3f s  |  Steps/s: %.1f  |  ns/day: %.3f  |  rebuilds: %d  sorts: %d\n",
           elapsed, nrun / elapsed, (nrun * params_.dt / 1000.0) / (elapsed / 86400.0),
           n_rebuilds_, n_sorts_);
    finish();
}

void Simulation::finish() {
    if (!params_.stress_on) return;
    correlator_.write(params_.stress_file, params_.stress_freq);
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
    free_dev(d_lj_params_);
    free_dev(d_lj_cap_);
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
    free_dev(d_status_);
    if (h_status_) cudaFreeHost(h_status_);
    free_dev(d_step_);
    free_dev(d_out4_);
    t_.free();
    p_.free();
    if (stream_) cudaStreamDestroy(stream_);
}
