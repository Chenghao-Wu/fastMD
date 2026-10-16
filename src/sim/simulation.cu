#include "simulation.cuh"
#include "../core/system.cuh"
#include "../neighbor/skin_trigger.cuh"
#include "../force/lj.cuh"
#include "../force/soft.cuh"
#include "../force/wall.cuh"
#include "../analysis/film.cuh"
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
    box_ = make_box(params_.box_L3);
    box_.xy = params_.box_xy;
    if (box_.xy != 0.0f) check_tilt_allowed(box_.xy);

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

    // NPT: size cells for 5% box compression before the grid must change
    verlet_.build_tpa = 4;
    verlet_.allocate(params_.natoms, params_.rc, params_.skin, box_,
                     params_.ensemble == Ensemble::NPT_NH ? 1.05f : 1.0f, params_.neigh_max_age);
    check_every_ = params_.check_every;
    CUDA_CHECK(cudaMalloc(&d_status_, sizeof(DeviceStatus)));
    CUDA_CHECK(cudaHostAlloc(&h_status_, sizeof(DeviceStatus), cudaHostAllocDefault));
    *h_status_ = DeviceStatus{};
    h_status_->box = box_;
    h_status_->V = box_.volume();
    CUDA_CHECK(cudaMemcpy(d_status_, h_status_, sizeof(DeviceStatus), cudaMemcpyHostToDevice));
    {
        double mass = 0.0;
        for (const float4& p : topo_.positions) {
            int t = unpack_type_id(p.w);
            mass += t < (int)topo_.masses.size() ? topo_.masses[t] : 1.0;
        }
        inv_total_mass_ = mass > 0.0 ? 1.0 / mass : 0.0;
    }
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

    ckpt_p_.allocate(params_.natoms);
    CUDA_CHECK(cudaMalloc(&d_ckpt_status_, sizeof(DeviceStatus)));
    CUDA_CHECK(cudaMalloc(&d_ckpt_step_, sizeof(int)));
    if (d_nh_state_) CUDA_CHECK(cudaMalloc(&d_ckpt_nh_, sizeof(NoseHooverDeviceState)));

    rebuild_neighbors(params_.sort_every > 0);
    launch_film_center();   // valid before the first step (depth-dependent friction)
    tune_pair_tpa();
    tune_bonded_tpa();
    launch_forces(true);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (params_.neigh_capacity > 0) {
        // testing knob: the first device-side build uses this capacity (it
        // overflows if too small and is then undone by the rollback path)
        verlet_.set_capacity(params_.neigh_capacity);
        verlet_.request_rebuild(stream_);
    }

    if (is_nose_hoover(params_.ensemble)) {
        // first pre-force thermostat half-step uses the raw kinetic energy
        float init_ke = compute_ke_only(p_.vel, p_.pos, params_.natoms, d_ke_buf_, stream_);
        CUDA_CHECK(cudaMemcpy(&d_nh_state_->chain_KE_carry, &init_ke, sizeof(float),
                              cudaMemcpyHostToDevice));
        if (params_.ensemble == Ensemble::NVT_NH) {
            // pre-force chain half-step of step 1; later ones run fused in
            // the previous step's post-force kernel
            nh_propagate_chain_kernel<<<1, 32, 0, stream_>>>(
                d_nh_state_, d_ke_buf_, true, nose_hoover_.M, nose_hoover_.Q1,
                nose_hoover_.Q_rest, 0.5f * params_.dt, params_.natoms);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
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
    bp.angle_style = d_angle_table_data_ ? 2 : d_angle_params_ ? (topo_.angle_cosine ? 3 : 1) : 0;
    bp.dihedral_type = t_.dihedral_type;
    bp.opls = d_dihedral_params_;
    bp.dihedral_style = d_dihedral_params_ ? 1 : 0;

    if (params_.ewald_on) {
        if (topo_.charges.empty())
            throw std::runtime_error("coulomb ewald requires charges (Atoms # full)");
        if (params_.ewald_tol > 0.0f) {
            // relative RMS force error target -> alpha and kmax / grid
            params_.ewald_alpha = ewald_alpha_for_tolerance(params_.ewald_tol, params_.rc);
            if (params_.kspace == 1)
                params_.pme_grid = pme_grid_for_tolerance(params_.ewald_tol, params_.ewald_alpha, box_.L.x);
            else
                params_.ewald_kmax = ewald_kmax_for_tolerance(params_.ewald_tol, params_.ewald_alpha, box_.L.x);
        }
        if (params_.kspace == 1) {
            pme_.allocate(params_.natoms, params_.pme_grid, params_.pme_order,
                          params_.ewald_alpha, params_.coul_const);
        } else {
            if (params_.ewald_kmax > 64)
                throw std::runtime_error("direct Ewald needs kmax " + std::to_string(params_.ewald_kmax) +
                                         " > 64 for this tolerance and box; use 'coulomb pme'");
            std::vector<int3> kvecs = ewald_build_kvectors(params_.ewald_kmax, box_.L.x,
                                                           params_.ewald_alpha);
            params_.ewald_kcount = (int)kvecs.size();
            d_ewald_kvecs_ = upload_vec(kvecs);
            CUDA_CHECK(cudaMalloc(&d_ewald_sfac_, 2 * kvecs.size() * sizeof(float)));
        }
        params_.ewald_self_offset = ewald_self_offset(topo_.charges, params_.coul_const,
                                                      params_.ewald_alpha, box_.L.x);
    }

}

void Simulation::setup_integrator() {
    langevin_.init(p_.natoms_padded, params_.Tdamp, params_.dt, params_.T_start, params_.seed,
                   params_.T_stop, params_.nsteps);
    if (params_.ensemble == Ensemble::NVE) {   // velocity Verlet: no friction, no noise
        langevin_.c1 = 1.0f;
        langevin_.c2 = 0.0f;
    }
    if (is_nose_hoover(params_.ensemble)) {
        nose_hoover_.init(params_);
        allocate_nh_device_state(d_nh_state_);
        init_nh_device_state(nose_hoover_, d_nh_state_);
        CUDA_CHECK(cudaMalloc(&d_ke_buf_, sizeof(float)));
    }
    if (params_.ensemble == Ensemble::NPT_NH)
        allocate_npt_scratch(npt_scratch_, &d_status_->V, &d_status_->box);
}

void Simulation::setup_analysis() {
    if (params_.stress_on) {
        correlator_.allocate_auto(6, params_.stress_p, params_.stress_m);
        stress_sampler_.allocate();
    }
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
    if (params_.traj_mode != 0) {
        if (params_.traj_mode == 1) traj_sched_.set_linear(params_.traj_every);
        else traj_sched_.set_log(params_.traj_every, params_.traj_base);
        std::vector<int> types(params_.natoms);
        for (int t = 0; t < params_.natoms; t++) types[t] = unpack_type_id(topo_.positions[t].w);
        traj_.open(params_.traj_file, types, topo_.mol_ids);
    }
    if (params_.dump_freq > 0)
        dumper_.open(params_.dump_file[0] ? params_.dump_file : "traj.bin", params_.natoms,
                     params_.ntypes);
}

void Simulation::print_setup() const {
    printf("[setup]  atoms=%d  box=%.2f x %.2f x %.2f  rc=%.2f  dt=%.4f  nsteps=%d",
           params_.natoms, box_.L.x, box_.L.y, box_.L.z, params_.rc, params_.dt, params_.nsteps);
    if (!topo_.bonds.empty()) printf("  bonds=%zu", topo_.bonds.size());
    if (!topo_.angles.empty()) printf("  angles=%zu", topo_.angles.size());
    if (!topo_.dihedrals.empty()) printf("  dihedrals=%zu", topo_.dihedrals.size());
    if (params_.walls.n > 0) printf("  walls=%d", params_.walls.n);
    printf("\n");
    if (params_.ewald_on) {
        if (params_.kspace == 1)
            printf("[setup]  coulomb=pme  alpha=%.4f  grid=%d^3  order=%d", params_.ewald_alpha,
                   params_.pme_grid, params_.pme_order);
        else
            printf("[setup]  coulomb=ewald  alpha=%.4f  kmax=%d  kcount=%d", params_.ewald_alpha,
                   params_.ewald_kmax, params_.ewald_kcount);
        if (params_.ewald_tol > 0.0f) printf("  (auto, tolerance %.1e)", params_.ewald_tol);
        printf("  coul_const=%.4f  lj14=%.2f  coul14=%.2f  self-energy=%.4f\n",
               params_.coul_const, params_.lj14, params_.ewald14_coul, params_.ewald_self_offset);
    }
    if (params_.ensemble == Ensemble::NVE) {
        printf("[setup]  ensemble=nve");
    } else if (is_nose_hoover(params_.ensemble)) {
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
    printf("\n[setup]  neighbor list: capacity=%d  max_count=%d  cells=%d^3  threads/atom: pair=%d bonded=%d  (device-side rebuilds, checks every %d steps)\n\n",
           verlet_.max_neighbors, verlet_.max_count, verlet_.nx, pair_tpa_, bonded_tpa_,
           check_every_);
}

// ---------------------------------------------------------------------------
// Neighbor list / spatial sorting
// ---------------------------------------------------------------------------

// Synchronous rebuild driven by the host (setup, periodic re-sorting,
// after a rollback). With sort, particles are first reordered into cell
// order (ParticleData moves every per-particle field, DeviceTopology remaps
// the bonded lists). Grows the capacity / regrids until the build is valid.
void Simulation::rebuild_neighbors(bool sort) {
    invalidate_graphs();   // capacity / grid (kernel parameters) may change
    if (box_.min_length() < 2.0f * params_.rc)
        throw std::runtime_error("shortest box edge " + std::to_string(box_.min_length()) +
                                 " is smaller than 2*rc: minimum image is invalid");
    if (!verlet_.grid_ok_for(box_)) verlet_.set_grid(box_);
    const Box* d_box = &d_status_->box;
    int* d_dr2 = &d_status_->max_dr2_bits;
    if (sort) {
        verlet_.bin(p_.pos, p_.pos_ref, d_box, d_dr2, stream_);
        p_.apply_permutation(verlet_.order_padded(), stream_);
        t_.remap(p_.rank, stream_);
        n_sorts_++;
        last_sort_step_ = step_;
    }
    for (int attempt = 0; attempt < 8; attempt++) {
        verlet_.request_rebuild(stream_);
        verlet_.step(p_.pos, p_.pos_ref, d_box, d_dr2, p_.tag, t_.excl_offsets, t_.excl_list,
                     stream_);
        verlet_.read_state(stream_);
        if (verlet_.h_state->grid_error) {
            verlet_.set_grid(box_);
            verlet_.clear_errors(stream_);
        } else if (verlet_.h_state->overflow) {
            verlet_.grow(verlet_.h_state->overflow);
            verlet_.clear_errors(stream_);
        } else {
            return;
        }
    }
    throw std::runtime_error("neighbor list could not be built");
}

void Simulation::read_status() {
    CUDA_CHECK(cudaMemcpyAsync(h_status_, d_status_, sizeof(DeviceStatus),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (params_.ensemble == Ensemble::NPT_NH) {
        box_ = h_status_->box;
    }
}

// The neighbor list is maintained on the device without per-step host
// checks; overflow of its capacity or a grid invalidated by box compression
// is detected here, at check points, and undone by rolling back to the last
// checkpoint (the Langevin noise depends only on seed, tag and step, so the
// re-run is the same trajectory up to floating-point summation order).
bool Simulation::validate() {
    if (params_.walls.n > 0) {
        read_status();
        if (h_status_->wall_error)
            throw std::runtime_error("a particle reached the plane of wall " +
                                     std::to_string(h_status_->wall_error - 1) +
                                     " (walls must not move through the system)");
    }
    verlet_.read_state(stream_);
    const DeviceNeighborState& st = *verlet_.h_state;
    if (!st.grid_error && !st.overflow) return true;
    read_status();
    if (st.grid_error) verlet_.set_grid(box_);
    if (st.overflow) verlet_.grow(st.overflow);
    verlet_.clear_errors(stream_);
    invalidate_graphs();
    return false;
}

void Simulation::save_checkpoint() {
    ckpt_p_.copy_from(p_, stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_ckpt_status_, d_status_, sizeof(DeviceStatus),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_ckpt_step_, d_step_, sizeof(int), cudaMemcpyDeviceToDevice, stream_));
    if (d_ckpt_nh_)
        CUDA_CHECK(cudaMemcpyAsync(d_ckpt_nh_, d_nh_state_, sizeof(NoseHooverDeviceState),
                                   cudaMemcpyDeviceToDevice, stream_));
    if (params_.stress_on) correlator_.save_snapshot(stream_);
    ckpt_step_ = step_;
    ckpt_vel_pending_ = velocities_pending_;
}

void Simulation::restore_checkpoint() {
    p_.copy_from(ckpt_p_, stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_status_, d_ckpt_status_, sizeof(DeviceStatus),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_step_, d_ckpt_step_, sizeof(int), cudaMemcpyDeviceToDevice, stream_));
    if (d_ckpt_nh_)
        CUDA_CHECK(cudaMemcpyAsync(d_nh_state_, d_ckpt_nh_, sizeof(NoseHooverDeviceState),
                                   cudaMemcpyDeviceToDevice, stream_));
    if (params_.stress_on) correlator_.restore_snapshot(stream_);
    t_.remap(p_.rank, stream_);
    step_ = ckpt_step_;
    velocities_pending_ = ckpt_vel_pending_;
    read_status();
    rebuild_neighbors(false);
    n_rollbacks_++;
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

void Simulation::launch_forces(bool ev) {
    const Box box = box_;
    const int n = params_.natoms;

    cudaStream_t st = stream_;
    const Box* d_box = &d_status_->box;   // kernels read the current box (NPT changes it)
    CUDA_CHECK(cudaMemsetAsync(d_virial_, 0, 6 * sizeof(double), st));

    // Nonbonded kernels overwrite force[i] (=), everything after accumulates.
    // The smaller kernels below always add energies/virial (harmless when
    // ev is false: neither is read then).
    launch_pair_kernel(pair_tpa_, ev);
    if (params_.walls.n > 0)
        launch_wall_kernel(p_.pos, p_.force, n, params_.walls, current_step_ptr(), params_.nsteps,
                           &d_status_->wall_error, st, ev);
    if (params_.ewald_on) {
        if (params_.kspace == 1) {
            pme_.compute(p_.pos, p_.charge, p_.force, d_virial_, d_box, st);
        } else {
            launch_ewald_sf_kernel(p_.pos, p_.charge, d_ewald_kvecs_, d_ewald_sfac_,
                                   n, params_.ewald_kcount, box.L.x, st, d_box);
            launch_ewald_recip_kernel(p_.pos, p_.charge, d_ewald_kvecs_, d_ewald_sfac_,
                                      p_.force, d_virial_, n, params_.ewald_kcount,
                                      params_.coul_const, params_.ewald_alpha, box.L.x, st, d_box);
        }
        launch_ewald_excl_kernel(p_.pos, p_.charge, p_.force, d_virial_,
                                 t_.excl13_offsets, t_.excl13_list, p_.rank,
                                 n, t_.nexcl13, params_.coul_const, params_.ewald_alpha,
                                 box, st, d_box);
    }
    if (t_.npairs14 > 0) {
        launch_pair14_kernel(p_.pos, p_.charge, p_.force, d_virial_, t_.pairs14, d_lj_params_,
                             t_.npairs14, params_.ntypes, params_.coul_const,
                             params_.ewald_alpha, params_.lj14, params_.ewald14_coul,
                             box, st, d_box);
    }
    if (d_table_params_) {
        launch_table_kernel(p_.pos, p_.force, d_virial_, d_table_idx_, d_table_params_,
                            d_table_data_, verlet_.neighbors, verlet_.num_neighbors,
                            n, params_.ntypes, params_.rc2, box, st, pair_tpa_, d_box);
    }

    // Bonded terms accumulate into the same force buffer, so they follow the
    // overwriting nonbonded kernel on the same stream.
    if (params_.bonded_gather) {
        launch_bonded_kernel(bonded_tpa_, ev);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t_.nbonds > 0 && d_bond_table_data_) {
        launch_bond_table_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                                 d_bond_table_idx_, d_bond_table_params_, d_bond_table_data_,
                                 t_.nbonds, box, st);
    } else if (t_.nbonds > 0 && d_hbond_params_) {
        launch_bond_harmonic_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                                    d_hbond_params_, t_.nbonds, box, st);
    } else if (t_.nbonds > 0 && d_fene_params_) {
        launch_fene_kernel(p_.pos, p_.force, d_virial_, t_.bonds, t_.bond_type,
                           d_fene_params_, t_.nbonds, 1, box, st);
    }
    if (t_.nangles > 0 && d_angle_table_data_) {
        launch_angle_table_kernel(p_.pos, p_.force, d_virial_, t_.angles, d_angle_table_idx_,
                                  d_angle_table_params_, d_angle_table_data_,
                                  t_.nangles, box, st);
    } else if (t_.nangles > 0 && d_angle_params_) {
        launch_angle_kernel(p_.pos, p_.force, d_virial_, t_.angles, d_angle_params_,
                            t_.nangles, box, st, topo_.angle_cosine);
    }
    if (t_.ndihedrals > 0 && d_dihedral_params_) {
        launch_dihedral_opls_kernel(p_.pos, p_.force, d_virial_, t_.dihedrals,
                                    t_.dihedral_type, d_dihedral_params_,
                                    t_.ndihedrals, box, st);
    }
    CUDA_CHECK(cudaGetLastError());
}

void Simulation::launch_pair_kernel(int tpa, bool ev) {
    const int n = params_.natoms;
    if (params_.pair_soft) {
        launch_soft_kernel(p_.pos, p_.force, d_virial_, verlet_.neighbors, verlet_.num_neighbors,
                           n, params_.rc, params_.soft_A_start, params_.soft_A_stop,
                           params_.nsteps, current_step_ptr(), box_, stream_, tpa,
                           &d_status_->box, ev);
    } else if (params_.ewald_on)
        launch_lj_coul_long_kernel(p_.pos, p_.charge, p_.force, d_virial_, d_lj_params_,
                                   verlet_.neighbors, verlet_.num_neighbors,
                                   n, params_.ntypes, params_.coul_const, params_.ewald_alpha,
                                   params_.rc2, box_, stream_, tpa, &d_status_->box, ev,
                                   params_.pair_shift != 0);
    else
        launch_lj_kernel(p_.pos, p_.force, d_virial_, d_lj_params_,
                         verlet_.neighbors, verlet_.num_neighbors,
                         n, params_.ntypes, params_.rc2, box_, stream_,
                         d_lj_cap_, params_.lj_force_cap, tpa, &d_status_->box, ev,
                         params_.pair_shift != 0);
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
        launch_pair_kernel(tpa, false);   // warm-up (most steps need forces only)
        CUDA_CHECK(cudaEventRecord(e0, stream_));
        const int reps = 10;
        for (int r = 0; r < reps; r++) launch_pair_kernel(tpa, false);
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

void Simulation::launch_bonded_kernel(int tpa, bool ev) {
    launch_bonded_gather_kernel(p_.pos, p_.force, d_virial_, p_.tag, t_.term_offsets,
                                t_.terms, t_.bonds, t_.angles, t_.dihedrals, bonded_params_,
                                params_.natoms, box_, stream_, tpa, &d_status_->box, ev);
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
        launch_bonded_kernel(tpa, false);
        CUDA_CHECK(cudaEventRecord(e0, stream_));
        for (int r = 0; r < 10; r++) launch_bonded_kernel(tpa, false);
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
    verlet_.step(p_.pos, p_.pos_ref, &d_status_->box, &d_status_->max_dr2_bits, p_.tag,
                 t_.excl_offsets, t_.excl_list, stream_);
    if (!validate()) rebuild_neighbors(false);
    launch_forces(true);
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

    if (!is_nose_hoover(params_.ensemble)) {   // Langevin or NVE
        launch_integrator_pre_force(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2,
                                    p_.image, p_.tag, d_step_, langevin_,
                                    n, box_, hdt, stream_);
    } else if (nose_hoover_.is_npt) {
        launch_nh_npt_ke_and_virial_trace(p_.vel, d_virial_, p_.pos, d_ke_buf_,
                                          npt_scratch_.d_virial_trace, n, stream_);
        nh_npt_chain_baro_kernel<<<1, 32, 0, stream_>>>(
            d_nh_state_, d_ke_buf_, npt_scratch_.d_virial_trace, true,
            nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest, nose_hoover_.W,
            hdt, params_.dt, nose_hoover_.V0, nose_hoover_.box0, n,
            npt_scratch_.d_V, npt_scratch_.d_box,
            npt_scratch_.d_baro_scale, npt_scratch_.d_exp_vW,
            npt_scratch_.d_v_eps_W, npt_scratch_.d_v_eps_W_dt);
        launch_nh_npt_pre_force_fused(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2,
                                      p_.image, d_nh_state_, npt_scratch_.d_baro_scale,
                                      npt_scratch_.d_exp_vW, npt_scratch_.d_v_eps_W_dt,
                                      hdt, params_.dt, n, npt_scratch_.d_box, stream_);
    } else {
        // nh_scale holds s_post(n-1) * s_pre(n) from the fused post-force kernel
        velocities_pending_ = false;
        launch_nh_nvt_pre_force_fused(p_.pos, p_.vel, p_.force, p_.pos_ref, d_max_dr2,
                                      p_.image, d_nh_state_, hdt, params_.dt,
                                      n, box_, stream_);
    }
}

void Simulation::integrate_post_force() {
    const float hdt = 0.5f * params_.dt;
    const int n = params_.natoms;
    if (!is_nose_hoover(params_.ensemble)) {
        launch_integrator_post_force(p_.vel, p_.force, p_.pos, n, hdt, stream_, d_step_);
    } else if (nose_hoover_.is_npt) {
        launch_nh_v_verlet_half(p_.vel, p_.force, p_.pos, n, hdt, stream_);
        launch_nh_npt_ke_and_virial_trace(p_.vel, d_virial_, p_.pos, d_ke_buf_,
                                          npt_scratch_.d_virial_trace, n, stream_);
        nh_npt_chain_baro_kernel<<<1, 32, 0, stream_>>>(
            d_nh_state_, d_ke_buf_, npt_scratch_.d_virial_trace, false,
            nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest, nose_hoover_.W,
            hdt, params_.dt, nose_hoover_.V0, nose_hoover_.box0, n,
            npt_scratch_.d_V, npt_scratch_.d_box,
            npt_scratch_.d_baro_scale, npt_scratch_.d_exp_vW,
            npt_scratch_.d_v_eps_W, npt_scratch_.d_v_eps_W_dt);
        launch_nh_barostat_vel_half(p_.vel, npt_scratch_.d_v_eps_W, 1.0f / (3.0f * n), n, hdt,
                                    stream_);
        launch_nh_global_scale_vel(p_.vel, d_nh_state_, n, stream_);
    } else {
        launch_nh_nvt_post_force_chain(p_.vel, p_.force, p_.pos, d_nh_state_, n, hdt,
                                       nose_hoover_.M, nose_hoover_.Q1, nose_hoover_.Q_rest,
                                       stream_);
        velocities_pending_ = true;
    }
    CUDA_CHECK(cudaGetLastError());
}

// Device work of one time step. Every per-step quantity (step counter,
// ramps, box, rebuild decision) lives in device memory, so the sequence is
// identical from step to step and can be captured as a CUDA graph.
void Simulation::launch_step(bool with_stress, bool ev) {
    launch_film_center();
    integrate_pre_force();
    verlet_.step(p_.pos, p_.pos_ref, &d_status_->box, &d_status_->max_dr2_bits, p_.tag,
                 t_.excl_offsets, t_.excl_list, stream_);
    launch_forces(ev);
    integrate_post_force();
    if (with_stress) {
        // device-only sample; a pending fused-NVT velocity scale is applied in the kernel
        stress_sampler_.sample(p_.vel, p_.pos, d_virial_, params_.natoms, &d_status_->box,
                               velocities_pending_ ? &d_nh_state_->s_post : nullptr, stream_);
        correlator_.push(stress_sampler_.d_stress, stream_);
    }
}

void Simulation::invalidate_graphs() {
    for (cudaGraphExec_t& g : graph_exec_)
        if (g) {
            CUDA_CHECK(cudaGraphExecDestroy(g));
            g = nullptr;
        }
}

// One time step. Energies and the virial are computed only when something
// reads them after this step (want_ev: outputs, the end of a run), on
// stress-sample steps, and every step under NPT (the barostat uses the virial).
void Simulation::advance_one(bool want_ev) {
    ++step_;
    const bool with_stress = params_.stress_on && step_ % params_.stress_freq == 0;
    const bool ev = want_ev || with_stress || params_.ensemble == Ensemble::NPT_NH;
    // the per-term bonded kernels take the box by value, so NPT then needs it
    // on the host every step and cannot be replayed from a graph
    const bool host_box = params_.ensemble == Ensemble::NPT_NH && !params_.bonded_gather;
    if (host_box) read_status();
    if (!params_.cuda_graphs || host_box) {
        launch_step(with_stress, ev);
        return;
    }
    cudaGraphExec_t& exec = graph_exec_[with_stress ? 2 : ev ? 1 : 0];
    if (!exec) {
        const bool pending = velocities_pending_;   // host bookkeeping replayed below
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        launch_step(with_stress, ev);
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        velocities_pending_ = pending;
        n_graph_captures_++;
    }
    CUDA_CHECK(cudaGraphLaunch(exec, stream_));
    // after a step the fused NVT path leaves the post-force scale pending
    velocities_pending_ = params_.ensemble == Ensemble::NVT_NH;
}

bool Simulation::output_due(int s) const {
    return traj_sched_.due(s) ||
           (params_.thermo_on && s % params_.thermo_freq == 0) ||
           (params_.rg_on && s % params_.rg_freq == 0) ||
           (params_.dump_freq > 0 && s % params_.dump_freq == 0) ||
           (params_.restart_freq > 0 && s % params_.restart_freq == 0);
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void Simulation::flush_velocities() {
    if (!velocities_pending_) return;
    launch_nh_flush_post_scale(p_.vel, d_nh_state_, params_.natoms, stream_);
    velocities_pending_ = false;
}

void Simulation::write_restart(const std::string& fname, int64_t step) {
    flush_velocities();
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
                      box_, step);
    printf("Restart written to %s (step %lld)\n", fname.c_str(), (long long)step);
}

void Simulation::write_outputs() {
    const int s = step_;
    bool need_thermo = params_.thermo_on && (s % params_.thermo_freq == 0);
    bool need_rg = params_.rg_on && s % params_.rg_freq == 0;
    bool need_dump = params_.dump_freq > 0 && s % params_.dump_freq == 0;
    bool need_restart = params_.restart_freq > 0 && s % params_.restart_freq == 0 &&
                        s != params_.nsteps;
    write_traj_frame_if_due();
    if (!(need_thermo || need_rg || need_dump || need_restart)) return;
    // the barostat may have changed the box during the post-force half step
    read_status();

    if (need_thermo) {
        flush_velocities();
        ThermoOutput thermo;
        compute_thermo(p_.vel, p_.pos, p_.force, d_virial_, params_.natoms, box_, &thermo,
                       thermo_bufs_, s, thermo_bufs_.fp, stream_,
                       params_.ewald_on ? params_.ewald_self_offset / params_.natoms : 0.0f);
        float etot = thermo.kinetic_energy + thermo.potential_energy;
        printf("\nStep %6d  T=%10.4f  KE=%11.2f  PE=%11.2f  Etot=%11.2f  Pxx=%8.2f  maxV=%5.2f  maxF=%5.2f\n",
               s, thermo.temperature, thermo.kinetic_energy, thermo.potential_energy, etot,
               thermo.stress[VXX], thermo.max_vel, thermo.max_force);
    }

    if (need_rg) {
        float rg;
        compute_rg(p_.pos, p_.image, p_.rank, rg_bufs_, box_, s, &rg, stream_);
    }

    if (need_dump) {
        p_.gather_float4_by_tag(d_out4_, p_.pos, stream_);
        dumper_.dump_frame(d_out4_, s, box_.L.x, stream_);   // cubic (config check)
    }

    if (need_restart) write_restart(build_restart_filename(params_.restart_file, s), s);
}

// Trajectory frames are written only at validated check points (output
// steps are check points), so a rolled-back segment never reaches the file.
void Simulation::write_traj_frame_if_due() {
    if (!traj_.is_open() || !traj_sched_.due(step_) || last_traj_step_ == step_) return;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    read_status();
    const double box[6] = {box_.L.x, box_.L.y, box_.L.z, box_.xy, box_.xz, box_.yz};
    traj_.write_frame(step_, box, p_.download_positions(), p_.download_images());
    last_traj_step_ = step_;
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

// Advance nsteps with periodic check points: neighbor-list overflow / grid
// errors roll back to the last checkpoint, particles are re-sorted every
// sort_every steps, and the configured outputs are written when due.
void Simulation::advance(int nsteps, bool progress) {
    const int target = step_ + nsteps;
    write_traj_frame_if_due();   // time origin of the run (e.g. step 0)
    save_checkpoint();
    while (step_ < target) {
        advance_one(output_due(step_ + 1) || step_ + 1 == target);
        bool at_output = output_due(step_) || step_ == target;
        if (at_output || step_ - ckpt_step_ >= check_every_) {
            if (!validate()) {
                restore_checkpoint();
                continue;
            }
            if (params_.sort_every > 0 && step_ - last_sort_step_ >= params_.sort_every) {
                read_status();
                rebuild_neighbors(true);
            }
            write_outputs();
            save_checkpoint();
        }
        if (progress) print_progress();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    read_status();
}

void Simulation::run(int nsteps) {
    if (nsteps < 0) throw std::invalid_argument("run: nsteps must be >= 0");
    advance(nsteps, false);
}

void Simulation::run() {
    progress_interval_ = 1000;
    if (params_.nsteps < 100) progress_interval_ = 1;
    else if (params_.nsteps < 1000) progress_interval_ = params_.nsteps / 2;
    t_start_ = last_progress_time_ = std::chrono::steady_clock::now();
    last_progress_step_ = step_;

    const int first = step_;
    advance(params_.nsteps, true);
    flush_velocities();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    verlet_.read_state(stream_);
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start_).count();
    int nrun = step_ - first;

    if (params_.restart_freq >= 0)
        write_restart(build_restart_final_filename(params_.restart_file), step_);

    printf("Wall time: %.3f s  |  Steps/s: %.1f  |  ns/day: %.3f  |  rebuilds: %d  sorts: %d  rollbacks: %d  graph captures: %d\n",
           elapsed, nrun / elapsed, (nrun * params_.dt / 1000.0) / (elapsed / 86400.0),
           verlet_.h_state->n_builds, n_sorts_, n_rollbacks_, n_graph_captures_);
    finish();
}

// ---------------------------------------------------------------------------
// State access (tag order)
// ---------------------------------------------------------------------------

std::vector<float4> Simulation::velocities() {
    flush_velocities();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return p_.download_velocities();
}

void Simulation::set_positions(const std::vector<float3>& xyz, const std::vector<int>* images) {
    const int n = params_.natoms;
    if ((int)xyz.size() != n)
        throw std::invalid_argument("set_positions: expected " + std::to_string(n) + " positions");
    if (images && (int)images->size() != 3 * n)
        throw std::invalid_argument("set_positions: expected " + std::to_string(3 * n) + " image counts");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    read_status();
    std::vector<float4> pos = p_.download_positions();   // keeps the packed types
    std::vector<int> img = images ? *images : std::vector<int>(3 * n, 0);
    for (int t = 0; t < n; t++) {
        float x = xyz[t].x, y = xyz[t].y, z = xyz[t].z;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            throw std::invalid_argument("set_positions: non-finite coordinate for particle " +
                                        std::to_string(t));
        wrap_position_host(x, y, z, &img[3 * t], box_);
        pos[t].x = x;
        pos[t].y = y;
        pos[t].z = z;
    }
    p_.set_positions(pos);
    p_.set_images(img);
    rebuild_neighbors(false);
    launch_forces(true);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Simulation::set_velocities(const std::vector<float3>& v) {
    const int n = params_.natoms;
    if ((int)v.size() != n)
        throw std::invalid_argument("set_velocities: expected " + std::to_string(n) + " velocities");
    // drop a pending fused-NVT scale (it belonged to the old velocities)
    flush_velocities();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::vector<float4> vel(n);
    for (int t = 0; t < n; t++) vel[t] = make_float4(v[t].x, v[t].y, v[t].z, 0.0f);
    p_.set_velocities(vel);
}

// Requirements for a sheared (xy-tilted) box.
void Simulation::check_tilt_allowed(float xy) const {
    if (std::fabs(xy) > 0.5f * box_.L.x)
        throw std::invalid_argument("xy tilt " + std::to_string(xy) + " exceeds Lx / 2");
    if (params_.ensemble == Ensemble::NPT_NH)
        throw std::invalid_argument("a tilted box cannot be combined with npt_nh");
    if (params_.ewald_on)
        throw std::invalid_argument("a tilted box cannot be combined with coulomb ewald/pme");
    if (!topo_.dihedrals.empty())
        throw std::invalid_argument("a tilted box cannot be combined with dihedrals (1-4 pair terms)");
    if (params_.dump_freq > 0)
        throw std::invalid_argument("the legacy dump format cannot describe a tilted box; use traj");
    for (int k = 0; k < params_.walls.n; k++)
        if (params_.walls.w[k].axis != 2)
            throw std::invalid_argument("with a tilted box walls must be along z");
}

// Change the xy tilt. affine: particles move with the box (x += dxy y / Ly,
// a homogeneous shear strain dxy / Ly; velocities unchanged), as in a step
// shear. Otherwise the unwrapped positions are kept and only re-wrapped.
void Simulation::set_tilt(float xy, bool affine) {
    if (xy != 0.0f) check_tilt_allowed(xy);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    read_status();
    const int n = params_.natoms;
    std::vector<float4> pos = p_.download_positions();
    std::vector<int> img = p_.download_images();
    const float dxy = xy - box_.xy;
    for (int t = 0; t < n; t++) {
        // unwrapped coordinates in the current box
        float ux = pos[t].x + img[3 * t] * box_.L.x + img[3 * t + 1] * box_.xy;
        float uy = pos[t].y + img[3 * t + 1] * box_.L.y;
        float uz = pos[t].z + img[3 * t + 2] * box_.L.z;
        if (affine) ux += dxy * uy / box_.L.y;
        img[3 * t] = img[3 * t + 1] = img[3 * t + 2] = 0;
        // re-wrap in the new box from the unwrapped coordinates
        Box nb = box_;
        nb.xy = xy;
        wrap_position_host(ux, uy, uz, &img[3 * t], nb);
        pos[t].x = ux;
        pos[t].y = uy;
        pos[t].z = uz;
    }
    box_.xy = xy;
    h_status_->box = box_;
    CUDA_CHECK(cudaMemcpy(&d_status_->box, &box_, sizeof(Box), cudaMemcpyHostToDevice));
    p_.set_positions(pos);
    p_.set_images(img);
    invalidate_graphs();
    rebuild_neighbors(false);
    launch_forces(true);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Simulation::apply_step_shear(float gamma) {
    set_tilt(box_.xy + gamma * box_.L.y, true);
}

void Simulation::launch_film_center() {
    if (params_.film_axis < 0) return;
    ::launch_film_center(p_.pos, p_.image, params_.natoms, params_.film_axis, &d_status_->box,
                         inv_total_mass_, &d_status_->film_accum, &d_status_->film_ticket,
                         &d_status_->film_center, stream_);
}

float Simulation::film_center() {
    if (params_.film_axis < 0) throw std::logic_error("film_center: no `film <axis>` in the config");
    launch_film_center();
    read_status();
    return h_status_->film_center;
}

void Simulation::set_walls(const WallSet& walls) {
    if (walls.n > 0 && params_.ensemble == Ensemble::NPT_NH)
        throw std::invalid_argument("walls cannot be combined with npt_nh");
    if (box_.xy != 0.0f)
        for (int k = 0; k < walls.n; k++)
            if (walls.w[k].axis != 2) throw std::invalid_argument("with a tilted box walls must be along z");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    params_.walls = walls;
    invalidate_graphs();   // the wall set is a kernel argument of the captured steps
    CUDA_CHECK(cudaMemsetAsync(&d_status_->wall_error, 0, sizeof(int), stream_));
    launch_forces(true);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ThermoOutput Simulation::thermo() {
    flush_velocities();
    read_status();
    ThermoOutput out;
    compute_thermo(p_.vel, p_.pos, p_.force, d_virial_, params_.natoms, box_, &out,
                   thermo_bufs_, step_, nullptr, stream_,
                   params_.ewald_on ? params_.ewald_self_offset / params_.natoms : 0.0f);
    return out;
}

void Simulation::finish() {
    if (!params_.stress_on) return;
    char vol[160];
    snprintf(vol, sizeof(vol), "# stress = (kinetic + virial) / V with V = %.10g (box volume); "
             "G(t) = V C(t) / (k_B T)\n", (double)box_.volume());
    correlator_.write(params_.stress_file, params_.stress_freq,
                      {"C_xx", "C_xy", "C_xz", "C_yy", "C_yz", "C_zz"}, vol);
    printf("Stress autocorrelation written to %s\n", params_.stress_file);
}

Simulation::~Simulation() {
    cudaDeviceSynchronize();
    if (params_.dump_freq > 0) dumper_.close();
    thermo_bufs_.close_file();
    thermo_bufs_.free();
    if (params_.rg_on) rg_bufs_.free();
    if (params_.stress_on) {
        correlator_.free();
        stress_sampler_.free();
    }
    ckpt_p_.free();
    free_dev(d_ckpt_nh_);
    if (d_ckpt_status_) cudaFree(d_ckpt_status_);
    free_dev(d_ckpt_step_);
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
    if (params_.ewald_on && params_.kspace == 1) pme_.free();
    free_dev(d_ewald_kvecs_);
    free_dev(d_ewald_sfac_);
    free_dev(d_virial_);
    free_dev(d_status_);
    if (h_status_) cudaFreeHost(h_status_);
    free_dev(d_step_);
    free_dev(d_out4_);
    t_.free();
    p_.free();
    invalidate_graphs();
    if (stream_) cudaStreamDestroy(stream_);
}
