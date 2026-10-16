#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static constexpr int TILE_SIZE = 32;

__host__ __device__ inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline float pack_type_id(int type_id) {
#ifdef __CUDA_ARCH__
    return __int_as_float(type_id);
#else
    float f;
    std::memcpy(&f, &type_id, sizeof(float));
    return f;
#endif
}

__host__ __device__ inline int unpack_type_id(float w) {
#ifdef __CUDA_ARCH__
    return __float_as_int(w);
#else
    int i;
    std::memcpy(&i, &w, sizeof(int));
    return i;
#endif
}

enum class Ensemble { Langevin = 0, NVT_NH = 1, NPT_NH = 2, NVE = 3 };
inline bool is_nose_hoover(Ensemble e) { return e == Ensemble::NVT_NH || e == Ensemble::NPT_NH; }

// Layout of every 6-component virial / stress array on the device and in
// ThermoOutput::stress. Text output uses the conventional order
// xx yy zz xy xz yz (see analysis/thermo.cu).
enum VirialIndex { VXX = 0, VXY = 1, VXZ = 2, VYY = 3, VYZ = 4, VZZ = 5 };

struct SimParams {
    float box_L;
    float inv_L;
    float half_L;
    float rc;
    float rc2;
    float skin;
    float dt;
    int   natoms;
    int   ntiles;
    int   ntypes;
    int   nsteps;
    int   dump_freq;
    uint64_t seed;

    int   thermo_on;
    int   thermo_freq;
    char  thermo_file[256];

    int   stress_on;
    int   stress_freq;
    char  stress_file[256];
    int   pair_shift = 0;     // 1: energy-shifted LJ (U(rc) = 0)
    int   stress_p = 16;      // correlator points per level
    int   stress_m = 2;       // correlator block-averaging factor

    int   restart_freq;
    char  restart_file[256];

    int   rg_on;
    int   rg_freq;
    char  rg_file[256];

    char  forces_dump_file[256];
    char  dump_file[256];       // binary trajectory (default traj.bin)
    char  traj_file[256];       // trajectory v2 (positions + images), see io/trajectory.hpp
    int   traj_mode = 0;        // 0 off, 1 linear, 2 log-periodic
    int   traj_every = 0;       // linear interval / log block length (steps)
    float traj_base = 2.0f;     // log: frames at block offsets round(base^j)

    // Nosé-Hoover
    Ensemble ensemble = Ensemble::Langevin;
    float T_start = 0.0f;
    float T_stop  = 0.0f;
    float Tdamp   = 0.0f;
    float P_start = 0.0f;
    float P_stop  = 0.0f;
    float Pdamp   = 0.0f;
    int   nh_chain_length = 3;

    // Ewald / OPLS Coulomb. ewald_on: any long-range Coulomb; kspace selects
    // the reciprocal solver. ewald_tol > 0: alpha and kmax / grid are chosen
    // at setup for that relative RMS force error.
    int   ewald_on   = 0;
    int   kspace     = 0;             // 0 direct Ewald, 1 PME
    float ewald_alpha = 0.0f;
    int   ewald_kmax = 0;
    float ewald_tol  = 0.0f;
    int   pme_grid   = 0;
    int   pme_order  = 5;
    float coul_const = 138.9354576f;  // e^2/(4 pi eps0) N_A in kJ*nm/(mol e^2)
    float ewald14_coul = 0.5f;        // OPLS 1-4 scaling
    float lj14 = 0.5f;
    float ewald_self_offset = 0.0f;   // total self-energy, PE correction
    int   ewald_kcount = 0;           // number of k-vectors (half-shell)

    // Optional, explicit LJ force cap (0 = off): below the radius where
    // |F_LJ| reaches lj_force_cap the force is held at the cap and the energy
    // continued linearly, so F = -dU/dr everywhere.
    float lj_force_cap = 0.0f;

    // 1-4 pairs: 0 = from bond-graph distance (LAMMPS default),
    // 1 = only end atoms of listed dihedrals (LAMMPS 'special_bonds dihedral yes').
    int   special_dihedral = 0;

    // Force a neighbor-list rebuild after this many steps even if no particle
    // moved more than the skin criterion allows (0 = never; the displacement
    // criterion alone is exact, as with LAMMPS 'neigh_modify check yes').
    int   neigh_max_age = 0;

    // Reorder particles into cell (Morton) order at the first check point at
    // least this many steps after the previous sort (0 = never). Improves
    // memory locality of the neighbor build and pair kernels; all
    // per-particle data and topology move together.
    int   sort_every = 1000;

    // Threads per particle in the neighbor-list pair kernels (1, 2, 4, 8);
    // 0 = pick the fastest by timing at start-up.
    int   pair_tpa = 0;

    // Bonded terms: 1 = per-particle gather kernel (no atomics),
    // 0 = per-term kernels with atomic scatter.
    int   bonded_gather = 1;

    // Device-side neighbor maintenance: steps between host checks of the
    // overflow/grid flags (and checkpoints); initial neighbor capacity per
    // particle (0 = estimate from density; mainly for testing the rollback).
    int   check_every = 100;
    int   neigh_capacity = 0;

    // Replay each time step as a captured CUDA graph (1) or launch kernels
    // individually (0).
    int   cuda_graphs = 1;
};
