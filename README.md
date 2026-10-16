# fastMD

GPU molecular dynamics in CUDA for bead-spring polymers and OPLS-AA liquids.

Multiscale models can speed up a simulation by changing the model. fastMD
speeds it up by optimizing the kernels.

## Performance

Measured on one RTX 3090 (CUDA 11.8) against LAMMPS (stable 22 Jul 2025)
with the KOKKOS CUDA package (`-k on g 1 -sf kk`), using the same system,
time step, cutoff and neighbor skin. Loop times exclude setup. Reproduce
with the scripts in [Benchmarking](#benchmarking).

**Kremer–Grest melt**, 30,000 beads (300 chains × 100), FENE + harmonic
angles + LJ (rc 2.5σ, skin 0.3σ), dt 0.001τ, thermo every 100 steps:

| Ensemble | fastMD (steps/s) | LAMMPS KOKKOS (steps/s) | Speedup |
|----------|------------------|-------------------------|---------|
| NVT (Nosé–Hoover) | 19,639 | 1,831 | 10.7× |
| NPT (Nosé–Hoover) | 13,918 | 945 | 14.7× |
| NVT (Langevin)    | 21,173 | 2,058 | 10.3× |

**OPLS-AA liquids with long-range electrostatics**: rc 10 Å, skin 2 Å,
dt 1 fs, NVT. Each code's reciprocal-space parameters are matched at equal
accuracy. LAMMPS uses PPPM at a relative accuracy of 1e-5. Its k-space
force error is measured against `kspace_style ewald 1e-8`. fastMD uses the
same α, with the smallest Ewald kmax or PME grid whose k-space force error
does not exceed LAMMPS'. The LAMMPS figure is the fastest of
neigh full/half × newton on/off.

| System | Atoms | LAMMPS KOKKOS PPPM | fastMD Ewald | fastMD PME | PME speedup |
|--------|-------|--------------------|--------------|------------|-------------|
| butane | 8,960 | 377 | 3,116 | 4,620 | 12.3× |
| methanol | 7,500 | 410 | 1,814 | 5,171 | 12.6× |
| butane (2×2×2 replicated) | 71,680 | 96 | 205 | 782 | 8.1× |

The 72k system shows why PME matters: direct Ewald costs O(N·K³), PME
O(N + K³ log K).

The original README quoted a 20× speedup. That benchmark passed `rc + skin`
to LAMMPS' `neighbor` command, which takes the skin alone, so LAMMPS
searched a 5.3σ neighbor radius against fastMD's 2.8σ. With the harness
fixed, the original code was 3.6× faster than LAMMPS KOKKOS on the NVT
case above.

## Quick start

**Requirements:** CUDA Toolkit ≥ 11.0, CMake ≥ 3.22, a C++17 compiler.

```bash
git clone https://github.com/Chenghao-Wu/fastMD.git
cd fastMD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
./build/fastmd run.conf
```

CMake options:

| Option | Default | |
|--------|---------|-|
| `CMAKE_CUDA_ARCHITECTURES` | SASS for sm_75/80/86/89(/90) + PTX for the newest | Set your GPU's architecture for a faster build, e.g. `86` |
| `FASTMD_BUILD_TESTS` | `ON` (top-level builds) | Unit tests; GoogleTest is fetched if not installed |
| `FASTMD_BUILD_PYTHON` | `OFF` (`ON` for wheels) | Python extension `fastmd._core` (needs pybind11) |

Tests: `ctest --test-dir build -L cpu` (no GPU needed) and
`ctest --test-dir build -L gpu`.

A minimal config (Kremer–Grest melt):

```
ntypes 1
rc 2.5
skin 0.3
dt 0.005
nsteps 100000
thermo 1 1000 thermo.dat
nvt_langevin 1.0 1.0 1.0 42
lj 0 0 1.0 1.0
bond_type 0 30.0 1.5 1.0 1.0
angle_type 0 5.0 180.0
lammps_data_file melt.data
```

### Preparing systems

Bead-spring melts and free-standing films can be prepared entirely with
fastMD:

```bash
# random-walk chains with the stiffness of the cosine bending potential (or --cinf),
# in a laterally wide box (--aspect) for films
python tools/make_kg_chains.py --nchains 400 --chain-length 50 --kbend 1.5 --aspect 3 -o raw.data
# Auhl-style fast push-off: soft repulsion ramp -> capped LJ -> full LJ; prints <R^2(n)>/n per stage
python tools/equilibrate_melt.py raw.data --kbend 1.5 --rc 2.5 --equil-steps 1000000 --workdir bulk
# film without cutting chains: unwrap along z, compress between moving walls, anneal, release
python tools/build_film.py bulk/equil_final.data --kbend 1.5 --workdir film
```

`tools/make_kg_melt.py` writes an overlap-free lattice melt (used by the
benchmarks).

## Python interface

`pip install .` builds the command-line program and the extension. Choose
the architecture with `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=86`.
`fastmd.Simulation` keeps a simulation resident on the GPU:

```python
import numpy as np
import fastmd

sim = fastmd.Simulation("run.conf")          # or fastmd.Simulation.from_text(config_text)
sim.run(1000)                                # repeatable; outputs in the config are written when due
x, img, f = sim.positions, sim.images, sim.forces   # (natoms, 3) numpy arrays, data-file atom order
t = sim.thermo()                             # total KE/PE, temperature, 3x3 pressure tensor
print(t.potential_energy, t.temperature, t.pressure)

sim.set_positions(x + 0.01 * np.random.randn(*x.shape), img)   # forces are recomputed
sim.velocities = np.zeros_like(x)
e_i = sim.per_atom_energy
```

Further methods, all acting on the resident simulation:

| | |
|-|-|
| `box`, `unwrapped_positions` | box (Lx, Ly, Lz, xy, xz, yz); unwrapped coordinates including the tilt |
| `minimize(ftol, max_iter, dt_max)` | FIRE energy minimisation (velocities are kept) |
| `apply_step_shear(gamma)`, `set_tilt(xy, affine)` | step shear strain in the xy plane; set the box tilt |
| `film_center` | film reference plane (needs `film`) |
| `set_friction_profile(z, factor)`, `set_friction_profile_analytic(a, ell, h)`, `clear_friction_profile()` | depth-dependent Langevin friction / Andersen collision rate |
| `layer_stress()` | extensive stress tensor of each film layer, (nlayers, 6) |
| `dyn_corr()` | on-the-fly multiple-origin correlations (needs `dyn_corr`) |
| `remove_walls()` | release a film after confined preparation |

Positions are wrapped into the box (the data-file box shifted to the
origin, sheared by the xy tilt). `fastmd.trajectory` reads trajectory
files, `fastmd.analysis` computes MSD, P(t), Rouse modes, F_s(q,t), G(t),
G'/G'', internal distances and film profiles (numpy only).
`get_binary_path()` finds the command-line program.

## Configuration reference

One keyword per line; `#` starts a comment. Unknown keywords, missing or
extra arguments, out-of-range values and inconsistent combinations are
errors that name the line. Type indices are 0-based, so LAMMPS type
minus 1.

### System

| Key | Arguments | |
|-----|-----------|-|
| `lammps_data_file` | `path` | Coordinates, image flags, masses, velocities, charges, molecule ids and topology. Atom styles `atomic`, `charge`, `molecular`/`bond`/`angle` and `full` are supported; with no style comment the style is inferred from the column count. Orthorhombic boxes; an `xy xz yz` line with only an xy tilt is accepted (sheared box). |
| `natoms` | `N` | Optional with a data file; must match it. |
| `ntypes` | `N` | Number of atom types (≤ 16). |
| `rc` | `r` | Nonbonded cutoff. LJ is truncated (see `pair_shift`). |
| `skin` | `s` | Neighbor-list skin. The list is rebuilt exactly when a pair could have crossed `rc`. |
| `dt` | `dt` | Time step. |
| `nsteps` | `N` | Steps for the command-line run. Temperature and pressure ramps span these steps. |

### Ensemble (exactly one)

| Key | Arguments | |
|-----|-----------|-|
| `nvt_langevin` | `T_start T_stop Tdamp [seed]` | Langevin (BAOAB). The noise is counter-based (seed, particle, step), so trajectories don't depend on particle order. |
| `nvt_nh` | `T_start T_stop Tdamp [chain]` | Nosé–Hoover chain (default length 3). |
| `npt_nh` | `T_start T_stop Tdamp P_start P_stop Pdamp [chain]` | Nosé–Hoover thermostat and isotropic barostat (all box edges scale together). |
| `nvt_andersen` | `T_start T_stop nu [seed]` | Velocity Verlet with Andersen collisions: velocities are redrawn from Maxwell–Boltzmann with probability ν dt per particle and step (counter-based random numbers). |
| `nve` | | Velocity Verlet without a thermostat. |

### Interactions

| Key | Arguments | |
|-----|-----------|-|
| `lj` | `ti tj eps sig` | LJ 12-6 for a type pair (symmetric). |
| `lj_type` | `t eps sig` | Per-type LJ with geometric mixing (OPLS). Not combinable with `lj`. |
| `lj_force_cap` | `F_max` | Softens LJ below the radius where \|F\| = F_max, keeping energy and force consistent. Use it for pushing apart overlapping starting structures. |
| `pair_shift` | `yes\|no` | Energy-shifted LJ, U(rc) = 0 (forces unchanged; default no). |
| `pair_soft` | `A_start A_stop` | Soft repulsion A[1 + cos(πr/rc)] instead of LJ, with A ramped linearly over the run (push-off of overlapping chains). |
| `table` | `ti tj file keyword [force\|force_over_r]` | Tabulated pair potential (LAMMPS `pair_style table` file). Column 4 is F = −dE/dr by default; `force_over_r` reads legacy F/r tables. |
| `coulomb` | `ewald alpha kmax` · `ewald auto tol` · `pme alpha grid [order]` · `pme auto tol [order]` | Long-range Coulomb: real-space erfc within `rc` plus direct Ewald or smooth PME (B-spline order 3–8, default 5). `auto` picks α, kmax or grid for a target relative RMS force error `tol` (as OpenMM does). Needs charges (`Atoms # full` or `charge`). |
| `coulomb_const` | `C` | Coulomb constant; default 138.9354576 kJ·nm/(mol·e²). |
| `special_bonds_14` | `lj14 coul14` | 1-4 scaling (default 0.5 0.5). 1-2 and 1-3 pairs are excluded. |
| `special_dihedral` | `yes\|no` | Take 1-4 pairs from listed dihedrals (`yes`) instead of bond-graph distance 3 (default, as LAMMPS does by default). |
| `bond_type` | `t k R0 eps sig` | FENE bond with WCA repulsion. |
| `bond_harmonic` | `t K r0` | Harmonic bond, E = K (r − r0)² (LAMMPS convention). |
| `bond_table` | `t file keyword [force\|force_over_r]` | Tabulated bond (LAMMPS `bond_style table`). |
| `angle_type` | `t k theta0` | Harmonic angle, E = k (θ − θ0)², θ0 in degrees. |
| `angle_cosine` | `t K` | Cosine bending potential E = K (1 + cos θ) (LAMMPS `angle_style cosine`; the Kremer–Grest stiffness potential). |
| `angle_table` | `t file keyword [degrees\|radians]` | Tabulated angle (LAMMPS `angle_style table`: θ in degrees, force per degree; `radians` for legacy tables). |
| `dihedral_opls` | `t V1 V2 V3 V4` | OPLS dihedral (LAMMPS `dihedral_style opls`). |

| `wall` | `face style eps sigma cutoff pos_start [pos_stop]` | Planar wall on `xlo`…`zhi`: `lj93` (energy-shifted LJ 9-3) or `harmonic` (ε (r − rc)²). The plane position is measured from the lower box edge and ramps from `pos_start` to `pos_stop` over the run (confined compression). Up to six walls; not with `npt_nh`. |

Every bond, angle and dihedral type present in the data file must have
parameters. Ewald/PME need a cubic box. A sheared box excludes
electrostatics, dihedrals, NPT, the legacy dump and x/y walls.

### Films

| Key | Arguments | |
|-----|-----------|-|
| `film` | `x\|y\|z` | Film normal. The film reference plane (mass-weighted mean unwrapped coordinate along it) is computed on the GPU every step; depth-resolved features are measured from it. |
| `friction_profile` | `analytic a ell h` · `table file` | Depth-dependent Langevin friction γ = γ_b f(d) (or Andersen collision rate ν f(d)), d = depth from the film reference plane. Analytic: f = exp(a [exp(−s/ℓ) + exp(−(h−s)/ℓ)]), s = d + h/2 clamped to [0, h]. Table: rows `d factor`. The friction step uses per-particle coefficients, so the equilibrium distribution is unchanged. |
| `layer_stress` | `nlayers width freq file [p m]` | Per-atom virial partition (pair terms half to each atom, bonded terms split equally) binned into layers of `width` about the film reference plane. Writes the multiple-tau cross-correlations ⟨P_a(t) P_b(0)⟩ of the extensive in-plane shear stress for all layer pairs; the layers sum to the total stress. LJ pair style with `bonded_mode gather` only. |

### Output

| Key | Arguments | |
|-----|-----------|-|
| `thermo` | `on freq file` | Columns `step KE PE T Pxx Pyy Pzz Pxy Pxz Pyz`. Energies are per atom, and PE includes the Ewald self energy. |
| `rg` | `on freq file` | Mean radius of gyration over molecules, using unwrapped coordinates. |
| `stress` | `on freq file [p m]` | Multiple-tau autocorrelation of the six stress components, sampled every `freq` steps; p points per level, block averaging by m (default 16, 2). The header records the volume and channel means; `fastmd.analysis.stress_relaxation` turns it into G(t). |
| `traj` | `file linear N` · `file log block [base]` | Trajectory frames (step, box, wrapped positions, image counts; data-file order). `log` writes frames at offsets 0 and round(base^j) of every block (default base 2), so each block start is a time origin. Read with `fastmd.trajectory.Trajectory`. |
| `dyn_corr` | `file origin_every norigins [bins n width] [q q] [modes p…]` | Multiple-origin correlations computed on the GPU during the run, per depth bin at the origin: bead MSD and F_s(q,t) (in-plane / normal, and restricted to beads that stayed in their bin), chain centre-of-mass MSD, P(t) and Rouse-mode autocorrelations, at log-spaced lags. |
| `dump_freq` | `N` | Legacy binary trajectory every N steps (0 = off; cubic boxes, no image counts). |
| `dump_file` | `path` | Legacy trajectory file (default `traj.bin`). |
| `restart` | `freq file` | LAMMPS data files (`Atoms # full` + `Velocities` + topology) every `freq` steps as `file_<step>.ext`, and at the end as `file_final.ext`. `freq` 0 writes only the final file. |
| `forces_dump_file` | `path` | Write the initial per-atom forces and energies, then exit (validation). |

### Preparation

| Key | Arguments | |
|-----|-----------|-|
| `minimize` | `ftol max_iter [dt_max]` | FIRE energy minimisation (FIRE 2.0 backtracking) before the run, at fixed box, until max \|F\| < `ftol`. |

### Performance and diagnostics

| Key | Default | |
|-----|---------|-|
| `sort_every` | 1000 | Re-sort particles along a Morton curve every N steps (0 = never). |
| `check_every` | 100 | Steps between host checks of neighbor-list overflow and the grid. On error the run rolls back to the last check point and repeats with a larger list. |
| `cuda_graphs` | 1 | Replay each step as a captured CUDA graph. |
| `pair_tpa` | 0 | Threads per particle in the pair and bonded kernels (1, 2, 4 or 8). 0 times all four at start-up and keeps the fastest; fix it for bitwise-reproducible runs. |
| `bonded_mode` | `gather` | `gather`: one thread group per atom, no atomics. `scatter`: one thread per term. |
| `neigh_max_age` | 0 | Also rebuild the list after N steps (0 = only when required). |
| `neigh_capacity` | 0 | Initial neighbors per particle (0 = estimate; it grows on overflow). |

Environment: `FASTMD_TPA=1|2|4|8` overrides the threads-per-atom
heuristic for kernels launched outside a simulation.
`FASTMD_CELL_DIV=1|2` forces full or half-size neighbor cells; by default
half cells are used above 64 particles per (rc + skin)³.

### Units

The code is unit-agnostic. The bead-spring examples use LJ units. The
OPLS path is validated in kJ/mol, nm, ps, e, with temperature in kJ/mol
(k_B T). `tools/opls_force_check.py` converts LAMMPS `real` units:
K[kJ/mol/nm²] = 418.4 × K[kcal/mol/Å²].

## Implementation notes

- **Particle order.** Each particle has a permanent tag (its position in the data file). Sorting permutes every per-particle array together, and topology, exclusions and outputs work in tag space.
- **Neighbor lists.** These are maintained on the GPU. The exact rebuild test (displacement and box scaling against `rc + skin`) runs in-kernel. Binning is a counting sort into Morton-ordered cells, half-size cells for dense systems. The build uses ballot-compacted thread groups. The host checks overflow flags only at check points.
- **Time step.** A step is a fixed sequence of kernels that read the step counter, ramps and box from device memory, so it can be replayed as a CUDA graph. Nosé–Hoover NVT chain updates are fused into the kinetic-energy reduction. Energies and the virial are computed only on steps that use them.
- **Kernels.** Pair and bonded kernels use 1–8 threads per particle, autotuned at start-up, with warp-shuffle reductions and double-precision global virial accumulation. All six virial components are computed.
- **Electrostatics.** PME uses B-spline charge spreading, cuFFT R2C/C2R, analytical differentiation and the full virial. Ewald and PME share the real-space, exclusion-correction and self-energy terms.
- **Boxes.** Orthorhombic, optionally sheared by an xy tilt. The minimum image reduces y first and shifts x by the tilt. Neighbor cells divide fractional coordinates with counts from the perpendicular cell widths. The hottest kernels choose a tilt-free variant at launch, since the tilt only changes at host events. A step shear moves particles affinely and adds γLy to the tilt.
- **Films.** Depth-resolved features use the film reference plane computed on the GPU, so they follow a drifting film. Per-layer stresses use per-atom virials written only on sample steps. Dynamic correlations store origin data per tag (positions and image counts), so spatial re-sorting is harmless; their kernels run after the replayed step graph. All accumulators are restored on rollback.

## Validation

Unit tests (`tests/`) check, among other things:

- forces against energy finite differences
- strain-derivative virials
- permutation invariance
- neighbor-list completeness under box deformation and overflow
- restart round trips
- correlator results against direct calculation
- PME against converged Ewald
- replicated systems in orthorhombic boxes, tilted-lattice equivalence and dU/dγ = −V P_xy in sheared boxes
- flat temperature profiles under strongly varying friction and Andersen collision rates
- layer stresses summing to the total stress, and layer cross-correlations to the global one
- GPU dynamic correlations against the same estimators computed from a trajectory
- FIRE convergence, walls, the soft push-off, and the full native melt → film pipeline
- config and data-file error handling

OPLS liquids are compared with LAMMPS (`lj/cut/coul/long`, `ewald`,
`special_bonds 0 0 0.5`, `dihedral_style opls`):

```bash
python tools/opls_force_check.py --data data/OPLS_butane/butane.data --workdir data/OPLS_butane
python tools/opls_nvt_check.py   --data data/OPLS_butane/butane.data --workdir data/OPLS_butane
```

The data files are generated with `data/OPLS_*/generate_*.py` (RDKit +
packmol). fastMD (fp32) agrees with LAMMPS (fp64) to a relative RMS force
error of 6e-5 on 9k-atom liquids, which is the fp32 floor.

## Benchmarking

```bash
# Kremer-Grest melt vs LAMMPS (validation run, then timed runs per ensemble)
python benchmarks/run.py --fastmd build/fastmd --lammps lmp_kk --data melt30k.data \
    --lammps-mode kokkos            # or gpu / cpu

# OPLS electrostatics at matched accuracy (optionally replicated n x n x n)
python benchmarks/opls_bench.py --data data/OPLS_butane_x10/butane.data \
    --lammps lmp_kk --fastmd build/fastmd --workdir bench_opls [--replicate 2]
```

Binaries and data can also be given with `FASTMD_BINARY`, `LAMMPS_BINARY`
and `FASTMD_BENCH_DATA`.

## Project structure

```
fastMD/
├── benchmarks/     # LAMMPS comparison harnesses (bead-spring, OPLS electrostatics)
├── data/           # OPLS system generators and parameters
├── python/         # Python package (fastmd) and pybind11 binding
├── reference/      # CPU reference implementation used by tests
├── src/
│   ├── analysis/   # thermo, correlators, Rg, film plane, layer stress, dynamic correlations
│   ├── api/        # std-only C++ facade used by the Python binding
│   ├── core/       # particle data, topology, reductions, PBC, Morton order
│   ├── force/      # pair, Ewald/PME, bonded and table kernels
│   ├── integrate/  # Langevin/Andersen/NVE, Nosé-Hoover, FIRE
│   ├── io/         # config and LAMMPS data parsing, restart and trajectory output
│   ├── neighbor/   # device-maintained Verlet lists
│   └── sim/        # Simulation: setup, time loop, check points
├── tests/          # GoogleTest unit tests and Python API tests
└── tools/          # melt/film builders, push-off, LAMMPS validation and conversion scripts
```

## License

MIT
