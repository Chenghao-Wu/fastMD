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

`tools/make_kg_melt.py` writes an overlap-free starting melt, for example
`python tools/make_kg_melt.py --nchains 300 --chain-length 100 -o melt.data`.

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

Positions are wrapped into `[0, L)`, which is the data-file box shifted to
the origin. Unwrapped coordinates are `positions + images * box_length`.
`get_binary_path()` finds the command-line program.

## Configuration reference

One keyword per line; `#` starts a comment. Unknown keywords, missing or
extra arguments, out-of-range values and inconsistent combinations are
errors that name the line. Type indices are 0-based, so LAMMPS type
minus 1.

### System

| Key | Arguments | |
|-----|-----------|-|
| `lammps_data_file` | `path` | Coordinates, image flags, masses, velocities, charges, molecule ids and topology. Atom styles `atomic`, `charge`, `molecular`/`bond`/`angle` and `full` are supported; with no style comment the style is inferred from the column count. Cubic, orthogonal boxes only. |
| `natoms` | `N` | Optional with a data file; must match it. |
| `ntypes` | `N` | Number of atom types (≤ 16). |
| `rc` | `r` | Nonbonded cutoff. LJ is truncated, not shifted. |
| `skin` | `s` | Neighbor-list skin. The list is rebuilt exactly when a pair could have crossed `rc`. |
| `dt` | `dt` | Time step. |
| `nsteps` | `N` | Steps for the command-line run. Temperature and pressure ramps span these steps. |

### Ensemble (exactly one)

| Key | Arguments | |
|-----|-----------|-|
| `nvt_langevin` | `T_start T_stop Tdamp [seed]` | Langevin (BAOAB). The noise is counter-based (seed, particle, step), so trajectories don't depend on particle order. |
| `nvt_nh` | `T_start T_stop Tdamp [chain]` | Nosé–Hoover chain (default length 3). |
| `npt_nh` | `T_start T_stop Tdamp P_start P_stop Pdamp [chain]` | Nosé–Hoover thermostat and isotropic barostat. |

### Interactions

| Key | Arguments | |
|-----|-----------|-|
| `lj` | `ti tj eps sig` | LJ 12-6 for a type pair (symmetric). |
| `lj_type` | `t eps sig` | Per-type LJ with geometric mixing (OPLS). Not combinable with `lj`. |
| `lj_force_cap` | `F_max` | Softens LJ below the radius where \|F\| = F_max, keeping energy and force consistent. Use it for pushing apart overlapping starting structures. |
| `table` | `ti tj file keyword [force\|force_over_r]` | Tabulated pair potential (LAMMPS `pair_style table` file). Column 4 is F = −dE/dr by default; `force_over_r` reads legacy F/r tables. |
| `coulomb` | `ewald alpha kmax` · `ewald auto tol` · `pme alpha grid [order]` · `pme auto tol [order]` | Long-range Coulomb: real-space erfc within `rc` plus direct Ewald or smooth PME (B-spline order 3–8, default 5). `auto` picks α, kmax or grid for a target relative RMS force error `tol` (as OpenMM does). Needs charges (`Atoms # full` or `charge`). |
| `coulomb_const` | `C` | Coulomb constant; default 138.9354576 kJ·nm/(mol·e²). |
| `special_bonds_14` | `lj14 coul14` | 1-4 scaling (default 0.5 0.5). 1-2 and 1-3 pairs are excluded. |
| `special_dihedral` | `yes\|no` | Take 1-4 pairs from listed dihedrals (`yes`) instead of bond-graph distance 3 (default, as LAMMPS does by default). |
| `bond_type` | `t k R0 eps sig` | FENE bond with WCA repulsion. |
| `bond_harmonic` | `t K r0` | Harmonic bond, E = K (r − r0)² (LAMMPS convention). |
| `bond_table` | `t file keyword [force\|force_over_r]` | Tabulated bond (LAMMPS `bond_style table`). |
| `angle_type` | `t k theta0` | Harmonic angle, E = k (θ − θ0)², θ0 in degrees. |
| `angle_table` | `t file keyword [degrees\|radians]` | Tabulated angle (LAMMPS `angle_style table`: θ in degrees, force per degree; `radians` for legacy tables). |
| `dihedral_opls` | `t V1 V2 V3 V4` | OPLS dihedral (LAMMPS `dihedral_style opls`). |

Every bond, angle and dihedral type present in the data file must have
parameters.

### Output

| Key | Arguments | |
|-----|-----------|-|
| `thermo` | `on freq file` | Columns `step KE PE T Pxx Pyy Pzz Pxy Pxz Pyz`. Energies are per atom, and PE includes the Ewald self energy. |
| `rg` | `on freq file` | Mean radius of gyration over molecules, using unwrapped coordinates. |
| `stress` | `on freq file` | Multiple-tau autocorrelation of the off-diagonal and diagonal stress, sampled every `freq` steps (for viscosity). Columns: lag (steps), C_xx…C_zz, sample count. |
| `dump_freq` | `N` | Binary trajectory every N steps (0 = off). |
| `dump_file` | `path` | Trajectory file (default `traj.bin`). |
| `restart` | `freq file` | LAMMPS data files (`Atoms # full` + `Velocities` + topology) every `freq` steps as `file_<step>.ext`, and at the end as `file_final.ext`. `freq` 0 writes only the final file. |
| `forces_dump_file` | `path` | Write the initial per-atom forces and energies, then exit (validation). |

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

## Validation

Unit tests (`tests/`) check, among other things:

- forces against energy finite differences
- strain-derivative virials
- permutation invariance
- neighbor-list completeness under box deformation and overflow
- restart round trips
- correlator results against direct calculation
- PME against converged Ewald
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
│   ├── analysis/   # thermo, stress correlator, Rg
│   ├── api/        # std-only C++ facade used by the Python binding
│   ├── core/       # particle data, topology, reductions, PBC, Morton order
│   ├── force/      # pair, Ewald/PME, bonded and table kernels
│   ├── integrate/  # Langevin and Nosé-Hoover integrators
│   ├── io/         # config and LAMMPS data parsing, restart and trajectory output
│   ├── neighbor/   # device-maintained Verlet lists
│   └── sim/        # Simulation: setup, time loop, check points
├── tests/          # GoogleTest unit tests and Python API tests
└── tools/          # system builders, LAMMPS validation and conversion scripts
```

## License

MIT
