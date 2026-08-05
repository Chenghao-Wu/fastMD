# fastMD

Why this package?

We can accelerate MD simulation via models (multiscale simulations), why not accelerate it via optimizing kernels?

Here it is:

GPU-accelerated molecular dynamics simulator written in CUDA.

Compared with LAMMPS for 30K bead-spring polymer systems with Langevin thermostat, fastMD gains 20 times acceleration on RTX4090D (10,000 steps per second).

## Quick Start

**Prerequisites:** CUDA Toolkit (>= 11.0), CMake (>= 3.22), C++17 compiler.

```bash
git clone https://github.com/<user>/fastMD.git
cd fastMD
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Create a config file (e.g. `run.conf`):

```
natoms 30000
ntypes 1
rc 2.5
skin 0.3
dt 0.001
nsteps 5000
dump_freq 0
thermo 1 1000 thermo.dat
nvt_langevin 1.0 1.0 1.0 42
lj 0 0 1.0 1.0
bond_type 0 30.0 1.5 1.0 1.0
angle_type 0 5.0 180.0
lammps_data_file data/input_000_minimized.data
```

Run:

```bash
./fastMD ../run.conf
```

## Configuration Reference

### Simulation parameters

| Key | Arguments | Description |
|-----|-----------|-------------|
| `natoms` | `N` | Number of atoms in the system |
| `ntypes` | `N` | Number of atom types |
| `rc` | `float` | Non-bonded cutoff radius |
| `skin` | `float` | Neighbor list skin distance |
| `dt` | `float` | Timestep size |
| `nsteps` | `N` | Number of simulation steps |
| `dump_freq` | `N` | Trajectory dump frequency (0 to disable) |

### Ensemble

Choose exactly one of the following ensemble keys. All support linear temperature ramping from `T_start` to `T_stop` over the simulation.

| Key | Arguments | Description |
|-----|-----------|-------------|
| `nvt_langevin` | `T_start T_stop Tdamp [seed]` | Langevin thermostat (NVT). `Tdamp` is the characteristic damping time. `seed` defaults to 42. |
| `nvt_nh` | `T_start T_stop Tdamp [chain_length]` | Nosé-Hoover thermostat (NVT). `chain_length` defaults to 3. |
| `npt_nh` | `T_start T_stop Tdamp P_start P_stop Pdamp [chain_length]` | Nosé-Hoover thermostat + barostat (NPT). `P_start`/`P_stop` for pressure ramping, `Pdamp` for barostat damping. |

### Interaction types

**`lj <ti> <tj> <eps> <sig>`** — Lennard-Jones non-bonded interaction between atom types `ti` and `tj`. Both types are 0-indexed. Parameters are symmetric (applied to both `ti,tj` and `tj,ti`). Required (or use `lj_type`).

**`lj_type <t> <eps> <sig>`** — OPLS-style per-type LJ parameters. All type-pair parameters are built with **geometric mixing** (`eps_ij = sqrt(eps_i eps_j)`, `sig_ij = sqrt(sig_i sig_j)`). Cannot be combined with `lj`.

**`coulomb ewald <alpha> <kmax>`** — Ewald electrostatics: real-space `erfc(alpha r)` within the `rc` cutoff plus a reciprocal-space sum over the half-shell of integer k-vectors with `|n| <= kmax`. Requires per-atom charges from a LAMMPS data file with `Atoms # full`. `alpha` in 1/length of the simulation units (e.g. 1/nm when lengths are nm). The Coulomb constant defaults to `138.9354576` (kJ·nm/mol·e², HOOMD md units) and can be overridden with `coulomb_const <value>`.

**`special_bonds_14 <lj14> <coul14>`** — 1-4 pair scaling for OPLS (defaults `0.5 0.5`). 1-2/1-3 pairs are excluded automatically from bonds/angles when dihedrals are present or Ewald is on; unique 1-4 pairs are taken from dihedral end atoms and handled by a dedicated kernel (scaled LJ + scaled Coulomb minus the Ewald short-range term, matching LAMMPS `special_bonds 0 0 0.5` with `lj/cut/coul/long`).

**`bond_harmonic <t> <K> <r0>`** — Harmonic bond `E = K (r - r0)^2` (LAMMPS `bond_style harmonic` convention) for bond type `t` (0-indexed). Cannot be combined with FENE `bond_type`.

**`dihedral_opls <t> <V1> <V2> <V3> <V4>`** — OPLS proper dihedral for dihedral type `t` (0-indexed, read from the `Dihedrals` section of the data file): `E = V1/2 (1+cos φ) + V2/2 (1-cos 2φ) + V3/2 (1+cos 3φ) + V4/2 (1-cos 4φ)` (LAMMPS `dihedral_style opls` convention).

**`bond_type <t> <k> <R0> <eps> <sig>`** — FENE bond parameters for bond type `t`. `t` is 0-indexed (corresponds to LAMMPS bond type minus 1). Parameters:
- `k` — FENE spring constant
- `R0` — equilibrium bond distance
- `eps`, `sig` — WCA wall parameters (repulsive LJ between bonded atoms)

**`angle_type <t> <k> <theta0>`** — Harmonic angle bending parameters for angle type `t`. `t` is 0-indexed (LAMMPS angle type minus 1). Parameters:
- `k` — angle force constant
- `theta0` — equilibrium angle in degrees

**`lammps_data_file <path>`** — Path to LAMMPS data file containing atom coordinates, bonds, angles, and (for OPLS) per-atom charges and dihedrals. Use `Atoms # full` style so charges are read.

To run without angle interactions, simply omit all `angle_type` lines and ensure your LAMMPS data file has no `Angles` section.

### Units for OPLS systems

The OPLS path is unit-agnostic but is validated in HOOMD **md units**: length nm, energy kJ/mol, time ps, temperature kJ/mol (divide by kB = 0.00831446 for Kelvin), charge e. The validation tool `tools/opls_force_check.py` converts LAMMPS real units (kcal/mol, Å) accordingly. Bond force constants convert as `K[kJ/mol/nm²] = 100 × 4.184 × K[kcal/mol/Å²]`.

### Analysis output

**`thermo <on> <freq> <file>`** — Thermodynamic output. Writes CSV with columns: `step, ke, pe, etotal, T, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz` at the given frequency. Set `<on>` to 1 to enable, 0 to disable.

**`rg <on> <freq> <file>`** — Radius of gyration. Computes ensemble-averaged Rg per chain using molecule IDs from the LAMMPS data file. Set `<on>` to 1 to enable, 0 to disable.

**`stress <on> <freq> <file>`** — Stress autocorrelation. Uses a multiple-tau correlator to compute the stress autocorrelation function for viscosity analysis. Requires a LAMMPS data file with a `Velocities` section.

## Features

- Lennard-Jones non-bonded potential
- **OPLS force field**: per-atom charges, Ewald electrostatics (real + reciprocal + self),
  harmonic bonds, harmonic angles, OPLS dihedrals, 1-2/1-3 exclusions and scaled 1-4 pairs
  (validated against LAMMPS `lj/cut/coul/long` + `dihedral_style opls` on liquid butane and methanol)
- FENE bonded potential
- Harmonic angle bending potential
- Langevin thermostat (NVT) with temperature damping and ramping
- Nosé-Hoover thermostat (NVT/NPT) with chain integration and barostat
- Verlet neighbor lists with automatic rebuild
- Morton spatial sorting for cache efficiency
- Stress autocorrelation analysis
- LAMMPS data file input
- Binary trajectory output

## Project Structure

```
fastMD/
├── benchmarks/     # benchmark runner and input generator
├── data/           # input data files
├── docs/           # design docs and plans
├── reference/      # CPU reference implementation
├── src/
│   ├── analysis/   # thermodynamics, correlators
│   ├── core/       # system state, PBC, Morton sorting
│   ├── force/      # LJ, FENE, angle force kernels
│   ├── integrate/  # Langevin and Nosé-Hoover integrators
│   ├── io/         # config parsing, LAMMPS parser, binary dump
│   └── neighbor/   # Verlet and tile neighbor lists
├── tests/          # unit tests
├── CMakeLists.txt
└── LICENSE
```

## Benchmarking

The `benchmarks/` directory contains a Python runner that compares fastMD output against LAMMPS for validation and timing:

```bash
cd benchmarks
python run.py
```

## OPLS validation

End-to-end validation against LAMMPS (`pair_style lj/cut/coul/long`, `kspace_style ewald`, `special_bonds 0 0 0.5`, `dihedral_style opls`) is provided for two liquid systems (RDKit + packmol generated):

```bash
# static run-0 force + energy comparison
python tools/opls_force_check.py --data data/OPLS_butane/butane.data --workdir data/OPLS_butane
python tools/opls_force_check.py --data data/OPLS_methanol/methanol.data --workdir data/OPLS_methanol

# 5 ps NVT dynamics comparison
python tools/opls_nvt_check.py --data data/OPLS_butane/butane.data --workdir data/OPLS_butane
python tools/opls_nvt_check.py --data data/OPLS_methanol/methanol.data --workdir data/OPLS_methanol
```

Current status (RTX4090D, fp32 fastMD vs fp64 LAMMPS):

| System | max per-atom ΔF | PE diff | NVT T / PE | speedup vs LAMMPS (1 core) |
|--------|-----------------|---------|------------|-----------------------------|
| butane (896 atoms)   | 0.31 kJ/mol/nm | 0.05%  | 0.9% / 1.2%  | ~19× |
| methanol (750 atoms) | 0.39 kJ/mol/nm | 0.009% | 0.7% / ~2%   | ~12× |

## License

MIT
