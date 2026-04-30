# fastMD

Why this package?

We can accelerate MD simulation via models (multiscale simulations), why not accelerate it via optimizing kernels?

Here it is:

GPU-accelerated molecular dynamics simulator written in CUDA.

Compared with LAMMPS for 30K bead-spring polymer systems, fastMD gains 20 times acceleration on RTX4090D.

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
temperature 1.0
gamma 1.0
nsteps 5000
dump_freq 0
thermo 1 1000 thermo.dat
seed 42
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
| `temperature` | `float` | Target thermostat temperature |
| `gamma` | `float` | Langevin friction coefficient |
| `nsteps` | `N` | Number of simulation steps |
| `dump_freq` | `N` | Trajectory dump frequency (0 to disable) |
| `seed` | `N` | Random number generator seed |

### Interaction types

**`lj <ti> <tj> <eps> <sig>`** — Lennard-Jones non-bonded interaction between atom types `ti` and `tj`. Both types are 0-indexed. Parameters are symmetric (applied to both `ti,tj` and `tj,ti`). Required.

**`bond_type <t> <k> <R0> <eps> <sig>`** — FENE bond parameters for bond type `t`. `t` is 0-indexed (corresponds to LAMMPS bond type minus 1). Parameters:
- `t` — bond type index (0 = LAMMPS bond type 1, 1 = LAMMPS bond type 2, etc.)
- `k` — FENE spring constant
- `R0` — equilibrium bond distance
- `eps`, `sig` — WCA wall parameters (repulsive LJ between bonded atoms)

**`angle_type <t> <k> <theta0>`** — Harmonic angle bending parameters for angle type `t`. `t` is 0-indexed (LAMMPS angle type minus 1). Parameters:
- `k` — angle force constant
- `theta0` — equilibrium angle in degrees

**`lammps_data_file <path>`** — Path to LAMMPS data file containing atom coordinates, bonds, and angles.

To run without angle interactions, simply omit all `angle_type` lines and ensure your LAMMPS data file has no `Angles` section.

### Analysis output

**`thermo <on> <freq> <file>`** — Thermodynamic output. Writes CSV with columns: `step, ke, pe, etotal, T, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz` at the given frequency. Set `<on>` to 1 to enable, 0 to disable.

**`rg <on> <freq> <file>`** — Radius of gyration. Computes ensemble-averaged Rg per chain using molecule IDs from the LAMMPS data file. Set `<on>` to 1 to enable, 0 to disable.

**`stress <on> <freq> <file>`** — Stress autocorrelation. Uses a multiple-tau correlator to compute the stress autocorrelation function for viscosity analysis. Requires a LAMMPS data file with a `Velocities` section.

## Features

- Lennard-Jones non-bonded potential
- FENE bonded potential
- Harmonic angle bending potential
- Langevin thermostat
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
│   ├── integrate/  # Langevin integrator
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

## License

MIT
