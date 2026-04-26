# fastMD

GPU-accelerated molecular dynamics simulator written in CUDA.

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
box_L 34.995
ntypes 1
rc 2.5
skin 0.3
dt 0.001
temperature 1.0
gamma 1.0
nsteps 5000
dump_freq 0
thermo_freq 1000
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
