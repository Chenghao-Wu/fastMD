# fastMD Repo Organization & README

## Goal

Prepare the fastMD package for GitHub hosting: rename from ultimateCGMD to fastMD, clean up repo hygiene, add license and README, organize input data.

## Approach

Approach A — keep existing `src/` structure, add missing pieces (license, README, gitignore, data directory).

## Changes

### 1. .gitignore

Expand from only ignoring `build/` and `.worktrees/` to cover all runtime outputs:

- Python cache (`__pycache__/`, `*.pyc`)
- Profiling output (`*.nsys-rep`, `*.sqlite`)
- Simulation output (`*.data`, `log.*`, `traj.bin`, `stress_acf.dat`, `benchmark_report.json`)
- Generated configs (`fastmd_benchmark.conf`, `lammps_benchmark.in`, `profile_fastmd.conf`)
- CTest (`Testing/`)

### 2. CMake rename

- `project(ultimateCGMD ...)` → `project(fastMD ...)`
- `add_executable(ultimateCGMD ...)` → `add_executable(fastMD ...)`
- Internal library target `core` stays unchanged

### 3. Directory organization

- Add `data/` directory, move `input_000.data` and `input_000_minimized.data` there
- Root stays: `src/`, `tests/`, `benchmarks/`, `docs/`, `reference/`
- Remove tracked artifacts no longer needed at root

### 4. Add LICENSE (MIT)

### 5. README.md

Concise, five sections:

- **Title** — GPU-accelerated molecular dynamics simulator in CUDA
- **Quick start** — prerequisites (CUDA, CMake, C++17), build, run
- **Features** — LJ, FENE, Angle, Langevin, Verlet lists, Morton sorting, stress ACF, LAMMPS I/O
- **Project structure** — directory tree
- **Benchmarking** — note about LAMMPS comparison
- **License** — MIT
