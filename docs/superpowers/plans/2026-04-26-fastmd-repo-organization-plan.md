# fastMD Repo Organization & README — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare fastMD for GitHub hosting — rename from ultimateCGMD, add .gitignore/README/LICENSE, organize data.

**Architecture:** Keep existing `src/` structure intact. Add missing repo-standard files at root. Move large data files to `data/`. Remove tracked runtime artifacts.

**Tech Stack:** C++/CUDA, CMake

---

### Task 1: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Replace .gitignore with expanded patterns**

```gitignore
build/
.worktrees/

# Python
__pycache__/
*.pyc

# Profiling
*.nsys-rep
*.sqlite

# Simulation output
*.data
log.*
traj.bin
stress_acf.dat
benchmark_report.json

# Generated configs
fastmd_benchmark.conf
lammps_benchmark.in
profile_fastmd.conf

# CTest
Testing/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: expand .gitignore to cover runtime artifacts
" -m "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Rename package in CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt:1,35`

- [ ] **Step 1: Rename project and executable**

In `CMakeLists.txt`, change line 2 from:
```cmake
project(ultimateCGMD LANGUAGES CXX CUDA)
```
to:
```cmake
project(fastMD LANGUAGES CXX CUDA)
```

Change line 35 from:
```cmake
add_executable(ultimateCGMD src/main.cu src/io/config.cpp)
```
to:
```cmake
add_executable(fastMD src/main.cu src/io/config.cpp)
```

- [ ] **Step 2: Verify build still works**

```bash
cd build && cmake .. && make -j$(nproc)
```

Expected: build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "chore: rename project from ultimateCGMD to fastMD
" -m "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Create data directory and move input files

**Files:**
- Create: `data/` (directory)
- Move: `input_000.data` → `data/input_000.data`
- Move: `input_000_minimized.data` → `data/input_000_minimized.data`

- [ ] **Step 1: Find and update references to input data files**

```bash
grep -rn "input_000" . --include="*.py" --include="*.cpp" --include="*.cu" --include="*.cuh" --include="*.hpp" --include="*.conf" --include="*.json" 2>/dev/null
```

Update any references. Expected hits: `profile_fastmd.conf`, `benchmarks/benchmark.json`.

In `profile_fastmd.conf`, change `input_000_minimized.data` to `data/input_000_minimized.data`.
In `benchmarks/benchmark.json`, change `input_000_minimized.data` to `data/input_000_minimized.data`.

- [ ] **Step 2: Create data directory and move files**

```bash
mkdir -p data
mv input_000.data data/input_000.data
mv input_000_minimized.data data/input_000_minimized.data
```

- [ ] **Step 3: Commit**

```bash
git add data/
git commit -m "chore: move input data files to data/ directory
" -m "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Add MIT LICENSE

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Write LICENSE**

```
MIT License

Copyright (c) 2026 fastMD contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license
" -m "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Write README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

````markdown
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
./fastMD ../data/profile_fastmd.conf
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
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README
" -m "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Verify final state

- [ ] **Step 1: Check git status is clean**

```bash
git status
```

Expected: no modified or untracked files.

- [ ] **Step 2: Verify build from clean**

```bash
cd build && cmake .. && make -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 3: List top-level directory for visual check**

```bash
ls -la
```

Expected: no `.data`, `.sqlite`, `.nsys-rep`, log, or bin files in root.
