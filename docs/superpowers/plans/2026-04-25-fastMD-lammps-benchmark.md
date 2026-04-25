# fastMD vs LAMMPS Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend fastMD to load bonds/angles from LAMMPS data files, then build a reusable Python benchmark harness that compares fastMD and LAMMPS speed on `input_000.data` (1M steps, NVT Langevin) while validating physics agreement.

**Architecture:** A C++ LAMMPS data parser populates fastMD's `TopologyData`, enabling bonded simulations. A Python harness generates matching input decks for both codes, runs a 1k-step validation comparing average temperature and potential energy, then runs a 1M-step timed benchmark and emits a JSON report.

**Tech Stack:** CUDA C++17, CMake, Python 3, LAMMPS (GPU package), GoogleTest

---

## File Structure

| File | Role |
|------|------|
| `src/io/lammps_data.hpp` | Parser declaration |
| `src/io/lammps_data.cpp` | Parses LAMMPS `Atoms`, `Bonds`, `Angles` sections into `TopologyData` |
| `src/io/config.hpp` | Extended with `bond_params`, `angle_params`, `data_file` fields |
| `src/io/config.cpp` | Populates new `TopologyData` fields from config keys |
| `src/main.cu` | Calls parser, uploads bond/angle arrays to device |
| `CMakeLists.txt` | Adds `lammps_data.cpp` to `core` library |
| `tests/test_lammps_data.cu` | Unit test for the parser |
| `tests/fixtures/mini.data` | Tiny 4-atom LAMMPS data file for parser tests |
| `tests/CMakeLists.txt` | Registers parser test |
| `benchmarks/benchmark.json` | Shared benchmark parameters |
| `benchmarks/generate_lammps_input.py` | Emits fastMD `.conf` and LAMMPS input script |
| `benchmarks/run.py` | Orchestrates validation + benchmark phases |
| `benchmarks/report.py` | Writes `benchmark_report.json` |

---

### Task 1: Extend TopologyData and parse_config for bond/angle params

fastMD's config parser reads `bond_type` and `angle_type` keys but currently discards them. We need to capture them so `main.cu` can upload them to the GPU.

**Files:**
- Modify: `src/io/config.hpp`
- Modify: `src/io/config.cpp`

- [ ] **Step 1: Add fields to TopologyData and bond/angle param vectors**

Edit `src/io/config.hpp`:
```cpp
#pragma once
#include "../core/types.cuh"
#include "../force/fene.cuh"
#include "../force/angle.cuh"
#include <string>
#include <vector>

struct TopologyData {
    std::vector<float4> positions;
    std::vector<float4> velocities;
    std::vector<int2>   bonds;
    std::vector<int>    bond_types;
    std::vector<int4>   angles;
    std::vector<float2> lj_params;
    std::vector<FENEParams> bond_params;
    std::vector<AngleParams> angle_params;
    std::string data_file;
};

SimParams parse_config(const std::string& filename, TopologyData& topo);
```

- [ ] **Step 2: Populate the new fields in parse_config**

Edit `src/io/config.cpp`. After the existing `bond_types_params` and `angle_types_params` loops, add at the end of the function (before the `return params;`):
```cpp
    topo.bond_params.resize(bond_types_params.size());
    for (size_t i = 0; i < bond_types_params.size(); i++) {
        topo.bond_params[i] = {bond_types_params[i].k,
                               bond_types_params[i].R0,
                               bond_types_params[i].eps,
                               bond_types_params[i].sig};
    }
    topo.angle_params.resize(angle_types_params.size());
    for (size_t i = 0; i < angle_types_params.size(); i++) {
        // fastMD angle kernel expects theta0 in radians
        float theta0_rad = angle_types_params[i].theta0 * 3.14159265358979323846f / 180.0f;
        topo.angle_params[i] = {angle_types_params[i].k_theta, theta0_rad};
    }
```

Also add the new config key parser inside the while loop:
```cpp
        else if (key == "lammps_data_file") iss >> topo.data_file;
```

- [ ] **Step 3: Build to check compilation**

Run:
```bash
cd /home/zhenghaowu/fastMD
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/io/config.hpp src/io/config.cpp
git commit -m "feat: capture bond/angle params and lammps_data_file in TopologyData"
```

---

### Task 2: Implement LAMMPS data file parser

**Files:**
- Create: `src/io/lammps_data.hpp`
- Create: `src/io/lammps_data.cpp`

- [ ] **Step 1: Write header**

Create `src/io/lammps_data.hpp`:
```cpp
#pragma once
#include "config.hpp"
#include <string>

void parse_lammps_data(const std::string& path, TopologyData& topo);
```

- [ ] **Step 2: Write implementation**

Create `src/io/lammps_data.cpp`:
```cpp
#include "lammps_data.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

void parse_lammps_data(const std::string& path, TopologyData& topo) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open LAMMPS data file: " + path);

    enum Section { NONE, ATOMS, BONDS, ANGLES };
    Section section = NONE;
    std::string line;

    while (std::getline(in, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        std::string trimmed = line.substr(start);

        if (trimmed == "Atoms")          { section = ATOMS;  continue; }
        if (trimmed == "Bonds")          { section = BONDS;  continue; }
        if (trimmed == "Angles")         { section = ANGLES; continue; }
        if (trimmed.empty() || trimmed[0] == '#') continue;

        // Stop parsing if we hit another header-style capitalized keyword
        bool is_new_section = false;
        if (std::isupper(trimmed[0])) {
            // Check if it's a known LAMMPS section header
            if (trimmed == "Velocities" || trimmed == "Dihedrals" ||
                trimmed == "Impropers"  || trimmed == "Masses" ||
                trimmed == "Pair Coeffs" || trimmed == "Bond Coeffs" ||
                trimmed == "Angle Coeffs")
                is_new_section = true;
        }
        if (is_new_section) { section = NONE; continue; }

        std::istringstream iss(trimmed);
        if (section == ATOMS) {
            int id, mol, type;
            float x, y, z;
            iss >> id >> mol >> type >> x >> y >> z;
            if (iss.fail()) continue;
            topo.positions.push_back(make_float4(x, y, z, pack_type_id(type - 1)));
        } else if (section == BONDS) {
            int id, type, a1, a2;
            iss >> id >> type >> a1 >> a2;
            if (iss.fail()) continue;
            topo.bonds.push_back(make_int2(a1 - 1, a2 - 1));
            topo.bond_types.push_back(type - 1);
        } else if (section == ANGLES) {
            int id, type, a1, a2, a3;
            iss >> id >> type >> a1 >> a2 >> a3;
            if (iss.fail()) continue;
            topo.angles.push_back(make_int4(a1 - 1, a2 - 1, a3 - 1, type - 1));
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add src/io/lammps_data.hpp src/io/lammps_data.cpp
git commit -m "feat: LAMMPS data file parser for atoms, bonds, and angles"
```

---

### Task 3: Wire parser into main.cu and build system

**Files:**
- Modify: `src/main.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Update CMakeLists.txt to include parser in core library**

Edit `CMakeLists.txt`, add `src/io/lammps_data.cpp` to the `core` source list:
```cmake
add_library(core STATIC
    src/core/system.cu
    src/core/morton.cu
    src/neighbor/tile_list.cu
    src/force/lj.cu
    src/force/fene.cu
    src/force/angle.cu
    src/integrate/langevin.cu
    src/analysis/thermo.cu
    src/analysis/correlator.cu
    src/io/dump.cu
    src/io/lammps_data.cpp
)
```

- [ ] **Step 2: Add parser call and device upload to main.cu**

Edit `src/main.cu`. After the existing `#include "io/config.hpp"`, add:
```cpp
#include "io/lammps_data.hpp"
```

Then, after the line `SimParams params = parse_config(argv[1], topo);`, add:
```cpp
    if (!topo.data_file.empty()) {
        parse_lammps_data(topo.data_file, topo);
        printf("Loaded %zu bonds, %zu angles from %s\n",
               topo.bonds.size(), topo.angles.size(), topo.data_file.c_str());
    }
```

After the existing `sys.allocate(params);`, add:
```cpp
    if (topo.bonds.size() > 0) {
        sys.nbonds = static_cast<int>(topo.bonds.size());
        CUDA_CHECK(cudaMalloc(&sys.bonds, sys.nbonds * sizeof(int2)));
        CUDA_CHECK(cudaMalloc(&sys.bond_param_idx, sys.nbonds * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(sys.bonds, topo.bonds.data(),
                              sys.nbonds * sizeof(int2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sys.bond_param_idx, topo.bond_types.data(),
                              sys.nbonds * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (topo.angles.size() > 0) {
        sys.nangles = static_cast<int>(topo.angles.size());
        CUDA_CHECK(cudaMalloc(&sys.angles, sys.nangles * sizeof(int4)));
        CUDA_CHECK(cudaMemcpy(sys.angles, topo.angles.data(),
                              sys.nangles * sizeof(int4), cudaMemcpyHostToDevice));
    }
```

After the existing `CUDA_CHECK(cudaMemcpy(sys.lj_params, ...));`, add:
```cpp
    if (topo.bond_params.size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_fene_params,
                              topo.bond_params.size() * sizeof(FENEParams)));
        CUDA_CHECK(cudaMemcpy(d_fene_params, topo.bond_params.data(),
                              topo.bond_params.size() * sizeof(FENEParams),
                              cudaMemcpyHostToDevice));
    }
    if (topo.angle_params.size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_angle_params,
                              topo.angle_params.size() * sizeof(AngleParams)));
        CUDA_CHECK(cudaMemcpy(d_angle_params, topo.angle_params.data(),
                              topo.angle_params.size() * sizeof(AngleParams),
                              cudaMemcpyHostToDevice));
    }
```

- [ ] **Step 3: Build and smoke-test on the full system**

Run:
```bash
cd /home/zhenghaowu/fastMD/build
make -j$(nproc)
```

Create a temporary config `test_bonds.conf`:
```
natoms 30000
box_L 34.99514024
ntypes 1
rc 2.5
skin 0.3
dt 0.001
temperature 1.0
gamma 1.0
nsteps 10
dump_freq 0
thermo_freq 1
seed 42
lj 0 0 1.0 1.0
bond_type 0 30.0 1.5 1.0 1.0
angle_type 0 5.0 180.0
lammps_data_file /home/zhenghaowu/fastMD/input_000.data
```

Run:
```bash
./ultimateCGMD test_bonds.conf
```
Expected output includes:
```
Loaded 29700 bonds, 29400 angles from /home/zhenghaowu/fastMD/input_000.data
```
and completes 10 steps without crash.

- [ ] **Step 4: Commit**

```bash
rm -f test_bonds.conf
git add src/main.cu CMakeLists.txt
git commit -m "feat: wire LAMMPS data parser into main simulation loop"
```

---

### Task 4: Unit test for LAMMPS data parser

**Files:**
- Create: `tests/fixtures/mini.data`
- Create: `tests/test_lammps_data.cu`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create mini LAMMPS data fixture**

Create `tests/fixtures/mini.data`:
```
LAMMPS data file for testing

4 atoms
2 bonds
1 angles

1 atom types
1 bond types
1 angle types

0 10.0 xlo xhi
0 10.0 ylo yhi
0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 1 1.0 2.0 3.0
2 1 1 2.0 3.0 4.0
3 1 1 3.0 4.0 5.0
4 1 1 4.0 5.0 6.0

Bonds

1 1 1 2
2 1 2 3

Angles

1 1 1 2 3
```

- [ ] **Step 2: Write test**

Create `tests/test_lammps_data.cu`:
```cpp
#include <gtest/gtest.h>
#include "io/config.hpp"
#include "io/lammps_data.hpp"

TEST(LammpsDataParser, MiniFile) {
    TopologyData topo;
    parse_lammps_data("tests/fixtures/mini.data", topo);

    ASSERT_EQ(topo.positions.size(), 4u);
    EXPECT_FLOAT_EQ(topo.positions[0].x, 1.0f);
    EXPECT_FLOAT_EQ(topo.positions[3].z, 6.0f);
    EXPECT_EQ(unpack_type_id(topo.positions[0].w), 0);

    ASSERT_EQ(topo.bonds.size(), 2u);
    EXPECT_EQ(topo.bonds[0].x, 0);
    EXPECT_EQ(topo.bonds[0].y, 1);
    EXPECT_EQ(topo.bonds[1].x, 1);
    EXPECT_EQ(topo.bonds[1].y, 2);
    ASSERT_EQ(topo.bond_types.size(), 2u);
    EXPECT_EQ(topo.bond_types[0], 0);
    EXPECT_EQ(topo.bond_types[1], 0);

    ASSERT_EQ(topo.angles.size(), 1u);
    EXPECT_EQ(topo.angles[0].x, 0);
    EXPECT_EQ(topo.angles[0].y, 1);
    EXPECT_EQ(topo.angles[0].z, 2);
    EXPECT_EQ(topo.angles[0].w, 0);
}
```

- [ ] **Step 3: Register test in CMake**

Edit `tests/CMakeLists.txt`, add at the bottom:
```cmake
add_cuda_test(test_lammps_data)
target_sources(test_lammps_data PRIVATE ${CMAKE_SOURCE_DIR}/src/io/lammps_data.cpp)
```

- [ ] **Step 4: Run test**

```bash
cd /home/zhenghaowu/fastMD/build
make -j$(nproc) test_lammps_data
./tests/test_lammps_data
```
Expected: `[  PASSED  ] 1 test.`

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/mini.data tests/test_lammps_data.cu tests/CMakeLists.txt
git commit -m "test: unit test for LAMMPS data parser"
```

---

### Task 5: Python benchmark input generator

**Files:**
- Create: `benchmarks/generate_lammps_input.py`
- Create: `benchmarks/benchmark.json`

- [ ] **Step 1: Write generate_lammps_input.py**

Create `benchmarks/generate_lammps_input.py`:
```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_fastmd_config(cfg: dict, out_path: str, nsteps: int, thermo_freq: int):
    lines = [
        f"natoms {cfg['natoms']}",
        f"box_L {cfg['box_L']}",
        f"ntypes {cfg['ntypes']}",
        f"rc {cfg['rc']}",
        f"skin {cfg['skin']}",
        f"dt {cfg['dt']}",
        f"temperature {cfg['temperature']}",
        f"gamma {cfg['gamma']}",
        f"nsteps {nsteps}",
        f"dump_freq {cfg['dump_freq']}",
        f"thermo_freq {thermo_freq}",
        f"seed {cfg['seed']}",
    ]
    for ti, tj, eps, sig in cfg["lj_params"]:
        lines.append(f"lj {ti} {tj} {eps} {sig}")
    for t, k, R0, eps, sig in cfg["bond_params"]:
        lines.append(f"bond_type {t} {k} {R0} {eps} {sig}")
    for t, k_theta, theta0_deg in cfg["angle_params"]:
        # fastMD config expects theta0 in degrees; parse_config converts to radians
        lines.append(f"angle_type {t} {k_theta} {theta0_deg}")
    lines.append(f"lammps_data_file {cfg['data_file']}")

    Path(out_path).write_text("\n".join(lines) + "\n")


def write_lammps_input(cfg: dict, out_path: str, nsteps: int, thermo_freq: int):
    lj = cfg["lj_params"][0]
    bond = cfg["bond_params"][0]
    angle = cfg["angle_params"][0]
    neighbor_dist = cfg["rc"] + cfg["skin"]

    lines = [
        "units lj",
        "atom_style molecular",
        "newton off",
        f"read_data {cfg['data_file']}",
        "",
        "package gpu 1",
        "suffix gpu",
        "",
        f"pair_style lj/cut {cfg['rc']}",
        f"pair_coeff {lj[0]+1} {lj[1]+1} {lj[2]} {lj[3]}",
        "",
        "bond_style fene",
        f"bond_coeff {bond[0]+1} {bond[1]} {bond[2]}",
        "",
        "angle_style harmonic",
        f"angle_coeff {angle[0]+1} {angle[1]} {angle[2]}",
        "",
        f"neighbor {neighbor_dist} bin",
        "neigh_modify delay 0 every 1 check yes",
        "",
        f"timestep {cfg['dt']}",
        f"fix 1 all nve",
        f"fix 2 all langevin {cfg['temperature']} {cfg['temperature']} {1.0/cfg['gamma']} {cfg['seed']}",
        "",
        f"thermo {thermo_freq}",
        "thermo_style custom step temp pe ke etotal",
        "",
        f"run {nsteps}",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_lammps_input.py <benchmark.json>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    phase = sys.argv[2] if len(sys.argv) > 2 else "benchmark"

    if phase == "validation":
        nsteps = cfg["nsteps_validation"]
        thermo_freq = cfg["thermo_freq_validation"]
    else:
        nsteps = cfg["nsteps_benchmark"]
        thermo_freq = cfg["thermo_freq_benchmark"]

    write_fastmd_config(cfg, "fastmd_benchmark.conf", nsteps, thermo_freq)
    write_lammps_input(cfg, "lammps_benchmark.in", nsteps, thermo_freq)
    print("Generated fastmd_benchmark.conf and lammps_benchmark.in")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write benchmark.json**

Create `benchmarks/benchmark.json`:
```json
{
  "fastmd_binary": "/home/zhenghaowu/fastMD/build/ultimateCGMD",
  "lammps_binary": "/home/zhenghaowu/lammps/build/lmp",
  "data_file": "/home/zhenghaowu/fastMD/input_000.data",
  "natoms": 30000,
  "box_L": 34.99514024,
  "ntypes": 1,
  "temperature": 1.0,
  "dt": 0.001,
  "rc": 2.5,
  "skin": 0.3,
  "gamma": 1.0,
  "seed": 42,
  "nsteps_validation": 1000,
  "nsteps_benchmark": 1000000,
  "thermo_freq_validation": 100,
  "thermo_freq_benchmark": 1000,
  "dump_freq": 0,
  "lj_params": [[0, 0, 1.0, 1.0]],
  "bond_params": [[0, 30.0, 1.5, 1.0, 1.0]],
  "angle_params": [[0, 5.0, 180.0]]
}
```

- [ ] **Step 3: Smoke-test generator**

```bash
cd /home/zhenghaowu/fastMD/benchmarks
python3 generate_lammps_input.py benchmark.json validation
cat fastmd_benchmark.conf
cat lammps_benchmark.in
```
Expected: both files exist and contain sensible values for 1000 steps.

```bash
python3 generate_lammps_input.py benchmark.json benchmark
cat fastmd_benchmark.conf | grep nsteps
cat lammps_benchmark.in | grep "^run"
```
Expected: `nsteps 1000000` and `run 1000000`.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/generate_lammps_input.py benchmarks/benchmark.json
git commit -m "feat: Python benchmark input generator"
```

---

### Task 6: Python benchmark runner

**Files:**
- Create: `benchmarks/run.py`

- [ ] **Step 1: Write run.py**

Create `benchmarks/run.py`:
```python
#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path

from report import write_report


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_fastmd(cfg: dict, phase: str) -> tuple:
    subprocess.run(
        [sys.executable, "benchmarks/generate_lammps_input.py",
         "benchmarks/benchmark.json", phase],
        check=True,
        cwd="/home/zhenghaowu/fastMD",
    )
    cmd = [cfg["fastmd_binary"], "fastmd_benchmark.conf"]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_time = time.perf_counter() - start
    return wall_time, result.stdout


def run_lammps(cfg: dict, phase: str) -> tuple:
    subprocess.run(
        [sys.executable, "benchmarks/generate_lammps_input.py",
         "benchmarks/benchmark.json", phase],
        check=True,
        cwd="/home/zhenghaowu/fastMD",
    )
    cmd = [cfg["lammps_binary"], "-in", "lammps_benchmark.in", "-log", "log.lammps"]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_time = time.perf_counter() - start
    return wall_time, Path("log.lammps").read_text()


def parse_fastmd_thermo(stdout: str):
    temps = []
    pes = []
    for line in stdout.splitlines():
        if line.startswith("Step "):
            parts = line.split()
            # Format: Step N: T=... KE=... PE=... Pxx=...
            for i, p in enumerate(parts):
                if p.startswith("T="):
                    temps.append(float(p[2:]))
                elif p.startswith("PE="):
                    pes.append(float(p[3:]))
    return temps, pes


def parse_lammps_thermo(log: str):
    temps = []
    pes = []
    in_data = False
    for line in log.splitlines():
        if line.startswith("Step"):
            in_data = True
            continue
        if in_data:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    temps.append(float(parts[1]))
                    pes.append(float(parts[2]))
                except ValueError:
                    in_data = False
    return temps, pes


def validate(temps_f, pes_f, temps_l, pes_l, cfg: dict):
    n = cfg.get("validation_window", 500)
    if len(temps_f) < n or len(temps_l) < n:
        raise RuntimeError(f"Not enough thermo outputs: fastMD={len(temps_f)}, LAMMPS={len(temps_l)}")

    avg_t_f = sum(temps_f[-n:]) / n
    avg_t_l = sum(temps_l[-n:]) / n
    avg_pe_f = sum(pes_f[-n:]) / n
    avg_pe_l = sum(pes_l[-n:]) / n

    rel_temp_diff = abs(avg_t_f - avg_t_l) / cfg["temperature"]
    rel_pe_diff = abs(avg_pe_f - avg_pe_l) / abs(avg_pe_l) if avg_pe_l != 0 else abs(avg_pe_f - avg_pe_l)

    print(f"Validation averages (last {n} steps):")
    print(f"  T  fastMD={avg_t_f:.4f} LAMMPS={avg_t_l:.4f} rel_diff={rel_temp_diff:.4f}")
    print(f"  PE fastMD={avg_pe_f:.4f} LAMMPS={avg_pe_l:.4f} rel_diff={rel_pe_diff:.4f}")

    passed = rel_temp_diff < 0.05 and rel_pe_diff < 1e-2
    return passed, rel_temp_diff, rel_pe_diff


def main():
    cfg = load_config("benchmarks/benchmark.json")

    # Phase 1: Validation
    print("=== Validation phase (1k steps) ===")
    t_f, out_f = run_fastmd(cfg, "validation")
    t_l, out_l = run_lammps(cfg, "validation")

    temps_f, pes_f = parse_fastmd_thermo(out_f)
    temps_l, pes_l = parse_lammps_thermo(out_l)

    passed, rel_temp_diff, rel_pe_diff = validate(temps_f, pes_f, temps_l, pes_l, cfg)
    if not passed:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("Validation passed.")

    # Phase 2: Benchmark
    print("\n=== Benchmark phase (1M steps) ===")
    t_f, _ = run_fastmd(cfg, "benchmark")
    print(f"fastMD wall time: {t_f:.2f} s")
    t_l, _ = run_lammps(cfg, "benchmark")
    print(f"LAMMPS wall time: {t_l:.2f} s")

    ns_per_day_f = (cfg["nsteps_benchmark"] * cfg["dt"]) / (t_f / 86400.0)
    ns_per_day_l = (cfg["nsteps_benchmark"] * cfg["dt"]) / (t_l / 86400.0)
    speedup = t_l / t_f

    print(f"fastMD: {ns_per_day_f:.2f} ns/day")
    print(f"LAMMPS: {ns_per_day_l:.2f} ns/day")
    print(f"Speedup: {speedup:.2f}x")

    write_report(
        cfg=cfg,
        fastmd_time=t_f,
        lammps_time=t_l,
        ns_per_day_f=ns_per_day_f,
        ns_per_day_l=ns_per_day_l,
        speedup=speedup,
        validation={"passed": passed, "rel_temp_diff": rel_temp_diff, "rel_pe_diff": rel_pe_diff},
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/run.py
git commit -m "feat: Python benchmark runner with validation and timing"
```

---

### Task 7: Python benchmark reporter

**Files:**
- Create: `benchmarks/report.py`

- [ ] **Step 1: Write report.py**

Create `benchmarks/report.py`:
```python
#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def write_report(
    cfg: dict,
    fastmd_time: float,
    lammps_time: float,
    ns_per_day_f: float,
    ns_per_day_l: float,
    speedup: float,
    validation: dict,
    out_path: str = "benchmark_report.json",
):
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": get_gpu_name(),
        "system": cfg["data_file"],
        "n_atoms": cfg["natoms"],
        "nsteps": cfg["nsteps_benchmark"],
        "fastMD": {
            "wall_time_s": round(fastmd_time, 2),
            "ns_per_day": round(ns_per_day_f, 2),
        },
        "lammps": {
            "wall_time_s": round(lammps_time, 2),
            "ns_per_day": round(ns_per_day_l, 2),
        },
        "speedup": round(speedup, 2),
        "validation": {
            "passed": validation["passed"],
            "rel_temp_diff": validation["rel_temp_diff"],
            "rel_pe_diff": validation["rel_pe_diff"],
        },
    }
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report written to {out_path}")
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/report.py
git commit -m "feat: JSON benchmark reporter with GPU detection"
```

---

### Task 8: End-to-end validation and benchmark run

**Files:**
- Modify: `benchmarks/benchmark.json` (if any path tweaks needed)

- [ ] **Step 1: Ensure fastMD is built in Release mode**

```bash
cd /home/zhenghaowu/fastMD/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

- [ ] **Step 2: Run validation on full system**

```bash
cd /home/zhenghaowu/fastMD
python3 benchmarks/run.py
```

This will first run the 1k-step validation. Watch for:
- `Validation passed.`
- If it fails, inspect `fastmd_benchmark.conf` and `lammps_benchmark.in` for parameter mismatches, then adjust `benchmark.json` or the generator.

- [ ] **Step 3: Let the full benchmark complete**

The 1M-step benchmark will take several minutes. After completion, verify:
```bash
cat benchmark_report.json
```

Expected keys: `timestamp`, `gpu`, `fastMD`, `lammps`, `speedup`, `validation`.

- [ ] **Step 4: Commit any final fixes**

If you had to tweak `benchmark.json` or the Python scripts, commit them:
```bash
git add benchmarks/
git commit -m "fix: benchmark parameter tuning after end-to-end test"
```

---

## Self-Review

**1. Spec coverage:**
- LAMMPS data parser for bonds/angles → Task 2 + Task 4
- Wire into main.cu → Task 3
- Python input generator → Task 5
- Validation phase (1k steps, average T/PE comparison) → Task 6
- Benchmark phase (1M steps, timing) → Task 6
- JSON reporter → Task 7
- Parameter mapping (gamma→damp, degrees→radians, etc.) → Task 5 code + Task 1 Step 2
- No gaps identified.

**2. Placeholder scan:**
- No TBD/TODO/fill-in-later found.
- All code blocks contain complete, compilable code.
- All commands have exact paths and expected outputs.

**3. Type consistency:**
- `TopologyData` fields used in Task 1 match usage in Task 2 and Task 3.
- `FENEParams` and `AngleParams` structs match their definitions in `fene.cuh` and `angle.cuh`.
- Config key `lammps_data_file` is parsed in Task 1 and consumed in Task 3.

Plan is clean and ready for execution.
