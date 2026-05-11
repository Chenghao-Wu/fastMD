# Tabulated Potential Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LAMMPS-style tabulated pair potentials to fastMD with linear GPU interpolation, coexisting with the existing LJ kernel.

**Architecture:** A host-side parser reads LAMMPS table files. Config parsing wires type pairs to tables. A new GPU kernel traverses the shared Verlet neighbor list and accumulates table forces via read-add-write after the LJ kernel. The existing LJ kernel is untouched.

**Tech Stack:** CUDA C++, CMake, GoogleTest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/io/table_parser.hpp` | Create | `TableParams`, `TableFileData`, `parse_table_file()` declaration |
| `src/io/table_parser.cpp` | Create | LAMMPS table file parser implementation |
| `src/force/table.cuh` | Create | `launch_table_kernel()` declaration |
| `src/force/table.cu` | Create | `table_verlet_kernel` and launcher |
| `src/io/config.hpp` | Modify | Add `table_idx`, `table_params`, `table_data` to `TopologyData` |
| `src/io/config.cpp` | Modify | Parse `table` keyword, validate against `lj` conflicts |
| `src/core/system.cuh` | Modify | Add `d_table_idx`, `d_table_params`, `d_table_data` to `System` |
| `src/core/system.cu` | Modify | Allocate/free table device memory conditionally |
| `src/main.cu` | Modify | Upload table data, launch table kernel after LJ |
| `CMakeLists.txt` | Modify | Add `table.cu` and `table_parser.cpp` to `core` |
| `tests/CMakeLists.txt` | Modify | Add `test_table_parser.cu` and `test_table.cu` targets |
| `tests/test_table_parser.cu` | Create | Parser unit tests (valid file, N-only header, errors) |
| `tests/test_table.cu` | Create | GPU kernel test: table reproduces LJ within interpolation tolerance |
| `tests/test_config.cu` | Create | Config parsing test for `table` keyword |

---

### Task 1: Table Parser

**Files:**
- Create: `src/io/table_parser.hpp`
- Create: `src/io/table_parser.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_table_parser.cu`

- [ ] **Step 1: Create parser header**

```cpp
// src/io/table_parser.hpp
#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

struct TableParams {
    float rmin;
    float rmax;
    float dr;
    float inv_dr;
    int   npoints;
    int   data_offset;
};

struct TableFileData {
    std::vector<TableParams> params;
    std::vector<float4>      data;
};

TableFileData parse_table_file(const std::string& filename,
                                const std::string& keyword);
```

- [ ] **Step 2: Create parser implementation**

```cpp
// src/io/table_parser.cpp
#include "table_parser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

TableFileData parse_table_file(const std::string& filename,
                                const std::string& keyword) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("Cannot open table file: " + filename);

    std::string line;
    bool found_keyword = false;
    int n = 0;
    float rlo = 0.0f, rhi = 0.0f;
    bool has_r = false;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string first;
        iss >> first;
        if (first == keyword) {
            found_keyword = true;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream hss(line);
                std::string key;
                hss >> key;
                if (key == "N") {
                    hss >> n;
                    std::string r_key;
                    if (hss >> r_key) {
                        if (r_key == "R") {
                            hss >> rlo >> rhi;
                            has_r = true;
                        }
                    }
                }
                break;
            }
            break;
        }
    }

    if (!found_keyword)
        throw std::runtime_error("Table keyword not found: " + keyword);

    std::vector<float4> points;
    points.reserve(n);

    while (std::getline(in, line) && (int)points.size() < n) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int idx;
        float r, e, f;
        iss >> idx >> r >> e >> f;
        points.push_back(make_float4(r, f, e, 0.0f));
    }

    if ((int)points.size() != n)
        throw std::runtime_error("Table data line count mismatch in " + filename);

    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x <= points[i - 1].x)
            throw std::runtime_error("Table r values not monotonically increasing");
    }

    if (!has_r) {
        rlo = points.front().x;
        rhi = points.back().x;
    }

    TableParams tp;
    tp.rmin = rlo;
    tp.rmax = rhi;
    tp.npoints = n;
    tp.dr = (rhi - rlo) / (n - 1);
    tp.inv_dr = 1.0f / tp.dr;
    tp.data_offset = 0;

    TableFileData result;
    result.params.push_back(tp);
    result.data = std::move(points);
    return result;
}
```

- [ ] **Step 3: Add parser to build**

Modify `CMakeLists.txt` `core` library sources:

```cmake
add_library(core STATIC
    src/core/system.cu
    src/core/morton.cu
    src/neighbor/tile_list.cu
    src/neighbor/verlet_list.cu
    src/force/lj.cu
    src/force/table.cu
    src/force/fene.cu
    src/force/angle.cu
    src/integrate/langevin.cu
    src/integrate/nose_hoover.cu
    src/analysis/thermo.cu
    src/analysis/rg.cu
    src/analysis/correlator.cu
    src/io/dump.cu
    src/io/lammps_data.cpp
    src/io/table_parser.cpp   # ADD
)
```

- [ ] **Step 4: Write failing parser test**

```cpp
// tests/test_table_parser.cu
#include <gtest/gtest.h>
#include "io/table_parser.hpp"
#include <fstream>
#include <sstream>

TEST(TableParser, ParsesValidFile) {
    std::string content = R"(# comment
LJ_TABLE
N 5 R 0.5 1.5

1 0.5000 100.0000 -50.0000
2 0.7500  50.0000 -25.0000
3 1.0000   0.0000   0.0000
4 1.2500 -10.0000   5.0000
5 1.5000  -5.0000   2.5000
)";
    std::string fname = "/tmp/test_table.txt";
    {
        std::ofstream f(fname);
        f << content;
    }
    auto result = parse_table_file(fname, "LJ_TABLE");
    ASSERT_EQ(result.params.size(), 1u);
    ASSERT_EQ(result.data.size(), 5u);
    EXPECT_FLOAT_EQ(result.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(result.params[0].rmax, 1.5f);
    EXPECT_FLOAT_EQ(result.data[0].y, -50.0f);
    EXPECT_FLOAT_EQ(result.data[4].z, -5.0f);
}

TEST(TableParser, ParsesNOnlyHeader) {
    std::string content = R"(PAIR_0
N 3

1 0.5 10.0 -5.0
2 1.0  5.0 -2.5
3 1.5  0.0  0.0
)";
    std::string fname = "/tmp/test_table_nonly.txt";
    {
        std::ofstream f(fname);
        f << content;
    }
    auto result = parse_table_file(fname, "PAIR_0");
    ASSERT_EQ(result.data.size(), 3u);
    EXPECT_FLOAT_EQ(result.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(result.params[0].rmax, 1.5f);
}

TEST(TableParser, MissingKeywordThrows) {
    std::string fname = "/tmp/test_table_bad.txt";
    {
        std::ofstream f(fname);
        f << "WRONG\nN 2 R 0 1\n\n1 0 0 0\n2 1 0 0\n";
    }
    EXPECT_THROW(parse_table_file(fname, "RIGHT"), std::runtime_error);
}
```

- [ ] **Step 5: Register test in CMake**

Add to `tests/CMakeLists.txt`:

```cmake
add_cuda_test(test_table_parser)
target_sources(test_table_parser PRIVATE
    ${CMAKE_SOURCE_DIR}/src/io/table_parser.cpp)
```

- [ ] **Step 6: Build and run parser test**

Run:
```bash
cd build && cmake .. && make test_table_parser -j$(nproc) && ./tests/test_table_parser
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/io/table_parser.hpp src/io/table_parser.cpp tests/test_table_parser.cu tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat: add LAMMPS table file parser

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Config Integration

**Files:**
- Modify: `src/io/config.hpp`
- Modify: `src/io/config.cpp`
- Test: `tests/test_config.cu`

- [ ] **Step 1: Modify config.hpp**

Add to `TopologyData`:

```cpp
// src/io/config.hpp
#include "table_parser.hpp"   // ADD at top

struct TopologyData {
    // ... existing fields ...
    std::vector<int>         table_idx;      // ADD: size ntypes*ntypes, -1 = none
    std::vector<TableParams> table_params;   // ADD
    std::vector<float4>      table_data;     // ADD
    // ... rest unchanged ...
};
```

- [ ] **Step 2: Modify config.cpp**

After the `lj_entries` collection, add `table_entries`:

```cpp
// src/io/config.cpp
#include "table_parser.hpp"   // ADD at top

// In parse_config, add after lj_entries vector:
std::vector<std::tuple<int,int,std::string,std::string>> table_entries;

// In parsing loop, add after "lj" block:
else if (key == "table") {
    int ti, tj; std::string filename, keyword;
    iss >> ti >> tj >> filename >> keyword;
    table_entries.push_back({ti, tj, filename, keyword});
}

// After lj_params population, add:
topo.table_idx.resize(params.ntypes * params.ntypes, -1);

std::map<std::pair<std::string,std::string>, int> file_keyword_to_idx;
for (auto& [ti, tj, filename, keyword] : table_entries) {
    if (ti < 0 || ti >= params.ntypes || tj < 0 || tj >= params.ntypes)
        throw std::runtime_error("table type index out of range");

    float2 lj_p = topo.lj_params[ti * params.ntypes + tj];
    if (lj_p.x != 0.0f || lj_p.y != 0.0f)
        throw std::runtime_error("Both lj and table defined for pair (" +
                                  std::to_string(ti) + "," + std::to_string(tj) + ")");

    auto fk = std::make_pair(filename, keyword);
    int idx;
    auto it = file_keyword_to_idx.find(fk);
    if (it == file_keyword_to_idx.end()) {
        TableFileData tfd = parse_table_file(filename, keyword);
        idx = (int)topo.table_params.size();
        file_keyword_to_idx[fk] = idx;
        topo.table_params.push_back(tfd.params[0]);
        topo.table_data.insert(topo.table_data.end(),
                                tfd.data.begin(), tfd.data.end());
    } else {
        idx = it->second;
    }
    topo.table_idx[ti * params.ntypes + tj] = idx;
    topo.table_idx[tj * params.ntypes + ti] = idx;
}
```

- [ ] **Step 3: Write config test**

```cpp
// tests/test_config.cu
#include <gtest/gtest.h>
#include "io/config.hpp"
#include <fstream>

TEST(Config, ParsesTableLine) {
    std::string cfg = "/tmp/test_config_table.conf";
    {
        std::ofstream f(cfg);
        f << "natoms 2\nntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\n"
          << "nvt_langevin 1.0 1.0 1.0\n"
          << "table 0 0 /tmp/test_table.txt LJ_TABLE\n";
    }
    {
        std::ofstream f("/tmp/test_table.txt");
        f << "LJ_TABLE\nN 2 R 0.5 1.0\n\n1 0.5 0 0\n2 1.0 0 0\n";
    }
    TopologyData topo;
    SimParams params = parse_config(cfg, topo);
    EXPECT_EQ(topo.table_idx.size(), 1u);
    EXPECT_EQ(topo.table_idx[0], 0);
    EXPECT_EQ(topo.table_params.size(), 1u);
    EXPECT_EQ(topo.table_data.size(), 2u);
}

TEST(Config, RejectsLjAndTableConflict) {
    std::string cfg = "/tmp/test_config_conflict.conf";
    {
        std::ofstream f(cfg);
        f << "natoms 2\nntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\n"
          << "nvt_langevin 1.0 1.0 1.0\n"
          << "lj 0 0 1.0 1.0\n"
          << "table 0 0 /tmp/test_table.txt LJ_TABLE\n";
    }
    {
        std::ofstream f("/tmp/test_table.txt");
        f << "LJ_TABLE\nN 2 R 0.5 1.0\n\n1 0.5 0 0\n2 1.0 0 0\n";
    }
    TopologyData topo;
    EXPECT_THROW(parse_config(cfg, topo), std::runtime_error);
}
```

- [ ] **Step 4: Register config test in CMake**

Add to `tests/CMakeLists.txt`:

```cmake
add_cuda_test(test_config)
target_sources(test_config PRIVATE
    ${CMAKE_SOURCE_DIR}/src/io/config.cpp
    ${CMAKE_SOURCE_DIR}/src/io/table_parser.cpp)
```

- [ ] **Step 5: Build and run config tests**

Run:
```bash
cd build && cmake .. && make test_config -j$(nproc) && ./tests/test_config
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/io/config.hpp src/io/config.cpp tests/test_config.cu tests/CMakeLists.txt
git commit -m "feat: parse table keyword in config, validate lj conflicts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: System Device Memory

**Files:**
- Modify: `src/core/system.cuh`
- Modify: `src/core/system.cu`

- [ ] **Step 1: Add table pointers to System**

```cpp
// src/core/system.cuh
struct System {
    // ... existing fields ...
    float2* lj_params;
    float*  virial;

    // ADD:
    int*         d_table_idx;
    TableParams* d_table_params;
    float4*      d_table_data;

    int*    d_max_dr2_int;
    // ... rest unchanged ...
};
```

- [ ] **Step 2: Allocate and free table memory**

```cpp
// src/core/system.cu
void System::allocate(const SimParams& params) {
    // ... existing allocation code ...
    CUDA_CHECK(cudaMalloc(&lj_params, lj_size * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&virial, 6 * sizeof(float)));

    // ADD:
    d_table_idx = nullptr;
    d_table_params = nullptr;
    d_table_data = nullptr;

    CUDA_CHECK(cudaMalloc(&d_max_dr2_int, sizeof(int)));
    // ... rest unchanged ...
}

void System::free() {
    // ... existing frees ...
    CUDA_CHECK(cudaFree(lj_params));
    CUDA_CHECK(cudaFree(virial));

    // ADD:
    if (d_table_idx)     CUDA_CHECK(cudaFree(d_table_idx));
    if (d_table_params)  CUDA_CHECK(cudaFree(d_table_params));
    if (d_table_data)    CUDA_CHECK(cudaFree(d_table_data));

    CUDA_CHECK(cudaFree(d_max_dr2_int));
    // ... rest unchanged ...
}
```

- [ ] **Step 3: Build to check compilation**

Run:
```bash
cd build && cmake .. && make core -j$(nproc)
```
Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/core/system.cuh src/core/system.cu
git commit -m "feat: add table device memory to System

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Table GPU Kernel

**Files:**
- Create: `src/force/table.cuh`
- Create: `src/force/table.cu`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Test: `tests/test_table.cu`

- [ ] **Step 1: Create table kernel header**

```cpp
// src/force/table.cuh
#pragma once
#include "io/table_parser.hpp"
#include "../core/types.cuh"

void launch_table_kernel(const float4* __restrict__ pos,
                         float4* __restrict__ force,
                         float* __restrict__ virial,
                         const int* __restrict__ table_idx,
                         const TableParams* __restrict__ table_params,
                         const float4* __restrict__ table_data,
                         const int* __restrict__ neighbors,
                         const int* __restrict__ num_neighbors,
                         int natoms, int ntypes,
                         float rc2, float L, float inv_L,
                         cudaStream_t stream = 0);
```

- [ ] **Step 2: Create table kernel implementation**

```cpp
// src/force/table.cu
#include "table.cuh"
#include "../core/pbc.cuh"

__global__ void table_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int* __restrict__ table_idx,
    const TableParams* __restrict__ table_params,
    const float4* __restrict__ table_data,
    const int* __restrict__ neighbors,
    const int* __restrict__ num_neighbors,
    int natoms, int ntypes,
    float rc2, float L, float inv_L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    if (i < natoms) {
        float4 pos_i = pos[i];
        int type_i = __float_as_int(pos_i.w);
        int nneigh = __ldg(&num_neighbors[i]);

        #pragma unroll 8
        for (int k = 0; k < nneigh; k++) {
            int j = neighbors[k * natoms + i];
            float4 pos_j = __ldg(&pos[j]);
            int type_j = __float_as_int(pos_j.w);

            float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
            float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
            float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2 && r2 > 1e-10f) {
                int tidx = __ldg(&table_idx[type_i * ntypes + type_j]);
                if (tidx >= 0) {
                    TableParams tp = table_params[tidx];
                    float r = sqrtf(r2);
                    if (r < tp.rmin) r = tp.rmin;
                    if (r > tp.rmax) r = tp.rmax;

                    int idx = (int)((r - tp.rmin) * tp.inv_dr);
                    if (idx < 0) idx = 0;
                    if (idx >= tp.npoints - 1) idx = tp.npoints - 2;

                    float t = (r - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
                    float4 p0 = __ldg(&table_data[tp.data_offset + idx]);
                    float4 p1 = __ldg(&table_data[tp.data_offset + idx + 1]);

                    float f_scalar = p0.y + t * (p1.y - p0.y);
                    float e = p0.z + t * (p1.z - p0.z);

                    float rinv = 1.0f / r;
                    float fpair = f_scalar * rinv;
                    float fxij = fpair * dx;
                    float fyij = fpair * dy;
                    float fzij = fpair * dz;

                    fx += fxij; fy += fyij; fz += fzij;
                    pe_i += 0.5f * e;
                    vir_xx += 0.5f * fxij * dx;
                    vir_xy += 0.5f * fxij * dy;
                    vir_xz += 0.5f * fxij * dz;
                    vir_yy += 0.5f * fyij * dy;
                    vir_yz += 0.5f * fyij * dz;
                    vir_zz += 0.5f * fzij * dz;
                }
            }
        }

        float4 f = force[i];
        f.x += fx; f.y += fy; f.z += fz; f.w += pe_i;
        force[i] = f;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        vir_xx += __shfl_down_sync(0xFFFFFFFF, vir_xx, offset);
        vir_xy += __shfl_down_sync(0xFFFFFFFF, vir_xy, offset);
        vir_xz += __shfl_down_sync(0xFFFFFFFF, vir_xz, offset);
        vir_yy += __shfl_down_sync(0xFFFFFFFF, vir_yy, offset);
        vir_yz += __shfl_down_sync(0xFFFFFFFF, vir_yz, offset);
        vir_zz += __shfl_down_sync(0xFFFFFFFF, vir_zz, offset);
    }

    extern __shared__ float s_virial[];
    if (lane == 0) {
        int wid = threadIdx.x >> 5;
        s_virial[wid * 6 + 0] = vir_xx;
        s_virial[wid * 6 + 1] = vir_xy;
        s_virial[wid * 6 + 2] = vir_xz;
        s_virial[wid * 6 + 3] = vir_yy;
        s_virial[wid * 6 + 4] = vir_yz;
        s_virial[wid * 6 + 5] = vir_zz;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x >> 5;
        float b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
        for (int w = 0; w < num_warps; w++) {
            b0 += s_virial[w * 6 + 0];
            b1 += s_virial[w * 6 + 1];
            b2 += s_virial[w * 6 + 2];
            b3 += s_virial[w * 6 + 3];
            b4 += s_virial[w * 6 + 4];
            b5 += s_virial[w * 6 + 5];
        }
        atomicAdd(&virial[0], b0);
        atomicAdd(&virial[1], b1);
        atomicAdd(&virial[2], b2);
        atomicAdd(&virial[3], b3);
        atomicAdd(&virial[4], b4);
        atomicAdd(&virial[5], b5);
    }
}

void launch_table_kernel(const float4* pos, float4* force, float* virial,
                         const int* table_idx,
                         const TableParams* table_params,
                         const float4* table_data,
                         const int* neighbors, const int* num_neighbors,
                         int natoms, int ntypes,
                         float rc2, float L, float inv_L,
                         cudaStream_t stream)
{
    int blocks = div_ceil(natoms, 256);
    int smem = (256 / 32) * 6 * sizeof(float);
    table_verlet_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, table_idx, table_params, table_data,
        neighbors, num_neighbors,
        natoms, ntypes, rc2, L, inv_L);
}
```

- [ ] **Step 3: Write table kernel test**

```cpp
// tests/test_table.cu
#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/lj.cuh"
#include "force/table.cuh"
#include "neighbor/verlet_list.cuh"
#include "io/table_parser.hpp"

TEST(Table, MatchesLJWithinInterpolationTolerance) {
    const int N = 64;
    const float L = 8.0f;
    const float inv_L = 1.0f / L;
    const float rc = 2.5f;
    const float rc2 = rc * rc;
    const float skin = 0.5f;
    const int ntypes = 1;
    const int ntiles = div_ceil(N, TILE_SIZE);
    const int np = ntiles * TILE_SIZE;

    std::vector<float4> h_pos(N);
    srand(123);
    for (int i = 0; i < N; i++) {
        h_pos[i] = make_float4(
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            L * (rand() / (float)RAND_MAX),
            pack_type_id(0));
    }

    // Build LJ-equivalent table
    int ntable = 1000;
    float rlo = 0.5f, rhi = rc;
    float dr = (rhi - rlo) / (ntable - 1);
    std::vector<float4> h_table_data(ntable);
    for (int k = 0; k < ntable; k++) {
        float r = rlo + k * dr;
        float r2 = r * r;
        float r2inv = 1.0f / r2;
        float sr2 = r2inv;
        float sr6 = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6;
        float force = 24.0f * r2inv * (2.0f * sr12 - sr6);
        float energy = 4.0f * (sr12 - sr6);
        h_table_data[k] = make_float4(r, force, energy, 0.0f);
    }
    TableParams tp;
    tp.rmin = rlo; tp.rmax = rhi; tp.dr = dr;
    tp.inv_dr = 1.0f / dr; tp.npoints = ntable; tp.data_offset = 0;

    std::vector<int> h_table_idx(ntypes * ntypes, 0);
    std::vector<float2> h_lj = {{1.0f, 1.0f}};

    std::vector<float4> h_pos_pad(np, make_float4(0,0,0, pack_type_id(-1)));
    std::copy(h_pos.begin(), h_pos.end(), h_pos_pad.begin());

    float4* d_pos = to_device(h_pos_pad);
    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    float* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(float)));
    float2* d_lj = to_device(h_lj);
    int* d_table_idx = to_device(h_table_idx);
    TableParams* d_table_params = to_device(std::vector<TableParams>{tp});
    float4* d_table_data = to_device(h_table_data);

    VerletList verlet;
    verlet.allocate(N, rc + skin, L);
    verlet.build(d_pos, N, L, inv_L, nullptr, nullptr, rc + skin);

    // Run LJ
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_lj_kernel(d_pos, d_force, d_virial, d_lj,
                     verlet.neighbors, verlet.num_neighbors,
                     N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto h_force_lj = to_host(d_force, N);
    float h_virial_lj[6];
    CUDA_CHECK(cudaMemcpy(h_virial_lj, d_virial, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    // Run table
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(float)));
    launch_table_kernel(d_pos, d_force, d_virial,
                        d_table_idx, d_table_params, d_table_data,
                        verlet.neighbors, verlet.num_neighbors,
                        N, ntypes, rc2, L, inv_L);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto h_force_table = to_host(d_force, N);
    float h_virial_table[6];
    CUDA_CHECK(cudaMemcpy(h_virial_table, d_virial, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float3 lj_f = make_float3(h_force_lj[i].x, h_force_lj[i].y, h_force_lj[i].z);
        float3 t_f  = make_float3(h_force_table[i].x, h_force_table[i].y, h_force_table[i].z);
        assert_float3_near(lj_f, t_f, 1e-3f);
        float rel_pe = fabsf(h_force_lj[i].w - h_force_table[i].w)
                       / fmaxf(fabsf(h_force_lj[i].w), 1e-6f);
        EXPECT_LT(rel_pe, 1e-3f) << "PE mismatch at atom " << i;
    }
    for (int k = 0; k < 6; k++) {
        float denom = fmaxf(fabsf(h_virial_lj[k]), 1e-6f);
        float rel = fabsf(h_virial_table[k] - h_virial_lj[k]) / denom;
        EXPECT_LT(rel, 1e-2f) << "virial component " << k;
    }

    verlet.free();
    free_device(d_pos); free_device(d_force); free_device(d_virial);
    free_device(d_lj); free_device(d_table_idx); free_device(d_table_params);
    free_device(d_table_data);
}
```

- [ ] **Step 4: Register kernel test in CMake**

Add to `tests/CMakeLists.txt`:

```cmake
add_cuda_test(test_table)
target_sources(test_table PRIVATE
    ${CMAKE_SOURCE_DIR}/src/force/table.cu
    ${CMAKE_SOURCE_DIR}/src/neighbor/verlet_list.cu
    ${CMAKE_SOURCE_DIR}/src/core/morton.cu
    ${CMAKE_SOURCE_DIR}/src/io/table_parser.cpp)
```

- [ ] **Step 5: Build and run table test**

Run:
```bash
cd build && cmake .. && make test_table -j$(nproc) && ./tests/test_table
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/force/table.cuh src/force/table.cu tests/test_table.cu tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat: add tabulated potential GPU kernel and unit test

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Main Dispatch

**Files:**
- Modify: `src/main.cu`

- [ ] **Step 1: Add table includes and uploads in main**

At the top of `src/main.cu`, add:
```cpp
#include "force/table.cuh"
```

After the existing `lj_params` upload (around line 99), add:
```cpp
if (!topo.table_params.empty()) {
    CUDA_CHECK(cudaMalloc(&sys.d_table_idx,
                          params.ntypes * params.ntypes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sys.d_table_params,
                          topo.table_params.size() * sizeof(TableParams)));
    CUDA_CHECK(cudaMalloc(&sys.d_table_data,
                          topo.table_data.size() * sizeof(float4)));

    CUDA_CHECK(cudaMemcpy(sys.d_table_idx, topo.table_idx.data(),
                          params.ntypes * params.ntypes * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sys.d_table_params, topo.table_params.data(),
                          topo.table_params.size() * sizeof(TableParams),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sys.d_table_data, topo.table_data.data(),
                          topo.table_data.size() * sizeof(float4),
                          cudaMemcpyHostToDevice));
}
```

- [ ] **Step 2: Launch table kernel after LJ**

In the force computation section (both initial setup and main loop), after `launch_lj_kernel` and before bonded kernels, add:

```cpp
if (!topo.table_params.empty()) {
    launch_table_kernel(sys.pos, sys.force, sys.virial,
                        sys.d_table_idx, sys.d_table_params, sys.d_table_data,
                        verlet.neighbors, verlet.num_neighbors,
                        params.natoms, params.ntypes,
                        params.rc2, L_f, inv_L_f, stream_lj);
}
```

The table kernel runs on `stream_lj`, sequentially after LJ, before the `force_done_lj` event is recorded.

- [ ] **Step 3: Build project**

Run:
```bash
cd build && cmake .. && make fastMD -j$(nproc)
```
Expected: links successfully.

- [ ] **Step 4: Integration test with CG_water**

Create `/tmp/cg_water_fastmd.conf`:

```
natoms 3000
ntypes 1
rc 15.0
skin 0.3
dt 2.0
nsteps 100
dump_freq 0
thermo 1 10 thermo.dat
nvt_langevin 300.0 300.0 100.0 12345
table 0 0 data/CG_water/pair_table.txt PAIR_0
lammps_data_file data/CG_water/system.data
```

Run:
```bash
cd build && ./fastMD /tmp/cg_water_fastmd.conf
```
Expected: simulation completes 100 steps without crashing; thermo output written.

- [ ] **Step 5: Commit**

```bash
git add src/main.cu
git commit -m "feat: wire table kernel into main simulation loop

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Regression Tests & Final Verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
cd build && ctest --output-on-failure
```
Expected: all existing tests plus new tests PASS.

- [ ] **Step 2: Run LJ benchmark to confirm no regression**

```bash
cd build && ./fastMD ../fastmd_test.conf
```
(Or any existing LJ-only config.) Expected: same output as before.

- [ ] **Step 3: Commit final state**

```bash
git commit --allow-empty -m "test: verify full suite passes and LJ benchmark shows no regression

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**
- LAMMPS table file parser with `N`-only header -> Task 1
- Config `table` keyword parsing and `lj` conflict validation -> Task 2
- `TopologyData` and `System` table fields -> Tasks 2, 3
- GPU kernel with linear interpolation and accumulation -> Task 4
- Main dispatch on `stream_lj` after LJ -> Task 5
- Error handling (file not found, missing keyword, monotonicity, conflicts) -> Tasks 1, 2
- Testing (parser, kernel, config, integration) -> Tasks 1, 2, 4, 5, 6
- CG_water verification example -> Task 5

**Placeholder scan:** No TBD, TODO, or vague instructions found. Every step includes exact code or exact commands.

**Type consistency:**
- `TableParams` used in `table_parser.hpp`, `config.hpp`, `system.cuh`, `table.cuh` — same struct everywhere.
- `parse_table_file` signature matches declaration and call sites.
- `launch_table_kernel` signature matches declaration and call sites.
