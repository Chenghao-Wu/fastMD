# Profile-Guided GPU Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile the Verlet-list-based fastMD codebase with Nsight Systems + Compute, identify the top 2-3 bottleneck kernels, and apply targeted optimizations targeting 20-40% overall step-time reduction.

**Architecture:** Two-phase approach. Phase 1 (Tasks 1-4): Profile with nsys to get the new GPU timeline, then ncu on the top kernels. Phase 2 (Tasks 5-11): Apply optimizations based on profiling results — low-hanging fruit (persistent allocations) always, plus targeted fixes for whichever kernels profiling ranks highest. Final validation against the LAMMPS benchmark harness.

**Tech Stack:** CUDA C++ (CUB for radix sort), Nsight Systems CLI, Nsight Compute CLI, CMake, Python benchmark harness.

---

## File Map

| File | Role | Change |
|------|------|--------|
| `src/analysis/thermo.cu` | Thermo kernels + host wrapper | Persistent allocation for `d_kin_stress`/`d_pe` |
| `src/analysis/thermo.cuh` | Thermo types | Add persistent buffer pointers to signature or struct |
| `src/io/dump.cu` | Binary trajectory dumper | Persistent allocation for `d_temp` |
| `src/force/fene.cu` | FENE bond kernel + launcher | Per-atom bond processing (remove atomics) |
| `src/force/fene.cuh` | FENE params + launcher signature | Add atom-bond mapping params |
| `src/force/angle.cu` | Angle kernel + launcher | Per-atom angle processing (remove atomics) |
| `src/force/angle.cuh` | Angle params + launcher signature | Add atom-angle mapping params |
| `src/core/system.cu` | System allocation/free | Add atom-bond and atom-angle mapping arrays |
| `src/core/system.cuh` | System struct | Add mapping array pointers |
| `src/io/lammps_data.hpp` | TopologyData struct | Add atom-bond/atom-angle mapping builder |
| `src/io/lammps_data.cpp` | Topology data parsing | Build atom-bond and atom-angle maps |
| `src/main.cu` | Main loop | Wire new kernel signatures, allocate/free maps |
| `tests/test_fene.cu` | FENE unit test | Build atom-bond map, update kernel call |
| `tests/test_angle.cu` | Angle unit test | Build atom-angle map, update kernel call |
| `src/force/lj.cu` | LJ kernel | Micro-optimizations if top-ranked |
| `src/integrate/langevin.cu` | Integrator kernel | RNG replacement if top-ranked |
| `src/neighbor/verlet_list.cu` | Verlet build | Build pipeline optimizations if top-ranked |

---

### Task 1: Build baseline and verify correctness

**Files:** None modified

- [ ] **Step 1: Clean rebuild in Release mode**

```bash
cd /home/zhenghaowu/fastMD/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Expected: build succeeds with no errors.

- [ ] **Step 2: Run unit tests**

```bash
cd /home/zhenghaowu/fastMD/build && ctest --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 3: Run benchmark validation (1K steps)**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: `"passed": true` in `benchmark_report.json`.

- [ ] **Step 4: Record baseline wall time**

Read `benchmark_report.json` and note `fastMD.wall_time_s` and `speedup` — this is the pre-optimization baseline.

- [ ] **Step 5: Commit baseline (no changes)**

Skip if clean — nothing to commit. This task is verification only.

---

### Task 2: Run Nsight Systems profile on validation workload

**Files:**
- Create: `nsys_profile.log` (temporary, not committed)

- [ ] **Step 1: Generate the benchmark config for validation phase**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/generate_lammps_input.py benchmarks/benchmark.json validation
```

Expected: `fastmd_benchmark.conf` created/updated.

- [ ] **Step 2: Run nsys profile**

```bash
cd /home/zhenghaowu/fastMD && nsys profile --stats=true --force-overwrite=true -o fastmd_nsys ./build/fastMD fastmd_benchmark.conf 2>&1 | tee nsys_profile.log
```

Expected: Profile completes. Look for the CUDA GPU kernel summary table in the output.

- [ ] **Step 3: Extract kernel timing summary from nsys output**

```bash
cd /home/zhenghaowu/fastMD && grep -A 50 "CUDA Kernel Statistics" nsys_profile.log | head -60
```

Expected: Table showing each kernel name, total time, % of total, invocation count.

- [ ] **Step 4: Save the kernel ranking to a reference file**

```bash
cd /home/zhenghaowu/fastMD && grep -A 50 "CUDA Kernel Statistics" nsys_profile.log > kernel_ranking.txt
```

Note: the top 2-3 kernels by total time (excluding `memset`, `memcpy`, and CUB internals) are the optimization targets. Record them — they determine which of Tasks 5-11 apply.

---

### Task 3: Analyze nsys results and classify bottlenecks

**Files:** None (analysis only)

- [ ] **Step 1: Identify top compute kernels**

Examine `kernel_ranking.txt`. Filter out:
- `cudaMemset`, `cudaMemcpy`, `cudaMemcpyAsync` — these are infrastructure
- `cub::DeviceRadixSort` internal kernels — these are library code, not our optimization target
- `curand_init` — one-time setup

Focus on the remaining kernels: `lj_verlet_kernel`, `fene_kernel`, `angle_kernel`, `integrator_pre_force_kernel`, `integrator_post_force_kernel`, `build_verlet_list`, `assign_cells`, `find_cell_starts`, `kinetic_stress_kernel`, `sum_pe_kernel`, `correlator_push_kernel`, `pack_float3`.

- [ ] **Step 2: Check stream overlap in the nsys timeline**

Open the `.nsys-rep` file in Nsight Systems GUI (if available) or check the API trace:

```bash
cd /home/zhenghaowu/fastMD && nsys stats fastmd_nsys.nsys-rep | grep -A 20 "Stream"
```

Expected: Verify that `stream_lj` and `stream_bonded` show concurrent execution. If not, note the serialization source.

- [ ] **Step 3: Document the top-3 bottleneck ranking**

Record in the commit message or a brief note:
1. Top kernel name and % of GPU time
2. Second kernel name and % of GPU time
3. Third kernel name and % of GPU time

This ranking determines which optimization tasks to execute next.

---

### Task 4: Run Nsight Compute on the top 2-3 kernels

**Files:** None

Run this task for each kernel identified in Task 3. For each kernel `KERNEL_NAME`:

- [ ] **Step 1: Run ncu on the kernel**

```bash
cd /home/zhenghaowu/fastMD && ncu --set full --launch-skip 0 --launch-count 10 -o fastmd_ncu_<short_name> -k <KERNEL_NAME> ./build/fastMD fastmd_benchmark.conf
```

Replace `<KERNEL_NAME>` with the actual kernel name (e.g., `lj_verlet_kernel`, `fene_kernel`).

If the kernel runs many times (e.g., every step for 1000 steps), skip past the first few launches to get steady-state metrics:

```bash
ncu --set full --launch-skip 100 --launch-count 5 -o fastmd_ncu_<short_name> -k <KERNEL_NAME> ./build/fastMD fastmd_benchmark.conf
```

- [ ] **Step 2: Extract key metrics for bottleneck classification**

For each profiled kernel, record:
- **Compute utilization:** `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- **Memory bandwidth:** `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- **Occupancy:** `sm__warps_active.avg.pct_of_peak_sustained_active`
- **L1 hit rate:** `l1tex__t_sector_hit_lookup.pct`
- **L2 hit rate:** `lts__t_sector_hit_lookup.pct`
- **Long scoreboard stalls:** `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
- **Atomic count:** `smsp__inst_executed.sum.atom` (if non-zero)

Classification rules:
- Compute-bound: `sm__throughput > 50%`, `dram__throughput < 30%`
- Memory-bound: `dram__throughput > 50%`, L1/L2 hit rate low
- Atomic-bottleneck: atomic instructions > 10% of total, paired with memory stalls
- Latency-bound: occupancy < 50%, long scoreboard stalls high

- [ ] **Step 3: Decide which optimization tasks to execute**

Based on the bottleneck classification:
- If `fene_kernel` or `angle_kernel` is in top-3 with atomics: **execute Tasks 7 and 8** (per-atom bonded kernels)
- If `lj_verlet_kernel` is still top-ranked and compute-bound: **execute Task 9** (LJ micro-optimizations)
- If `integrator_pre_force_kernel` is in top-3: **execute Task 10** (integrator RNG)
- If `build_verlet_list` or `assign_cells` is in top-3: **execute Task 11** (Verlet build)
- Always execute **Tasks 5 and 6** (persistent allocations — low-hanging fruit)

---

### Task 5: Persistent allocations in thermo computation

**Files:**
- Modify: `src/analysis/thermo.cu:51-92`
- Modify: `src/analysis/thermo.cuh`

The `compute_thermo` function currently allocates and frees `d_kin_stress` (6 floats) and `d_pe` (1 float) on every thermo step. These are tiny but the `cudaMalloc`/`cudaFree` calls add API overhead.

- [ ] **Step 1: Add persistent buffers to the thermo header**

Modify `src/analysis/thermo.cuh`. Current file:

```cuda
#pragma once
#include "../core/types.cuh"

#define STRESS_COMPONENTS 6

struct ThermoOutput {
    float temperature;
    float kinetic_energy;
    float potential_energy;
    float stress[6];
};

void compute_thermo(const float4* vel, const float4* force,
                     const float* virial, int natoms, float box_L,
                     ThermoOutput* h_output, cudaStream_t stream = 0);
```

Change to:

```cuda
#pragma once
#include "../core/types.cuh"

#define STRESS_COMPONENTS 6

struct ThermoOutput {
    float temperature;
    float kinetic_energy;
    float potential_energy;
    float stress[6];
};

struct ThermoBuffers {
    float* d_kin_stress;
    float* d_pe;
    bool allocated;

    void allocate();
    void free();
};

void compute_thermo(const float4* vel, const float4* force,
                     const float* virial, int natoms, float box_L,
                     ThermoOutput* h_output, ThermoBuffers& bufs,
                     cudaStream_t stream = 0);
```

- [ ] **Step 2: Implement ThermoBuffers in thermo.cu**

Add before the existing kernels in `src/analysis/thermo.cu`:

```cuda
void ThermoBuffers::allocate() {
    CUDA_CHECK(cudaMalloc(&d_kin_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pe, sizeof(float)));
    allocated = true;
}

void ThermoBuffers::free() {
    CUDA_CHECK(cudaFree(d_kin_stress));
    CUDA_CHECK(cudaFree(d_pe));
    allocated = false;
}
```

- [ ] **Step 3: Update compute_thermo to use persistent buffers**

In `src/analysis/thermo.cu`, replace the `compute_thermo` function body. Old code (lines 51-92):

```cuda
void compute_thermo(const float4* vel, const float4* force,
                     const float* virial, int natoms, float box_L,
                     ThermoOutput* h_output, cudaStream_t stream) {
    float* d_kin_stress;
    float* d_pe;
    CUDA_CHECK(cudaMalloc(&d_kin_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pe, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_kin_stress, 0, 6 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_pe, 0, sizeof(float), stream));

    int blocks = div_ceil(natoms, 256);
    kinetic_stress_kernel<<<blocks, 256, 0, stream>>>(vel, d_kin_stress, natoms);
    sum_pe_kernel<<<blocks, 256, 0, stream>>>(force, d_pe, natoms);

    float h_kin_stress[6];
    float h_pe;
    CUDA_CHECK(cudaMemcpyAsync(h_kin_stress, d_kin_stress, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_pe, d_pe, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    float h_virial[6];
    CUDA_CHECK(cudaMemcpyAsync(h_virial, virial, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ke = 0.5f * (h_kin_stress[0] + h_kin_stress[3] + h_kin_stress[5]);

    h_output->kinetic_energy = ke;
    h_output->potential_energy = h_pe;
    h_output->temperature = 2.0f * ke / (3.0f * natoms);

    float vol = box_L * box_L * box_L;
    float inv_vol = 1.0f / vol;
    for (int c = 0; c < 6; c++) {
        h_output->stress[c] = (h_kin_stress[c] + h_virial[c]) * inv_vol;
    }

    CUDA_CHECK(cudaFree(d_pe));
    CUDA_CHECK(cudaFree(d_kin_stress));
}
```

New code:

```cuda
void compute_thermo(const float4* vel, const float4* force,
                     const float* virial, int natoms, float box_L,
                     ThermoOutput* h_output, ThermoBuffers& bufs,
                     cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(bufs.d_kin_stress, 0, 6 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(bufs.d_pe, 0, sizeof(float), stream));

    int blocks = div_ceil(natoms, 256);
    kinetic_stress_kernel<<<blocks, 256, 0, stream>>>(vel, bufs.d_kin_stress, natoms);
    sum_pe_kernel<<<blocks, 256, 0, stream>>>(force, bufs.d_pe, natoms);

    float h_kin_stress[6];
    float h_pe;
    CUDA_CHECK(cudaMemcpyAsync(h_kin_stress, bufs.d_kin_stress, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_pe, bufs.d_pe, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    float h_virial[6];
    CUDA_CHECK(cudaMemcpyAsync(h_virial, virial, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ke = 0.5f * (h_kin_stress[0] + h_kin_stress[3] + h_kin_stress[5]);

    h_output->kinetic_energy = ke;
    h_output->potential_energy = h_pe;
    h_output->temperature = 2.0f * ke / (3.0f * natoms);

    float vol = box_L * box_L * box_L;
    float inv_vol = 1.0f / vol;
    for (int c = 0; c < 6; c++) {
        h_output->stress[c] = (h_kin_stress[c] + h_virial[c]) * inv_vol;
    }
}
```

- [ ] **Step 4: Update call sites in main.cu**

In `src/main.cu`, after `correlator.allocate();` (line 105), add:

```cuda
    ThermoBuffers thermo_bufs;
    thermo_bufs.allocate();
```

In the thermo output block (around line 199), update the call from:

```cuda
            compute_thermo(sys.vel, sys.force, sys.virial,
                            params.natoms, params.box_L, &thermo);
```

To:

```cuda
            compute_thermo(sys.vel, sys.force, sys.virial,
                            params.natoms, params.box_L, &thermo, thermo_bufs);
```

Before `dumper.close();` at cleanup, add:

```cuda
    thermo_bufs.free();
```

Also add `#include "analysis/thermo.cuh"` at the top if not already present (it is, line 11).

- [ ] **Step 5: Build and run tests**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc) && ctest --output-on-failure
```

Expected: build succeeds, all tests pass.

- [ ] **Step 6: Run benchmark validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: `"passed": true`.

- [ ] **Step 7: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/analysis/thermo.cuh src/analysis/thermo.cu src/main.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: persistent allocations for thermo compute buffers

Replace per-step cudaMalloc/cudaFree of d_kin_stress and d_pe with
persistent ThermoBuffers allocated once at startup.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Persistent allocation in binary dumper

**Files:**
- Modify: `src/io/dump.cu:35-57`

The `dump_frame` function allocates and frees `d_temp` (natoms * 12 bytes) on every dump frame. This is larger than the thermo buffers and more impactful.

- [ ] **Step 1: Add persistent d_temp to BinaryDumper**

Modify `src/io/dump.cuh`. Current header:

```cuda
#pragma once
#include "../core/types.cuh"

struct BinaryDumper {
    FILE* fp;
    float3* h_buf[2];
    int current_buf;
    int natoms;

    void open(const char* filename, int natoms, int ntypes);
    void close();
    void dump_frame(const float4* d_pos, int64_t step, float box_L,
                    cudaStream_t stream = 0);
};
```

Change to add `d_temp` pointer and allocate in `open`:

```cuda
#pragma once
#include "../core/types.cuh"

struct BinaryDumper {
    FILE* fp;
    float3* h_buf[2];
    float3* d_temp;
    int current_buf;
    int natoms;

    void open(const char* filename, int natoms, int ntypes);
    void close();
    void dump_frame(const float4* d_pos, int64_t step, float box_L,
                    cudaStream_t stream = 0);
};
```

- [ ] **Step 2: Allocate d_temp in open(), free in close()**

In `src/io/dump.cu`, modify `open()` — add after `CUDA_CHECK(cudaHostAlloc(&h_buf[1], ...))`:

```cuda
    CUDA_CHECK(cudaMalloc(&d_temp, natoms * sizeof(float3)));
```

Modify `close()` — add before `if (fp) fclose(fp)`:

```cuda
    CUDA_CHECK(cudaFree(d_temp));
```

- [ ] **Step 3: Remove per-frame allocation in dump_frame**

In `src/io/dump.cu`, modify `dump_frame()`. Old code (lines 35-57):

```cuda
void BinaryDumper::dump_frame(const float4* d_pos, int64_t step, float box_L,
                                cudaStream_t stream) {
    float3* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, natoms * sizeof(float3)));

    int blocks = div_ceil(natoms, 256);
    pack_float3<<<blocks, 256, 0, stream>>>(d_temp, d_pos, natoms);

    int buf_idx = current_buf;
    CUDA_CHECK(cudaMemcpyAsync(h_buf[buf_idx], d_temp,
                                natoms * sizeof(float3),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_temp));

    fwrite(&step, sizeof(int64_t), 1, fp);
    int32_t n = natoms;
    fwrite(&n, sizeof(int32_t), 1, fp);
    fwrite(&box_L, sizeof(float), 1, fp);
    fwrite(h_buf[buf_idx], sizeof(float3), natoms, fp);

    current_buf = 1 - current_buf;
}
```

New code:

```cuda
void BinaryDumper::dump_frame(const float4* d_pos, int64_t step, float box_L,
                                cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    pack_float3<<<blocks, 256, 0, stream>>>(d_temp, d_pos, natoms);

    int buf_idx = current_buf;
    CUDA_CHECK(cudaMemcpyAsync(h_buf[buf_idx], d_temp,
                                natoms * sizeof(float3),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    fwrite(&step, sizeof(int64_t), 1, fp);
    int32_t n = natoms;
    fwrite(&n, sizeof(int32_t), 1, fp);
    fwrite(&box_L, sizeof(float), 1, fp);
    fwrite(h_buf[buf_idx], sizeof(float3), natoms, fp);

    current_buf = 1 - current_buf;
}
```

- [ ] **Step 4: Build and run tests**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc) && ctest --output-on-failure
```

Expected: build succeeds, all tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/io/dump.cuh src/io/dump.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: persistent d_temp allocation in binary dumper

Replace per-frame cudaMalloc/cudaFree of the pack intermediate buffer
with a persistent allocation held for the dumper's lifetime.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Build atom-to-bond map and per-atom FENE kernel

**Context:** The current `fene_kernel` launches one thread per bond. Each bond thread does 9 `atomicAdd` operations (6 force components + 2 PE + virial). In a polymer chain, backbone atoms are shared by ~2 bonds each, so threads serialize on atomics. The fix: restructure so one thread handles all bonds involving a given atom. This requires a precomputed "which bonds touch each atom" map.

**Files:**
- Modify: `src/io/lammps_data.hpp` — add `atom_bond_offsets`, `atom_bond_list`
- Modify: `src/io/lammps_data.cpp` — build the mapping after parsing bonds
- Modify: `src/core/system.cuh` — add mapping array pointers
- Modify: `src/core/system.cu` — allocate/free mapping arrays
- Modify: `src/force/fene.cuh` — update launcher signature for per-atom kernel
- Modify: `src/force/fene.cu` — new per-atom kernel + launcher (full rewrite)
- Modify: `src/main.cu` — wire new signature

- [ ] **Step 1: Add atom-bond mapping to TopologyData**

In `src/io/lammps_data.hpp`, add to the `TopologyData` struct after `exclusion_list`:

```cuda
    // Atom-to-bond mapping for per-atom bonded force kernel
    std::vector<int> atom_bond_offsets;
    std::vector<int> atom_bond_list;
```

In the `TopologyData` constructor (or init), these vectors start empty.

- [ ] **Step 2: Build the mapping after parsing bonds**

In `src/io/lammps_data.cpp`, after `bonds` and `bond_types` are populated (after `parse_lammps_data` completes the bond parsing loop), add:

```cpp
    // Build atom-to-bond mapping
    int nbonds = static_cast<int>(topo.bonds.size());
    topo.atom_bond_offsets.assign(natoms + 1, 0);

    // Count bonds per atom
    for (int b = 0; b < nbonds; b++) {
        topo.atom_bond_offsets[topo.bonds[b].x]++;
        topo.atom_bond_offsets[topo.bonds[b].y]++;
    }

    // Prefix sum to get offsets
    int sum = 0;
    for (int i = 0; i < natoms; i++) {
        int count = topo.atom_bond_offsets[i];
        topo.atom_bond_offsets[i] = sum;
        sum += count;
    }
    topo.atom_bond_offsets[natoms] = sum;

    // Fill bond list
    topo.atom_bond_list.assign(sum, -1);
    std::vector<int> cursors = topo.atom_bond_offsets; // copy for write cursors
    for (int b = 0; b < nbonds; b++) {
        int i = topo.bonds[b].x;
        int j = topo.bonds[b].y;
        topo.atom_bond_list[cursors[i]++] = b;
        topo.atom_bond_list[cursors[j]++] = b;
    }
```

Note: `natoms` is available in the function scope as `positions.size()` or from the data file header. Check the exact variable name in `parse_lammps_data` and adjust. If `natoms` is `topo.natoms` or `topo.num_atoms`, use that.

- [ ] **Step 3: Add atom-bond mapping arrays to System struct**

In `src/core/system.cuh`, add to the `System` struct after `nexclusions`:

```cuda
    int*    atom_bond_offsets;
    int*    atom_bond_list;
    int     atom_bond_list_len;
```

- [ ] **Step 4: Allocate/free the mapping in System**

In `src/core/system.cu`, in `System::allocate()`, add after the exclusion allocation block:

```cuda
    atom_bond_offsets = nullptr;
    atom_bond_list = nullptr;
    atom_bond_list_len = 0;
```

In `main.cu`, after copying bond data to device (the existing bond allocation block around lines 37-44), add:

```cuda
        // Allocate and copy atom-to-bond mapping
        sys.atom_bond_list_len = static_cast<int>(topo.atom_bond_list.size());
        CUDA_CHECK(cudaMalloc(&sys.atom_bond_offsets,
                              (params.natoms + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&sys.atom_bond_list,
                              sys.atom_bond_list_len * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(sys.atom_bond_offsets, topo.atom_bond_offsets.data(),
                              (params.natoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sys.atom_bond_list, topo.atom_bond_list.data(),
                              sys.atom_bond_list_len * sizeof(int), cudaMemcpyHostToDevice));
```

In `System::free()`, add:

```cuda
    if (atom_bond_offsets) CUDA_CHECK(cudaFree(atom_bond_offsets));
    if (atom_bond_list) CUDA_CHECK(cudaFree(atom_bond_list));
```

- [ ] **Step 5: Rewrite the FENE kernel as per-atom (no atomicAdd)**

Replace the entire content of `src/force/fene.cu`:

```cuda
#include "fene.cuh"
#include "../core/pbc.cuh"

__global__ void fene_per_atom_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int2* __restrict__ bonds,
    const int* __restrict__ bond_types,
    const FENEParams* __restrict__ params,
    const int* __restrict__ atom_bond_offsets,
    const int* __restrict__ atom_bond_list,
    int natoms, float L, float inv_L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (i < natoms) {
        int b_start = atom_bond_offsets[i];
        int b_end = atom_bond_offsets[i + 1];

        for (int idx = b_start; idx < b_end; idx++) {
            int b = atom_bond_list[idx];
            int2 bond = bonds[b];
            int type = bond_types[b];
            FENEParams p = params[type];

            int j = (bond.x == i) ? bond.y : bond.x;
            bool i_am_x = (bond.x == i);

            float4 ri = pos[i];
            float4 rj = pos[j];

            float dx = min_image(ri.x - rj.x, L, inv_L);
            float dy = min_image(ri.y - rj.y, L, inv_L);
            float dz = min_image(ri.z - rj.z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;

            float R02 = p.R0 * p.R0;
            float r2_safe = fminf(fmaxf(r2, 1e-6f), R02 * 0.9999f);
            float fene_ff = -p.k * R02 / (R02 - r2_safe);

            float sig2 = p.sig * p.sig;
            float r_cut2 = sig2 * 1.2599210498948732f;
            float wca_ff = 0.0f;
            float wca_pe = 0.0f;
            if (r2_safe < r_cut2) {
                float r2inv = 1.0f / r2_safe;
                float sr2 = sig2 * r2inv;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                wca_ff = 24.0f * p.eps * r2inv * (2.0f * sr12 - sr6);
                wca_pe = 4.0f * p.eps * (sr12 - sr6) + p.eps;
            }

            float total_ff = fene_ff + wca_ff;

            // Force on atom i: sign depends on whether i is first or second in bond
            float sign = i_am_x ? 1.0f : -1.0f;
            fx += sign * total_ff * dx;
            fy += sign * total_ff * dy;
            fz += sign * total_ff * dz;

            float fene_pe = -0.5f * p.k * R02 * logf(1.0f - r2_safe / R02);
            float pair_pe = fene_pe + wca_pe;
            pe_i += 0.5f * pair_pe;

            // Virial: only accumulate once per bond (when i is x, the first atom)
            if (i_am_x) {
                float fix = total_ff * dx;
                float fiy = total_ff * dy;
                float fiz = total_ff * dz;
                v0 += fix * dx;
                v1 += fix * dy;
                v2 += fix * dz;
                v3 += fiy * dy;
                v4 += fiy * dz;
                v5 += fiz * dz;
            }
        }

        force[i] = make_float4(fx, fy, fz, pe_i);
    }

    // Block-level virial reduction (same pattern as current kernel)
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xFFFFFFFF, v0, offset);
        v1 += __shfl_down_sync(0xFFFFFFFF, v1, offset);
        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
        v3 += __shfl_down_sync(0xFFFFFFFF, v3, offset);
        v4 += __shfl_down_sync(0xFFFFFFFF, v4, offset);
        v5 += __shfl_down_sync(0xFFFFFFFF, v5, offset);
    }

    extern __shared__ float s_virial[];
    if (lane == 0) {
        int wid = threadIdx.x >> 5;
        s_virial[wid * 6 + 0] = v0;
        s_virial[wid * 6 + 1] = v1;
        s_virial[wid * 6 + 2] = v2;
        s_virial[wid * 6 + 3] = v3;
        s_virial[wid * 6 + 4] = v4;
        s_virial[wid * 6 + 5] = v5;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x >> 5;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
        for (int w = 0; w < num_warps; w++) {
            s0 += s_virial[w * 6 + 0];
            s1 += s_virial[w * 6 + 1];
            s2 += s_virial[w * 6 + 2];
            s3 += s_virial[w * 6 + 3];
            s4 += s_virial[w * 6 + 4];
            s5 += s_virial[w * 6 + 5];
        }
        atomicAdd(&virial[0], s0);
        atomicAdd(&virial[1], s1);
        atomicAdd(&virial[2], s2);
        atomicAdd(&virial[3], s3);
        atomicAdd(&virial[4], s4);
        atomicAdd(&virial[5], s5);
    }
}

void launch_fene_kernel(const float4* pos, float4* force, float* virial,
                         const int2* bonds, const int* bond_types,
                         const FENEParams* params,
                         int nbonds, int nparamtypes,
                         float L, float inv_L, cudaStream_t stream) {
    (void)nbonds;       // no longer needed for launch config
    (void)nparamtypes;  // no longer needed for launch config

    // Launch one thread per atom
    int natoms;  // We need natoms — pass it or extract from force array context
    // The caller knows natoms. We'll read it from a new parameter.
    // For now, we get natoms from the atom_bond_offsets context — but that's
    // passed to the kernel, not the launcher. The launcher needs natoms
    // for grid size. Update the header signature accordingly.

    // See Step 6 — the header signature change adds `int natoms` param
}
```

Wait — there's an issue. The launcher needs `natoms` for grid size but currently receives `nbonds`. We need to update the launcher signature. Let me handle that in the header.

- [ ] **Step 6: Update FENE header signature**

Replace `src/force/fene.cuh`:

```cuda
#pragma once
#include "../core/types.cuh"

struct FENEParams {
    float k;
    float R0;
    float eps;
    float sig;
};

void launch_fene_kernel(const float4* __restrict__ pos,
                         float4* __restrict__ force,
                         float* __restrict__ virial,
                         const int2* __restrict__ bonds,
                         const int* __restrict__ bond_types,
                         const FENEParams* __restrict__ params,
                         const int* __restrict__ atom_bond_offsets,
                         const int* __restrict__ atom_bond_list,
                         int natoms, float L, float inv_L,
                         cudaStream_t stream = 0);
```

- [ ] **Step 7: Rewrite the launcher**

Replace the `launch_fene_kernel` function body at the bottom of `src/force/fene.cu`:

```cuda
void launch_fene_kernel(const float4* pos, float4* force, float* virial,
                         const int2* bonds, const int* bond_types,
                         const FENEParams* params,
                         const int* atom_bond_offsets,
                         const int* atom_bond_list,
                         int natoms, float L, float inv_L,
                         cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    int smem = (256 / 32) * 6 * sizeof(float);
    fene_per_atom_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, bonds, bond_types, params,
        atom_bond_offsets, atom_bond_list,
        natoms, L, inv_L);
}
```

- [ ] **Step 8: Update main.cu call sites**

In `src/main.cu`, update the two `launch_fene_kernel` calls.

Old call at line 135-138 (pre-loop init):
```cuda
        launch_fene_kernel(sys.pos, sys.force, sys.virial,
                            sys.bonds, sys.bond_param_idx, d_fene_params,
                            sys.nbonds, 1, params.box_L, params.inv_L);
```

New call:
```cuda
        launch_fene_kernel(sys.pos, sys.force, sys.virial,
                            sys.bonds, sys.bond_param_idx, d_fene_params,
                            sys.atom_bond_offsets, sys.atom_bond_list,
                            params.natoms, params.box_L, params.inv_L);
```

Old call at lines 177-181 (per-step loop):
```cuda
            launch_fene_kernel(sys.pos, sys.force, sys.virial,
                                sys.bonds, sys.bond_param_idx, d_fene_params,
                                sys.nbonds, 1, params.box_L, params.inv_L,
                                stream_bonded);
```

New call:
```cuda
            launch_fene_kernel(sys.pos, sys.force, sys.virial,
                                sys.bonds, sys.bond_param_idx, d_fene_params,
                                sys.atom_bond_offsets, sys.atom_bond_list,
                                params.natoms, params.box_L, params.inv_L,
                                stream_bonded);
```

- [ ] **Step 9: Update test_fene.cu to build atom-bond map and use new signature**

In `tests/test_fene.cu`, after copying `h_bonds` and `h_types` to device (around line 41), build the atom-bond map and update the kernel call.

Add after the `cudaMemcpy(d_params, ...)` block:

```cuda
    // Build atom-to-bond mapping for per-atom kernel
    int natoms = N;
    std::vector<int> atom_bond_offsets(natoms + 1, 0);
    for (int b = 0; b < 2; b++) {
        atom_bond_offsets[h_bonds[b].x]++;
        atom_bond_offsets[h_bonds[b].y]++;
    }
    int sum = 0;
    for (int i = 0; i < natoms; i++) {
        int cnt = atom_bond_offsets[i];
        atom_bond_offsets[i] = sum;
        sum += cnt;
    }
    atom_bond_offsets[natoms] = sum;
    std::vector<int> atom_bond_list(sum, -1);
    std::vector<int> cursors = atom_bond_offsets;
    for (int b = 0; b < 2; b++) {
        atom_bond_list[cursors[h_bonds[b].x]++] = b;
        atom_bond_list[cursors[h_bonds[b].y]++] = b;
    }
    int* d_atom_bond_offsets = to_device(atom_bond_offsets);
    int* d_atom_bond_list = to_device(atom_bond_list);
```

Replace the `launch_fene_kernel` call line (lines 43-44):

```cuda
    launch_fene_kernel(d_pos, d_force, d_virial, d_bonds, d_btypes, d_params,
                        d_atom_bond_offsets, d_atom_bond_list,
                        natoms, L, inv_L);
```

Add cleanup lines after the kernel call:

```cuda
    free_device(d_atom_bond_offsets);
    free_device(d_atom_bond_list);
```

The `nparamtypes` argument (previously `1`) is removed from the signature.

- [ ] **Step 10: Build and run FENE unit test**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc) && ./tests/test_fene
```

Expected: build succeeds, FENE test passes (forces match CPU reference within tolerance).

If the test fails, check:
- That `atom_bond_offsets` and `atom_bond_list` are correctly copied to device
- That the `i_am_x` sign logic is correct
- That virial accumulation only happens once per bond

- [ ] **Step 11: Run all unit tests**

```bash
cd /home/zhenghaowu/fastMD/build && ctest --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 12: Run benchmark validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: `"passed": true`.

- [ ] **Step 13: Commit**

```bash
git -C /home/zhenghaowu/fastMD add \
  src/force/fene.cuh src/force/fene.cu \
  src/core/system.cuh src/core/system.cu \
  src/io/lammps_data.hpp src/io/lammps_data.cpp \
  src/main.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: per-atom FENE kernel with atom-bond mapping

Replace thread-per-bond FENE kernel (9 atomicAdd/bond) with per-atom
kernel that iterates atom_bond_list without atomics. Build atom-to-bond
mapping in TopologyData during data parsing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Per-atom angle kernel (no atomicAdd)

**Context:** Same pattern as Task 7 but for angles. Each 3-body angle (i,j,k) currently does 12 atomicAdd. With per-atom processing, each thread handles all angles involving its atom.

**Files:**
- Modify: `src/io/lammps_data.hpp` — add `atom_angle_offsets`, `atom_angle_list`
- Modify: `src/io/lammps_data.cpp` — build the mapping
- Modify: `src/core/system.cuh` — add mapping pointers
- Modify: `src/core/system.cu` — allocate/free
- Modify: `src/force/angle.cuh` — update launcher signature
- Modify: `src/force/angle.cu` — new per-atom kernel
- Modify: `src/main.cu` — wire new signature

- [ ] **Step 1: Add atom-angle mapping to TopologyData**

In `src/io/lammps_data.hpp`, add after `atom_bond_list`:

```cuda
    // Atom-to-angle mapping for per-atom angle force kernel
    std::vector<int> atom_angle_offsets;
    std::vector<int> atom_angle_list;
```

- [ ] **Step 2: Build atom-angle mapping in lammps_data.cpp**

In `src/io/lammps_data.cpp`, after the atom-bond mapping build, add:

```cpp
    // Build atom-to-angle mapping
    int nangles = static_cast<int>(topo.angles.size());
    topo.atom_angle_offsets.assign(natoms + 1, 0);

    for (int a = 0; a < nangles; a++) {
        topo.atom_angle_offsets[topo.angles[a].x]++;
        topo.atom_angle_offsets[topo.angles[a].y]++;
        topo.atom_angle_offsets[topo.angles[a].z]++;
    }

    int sum = 0;
    for (int i = 0; i < natoms; i++) {
        int count = topo.atom_angle_offsets[i];
        topo.atom_angle_offsets[i] = sum;
        sum += count;
    }
    topo.atom_angle_offsets[natoms] = sum;

    topo.atom_angle_list.assign(sum, -1);
    std::vector<int> cursors = topo.atom_angle_offsets;
    for (int a = 0; a < nangles; a++) {
        int i = topo.angles[a].x;
        int j = topo.angles[a].y;
        int k = topo.angles[a].z;
        topo.atom_angle_list[cursors[i]++] = a;
        topo.atom_angle_list[cursors[j]++] = a;
        topo.atom_angle_list[cursors[k]++] = a;
    }
```

- [ ] **Step 3: Add to System struct**

In `src/core/system.cuh`, after `atom_bond_list_len`:

```cuda
    int*    atom_angle_offsets;
    int*    atom_angle_list;
    int     atom_angle_list_len;
```

- [ ] **Step 4: Allocate in System / main.cu**

In `src/core/system.cu`, in `allocate()`:

```cuda
    atom_angle_offsets = nullptr;
    atom_angle_list = nullptr;
    atom_angle_list_len = 0;
```

In `main.cu`, after the atom-bond mapping allocation block (Task 7 Step 4), add:

```cuda
        sys.atom_angle_list_len = static_cast<int>(topo.atom_angle_list.size());
        CUDA_CHECK(cudaMalloc(&sys.atom_angle_offsets,
                              (params.natoms + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&sys.atom_angle_list,
                              sys.atom_angle_list_len * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(sys.atom_angle_offsets, topo.atom_angle_offsets.data(),
                              (params.natoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sys.atom_angle_list, topo.atom_angle_list.data(),
                              sys.atom_angle_list_len * sizeof(int), cudaMemcpyHostToDevice));
```

In `System::free()`, add:

```cuda
    if (atom_angle_offsets) CUDA_CHECK(cudaFree(atom_angle_offsets));
    if (atom_angle_list) CUDA_CHECK(cudaFree(atom_angle_list));
```

- [ ] **Step 5: Rewrite the angle kernel as per-atom**

Replace the entire content of `src/force/angle.cu`:

```cuda
#include "angle.cuh"
#include "../core/pbc.cuh"

__global__ void angle_per_atom_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int4* __restrict__ angles,
    const AngleParams* __restrict__ params,
    const int* __restrict__ atom_angle_offsets,
    const int* __restrict__ atom_angle_list,
    int natoms, float L, float inv_L)
{
    int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_a = 0.0f;
    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (a_idx < natoms) {
        int a_start = atom_angle_offsets[a_idx];
        int a_end = atom_angle_offsets[a_idx + 1];

        for (int idx = a_start; idx < a_end; idx++) {
            int a = atom_angle_list[idx];
            int4 ang = angles[a];
            int i = ang.x, j = ang.y, k = ang.z, type = ang.w;
            AngleParams p = params[type];

            float4 ri = pos[i], rj = pos[j], rk = pos[k];

            float dxij = min_image(ri.x - rj.x, L, inv_L);
            float dyij = min_image(ri.y - rj.y, L, inv_L);
            float dzij = min_image(ri.z - rj.z, L, inv_L);
            float dxkj = min_image(rk.x - rj.x, L, inv_L);
            float dykj = min_image(rk.y - rj.y, L, inv_L);
            float dzkj = min_image(rk.z - rj.z, L, inv_L);

            float rij2 = dxij*dxij + dyij*dyij + dzij*dzij;
            float rkj2 = dxkj*dxkj + dykj*dykj + dzkj*dzkj;
            float rij_inv = rsqrtf(rij2);
            float rkj_inv = rsqrtf(rkj2);
            float rij = rij2 * rij_inv;
            float rkj = rkj2 * rkj_inv;

            float cos_theta = (dxij*dxkj + dyij*dykj + dzij*dzkj) * rij_inv * rkj_inv;
            cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
            float theta = acosf(cos_theta);
            float dtheta = theta - p.theta0;
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            sin_theta = fmaxf(sin_theta, 1e-6f);

            float prefactor = 2.0f * p.k_theta * dtheta / sin_theta;

            float rij2_inv = 1.0f / rij2;
            float rkj2_inv = 1.0f / rkj2;
            float rij_rkj_inv = rij_inv * rkj_inv;

            float fi_x = prefactor * (dxkj * rij_rkj_inv - cos_theta * dxij * rij2_inv);
            float fi_y = prefactor * (dykj * rij_rkj_inv - cos_theta * dyij * rij2_inv);
            float fi_z = prefactor * (dzkj * rij_rkj_inv - cos_theta * dzij * rij2_inv);
            float fk_x = prefactor * (dxij * rij_rkj_inv - cos_theta * dxkj * rkj2_inv);
            float fk_y = prefactor * (dyij * rij_rkj_inv - cos_theta * dykj * rkj2_inv);
            float fk_z = prefactor * (dzij * rij_rkj_inv - cos_theta * dzkj * rkj2_inv);

            // Accumulate force for this atom based on its role
            if (a_idx == i) {
                fx += fi_x; fy += fi_y; fz += fi_z;
            } else if (a_idx == j) {
                fx += -(fi_x + fk_x);
                fy += -(fi_y + fk_y);
                fz += -(fi_z + fk_z);
            } else { // a_idx == k
                fx += fk_x; fy += fk_y; fz += fk_z;
            }

            float pe = p.k_theta * dtheta * dtheta;
            pe_a += pe / 3.0f;

            // Virial: accumulate once per angle (when a_idx == i)
            if (a_idx == i) {
                v0 += fi_x * dxij + fk_x * dxkj;
                v1 += fi_x * dyij + fk_x * dykj;
                v2 += fi_x * dzij + fk_x * dzkj;
                v3 += fi_y * dyij + fk_y * dykj;
                v4 += fi_y * dzij + fk_y * dzkj;
                v5 += fi_z * dzij + fk_z * dzkj;
            }
        }

        force[a_idx] = make_float4(fx, fy, fz, pe_a);
    }

    // Block-level virial reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xFFFFFFFF, v0, offset);
        v1 += __shfl_down_sync(0xFFFFFFFF, v1, offset);
        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
        v3 += __shfl_down_sync(0xFFFFFFFF, v3, offset);
        v4 += __shfl_down_sync(0xFFFFFFFF, v4, offset);
        v5 += __shfl_down_sync(0xFFFFFFFF, v5, offset);
    }

    extern __shared__ float s_virial[];
    if (lane == 0) {
        int wid = threadIdx.x >> 5;
        s_virial[wid * 6 + 0] = v0;
        s_virial[wid * 6 + 1] = v1;
        s_virial[wid * 6 + 2] = v2;
        s_virial[wid * 6 + 3] = v3;
        s_virial[wid * 6 + 4] = v4;
        s_virial[wid * 6 + 5] = v5;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x >> 5;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
        for (int w = 0; w < num_warps; w++) {
            s0 += s_virial[w * 6 + 0];
            s1 += s_virial[w * 6 + 1];
            s2 += s_virial[w * 6 + 2];
            s3 += s_virial[w * 6 + 3];
            s4 += s_virial[w * 6 + 4];
            s5 += s_virial[w * 6 + 5];
        }
        atomicAdd(&virial[0], s0);
        atomicAdd(&virial[1], s1);
        atomicAdd(&virial[2], s2);
        atomicAdd(&virial[3], s3);
        atomicAdd(&virial[4], s4);
        atomicAdd(&virial[5], s5);
    }
}

void launch_angle_kernel(const float4* pos, float4* force, float* virial,
                          const int4* angles, const AngleParams* params,
                          const int* atom_angle_offsets,
                          const int* atom_angle_list,
                          int natoms, float L, float inv_L,
                          cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    int smem = (256 / 32) * 6 * sizeof(float);
    angle_per_atom_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, angles, params,
        atom_angle_offsets, atom_angle_list,
        natoms, L, inv_L);
}
```

- [ ] **Step 6: Update angle header signature**

Replace `src/force/angle.cuh`:

```cuda
#pragma once
#include "../core/types.cuh"

struct AngleParams {
    float k_theta;
    float theta0;
};

void launch_angle_kernel(const float4* __restrict__ pos,
                          float4* __restrict__ force,
                          float* __restrict__ virial,
                          const int4* __restrict__ angles,
                          const AngleParams* __restrict__ params,
                          const int* __restrict__ atom_angle_offsets,
                          const int* __restrict__ atom_angle_list,
                          int natoms, float L, float inv_L,
                          cudaStream_t stream = 0);
```

- [ ] **Step 7: Update main.cu call sites** (for angle kernel)

Old pre-loop call (lines 140-143):
```cuda
        launch_angle_kernel(sys.pos, sys.force, sys.virial,
                             sys.angles, d_angle_params,
                             sys.nangles, params.box_L, params.inv_L);
```

New:
```cuda
        launch_angle_kernel(sys.pos, sys.force, sys.virial,
                             sys.angles, d_angle_params,
                             sys.atom_angle_offsets, sys.atom_angle_list,
                             params.natoms, params.box_L, params.inv_L);
```

Old per-step loop call (lines 183-186):
```cuda
            launch_angle_kernel(sys.pos, sys.force, sys.virial,
                                 sys.angles, d_angle_params,
                                 sys.nangles, params.box_L, params.inv_L,
                                 stream_bonded);
```

New:
```cuda
            launch_angle_kernel(sys.pos, sys.force, sys.virial,
                                 sys.angles, d_angle_params,
                                 sys.atom_angle_offsets, sys.atom_angle_list,
                                 params.natoms, params.box_L, params.inv_L,
                                 stream_bonded);
```

- [ ] **Step 8: Update test_angle.cu to build atom-angle map and use new signature**

In `tests/test_angle.cu`, after copying `h_angles` and `h_params` to device (around line 41), build the atom-angle map and update the kernel call.

Add after the `cudaMemcpy(d_params, ...)` block:

```cuda
    // Build atom-to-angle mapping for per-atom kernel
    int natoms = N;
    std::vector<int> atom_angle_offsets(natoms + 1, 0);
    for (int a = 0; a < 1; a++) {
        atom_angle_offsets[h_angles[a].x]++;
        atom_angle_offsets[h_angles[a].y]++;
        atom_angle_offsets[h_angles[a].z]++;
    }
    int sum = 0;
    for (int i = 0; i < natoms; i++) {
        int cnt = atom_angle_offsets[i];
        atom_angle_offsets[i] = sum;
        sum += cnt;
    }
    atom_angle_offsets[natoms] = sum;
    std::vector<int> atom_angle_list(sum, -1);
    std::vector<int> cursors = atom_angle_offsets;
    for (int a = 0; a < 1; a++) {
        atom_angle_list[cursors[h_angles[a].x]++] = a;
        atom_angle_list[cursors[h_angles[a].y]++] = a;
        atom_angle_list[cursors[h_angles[a].z]++] = a;
    }
    int* d_atom_angle_offsets = to_device(atom_angle_offsets);
    int* d_atom_angle_list = to_device(atom_angle_list);
```

Replace the `launch_angle_kernel` call (lines 42-43):

```cuda
    launch_angle_kernel(d_pos, d_force, d_virial, d_angles, d_params,
                         d_atom_angle_offsets, d_atom_angle_list,
                         natoms, L, inv_L);
```

Add cleanup lines after the kernel call:

```cuda
    free_device(d_atom_angle_offsets);
    free_device(d_atom_angle_list);
```

- [ ] **Step 9: Build and run angle unit test**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc) && ./tests/test_angle
```

Expected: build succeeds, angle test passes.

- [ ] **Step 10: Run all tests and benchmark validation**

```bash
cd /home/zhenghaowu/fastMD/build && ctest --output-on-failure
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: all tests pass, validation passes.

- [ ] **Step 11: Commit**

```bash
git -C /home/zhenghaowu/fastMD add \
  src/force/angle.cuh src/force/angle.cu \
  src/core/system.cuh src/core/system.cu \
  src/io/lammps_data.hpp src/io/lammps_data.cpp \
  src/main.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: per-atom angle kernel with atom-angle mapping

Replace thread-per-angle kernel (12 atomicAdd/angle) with per-atom
kernel that iterates atom_angle_list without atomics.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: LJ kernel micro-optimizations

**Context:** Execute this task only if `lj_verlet_kernel` is still the #1 or #2 kernel after the Verlet list implementation. The kernel is already lean, so improvements are small and targeted.

**Files:**
- Modify: `src/force/lj.cu`

- [ ] **Step 1: Add `__ldg` for all read-only global loads**

In `src/force/lj.cu`, in `lj_verlet_kernel`, change the `pos[j]` load to use `__ldg`:

Old (line 29):
```cuda
            float4 pos_j = pos[j];
```

New:
```cuda
            float4 pos_j = __ldg(&pos[j]);
```

Also add `__ldg` to the `num_neighbors[i]` read (line 25):
```cuda
        int nneigh = __ldg(&num_neighbors[i]);
```

- [ ] **Step 2: Precompute rcp for rc2**

The `r2 > 1e-10f` check is already in place. Add an early cut check before the expensive LJ computation by pre-checking `r2 < rc2` using the already-loaded `rc2` parameter — this is already done at line 37. No change needed.

- [ ] **Step 3: Use `__fdividef` for division**

In the LJ computation, replace `1.0f / r2` with `__fdividef(1.0f, r2)` for faster division on GPU:

Old (line 41):
```cuda
                float r2inv = 1.0f / r2;
```

New:
```cuda
                float r2inv = __fdividef(1.0f, r2);
```

`__fdividef` is ~3x faster than `/` and with `--use_fast_math` the precision is already relaxed.

- [ ] **Step 4: Build and run LJ test**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc) && ./tests/test_lj
```

Expected: build succeeds, LJ test passes.

- [ ] **Step 5: Run full test suite and benchmark validation**

```bash
cd /home/zhenghaowu/fastMD/build && ctest --output-on-failure
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/force/lj.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: LJ kernel micro-optimizations (__ldg, __fdividef)

Add __ldg for scattered pos[j] loads and num_neighbors read.
Replace 1.0f/r2 with __fdividef for faster division.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Replace curand_normal4 with Box-Muller in integrator

**Context:** Execute this task only if `integrator_pre_force_kernel` is in the top 3. `curand_normal4` generates 4 Gaussian random numbers, which internally uses Box-Muller on uniform randoms. Switching to `curand_uniform4` + explicit Box-Muller may save the transform overhead. A more aggressive option: pre-generate random numbers into a buffer.

**Files:**
- Modify: `src/integrate/langevin.cu:32-76` (integrator_pre_force_kernel)

- [ ] **Step 1: Replace curand_normal4 with curand_uniform4 + Box-Muller**

In `src/integrate/langevin.cu`, in `integrator_pre_force_kernel`, replace lines 57-63:

Old:
```cuda
    curandStatePhilox4_32_10_t local_state = rng_states[i];
    float4 rand4 = curand_normal4(&local_state);
    rng_states[i] = local_state;

    float noise_scale = c2 * sqrtf(kT);
    v.x = c1 * v.x + noise_scale * rand4.x;
    v.y = c1 * v.y + noise_scale * rand4.y;
    v.z = c1 * v.z + noise_scale * rand4.z;
```

New:
```cuda
    curandStatePhilox4_32_10_t local_state = rng_states[i];
    float4 u = curand_uniform4(&local_state);
    rng_states[i] = local_state;

    // Box-Muller: convert uniform(0,1) pairs to standard normal
    // u.x, u.y -> two Gaussians; u.z, u.w -> another two
    float log_u1 = logf(u.x > 1e-10f ? u.x : 1e-10f);
    float sqrt_term1 = sqrtf(-2.0f * log_u1);
    float n1 = sqrt_term1 * cosf(2.0f * CUDART_PI_F * u.y);
    float n2 = sqrt_term1 * sinf(2.0f * CUDART_PI_F * u.y);

    float log_u2 = logf(u.z > 1e-10f ? u.z : 1e-10f);
    float sqrt_term2 = sqrtf(-2.0f * log_u2);
    float n3 = sqrt_term2 * cosf(2.0f * CUDART_PI_F * u.w);
    float n4 = sqrt_term2 * sinf(2.0f * CUDART_PI_F * u.w);

    float noise_scale = c2 * sqrtf(kT);
    v.x = c1 * v.x + noise_scale * n1;
    v.y = c1 * v.y + noise_scale * n2;
    v.z = c1 * v.z + noise_scale * n3;
    // n4 is unused — we only need 3 normals per atom
```

Note: `CUDART_PI_F` is defined in `math_constants.h`. Add `#include <math_constants.h>` at the top of `langevin.cu` if not already present.

- [ ] **Step 2: Build and run integrator test**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc) && ./tests/test_integrator
```

Expected: build succeeds, integrator test passes. Note that the random sequence changes, so exact trajectory values will differ, but statistical properties (temperature, energy conservation) should match within tolerance.

- [ ] **Step 3: Run benchmark validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: `"passed": true` (temperature and PE within 5% and 1% of LAMMPS respectively).

- [ ] **Step 4: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/integrate/langevin.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: replace curand_normal4 with curand_uniform4 + Box-Muller

Box-Muller transform on uniform randoms avoids the overhead of
curand_normal4's internal Gaussian generation path.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Verlet list build optimizations

**Context:** Execute this task only if `build_verlet_list` or `assign_cells` appears in the top 3 kernels.

**Files:**
- Modify: `src/neighbor/verlet_list.cu`

- [ ] **Step 1: Replace floorf with __float2int_rd in assign_cells**

In `assign_cells` kernel, change the cell coordinate computation (lines 58-60):

Old:
```cuda
    int cx = int(floorf(r.x * inv_cell_size));
    int cy = int(floorf(r.y * inv_cell_size));
    int cz = int(floorf(r.z * inv_cell_size));
```

New:
```cuda
    int cx = __float2int_rd(r.x * inv_cell_size);
    int cy = __float2int_rd(r.y * inv_cell_size);
    int cz = __float2int_rd(r.z * inv_cell_size);
```

`__float2int_rd` (round toward negative infinity) is hardware-accelerated and avoids the `floorf` math library call.

- [ ] **Step 2: Use __ldg for scattered pos loads in build_verlet_list**

In `build_verlet_list`, the `pos[j]` load inside the inner loop (line 167) is a scattered read from global memory. Add `__ldg`:

Old (line 167):
```cuda
                            float4 pos_j = pos[j];
```

New (both in dedup and non-dedup paths):
```cuda
                            float4 pos_j = __ldg(&pos[j]);
```

- [ ] **Step 3: Remove redundant PBC wrap in cell coordinate calculation**

The `assign_cells` kernel already accounts for PBC via the mod operation. No change needed — already correct.

- [ ] **Step 4: Build and run Verlet list test (if exists)**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 5: Run benchmark validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: `"passed": true`.

- [ ] **Step 6: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/neighbor/verlet_list.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: Verlet build micro-optimizations (__float2int_rd, __ldg)

Replace floorf with __float2int_rd in cell assignment and add __ldg
for scattered pos[j] loads in the build kernel.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Final benchmark validation and report

**Files:**
- Modify: `benchmark_report.json` (regenerated)

- [ ] **Step 1: Run the full benchmark suite**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py
```

Expected: validation passes, benchmark completes.

- [ ] **Step 2: Compare with baseline and record improvement**

Read `benchmark_report.json` and compare `speedup` and `fastMD.wall_time_s` with the pre-optimization baseline from Task 1.

Calculate:
- Step-time reduction: `(old_wall_time - new_wall_time) / old_wall_time * 100%`
- Speedup vs LAMMPS change: `new_speedup - old_speedup`

- [ ] **Step 3: Run a quick nsys profile to confirm the new profile**

```bash
cd /home/zhenghaowu/fastMD && nsys profile --stats=true --force-overwrite=true -o fastmd_nsys_optimized ./build/fastMD fastmd_benchmark.conf 2>&1 | grep -A 50 "CUDA Kernel Statistics"
```

Expected: The top kernels' % share should have shifted. Optimized kernels should take less total time.

- [ ] **Step 4: Commit the updated benchmark report**

```bash
git -C /home/zhenghaowu/fastMD add benchmark_report.json
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
bench: update report with post-optimization results

Profile-guided kernel optimizations delivered step-time reduction on
30K-atom polymer system vs LAMMPS baseline.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Execution Notes

**Task ordering:** Tasks 1-4 MUST run first. Task 4 Step 3 determines which of Tasks 5-11 execute:
- Always: Tasks 5, 6 (persistent allocations — safe, low risk)
- If FENE is a top bottleneck: Tasks 7
- If Angle is a top bottleneck: Task 8
- If LJ is a top bottleneck: Task 9
- If integrator is a top bottleneck: Task 10
- If Verlet build is a top bottleneck: Task 11
- Always: Task 12 (final validation)

**If profiling shows an unexpected kernel as bottleneck:** Add a new optimization task before proceeding. Don't force-fit an existing task to a different problem.

**Rollback safety:** Each optimization task is an independent commit. If any task regresses validation, `git revert` that commit and continue with the next.
