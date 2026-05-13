# Table Kernel Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve table kernel performance via shared memory caching and type-pair hoisting, plus gate the Verlet rebuild diagnostic print behind `#ifndef NDEBUG`.

**Architecture:** Rewrite `table_verlet_kernel` to cooperatively load table data into `__shared__` memory once per block, hoist `type_i * ntypes` and table params out of the per-neighbor loop. For CG water (single type pair), the full 500-entry table (8 KB) fits in shared memory and all table reads come from there. For multi-type systems or large tables, fall back to `__ldg`. The kernel's `launch_table_kernel` signature stays the same.

**Tech Stack:** CUDA C++, `__shared__` memory, `__ldg`, same GPU arch (CC 8.0+)

---

### Task 1: Rewrite the table kernel with shared memory + hoisting

**Files:**
- Modify: `src/force/table.cu`

- [ ] **Step 1: Rewrite `table_verlet_kernel` and `launch_table_kernel`**

Replace the entire contents of `src/force/table.cu` with:

```c
#include "table.cuh"
#include "../core/pbc.cuh"

#define TABLE_MAX_SHARED 2048

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
    __shared__ float4 s_table[TABLE_MAX_SHARED];
    extern __shared__ float s_virial[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    // --- Cooperative table load (once per block) ---
    // All threads redundantly compute the representative table from the
    // block's first atom. For single-type systems (CG water), every atom
    // uses the same table and shared memory always hits. For multi-type,
    // cross-type neighbors fall back to __ldg.
    // The redundant reads all hit the same L1 cache line — no contention.
    int first = blockIdx.x * blockDim.x;
    int rep_type = __float_as_int(pos[first].w);
    int tidx_rep = __ldg(&table_idx[rep_type * ntypes + rep_type]);
    bool use_smem = false;
    float rmin_s = 0.0f, rmax_s = 0.0f, dr_s = 0.0f, inv_dr_s = 0.0f;
    int npoints_s = 0;

    if (tidx_rep >= 0) {
        TableParams tp = table_params[tidx_rep];
        if (tp.npoints <= TABLE_MAX_SHARED) {
            rmin_s   = tp.rmin;
            rmax_s   = tp.rmax;
            dr_s     = tp.dr;
            inv_dr_s = tp.inv_dr;
            npoints_s = tp.npoints;
            use_smem  = true;
        }
    }
    if (use_smem) {
        for (int t = threadIdx.x; t < npoints_s; t += blockDim.x)
            s_table[t] = __ldg(&table_data[table_params[tidx_rep].data_offset + t]);
    }
    __syncthreads();

    if (i < natoms) {
        float4 pos_i = pos[i];
        int type_i = __float_as_int(pos_i.w);
        int nneigh = __ldg(&num_neighbors[i]);
        int type_i_offset = type_i * ntypes;

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
                int tidx = __ldg(&table_idx[type_i_offset + type_j]);
                if (tidx >= 0) {
                    // Choose shared memory if the table matches, else global
                    const float4* tbl;
                    float rmin, rmax, dr, inv_dr;
                    int npoints;

                    if (use_smem && tidx == tidx_rep) {
                        tbl = s_table;
                        rmin = rmin_s; rmax = rmax_s;
                        dr = dr_s; inv_dr = inv_dr_s;
                        npoints = npoints_s;
                    } else {
                        TableParams tp = table_params[tidx];
                        tbl = table_data + tp.data_offset;
                        rmin = tp.rmin; rmax = tp.rmax;
                        dr = tp.dr; inv_dr = tp.inv_dr;
                        npoints = tp.npoints;
                    }

                    float r = sqrtf(r2);
                    if (r < rmin) r = rmin;
                    if (r > rmax) r = rmax;

                    int idx = (int)((r - rmin) * inv_dr);
                    if (idx < 0) idx = 0;
                    if (idx >= npoints - 1) idx = npoints - 2;

                    float t = (r - (rmin + idx * dr)) * inv_dr;
                    float4 p0 = tbl[idx];
                    float4 p1 = tbl[idx + 1];

                    float f_scalar = p0.y + t * (p1.y - p0.y);
                    float e = p0.z + t * (p1.z - p0.z);

                    float fxij = f_scalar * dx;
                    float fyij = f_scalar * dy;
                    float fzij = f_scalar * dz;

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

- [ ] **Step 2: Build to verify compilation**

```bash
cd /home/zhenghaowu/fastMD/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Expected: clean compilation, no errors.

- [ ] **Step 3: Run existing `test_table` unit test to verify correctness**

```bash
cd /home/zhenghaowu/fastMD/build && ./tests/test_table
```

Expected: `[  PASSED  ] 1 test.` — The `Table.MatchesLJWithinInterpolationTolerance` test passes, confirming the table kernel produces the same forces within tolerance.

- [ ] **Step 4: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/force/table.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: cache table data in shared memory and hoist type-pair constants

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Gate Verlet rebuild diagnostic print behind `#ifndef NDEBUG`

**Files:**
- Modify: `src/main.cu`

- [ ] **Step 1: Wrap the printf with `#ifndef NDEBUG`**

In `src/main.cu`, change lines 523-525 from:

```c
            printf("  [step %d] verlet rebuild: max_cell=%d ncells=%d nx=%d\n",
                   step, verlet.max_cell_atoms, verlet.ncells, verlet.nx);
```

To:

```c
#ifndef NDEBUG
            printf("  [step %d] verlet rebuild: max_cell=%d ncells=%d nx=%d\n",
                   step, verlet.max_cell_atoms, verlet.ncells, verlet.nx);
#endif
```

- [ ] **Step 2: Build to verify compilation**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc)
```

Expected: clean compilation. Release builds define `NDEBUG` automatically.

- [ ] **Step 3: Quick smoke test — run CG water simulation for 100 steps, verify no regression**

```bash
cd /home/zhenghaowu/fastMD && ./build/fastMD data/CG_water_md/fastmd_nvt.conf
```

Watch the first 30 seconds of output. The simulation should complete without errors and produce normal thermo output. With `NDEBUG` defined (Release build), the Verlet rebuild print is suppressed.

- [ ] **Step 4: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/main.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: gate Verlet rebuild diagnostic print behind #ifndef NDEBUG

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```
