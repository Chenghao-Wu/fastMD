# Virial Reduction and Force Zeroing Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GPU atomic contention and wasted memory bandwidth in the fastMD main loop to improve throughput 10-20% for large polymer systems.

**Architecture:** Five independent kernel-level optimizations: (1) remove redundant `cudaMemset` on the force array that the LJ kernel fully overwrites, (2-4) replace per-thread/per-warp `atomicAdd` on the 6-component virial array with block-level shared-memory reduction in all three force kernels, (5) skip the unnecessary visited-cell dedup loop in the Verlet list builder when the cell grid is >=3 in all dimensions.

**Critical design constraint:** All three force kernels have early returns for out-of-range threads (`if (i >= natoms) return;`). Adding `__syncthreads()` after an early return is a deadlock. Each kernel must be restructured so padding threads skip computation but still participate in the shared memory barrier, contributing zero to the virial accumulation.

**Tech Stack:** CUDA C++ (CUB for radix sort), CMake build system, Python benchmark harness comparing against LAMMPS for validation.

---

## File Map

| File | Role | Change |
|------|------|--------|
| `src/main.cu` | Main simulation loop | Remove two `sys.zero_forces()` calls |
| `src/force/lj.cu` | LJ pair force kernel + launcher | Restructure to eliminate early return; block-level virial reduction |
| `src/force/fene.cu` | FENE bond force kernel + launcher | Restructure to eliminate early return; block-level virial reduction |
| `src/force/angle.cu` | Angle force kernel + launcher | Restructure to eliminate early return; block-level virial reduction |
| `src/neighbor/verlet_list.cuh` | VerletList struct definition | Add `dedup_needed` member |
| `src/neighbor/verlet_list.cu` | Verlet list build kernels + methods | Branch on `dedup_needed`, skip visited-array when false |

No new files. No API changes. All kernel launch signatures stay compatible.

---

### Task 1: Remove redundant `zero_forces()` calls

**Files:**
- Modify: `src/main.cu`

- [ ] **Step 1: Delete `sys.zero_forces()` at line 129 (pre-loop init)**

Delete line 129:
```cuda
    sys.zero_forces();
```

- [ ] **Step 2: Delete `sys.zero_forces()` at line 169 (per-step loop body)**

Delete line 169:
```cuda
        sys.zero_forces();
```

Keep `sys.zero_virial()` at lines 130 and 170 — all three kernels use `atomicAdd` on virial.

- [ ] **Step 3: Build and verify compilation**

```bash
cd /home/zhenghaowu/fastMD && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Expected: build succeeds with no errors.

- [ ] **Step 4: Run benchmark with validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py --config fastmd_benchmark.conf --validate
```

Expected: validation passes (`"passed": true` in benchmark_report.json).

- [ ] **Step 5: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/main.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: remove redundant zero_forces calls before LJ kernel

The LJ kernel unconditionally overwrites force[i] for every atom, so
the preceding cudaMemset on the entire force array is wasted bandwidth.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Block-level virial reduction in LJ kernel

**Files:**
- Modify: `src/force/lj.cu` (full kernel rewrite + launcher update)

The LJ kernel currently has `if (i >= natoms) return;` at line 15 followed by computation and warp-level virial reduction. Since `__syncthreads()` requires ALL threads in the block to participate, we must move the early return inside the computation block and let padding threads reach the barrier with zero accumulators.

- [ ] **Step 1: Replace the `lj_verlet_kernel` function body (lines 4-87)**

Old kernel (lines 4-87 of `src/force/lj.cu`):
```cuda
__global__ void lj_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const float2* __restrict__ lj_params,
    const int* __restrict__ neighbors,
    const int* __restrict__ num_neighbors,
    int natoms, int ntypes,
    float rc2, float L, float inv_L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    int lane = threadIdx.x & 31;

    float4 pos_i = pos[i];
    int type_i = __float_as_int(pos_i.w);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float pe_i = 0.0f;
    float vir_xx = 0.0f, vir_xy = 0.0f, vir_xz = 0.0f;
    float vir_yy = 0.0f, vir_yz = 0.0f, vir_zz = 0.0f;

    int nneigh = num_neighbors[i];
    #pragma unroll 8
    for (int k = 0; k < nneigh; k++) {
        int j = neighbors[k * natoms + i];
        float4 pos_j = pos[j];
        int type_j = __float_as_int(pos_j.w);

        float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
        float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
        float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
        float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < rc2 && r2 > 1e-10f) {
            float2 params = __ldg(&lj_params[type_i * ntypes + type_j]);
            float eps = params.x, sig = params.y;
            float sig2 = sig * sig;
            float r2inv = 1.0f / r2;
            float sr2 = sig2 * r2inv;
            float sr6 = sr2 * sr2 * sr2;
            float sr12 = sr6 * sr6;
            float force_r = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);

            float fxij = force_r * dx;
            float fyij = force_r * dy;
            float fzij = force_r * dz;

            fx += fxij;
            fy += fyij;
            fz += fzij;

            pe_i += 0.5f * 4.0f * eps * (sr12 - sr6);

            vir_xx += 0.5f * fxij * dx;
            vir_xy += 0.5f * fxij * dy;
            vir_xz += 0.5f * fxij * dz;
            vir_yy += 0.5f * fyij * dy;
            vir_yz += 0.5f * fyij * dz;
            vir_zz += 0.5f * fzij * dz;
        }
    }

    if (i < natoms) {
        force[i] = make_float4(fx, fy, fz, pe_i);
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
        atomicAdd(&virial[0], vir_xx);
        atomicAdd(&virial[1], vir_xy);
        atomicAdd(&virial[2], vir_xz);
        atomicAdd(&virial[3], vir_yy);
        atomicAdd(&virial[4], vir_yz);
        atomicAdd(&virial[5], vir_zz);
    }
}
```

New kernel (full replacement):
```cuda
__global__ void lj_verlet_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const float2* __restrict__ lj_params,
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

        int nneigh = num_neighbors[i];
        #pragma unroll 8
        for (int k = 0; k < nneigh; k++) {
            int j = neighbors[k * natoms + i];
            float4 pos_j = pos[j];
            int type_j = __float_as_int(pos_j.w);

            float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
            float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
            float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2 && r2 > 1e-10f) {
                float2 params = __ldg(&lj_params[type_i * ntypes + type_j]);
                float eps = params.x, sig = params.y;
                float sig2 = sig * sig;
                float r2inv = 1.0f / r2;
                float sr2 = sig2 * r2inv;
                float sr6 = sr2 * sr2 * sr2;
                float sr12 = sr6 * sr6;
                float force_r = 24.0f * eps * r2inv * (2.0f * sr12 - sr6);

                float fxij = force_r * dx;
                float fyij = force_r * dy;
                float fzij = force_r * dz;

                fx += fxij; fy += fyij; fz += fzij;
                pe_i += 0.5f * 4.0f * eps * (sr12 - sr6);
                vir_xx += 0.5f * fxij * dx;
                vir_xy += 0.5f * fxij * dy;
                vir_xz += 0.5f * fxij * dz;
                vir_yy += 0.5f * fyij * dy;
                vir_yz += 0.5f * fyij * dz;
                vir_zz += 0.5f * fzij * dz;
            }
        }

        force[i] = make_float4(fx, fy, fz, pe_i);
    }

    // Warp reduction — ALL threads participate (padding = zero accumulators)
    for (int offset = 16; offset > 0; offset >>= 1) {
        vir_xx += __shfl_down_sync(0xFFFFFFFF, vir_xx, offset);
        vir_xy += __shfl_down_sync(0xFFFFFFFF, vir_xy, offset);
        vir_xz += __shfl_down_sync(0xFFFFFFFF, vir_xz, offset);
        vir_yy += __shfl_down_sync(0xFFFFFFFF, vir_yy, offset);
        vir_yz += __shfl_down_sync(0xFFFFFFFF, vir_yz, offset);
        vir_zz += __shfl_down_sync(0xFFFFFFFF, vir_zz, offset);
    }

    // Block-level reduction via shared memory — ALL threads reach this
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
```

Key structural changes:
- Accumulators declared at function top (initialized to zero).
- `if (i >= natoms) return;` removed — replaced with `if (i < natoms) { ... }` wrapping only the computation and force write.
- Warp reduction and shared memory barrier are AFTER the `if` block, so ALL threads (including padding) reach them with zero accumulators.
- The redundant `if (i < natoms)` guard before `force[i] = ...` is now the outer `if` block — removed.

- [ ] **Step 2: Update launcher to pass dynamic shared memory size**

Replace lines 89-101 of `src/force/lj.cu`:

Old:
```cuda
void launch_lj_kernel(const float4* pos, float4* force, float* virial,
                       const float2* lj_params,
                       const int* neighbors, const int* num_neighbors,
                       int natoms, int ntypes,
                       float rc2, float L, float inv_L,
                       cudaStream_t stream)
{
    int blocks = div_ceil(natoms, 256);
    lj_verlet_kernel<<<blocks, 256, 0, stream>>>(
        pos, force, virial, lj_params,
        neighbors, num_neighbors,
        natoms, ntypes, rc2, L, inv_L);
}
```

New:
```cuda
void launch_lj_kernel(const float4* pos, float4* force, float* virial,
                       const float2* lj_params,
                       const int* neighbors, const int* num_neighbors,
                       int natoms, int ntypes,
                       float rc2, float L, float inv_L,
                       cudaStream_t stream)
{
    int blocks = div_ceil(natoms, 256);
    int smem = (256 / 32) * 6 * sizeof(float);
    lj_verlet_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, lj_params,
        neighbors, num_neighbors,
        natoms, ntypes, rc2, L, inv_L);
}
```

- [ ] **Step 3: Build**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 4: Run benchmark with validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py --config fastmd_benchmark.conf --validate
```

Expected: validation passes.

- [ ] **Step 5: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/force/lj.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: block-level virial reduction in LJ kernel via shared memory

Restructure kernel to eliminate early return so all threads reach the
shared memory barrier, then replace 6 atomicAdd per warp (48/block)
with shared memory accumulation + 6 atomicAdd per block (8x reduction).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Block-level virial reduction in FENE kernel

**Files:**
- Modify: `src/force/fene.cu` (full kernel rewrite + launcher update)

Same pattern as LJ: the `if (b >= nbonds) return;` must become a guard around computation only, so padding threads reach the `__syncthreads__()` barrier.

- [ ] **Step 1: Replace the `fene_kernel` function body (lines 4-71)**

Old kernel (lines 4-71 of `src/force/fene.cu`):
```cuda
__global__ void fene_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int2* __restrict__ bonds,
    const int* __restrict__ bond_types,
    const FENEParams* __restrict__ params,
    int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbonds) return;

    int2 bond = bonds[b];
    int type = bond_types[b];
    FENEParams p = params[type];

    float4 ri = pos[bond.x];
    float4 rj = pos[bond.y];

    float dx = min_image(ri.x - rj.x, L, inv_L);
    float dy = min_image(ri.y - rj.y, L, inv_L);
    float dz = min_image(ri.z - rj.z, L, inv_L);
    float r2 = dx*dx + dy*dy + dz*dz;

    float R02 = p.R0 * p.R0;

    // Clamp r2 to avoid NaN from log/sqrt/div by zero
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
    float fix = total_ff * dx;
    float fiy = total_ff * dy;
    float fiz = total_ff * dz;

    atomicAdd(&force[bond.x].x, fix);
    atomicAdd(&force[bond.x].y, fiy);
    atomicAdd(&force[bond.x].z, fiz);
    atomicAdd(&force[bond.y].x, -fix);
    atomicAdd(&force[bond.y].y, -fiy);
    atomicAdd(&force[bond.y].z, -fiz);

    float fene_pe = -0.5f * p.k * R02 * logf(1.0f - r2_safe / R02);
    float pair_pe = fene_pe + wca_pe;
    atomicAdd(&force[bond.x].w, 0.5f * pair_pe);
    atomicAdd(&force[bond.y].w, 0.5f * pair_pe);

    atomicAdd(&virial[0], fix * dx);
    atomicAdd(&virial[1], fix * dy);
    atomicAdd(&virial[2], fix * dz);
    atomicAdd(&virial[3], fiy * dy);
    atomicAdd(&virial[4], fiy * dz);
    atomicAdd(&virial[5], fiz * dz);
}
```

New kernel (full replacement):
```cuda
__global__ void fene_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int2* __restrict__ bonds,
    const int* __restrict__ bond_types,
    const FENEParams* __restrict__ params,
    int nbonds, float L, float inv_L)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (b < nbonds) {
        int2 bond = bonds[b];
        int type = bond_types[b];
        FENEParams p = params[type];

        float4 ri = pos[bond.x];
        float4 rj = pos[bond.y];

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
        float fix = total_ff * dx;
        float fiy = total_ff * dy;
        float fiz = total_ff * dz;

        atomicAdd(&force[bond.x].x, fix);
        atomicAdd(&force[bond.x].y, fiy);
        atomicAdd(&force[bond.x].z, fiz);
        atomicAdd(&force[bond.y].x, -fix);
        atomicAdd(&force[bond.y].y, -fiy);
        atomicAdd(&force[bond.y].z, -fiz);

        float fene_pe = -0.5f * p.k * R02 * logf(1.0f - r2_safe / R02);
        float pair_pe = fene_pe + wca_pe;
        atomicAdd(&force[bond.x].w, 0.5f * pair_pe);
        atomicAdd(&force[bond.y].w, 0.5f * pair_pe);

        v0 = fix * dx;
        v1 = fix * dy;
        v2 = fix * dz;
        v3 = fiy * dy;
        v4 = fiy * dz;
        v5 = fiz * dz;
    }

    // Block-level virial reduction — ALL threads participate (padding = zero)
    extern __shared__ float s_virial[];
    s_virial[tid * 6 + 0] = v0;
    s_virial[tid * 6 + 1] = v1;
    s_virial[tid * 6 + 2] = v2;
    s_virial[tid * 6 + 3] = v3;
    s_virial[tid * 6 + 4] = v4;
    s_virial[tid * 6 + 5] = v5;
    __syncthreads();

    if (tid == 0) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
        for (int t = 0; t < blockDim.x; t++) {
            s0 += s_virial[t * 6 + 0];
            s1 += s_virial[t * 6 + 1];
            s2 += s_virial[t * 6 + 2];
            s3 += s_virial[t * 6 + 3];
            s4 += s_virial[t * 6 + 4];
            s5 += s_virial[t * 6 + 5];
        }
        atomicAdd(&virial[0], s0);
        atomicAdd(&virial[1], s1);
        atomicAdd(&virial[2], s2);
        atomicAdd(&virial[3], s3);
        atomicAdd(&virial[4], s4);
        atomicAdd(&virial[5], s5);
    }
}
```

- [ ] **Step 2: Update launcher to pass dynamic shared memory size**

Replace the launch line in `launch_fene_kernel` (currently `fene_kernel<<<blocks, 256, 0, stream>>>` at line 80):

Old (lines 79-82 of `src/force/fene.cu`):
```cuda
    int blocks = div_ceil(nbonds, 256);
    fene_kernel<<<blocks, 256, 0, stream>>>(
        pos, force, virial, bonds, bond_types, params,
        nbonds, L, inv_L);
```

New:
```cuda
    int blocks = div_ceil(nbonds, 256);
    int smem = 256 * 6 * sizeof(float);
    fene_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, bonds, bond_types, params,
        nbonds, L, inv_L);
```

- [ ] **Step 3: Build**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 4: Run benchmark with validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py --config fastmd_benchmark.conf --validate
```

Expected: validation passes.

- [ ] **Step 5: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/force/fene.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: block-level virial reduction in FENE kernel via shared memory

Restructure to eliminate early return, then replace 6 atomicAdd per
thread on the global virial with shared memory accumulation + 6
atomicAdd per block (256x reduction in virial atomics).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Block-level virial reduction in Angle kernel

**Files:**
- Modify: `src/force/angle.cu` (full kernel rewrite + launcher update)

Same pattern: eliminate `if (a >= nangles) return;`, guard only the computation, let all threads reach the barrier.

- [ ] **Step 1: Replace the `angle_kernel` function body (lines 4-77)**

Old kernel (lines 4-77 of `src/force/angle.cu`):
```cuda
__global__ void angle_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int4* __restrict__ angles,
    const AngleParams* __restrict__ params,
    int nangles, float L, float inv_L)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nangles) return;

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

    atomicAdd(&force[i].x, fi_x);
    atomicAdd(&force[i].y, fi_y);
    atomicAdd(&force[i].z, fi_z);
    atomicAdd(&force[k].x, fk_x);
    atomicAdd(&force[k].y, fk_y);
    atomicAdd(&force[k].z, fk_z);
    atomicAdd(&force[j].x, -(fi_x + fk_x));
    atomicAdd(&force[j].y, -(fi_y + fk_y));
    atomicAdd(&force[j].z, -(fi_z + fk_z));

    float pe = p.k_theta * dtheta * dtheta;
    float pe_third = pe / 3.0f;
    atomicAdd(&force[i].w, pe_third);
    atomicAdd(&force[j].w, pe_third);
    atomicAdd(&force[k].w, pe_third);

    atomicAdd(&virial[0], fi_x * dxij + fk_x * dxkj);
    atomicAdd(&virial[1], fi_x * dyij + fk_x * dykj);
    atomicAdd(&virial[2], fi_x * dzij + fk_x * dzkj);
    atomicAdd(&virial[3], fi_y * dyij + fk_y * dykj);
    atomicAdd(&virial[4], fi_y * dzij + fk_y * dzkj);
    atomicAdd(&virial[5], fi_z * dzij + fk_z * dzkj);
}
```

New kernel (full replacement):
```cuda
__global__ void angle_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    float* __restrict__ virial,
    const int4* __restrict__ angles,
    const AngleParams* __restrict__ params,
    int nangles, float L, float inv_L)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (a < nangles) {
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

        atomicAdd(&force[i].x, fi_x);
        atomicAdd(&force[i].y, fi_y);
        atomicAdd(&force[i].z, fi_z);
        atomicAdd(&force[k].x, fk_x);
        atomicAdd(&force[k].y, fk_y);
        atomicAdd(&force[k].z, fk_z);
        atomicAdd(&force[j].x, -(fi_x + fk_x));
        atomicAdd(&force[j].y, -(fi_y + fk_y));
        atomicAdd(&force[j].z, -(fi_z + fk_z));

        float pe = p.k_theta * dtheta * dtheta;
        float pe_third = pe / 3.0f;
        atomicAdd(&force[i].w, pe_third);
        atomicAdd(&force[j].w, pe_third);
        atomicAdd(&force[k].w, pe_third);

        v0 = fi_x * dxij + fk_x * dxkj;
        v1 = fi_x * dyij + fk_x * dykj;
        v2 = fi_x * dzij + fk_x * dzkj;
        v3 = fi_y * dyij + fk_y * dykj;
        v4 = fi_y * dzij + fk_y * dzkj;
        v5 = fi_z * dzij + fk_z * dzkj;
    }

    // Block-level virial reduction — ALL threads participate (padding = zero)
    extern __shared__ float s_virial[];
    s_virial[tid * 6 + 0] = v0;
    s_virial[tid * 6 + 1] = v1;
    s_virial[tid * 6 + 2] = v2;
    s_virial[tid * 6 + 3] = v3;
    s_virial[tid * 6 + 4] = v4;
    s_virial[tid * 6 + 5] = v5;
    __syncthreads();

    if (tid == 0) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
        for (int t = 0; t < blockDim.x; t++) {
            s0 += s_virial[t * 6 + 0];
            s1 += s_virial[t * 6 + 1];
            s2 += s_virial[t * 6 + 2];
            s3 += s_virial[t * 6 + 3];
            s4 += s_virial[t * 6 + 4];
            s5 += s_virial[t * 6 + 5];
        }
        atomicAdd(&virial[0], s0);
        atomicAdd(&virial[1], s1);
        atomicAdd(&virial[2], s2);
        atomicAdd(&virial[3], s3);
        atomicAdd(&virial[4], s4);
        atomicAdd(&virial[5], s5);
    }
}
```

- [ ] **Step 2: Update launcher to pass dynamic shared memory size**

Replace the launch line in `launch_angle_kernel` (lines 84-86 of `src/force/angle.cu`):

Old:
```cuda
    int blocks = div_ceil(nangles, 256);
    angle_kernel<<<blocks, 256, 0, stream>>>(
        pos, force, virial, angles, params, nangles, L, inv_L);
```

New:
```cuda
    int blocks = div_ceil(nangles, 256);
    int smem = 256 * 6 * sizeof(float);
    angle_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, angles, params, nangles, L, inv_L);
```

- [ ] **Step 3: Build**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 4: Run benchmark with validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py --config fastmd_benchmark.conf --validate
```

Expected: validation passes.

- [ ] **Step 5: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/force/angle.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: block-level virial reduction in Angle kernel via shared memory

Same pattern as FENE: restructure to eliminate early return, replace
per-thread atomicAdd on the 6-component global virial with shared
memory accumulation + per-block atomics (256x reduction).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Skip cell dedup in Verlet list build for large grids

**Files:**
- Modify: `src/neighbor/verlet_list.cuh` (add member)
- Modify: `src/neighbor/verlet_list.cu` (set flag, kernel param, branch, pass flag)

- [ ] **Step 1: Add `dedup_needed` member to VerletList struct**

In `src/neighbor/verlet_list.cuh`, after line 18 (`int   ncells;`), add:

```cuda
    bool  dedup_needed;
```

- [ ] **Step 2: Set `dedup_needed` in `allocate()`**

In `src/neighbor/verlet_list.cu`, in `VerletList::allocate()`, after `ncells = nx * ny * nz;` (line 9), add:

```cuda
    dedup_needed = (nx < 3 || ny < 3 || nz < 3);
```

- [ ] **Step 3: Add `int dedup_needed` parameter to kernel, pass it from `build()`**

Change the `build_verlet_list` kernel signature to add `int dedup_needed` as the last parameter.

In `VerletList::build()`, pass `dedup_needed ? 1 : 0` to the kernel launch:
```cuda
    build_verlet_list<<<ncells, 256, 0, stream>>>(
        pos, sorted_atoms, cell_starts, cell_ends,
        exclusion_offsets, exclusion_list,
        neighbors, num_neighbors,
        natoms, nx, ny, nz,
        rc_skin2, box_L, inv_L,
        max_neighbors, dedup_needed ? 1 : 0);
```

- [ ] **Step 4: Branch the 27-neighbor loop on `dedup_needed`**

Wrap the existing inner loop (the 3D for-loops with visited-array logic from lines 127-197) in `if (dedup_needed) { ...existing code... } else { ...same loops without visited array... }`.

The `else` branch is the same triple-nested loop but without the `visited[27]` array, the `n_visited` counter, the `already` check, and the `visited[n_visited++]` assignment — each neighbor cell is processed directly since all 27 offsets produce distinct cells.

- [ ] **Step 5: Build**

```bash
cd /home/zhenghaowu/fastMD/build && make -j$(nproc)
```

Expected: build succeeds.

- [ ] **Step 6: Run benchmark with validation**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py --config fastmd_benchmark.conf --validate
```

Expected: validation passes.

- [ ] **Step 7: Commit**

```bash
git -C /home/zhenghaowu/fastMD add src/neighbor/verlet_list.cuh src/neighbor/verlet_list.cu
git -C /home/zhenghaowu/fastMD commit -m "$(cat <<'EOF'
perf: skip visited-cell dedup in Verlet build when grid >= 3

When nx, ny, nz are all >= 3, the 27 neighbor cell offsets are all
distinct — dedup is unnecessary. Branch on a dedup_needed flag set
in allocate() based on grid dimensions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Full benchmark validation

**Files:** None (validation only)

- [ ] **Step 1: Run full benchmark and compare against baseline**

```bash
cd /home/zhenghaowu/fastMD && python benchmarks/run.py --config fastmd_benchmark.conf --validate
```

Expected: `"passed": true`, ns/day improved vs baseline.

- [ ] **Step 2: Check git log for commit history**

```bash
git -C /home/zhenghaowu/fastMD log --oneline -7
```

Expected: 6 new commits, all with clean messages.
