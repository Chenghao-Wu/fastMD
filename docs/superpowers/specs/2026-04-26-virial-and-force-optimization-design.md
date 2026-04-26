# Design: Virial reduction and force zeroing optimization

## Goal

Reduce unnecessary GPU memory bandwidth and atomic operation contention in the fastMD
main simulation loop to improve throughput for large polymer systems.

## Target

Large systems (30K-100K+ atoms) with polymers (LJ + FENE bonds + angle terms),
running on NVIDIA Ada architecture (RTX 4090 D).

## Changes

### 1. Remove redundant `zero_forces()` calls

**Why:** The LJ kernel unconditionally writes `force[i] = make_float4(fx, fy, fz, pe_i)`
for every `i < natoms` — a full overwrite, not an atomic add. The preceding
`cudaMemset(force, 0, natoms_padded * sizeof(float4))` is pure overhead.

**What changes:**
- Delete `sys.zero_forces()` from `main.cu` line 129 (pre-loop init) and line 169 (per-step loop body).
- Keep `sys.zero_virial()` — all three force kernels use `atomicAdd` on the virial array.
- No changes to any kernel.

**Files:** `src/main.cu`

### 2. Block-level virial reduction in LJ kernel

**Why:** The LJ kernel does warp shuffle reduction then 6 `atomicAdd` per warp on
the global virial array. At 256 threads/block (8 warps), that's 48 atomics per block
all targeting the same 6 global addresses — unnecessary serialization at the L2.

**What changes:**
- After the warp shuffle reduction, lane 0 writes its 6 accumulated virial
  components to shared memory instead of hitting global.
- `__syncthreads()` then thread 0 loops over all warps, sums, and does 6 `atomicAdd`.
- Launch gains a third argument for dynamic shared memory:
  `(256 / 32) * 6 * sizeof(float)` = 192 bytes.
- Kernel gains `extern __shared__ float s_virial[]` parameter.

**Atomic count:** 48/block → 6/block (8x reduction).

**Files:** `src/force/lj.cu`

### 3. Block-level virial reduction in FENE kernel

**Why:** The FENE kernel does 6 `atomicAdd` per *thread* on the global virial —
all threads hitting the same 6 addresses. For 100K bonds, that's ~600K serialized
atomics per step.

**What changes:**
- Each thread computes its 6 virial contributions locally (already done),
  then stores to `s_block[threadIdx.x * 6 + comp]` in shared memory.
- `__syncthreads()`, then thread 0 sums and does 6 `atomicAdd`.
- Launch adds dynamic shared memory: `blockDim.x * 6 * sizeof(float)` = 6144 bytes.
- Force `atomicAdd` calls (targeting per-atom force components) are unchanged —
  those hit distinct addresses and have low contention (~2 bonds per atom).

**Atomic count:** 6/thread → 6/block (256x reduction on virial only).

**Files:** `src/force/fene.cu`

### 4. Block-level virial reduction in Angle kernel

**Why:** Same pattern as FENE — 6 `atomicAdd` per thread on global virial.

**What changes:** Identical to FENE kernel change.

**Files:** `src/force/angle.cu`

### 5. Skip cell dedup in Verlet list build when grid >= 3 in all dimensions

**Why:** The `build_verlet_list` kernel's 27-neighbor loop deduplicates visited
cells using a `visited[27]` array with linear scan. When `nx >= 3 && ny >= 3 && nz >= 3`,
all 27 neighbor offsets map to distinct cells — dedup is unnecessary.

Since `nx = max(1, int(box_L / rc_skin))`, the condition holds whenever
`box_L >= 3 * rc_skin`, which is true for all real simulations.

**What changes:**
- Add `bool dedup_needed` member to `VerletList`, set during `allocate()`:
  `dedup_needed = (nx < 3 || ny < 3 || nz < 3)`.
- In `build_verlet_list` kernel, branch on a new `int dedup_needed` kernel parameter:
  - `dedup_needed == 0`: skip the visited array and dedup loop entirely.
  - `dedup_needed == 1`: keep existing dedup logic (only for tiny boxes).
- Optionally, replace the visited array with a `uint32_t` bitmask to make the
  fallback path faster, though this is low priority given how rare the case is.

**Files:** `src/neighbor/verlet_list.cuh`, `src/neighbor/verlet_list.cu`

## Impact summary

| Change | Mechanism | Expected gain |
|--------|-----------|---------------|
| Remove zero_forces | Eliminates 1 full-array write/step | ~1% throughput |
| LJ virial reduction | 48→6 atomics/block | ~2-5% LJ kernel speedup |
| FENE virial reduction | 6/thread→6/block atomics | ~10-20% FENE kernel speedup |
| Angle virial reduction | 6/thread→6/block atomics | ~10-20% Angle kernel speedup |
| Skip cell dedup | Branch elimination in inner loop | ~1-3% Verlet build speedup |

**Combined expectation:** 10-20% throughput improvement for large polymer systems.

## What is NOT changed

- Force `atomicAdd` in FENE/Angle kernels — these target distinct per-atom addresses
  with low contention and are not a bottleneck.
- Stream synchronization in the main loop — L1 must complete before bonded forces
  because LJ does a non-atomic overwrite of the force array.
- Neighbor list data layout — the transposed storage `neighbors[k * natoms + i]`
  is already optimal for coalesced access.
- Virial zeroing — still needed since all three kernels use `atomicAdd` on virial.
