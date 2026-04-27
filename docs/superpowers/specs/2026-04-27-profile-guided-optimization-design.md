# Profile-Guided GPU Optimization — Design Document

**Date:** 2026-04-27
**Status:** Approved
**Scope:** Profile the Verlet-list-based fastMD codebase with Nsight Systems + Nsight Compute, then optimize the top 1-3 bottleneck kernels

---

## 1. Motivation

The Verlet neighbor list (delivered 2026-04-25) fundamentally changed the GPU time profile. The old Nsight data — LJ kernel at 82.3% of GPU time — is stale. Fresh profiling is needed to identify the new bottlenecks, and the optimization effort will target the top 1-3 kernels with specific data-driven fixes.

**Goal:** 20-40% overall step-time reduction via targeted kernel optimization after profiling.

---

## 2. Profiling Setup

### 2.1 Workload

- 30K-atom polymer system from `benchmarks/benchmark.json` validation phase (1K steps)
- Short enough for profiling runs, still representative of production

### 2.2 Nsight Systems (`nsys`) — Timeline

```
nsys profile --stats=true --force-overwrite=true \
  -o fastmd_nsys ./build/fastMD fastmd_benchmark.conf
```

Captures: GPU timeline, per-kernel durations, stream overlap, CUDA API calls.

### 2.3 Nsight Compute (`ncu`) — Kernel Detail

```
ncu --set full -o fastmd_ncu -k <kernel_name> \
  ./build/fastMD fastmd_benchmark.conf
```

Run on the top 2-3 kernels from nsys. Key metrics: compute throughput (% of peak), memory bandwidth utilization, occupancy, warp stall reasons, L1/L2 hit rates.

---

## 3. Bottleneck Classification Framework

Each kernel is classified by its primary stall source:

| Bottleneck type | ncu signature | Typical fix |
|---|---|---|
| **Compute-bound** | High SM utilization, math pipe saturated, low memory bandwidth | Reduce instruction count, fast-math intrinsics, simplify arithmetic |
| **Memory-bound** | High L1/L2 miss rate, DRAM bandwidth near peak | Reorder data, use `__ldg`, increase locality |
| **Latency-bound** | Low occupancy, long scoreboard stalls, eligible warps < 2 | Increase occupancy, prefetch, ILP |
| **Atomic contention** | High Mem pipe utilization, system-wide atomics count | Shared memory reduction, warp shuffle, remove atomics |

---

## 4. Pre-Profiling Hypotheses

Based on code inspection of the current Verlet-based code:

| Kernel | Likely bottleneck | Reasoning |
|---|---|---|
| `lj_verlet_kernel` | Still compute-bound, leaner | ~4M pairs vs old ~76M; PBC + LJ math is still the bulk but false-positive ratio dropped 49x→1.4x |
| `fene_kernel` | Atomic contention | 9 `atomicAdd` per bond; backbone atoms shared across bonds cause serialization |
| `angle_kernel` | Atomic contention | 12 `atomicAdd` per angle; same atom-sharing problem |
| `build_verlet_list` | Memory divergence | 27-cell stencil with scattered `pos[j]` loads and exclusion linear scan |
| `integrator_pre_force_kernel` | Compute (curand) | `curand_normal4` generates 4 Gaussians per atom |
| `compute_thermo` (both kernels) | Allocation overhead | Allocates/frees `d_kin_stress` and `d_pe` every thermo step |

---

## 5. Optimization Scope & Targets

Top 1-3 kernels from profiling. Conservative targets:

- **Overall step-time reduction:** 20-40%
- **Bonded forces (FENE + Angle):** If atomic contention is confirmed, 2-4x speedup is achievable by eliminating atomics via warp-level accumulation
- **Integrator:** If `curand_normal4` dominates, switching to `curand_uniform4` + Box-Muller or pre-generated noise can significantly cut cost
- **Verlet build:** Smaller impact since rebuilds happen every ~10-20 steps; amortized cost is low

---

## 6. Implementation Principles

- Profile first, code second — no optimization without data
- Prefer local kernel changes over architectural rewrites
- Follow existing patterns: warp shuffle for reductions, `__ldg` for read-only, block-per-cell launch
- Final gate: `benchmarks/run.py` validation (fastMD vs LAMMPS, temp within 5%, PE within 1%)

---

## 7. Validation Strategy

1. **Per-optimization:** Run affected unit tests (`test_lj`, `test_fene`, `test_angle`, `test_integrator`)
2. **End-to-end:** `benchmarks/run.py` — fastMD vs LAMMPS, 1K validation steps, `rel_temp_diff < 0.05`, `rel_pe_diff < 1e-2`
3. **Performance confirmation:** Before/after `nsys` comparison

---

## 8. Files in Scope

| File | Potential change |
|------|-----------------|
| `src/force/fene.cu` | Eliminate atomicAdd for forces if confirmed bottleneck |
| `src/force/angle.cu` | Eliminate atomicAdd for forces if confirmed bottleneck |
| `src/force/lj.cu` | Micro-optimizations if still top-ranked |
| `src/neighbor/verlet_list.cu` | Build pipeline optimizations if ranked high enough |
| `src/integrate/langevin.cu` | RNG replacement if curand dominates |
| `src/analysis/thermo.cu` | Persistent allocation for d_kin_stress/d_pe |
| `src/io/dump.cu` | Persistent allocation for d_temp |

Actual changes determined by profiling results.

---

## 9. Deliverables

- Nsight profile report with bottleneck analysis
- 2-4 targeted kernel optimizations
- Updated `benchmark_report.json` with new speedup numbers
