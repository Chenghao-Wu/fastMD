# Table Kernel Optimization — Design Spec

**Date:** 2026-05-13
**Target:** CG water (table-only, single type pair, 500-point table)

## Goal

Improve the performance of the tabulated potential kernel for CG water MD simulations without changing the LAMMPS table file format or the neighbor list structure.

## Approach: Shared memory + hoisting (conservative)

Keep r-based interpolation with sqrtf. Load table data into shared memory once per block and hoist type-pair constants out of the inner loop.

## Changes

### 1. `src/force/table.cu` — Kernel rewrite

**Shared memory table cache:**
- Cooperatively load table data into `__shared__ float4[]` at block start
- All inner-loop table reads come from shared memory instead of global memory
- For CG water: 500 entries × 16 bytes = 8 KB, fits in shared memory

**Hoist type-pair constants:**
- Load `table_idx[type_i * ntypes + type_j]` once per thread, outside the neighbor loop
- Load `TableParams` (rmin, rmax, dr, inv_dr, npoints, data_offset) into registers once
- Since CG water has only type 0, `type_i` and the table index are constant across all rows, but the kernel remains general for multi-type

**Kernel structure:**
```
__global__ void table_verlet_kernel(...) {
    // 1. Cooperative determination of table params (thread 0 or all equal)
    // 2. Cooperative load of table data into __shared__ s_table[]
    // 3. __syncthreads()
    // 4. Per-atom loop: hoist type_i, tidx, tp fields into registers
    // 5. Neighbor loop: read from s_table[], same interpolation math
    // 6. Virial reduction (unchanged)
}
```

**Multi-type handling:** For systems where total table data exceeds shared memory capacity, the kernel falls back to `__ldg` reads from global memory. For CG water (single type pair, 8 KB table), shared memory is always used.

**Accuracy:** Numerically identical — same sqrtf, same interpolation formula, same data values. The only difference is the memory source (shared vs global).

### 2. `src/main.cu` — Gate diagnostic print

Wrap the Verlet rebuild diagnostic print behind `#ifndef NDEBUG`:

```c
#ifndef NDEBUG
            printf("  [step %d] verlet rebuild: max_cell=%d ncells=%d nx=%d\n",
                   step, verlet.max_cell_atoms, verlet.ncells, verlet.nx);
#endif
```

Release builds (`-DCMAKE_BUILD_TYPE=Release`, which defines `NDEBUG`) skip the print. Debug builds still show it. The underlying tracking infrastructure (`h_cell_max`, `atomicMax`, `cudaStreamSynchronize`) stays in place for diagnostic use.

### 3. No changes to

- `src/io/table_parser.hpp` / `.cpp` — same LAMMPS table parsing
- `src/neighbor/verlet_list.cuh` / `.cu` — neighbor list unchanged
- `src/force/table.cuh` — same function signature

## Validation

1. Static force comparison: dump forces from both old and new kernel on the same CG water system, assert equality (within float rounding)
2. Run a short CG water simulation, compare thermo output against existing reference data
