# Tabulated Potential Support (LAMMPS-style)

## Overview

Add support for LAMMPS-style tabulated pair potentials to fastMD, coexisting with the existing analytical Lennard-Jones potential. The implementation follows a "separate kernels, shared neighbor list" strategy: the existing LJ kernel remains untouched, and a new table kernel traverses the same Verlet list, skipping pairs that are not assigned to a table.

## Motivation

Some force fields (e.g., coarse-grained models, machine-learning potentials, or exotic functional forms) are easier to specify as tabulated r/energy/force data than as closed-form expressions. LAMMPS users are already familiar with `pair_style table`; fastMD should support the same file format and semantics.

## Design Decisions

- **Approach:** Separate kernels with shared neighbor list (Approach 2 from brainstorming).
  - Rationale: In the common case a simulation uses either all-LJ or all-table, so the double-traversal overhead is only paid in hybrid simulations. This keeps the existing LJ kernel completely untouched.
- **Interpolation:** Linear on the GPU (user choice).
- **File format:** LAMMPS native table format with section keywords.
- **Config syntax:** `table <ti> <tj> <filename> <keyword>` (mirrors existing `lj` keyword).

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `src/io/table_parser.hpp` | `TableParams` struct; `parse_table_file()` declaration |
| `src/io/table_parser.cpp` | LAMMPS table file parser |
| `src/force/table.cuh` | `launch_table_kernel()` declaration |
| `src/force/table.cu` | `table_verlet_kernel` and launcher |

### Modified Files

| File | Change |
|------|--------|
| `src/io/config.hpp` | Add `table_idx`, `table_params`, `table_data` to `TopologyData` |
| `src/io/config.cpp` | Parse `table <ti> <tj> <filename> <keyword>` lines; validate against `lj` conflicts |
| `src/core/system.cuh` / `src/core/system.cu` | Add device pointers for table data to `System`; allocate/free them |
| `src/main.cu` | Allocate/upload table device memory; launch `launch_table_kernel` alongside LJ |
| `CMakeLists.txt` | Add `src/io/table_parser.cpp` and `src/force/table.cu` to the `core` library |
| `tests/CMakeLists.txt` | Add `test_table.cu` and `test_table_parser.cu` targets |

## Data Structures

```cpp
struct TableParams {
    float rmin;
    float rmax;
    float dr;
    float inv_dr;
    int   npoints;
    int   data_offset; // index into the global flat table_data array
};
```

### TopologyData additions

```cpp
std::vector<int>         table_idx;      // size = ntypes * ntypes, -1 = no table
std::vector<TableParams> table_params;   // one entry per unique table section
std::vector<float4>      table_data;     // packed: x=r, y=force, z=energy, w=0
```

### System additions

```cpp
int*          d_table_idx;
TableParams*  d_table_params;
float4*       d_table_data;
```

`System::allocate` allocates these arrays when table pairs are present; `System::free` releases them if non-null.

## Config Syntax

```
table <ti> <tj> <filename> <keyword>
```

- `ti`, `tj`: 0-indexed atom types (symmetric, same semantics as `lj`).
- `filename`: path to a LAMMPS table file.
- `keyword`: section name inside the file.

Example:

```
lj 0 0 1.0 1.0
table 1 1 morse.table MORSE
```

Validation: if both `lj` and `table` are specified for the same `(ti, tj)`, parsing throws a `std::runtime_error`.

Like `lj`, `table` entries are symmetric: setting `table 0 1 file.txt KEY` automatically applies to `(1, 0)` as well.

## Data Flow

1. **Config parse:** `config.cpp` collects `table` entries. For each unique `(filename, keyword)`, it calls `parse_table_file(filename, keyword)`.
2. **File parse:** `table_parser.cpp` opens the file, locates the section by keyword, reads the header `N <n> R <rlo> <rhi>`, then reads `n` lines of `index r energy force`.
3. **Host storage:** Points are stored as `float4(r, force, energy, 0)` in `TopologyData.table_data`. A `TableParams` entry records metadata. `TopologyData.table_idx[ti * ntypes + tj]` points to the params entry (or `-1`).
4. **Device upload:** In `main.cu`, after `sys.allocate()`, device arrays are allocated and uploaded with `cudaMemcpy`.
5. **Runtime dispatch:** Every force step:
   - `launch_lj_kernel(...)` runs unchanged.
   - If any table pairs exist, `launch_table_kernel(...)` is also called on the same `stream_lj`.

## GPU Kernel Design

`table_verlet_kernel` mirrors `lj_verlet_kernel` in structure (same thread layout, same virial reduction), but instead of overwriting `force[i]` it **accumulates** into it:

```cpp
float4 f = force[i];
// ... compute table contributions for all neighbors ...
f.x += fx; f.y += fy; f.z += fz; f.w += pe_i;
force[i] = f;
```

This read-modify-write is safe because each thread handles a unique atom `i`, and the table kernel is launched sequentially after the LJ kernel on the same `stream_lj`.

Inner-loop pseudocode:

```cpp
int tidx = d_table_idx[type_i * ntypes + type_j];
if (tidx < 0) continue;

TableParams tp = d_table_params[tidx];
float r = sqrtf(r2);
if (r < tp.rmin) r = tp.rmin;
if (r > tp.rmax) r = tp.rmax;

int idx = (int)((r - tp.rmin) * tp.inv_dr);
if (idx < 0) idx = 0;
if (idx >= tp.npoints - 1) idx = tp.npoints - 2;

float t = (r - (tp.rmin + idx * tp.dr)) * tp.inv_dr;
float4 p0 = __ldg(&d_table_data[tp.data_offset + idx]);
float4 p1 = __ldg(&d_table_data[tp.data_offset + idx + 1]);

float f_scalar = p0.y + t * (p1.y - p0.y); // F = -dU/dr
float e       = p0.z + t * (p1.z - p0.z);

float rinv = 1.0f / r;
float fpair = f_scalar * rinv;
float fxij = dx * fpair;
float fyij = dy * fpair;
float fzij = dz * fpair;

fx += fxij; fy += fyij; fz += fzij;
pe_i += 0.5f * e;
// virial accumulation identical to lj_verlet_kernel
```

**Execution order on `stream_lj`:**
1. `launch_lj_kernel` writes `force[i]` (full LJ forces, or zeros if no LJ pairs).
2. `launch_table_kernel` reads `force[i]`, adds table contributions, writes back.
3. Bonded kernels (FENE/angle) then use `atomicAdd` on `stream_bonded` after waiting for the LJ event.

## Parser Specification

Supported header style (first version):

```
KEYWORD
N 1000 R 0.5 10.0
```

- `KEYWORD`: arbitrary alphanumeric section name.
- `N`: number of data points.
- `R`: linear spacing in `r`. `rlo` and `rhi` define the range.

Data lines:
```
1 0.5000 100.0000 -50.0000
2 0.5095  99.5000 -49.8000
...
```

Each line contains: `index r energy force`
- `force` is the scalar force magnitude `F = -dU/dr` (same convention as LAMMPS).

Validation performed by the parser:
- `r` values must be strictly monotonically increasing.
- Exact line count must match `N`.
- `rlo` must equal the first point's `r`; `rhi` must equal the last point's `r`.

`RSQ` spacing style may be added in a future iteration without kernel changes.

## Error Handling

### Fatal (thrown during config parse)
- Table file does not exist or is unreadable.
- Section `keyword` not found in file.
- Malformed or missing `N` / `R` header.
- Data line count does not match `N`.
- `r` values not strictly increasing.
- Both `lj` and `table` defined for the same `(ti, tj)`.
- `ti` or `tj` out of range.

### Warnings (printed once at setup)
- If `rc` exceeds a table's `rmax`, a warning is printed because the kernel will clamp to `rmax`.

### Kernel clamping
- `r < rmin`: clamped to `rmin`.
- `r > rmax`: clamped to `rmax`.
- This prevents out-of-bounds access and gives defined behavior for slightly mismatched cutoffs.

## Testing Plan

### `tests/test_table_parser.cu`
- Create an in-memory LAMMPS table file string (parsed via a host-only function inside a `.cu` test).
- Parse and verify `TableParams` and point values.
- Test error paths: missing keyword, bad `N`, non-monotonic `r`.

### `tests/test_table.cu`
- Build a small 2-atom system.
- Generate a table that reproduces LJ (`eps=1.0, sig=1.0`) at the same `r` values.
- Run `launch_table_kernel` and compare force/energy/virial against `launch_lj_kernel`.
- Tolerance: `1e-4` (accounting for linear interpolation error).

### Integration / benchmark
- Add a benchmark config that replaces `lj` with an equivalent `table` for all pairs.
- Verify thermodynamic output matches the analytical LJ benchmark within tolerance.

## Performance Notes

- Table kernel adds one additional `__ldg` load (`d_table_idx`) per neighbor pair, plus two `__ldg` loads from `d_table_data` for table pairs.
- For all-LJ simulations the table kernel is not launched, so there is zero overhead.
- For all-table simulations, the LJ kernel still runs but all pairs have `eps=0, sig=0`, so the force computation is skipped after the cheap parameter load. The kernel still performs the full neighbor traversal and virial reduction; a future optimization could skip the LJ launch entirely when no LJ pairs are defined.
