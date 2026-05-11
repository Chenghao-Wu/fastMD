# CG Water Table Potential Validation in Reduced (HOOMD MD) Units

## Overview

Convert the existing LAMMPS real-units CG water dataset (`data/CG_water/`) into HOOMD-style "MD" reduced units that fastMD can consume, add per-type mass support to fastMD, and validate the table potential end-to-end by comparing fastMD against LAMMPS for the same Langevin NVT trajectory.

The work is split into three phases. Phase 1 isolates the table kernel via a static (zero-step) force comparison so any disagreement is unambiguously a kernel/parser/unit-conversion issue. Phase 2 lands mass support in fastMD without breaking existing benchmarks. Phase 3 runs the full Langevin NVT trajectory comparison.

## Motivation

The table kernel and parser were landed (per `2026-05-11-tabulated-potential-design.md`) but have never been exercised against a real reference dataset. The repository contains a LAMMPS CG water example at `data/CG_water/` (1000 beads, 500-point pair table, 15 Å cutoff, Langevin NVT at 300 K) which is ideal for validation — but it is in LAMMPS `real` units, while fastMD assumes a self-consistent reduced unit system with implicit `m = 1`. Closing that gap requires (a) deciding on a unit convention, (b) generating the converted dataset, and (c) extending fastMD to handle non-unit masses so dynamics is physically correct.

## Design Decisions

- **Unit system:** HOOMD MD — length = nm, energy = kJ/mol, mass = atomic mass unit. Time follows: `[time] = [length]·sqrt([mass]/[energy]) = 1 ps`. Boltzmann constant is dimensional: `kB = 0.00831446 kJ/(mol·K)`. Reference: <https://hoomd-blue.readthedocs.io/en/stable/units.html>.
- **Mass model:** per-type, parsed from the LAMMPS `Masses` section. When the section is absent, all masses default to `1.0` so existing FENE/LJ benchmarks remain bit-identical.
- **Conversion tooling:** standalone Python script `tools/convert_lammps_units.py` (reusable for future LAMMPS→fastMD ports), no third-party deps. Output goes to `data/CG_water_md/`, leaving the original untouched.
- **Validation strategy:** three-phase, in order — (1) static initial-state force/PE comparison, (2) mass support landed and regression-tested, (3) Langevin NVT trajectory comparison via time-averaged thermodynamics.
- **fastMD code change in Phase 1:** a single new config key `forces_dump_file <path>` that, when set, computes forces once on the initial configuration, writes them, and exits without time-stepping. ~25 lines in `main.cu` plus a config key.

## Unit Conversion Reference

### Conversion factors (LAMMPS `real` → HOOMD `md`)

| Quantity | real | md | Factor `× q_real` |
|---|---|---|---|
| length (positions, box, table `r`) | Å | nm | × 0.1 |
| energy (table `U`, PE) | kcal/mol | kJ/mol | × 4.184 |
| mass | g/mol = amu | amu | × 1 |
| force (table column 4) | kcal/(mol·Å) | kJ/(mol·nm) | × 41.84 |
| time (`dt`, `Tdamp`) | fs | ps | × 0.001 |
| temperature `T` → fastMD `kT` input | K | kJ/mol | × 0.00831446 |
| pressure | atm | kJ/(mol·nm³) | × 0.061027 |

### CG_water concrete values

| Quantity | real | md |
|---|---|---|
| Box edge | 30.899 Å | 3.0899 nm |
| Water bead mass | 18.0153 g/mol | 18.0153 amu |
| Cutoff `rc` | 15.0 Å | 1.50 nm |
| Skin (suggested) | 0.5 Å | 0.05 nm |
| Timestep `dt` | 2 fs | 0.002 ps |
| `Tdamp` | 100 fs | 0.1 ps |
| Temperature → `kT` | 300 K | 2.49434 kJ/mol |
| Table `r` range | 0.5 – 15.0 Å | 0.05 – 1.50 nm |
| Sample `U(r=0.5 Å)` | 4.0144e5 kcal/mol | 1.6796e6 kJ/mol |
| Sample `F(r=0.5 Å)` | 5.0231e6 kcal/(mol·Å) | 2.1017e8 kJ/(mol·nm) |
| Run length | 150 000 × 2 fs = 300 ps | 150 000 × 0.002 ps = 300 ps |

## Phase 1 — Conversion script + static force check

### Goal

Validate that the table kernel + parser + unit conversion produce LAMMPS-equivalent per-atom forces and total PE on the initial CG water configuration. No integrator changes — forces are mass-independent.

### W2: `tools/convert_lammps_units.py`

Pure-Python tool, line-oriented. Standard library only.

```
tools/convert_lammps_units.py \
  --data data/CG_water/system.data \
  --table data/CG_water/pair_table.txt \
  --input data/CG_water/input.lmp \
  --src real --dst md \
  --out data/CG_water_md/
```

Outputs:
- `data/CG_water_md/system.data` — positions × 0.1, box bounds × 0.1, masses unchanged, Atom Style header preserved.
- `data/CG_water_md/pair_table.txt` — `r × 0.1`, `U × 4.184`, `F × 41.84`. Section keyword (`PAIR_0`) and `N 500` header preserved.
- `data/CG_water_md/fastmd.conf` — generated fastMD config with all converted parameters and the `forces_dump_file` key set for the Phase 1 static check (it can be removed/commented out for the Phase 3 trajectory run, or the script emits two configs — `fastmd_static.conf` and `fastmd_nvt.conf`).

Conversion factors are hard-coded for the `real → md` direction. The `--src` and `--dst` flags are scaffolding for future unit systems; only `real → md` is implemented in this iteration.

### W3: `data/CG_water_md/` dataset

Generated artifact, committed to the repo so reviewers can reproduce the validation without re-running the script. The script is rerun whenever the source files change.

### Phase 1 fastMD change: `forces_dump_file` config key

Single new config key:

```
forces_dump_file <path>
```

Behavior in `main.cu`: after `sys.allocate()` and the initial force computation pass, if `params.forces_dump_file[0] != '\0'`, copy `d_force` back to host, write each atom's `id fx fy fz pe` to the file as text, and `return 0` before entering the time-step loop. The key is silently ignored if absent. No effect on existing benchmarks.

Sketch:

```cpp
if (params.forces_dump_file[0] != '\0') {
    std::vector<float4> h_force(natoms);
    CUDA_CHECK(cudaMemcpy(h_force.data(), sys.d_force,
                          natoms * sizeof(float4), cudaMemcpyDeviceToHost));
    FILE* fp = fopen(params.forces_dump_file, "w");
    fprintf(fp, "# id fx fy fz pe\n");
    for (int i = 0; i < natoms; ++i)
        fprintf(fp, "%d %.6e %.6e %.6e %.6e\n", i,
                h_force[i].x, h_force[i].y, h_force[i].z, h_force[i].w);
    fclose(fp);
    return 0;
}
```

### `tools/static_force_check.py`

Orchestration script that:

1. Runs LAMMPS with a copy of `input.lmp` modified to `run 0` and to add `dump 1 all custom 1 forces_lammps.dump id fx fy fz` and a `thermo_style` including `pe`. Captures `forces_lammps.dump` and the step-0 PE.
2. Runs fastMD with `fastmd_static.conf` (which sets `forces_dump_file`). Captures `forces_fastmd.dat`.
3. Converts LAMMPS forces to MD units (× 41.84) and LAMMPS PE (× 4.184).
4. Builds atom-id-aligned arrays from both engines.
5. Reports max-abs per-component force error, RMS force error, and relative PE error.
6. Exits 0 if `max-abs < 1e-3 kJ/(mol·nm)` and `|ΔPE|/|PE| < 1e-4`, nonzero otherwise.

### Phase 1 file inventory

| File | New / Modified |
|---|---|
| `tools/convert_lammps_units.py` | new |
| `tools/static_force_check.py` | new |
| `data/CG_water_md/system.data` | new (generated) |
| `data/CG_water_md/pair_table.txt` | new (generated) |
| `data/CG_water_md/fastmd_static.conf` | new (generated) |
| `data/CG_water_md/fastmd_nvt.conf` | new (generated; used in Phase 3) |
| `src/io/config.hpp` | add `char forces_dump_file[256]` to `SimParams` |
| `src/io/config.cpp` | parse `forces_dump_file` key (~3 lines) |
| `src/main.cu` | early-return block (~20 lines) |

## Phase 2 — Mass support in fastMD

### Goal

Replace the implicit `m = 1` assumption with per-type masses, while keeping existing benchmarks bit-identical when no `Masses` section is present.

### Data path

1. **`src/io/lammps_data.cpp`** — currently skips `Masses` (line ~89). New behavior: add `Section::Masses` to the enum; when `current = Section::Masses`, parse `<type_id_1based> <mass>` lines into a vector keyed by 0-based type. Validate `mass > 0` and throw `std::runtime_error("mass must be > 0 for type N")` otherwise. Resize `topo.masses` to at least `ntypes` and default any unset entries to `1.0f`.
2. **`src/io/config.hpp`** — add `std::vector<float> masses` to `TopologyData`.
3. **`src/core/system.{cuh,cu}`** — masses live in a `__constant__ float c_masses[MAX_TYPES]` symbol (constant memory, cached, no register pressure). `MAX_TYPES = 16` is enough for any realistic CG system; static-assert if exceeded. Set via `cudaMemcpyToSymbol(c_masses, ...)` once after `System::allocate`.
4. **Type lookup at runtime:** existing helper `unpack_type_id(pos[i].w)` returns 0-based type id. Mass is `c_masses[unpack_type_id(pos[i].w)]`.

### Kernels touched

| File | Change |
|---|---|
| `src/integrate/langevin.cu` `integrator_pre_force_kernel` | Lookup `m_i = c_masses[type_i]` once at top. Replace `v += half_dt · F` with `v += (half_dt / m_i) · F`. Replace `noise_scale = c2 · sqrtf(kT)` with `noise_scale = c2 · sqrtf(kT / m_i)`. |
| `src/integrate/langevin.cu` `integrator_post_force_kernel` | Same `m_i` lookup; `v += (half_dt / m_i) · F`. |
| `src/integrate/nose_hoover.cu` (NVT and NPT) | Same `v += (half_dt / m_i) · F` change in the propagation kernels. Mass-weighted KE in the chain-update math (where `2 * KE = Σ m·v²` already appears symbolically; reads from the kinetic-stress accumulator below). |
| `src/analysis/thermo.cu` `kinetic_stress_kernel` | Multiply the contribution by `m_i`: `s[c] = m_i · v_α · v_β`. |
| `src/analysis/thermo.cu` final reduction | `T = 2·KE/(3·N)` formula unchanged (KE is now correctly mass-weighted, so the form still holds). |

### Kernels NOT touched

- Force kernels: `lj.cu`, `table.cu`, `fene.cu`, `angle.cu` — forces are independent of mass.
- Neighbor list, Morton sort, PBC update, FENE-bond traversal, binary dump — unaffected.
- Stress autocorrelation correlator — operates on pressure tensor output, which is now correctly mass-weighted upstream.

### Tests

- **Regression:** before/after diff of `thermo.dat` for `fastmd_benchmark.conf` (LJ + FENE polymer, no Masses section). Expected: byte-identical output because every atom's mass defaults to `1.0`.
- **New `tests/test_mass.cu`:**
  - Two-type system, `m_0 = 1.0`, `m_1 = 2.0`, same initial velocity, same applied force.
  - After one velocity-Verlet half-step, assert `(v_1 - v_1_0) = 0.5 · (v_0 - v_0_0)` (mass-2 receives half the velocity change of mass-1).
  - Compute KE via the thermo kernel and assert it equals analytic `0.5 · (m_0 · v_0² + m_1 · v_1²)` to 1e-5 relative.
- **Build target:** add `test_mass` to `tests/CMakeLists.txt`.

### Phase 2 file inventory

| File | New / Modified |
|---|---|
| `src/io/lammps_data.cpp` | parse Masses section |
| `src/io/config.hpp` | add `masses` to `TopologyData` |
| `src/core/system.cuh` | declare `__constant__ float c_masses[16]` |
| `src/core/system.cu` | upload via `cudaMemcpyToSymbol` in `System::allocate` |
| `src/integrate/langevin.cu` | mass-weighted velocity update + noise |
| `src/integrate/nose_hoover.cu` | mass-weighted velocity update; verify KE flows correctly |
| `src/analysis/thermo.cu` | mass-weighted kinetic stress |
| `tests/test_mass.cu` | new |
| `tests/CMakeLists.txt` | register `test_mass` |

### Scope NOT included in Phase 2

- Per-atom (non-per-type) masses.
- Mass-rescaling or generalized barostat coupling beyond what is required to keep NPT NH self-consistent under non-unit masses.
- Initial velocity sampling from Maxwell-Boltzmann. CG water's `system.data` has no Velocities section, so fastMD starts at `v=0` and the Langevin thermostat thermalizes over the first few `Tdamp` intervals. LAMMPS' `velocity create 300` thermalizes immediately. Both engines converge to the same equilibrium ensemble; only the equilibration window differs. This is accounted for by the `--equilibration-skip` argument in the comparison script.

## Phase 3 — Langevin NVT validation harness

### Goal

Run identical 300-ps Langevin NVT trajectories in fastMD and LAMMPS, compare time-averaged `<T>`, `<PE>`, `<P>` after equilibration.

### Generated fastMD config (`data/CG_water_md/fastmd_nvt.conf`)

```
natoms 1000
ntypes 1
rc 1.5
skin 0.05
dt 0.002
nsteps 150000
dump_freq 0
thermo 1 1200 thermo_fastmd.dat
nvt_langevin 2.49434 2.49434 0.1 12345
table 0 0 data/CG_water_md/pair_table.txt PAIR_0
lammps_data_file data/CG_water_md/system.data
```

Parameters mirror `input.lmp`: same total wall-time (300 ps), same Tdamp (0.1 ps = 100 fs), same thermo-output interval (every 1200 steps = 2.4 ps).

### LAMMPS reference run

Use the unmodified `data/CG_water/input.lmp`. Its existing `thermo_style custom step temp press pe ke etotal vol` is sufficient. The Langevin seed (`12345`) is the same as fastMD's — this does NOT make trajectories microscopically identical (LAMMPS Marsaglia vs fastMD Philox), but does keep the engines comparably stochastic.

### Comparison: `tools/compare_thermo.py`

```
tools/compare_thermo.py \
  --lammps data/CG_water/log.lammps \
  --fastmd thermo_fastmd.dat \
  --equilibration-skip 50.0    # ps
```

Steps:
1. Parse both logs into `(t_ps, T, PE, P)` arrays. fastMD's `Pxx, Pyy, Pzz` are averaged into scalar `P_md`.
2. Convert LAMMPS columns: `T_K → kT_md = T_K · 0.00831446`, `PE_kcalpermol → PE_kJpermol = × 4.184`, `P_atm → P_md = × 0.061027`.
3. Filter rows with `t < equilibration_skip`.
4. Compute time-averaged `<T>`, `<PE>`, `<P>` and standard error of the mean (SEM = stdev / √N). With dump interval 2.4 ps ≫ Tdamp = 0.1 ps, samples are already decorrelated, so block-averaging is unnecessary.
5. Print a side-by-side table: absolute values, deltas, relative errors, and SEM-normalized z-scores.
6. Exit code 0 if all relative errors are within tolerance; nonzero otherwise.

### Pass criteria

| Quantity | Tolerance | Rationale |
|---|---|---|
| Step-0 max-abs per-atom force error | < 1e-3 kJ/(mol·nm) | Validated in Phase 1 — re-checked in Phase 3 as a sanity precondition |
| Step-0 total PE relative error | < 1e-4 | Validated in Phase 1 |
| `<T>` relative error | < 1% | SEM at 100 independent samples is ~0.06% — 1% gives ~16σ margin |
| `<PE>` relative error | < 1% | Same statistical reasoning as `<T>` |
| `<P>` relative error | < 5% | Pressure is noisier (virial vs kinetic near-cancellation) |

### Failure-mode triage

- Phase 1 fails → kernel / parser / unit conversion bug. Do **not** proceed to Phase 2 until fixed.
- Phase 1 passes, Phase 3 `<T>` wrong → mass not propagating through Langevin. Check `c_masses` upload and the `sqrtf(kT/m)` line.
- Phase 1 passes, `<T>` correct, `<PE>` wrong → table kernel under finite density / PBC differs from snapshot. Check virial reduction and PBC handling in `table.cu`.
- Everything passes → table potential is end-to-end validated.

### Phase 3 file inventory

| File | New / Modified |
|---|---|
| `tools/compare_thermo.py` | new |
| `tools/run_validation.sh` | new — orchestrates LAMMPS run, fastMD run, comparison |
| `data/CG_water_md/fastmd_nvt.conf` | generated in Phase 1 by W2 |

## Out of Scope

- Other unit systems (`metal`, `electron`, custom). The script's `--src/--dst` flags are scaffolding only.
- Per-atom masses or isotope mixing.
- Mass-rescaling barostats or velocity-rescaling thermostats beyond what existing Langevin/NH already do.
- Maxwell-Boltzmann initial velocity sampling in fastMD.
- Radial distribution function comparison (would require new analysis module).
- Performance optimization. Phase 2 may slow the integrator slightly due to the constant-memory mass lookup; this is acceptable.

## Open Questions

None at design time. Open items will be tracked in the implementation plan written by the writing-plans skill.
