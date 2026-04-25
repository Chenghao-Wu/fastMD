# fastMD vs LAMMPS Benchmark Harness Design

## Goal
Create a reusable Python-based benchmark that compares the raw simulation speed of fastMD and LAMMPS for the sample system `input_000.data` (30 000 atoms, 29 700 bonds, 29 400 angles), while ensuring both codes use identical physics.

## Non-Goals
- Automated regression tracking or CI integration (the harness outputs structured data; CI wiring is future work).
- Benchmarking on multiple GPUs or strong/weak scaling studies.

## Architecture

The benchmark lives in `benchmarks/` and has three parts:

1. **Config generator** (`benchmarks/generate_lammps_input.py`) — Reads a shared benchmark JSON config and emits both a fastMD `.conf` file and a LAMMPS input script with matching physics parameters.
2. **Runner** (`benchmarks/run.py`) — Orchestrates validation and benchmark phases, collects timing, and invokes the reporter.
3. **Reporter** (`benchmarks/report.py`) — Writes a JSON report with timing, speedup, and validation results.

## fastMD Extension: LAMMPS Data File Parser

fastMD's `main.cu` currently does not load bonds or angles from the LAMMPS data file (`sys.nbonds` and `sys.nangles` remain zero). Because the sample system contains bonded interactions, a fair benchmark requires parsing them.

- **New file**: `src/io/lammps_data.cpp` / `lammps_data.hpp`
  - Function `parse_lammps_data(const std::string& path, TopologyData& topo)` parses the `Atoms`, `Bonds`, and `Angles` sections of a LAMMPS data file.
  - Populates `topo.positions`, `topo.bonds`, `topo.angles`, and `topo.bond_types`.
  - Zero-indexed atom IDs (LAMMPS uses one-based).
- **`main.cu` change**: After `parse_config()`, call `parse_lammps_data()` if the config specifies a LAMMPS data file. Upload the resulting bond/angle arrays to device memory and initialize `d_fene_params` and `d_angle_params` from the config-read parameters.

## Input Generation

The benchmark config (`benchmarks/benchmark.json`) specifies:
- `data_file`: path to LAMMPS data file (e.g. `input_000.data`)
- `temperature`: 1.0
- `dt`: timestep
- `rc`: LJ cutoff
- `skin`: neighbor-list skin
- `nsteps`: 1 000 000 for benchmark, 1 000 for validation
- `thermo_freq`: log interval
- `dump_freq`: 0 (no I/O during benchmark)
- `seed`: random seed
- `lj_params`, `bond_params`, `angle_params`: potential parameters

`generate_lammps_input.py` writes a LAMMPS input that:
- Uses `package gpu 1` and `suffix gpu`
- Reads the data file with `read_data`
- Sets `pair_style lj/cut`, `bond_style fene`, `angle_style harmonic`
- Sets `fix langevin` for NVT at the target temperature
- Runs for the requested number of steps with `run`
- Outputs thermo data every `thermo_freq` steps

The fastMD `.conf` file uses the same numerical values.

**Parameter mapping notes:**
- fastMD `gamma` maps to LAMMPS `fix langevin` damping parameter (same units, both are friction coefficients).
- fastMD FENE bond parameters `{k, R0, eps, sig}` map to LAMMPS `bond_style fene` with `K = k`, `R0 = R0`. The WCA repulsion between all atom pairs is handled by `pair_style lj/cut` using the same `eps` and `sig`.
- fastMD harmonic angle parameters `{k_theta, theta0}` map directly to LAMMPS `angle_style harmonic`.
- fastMD uses zero-based atom IDs internally; the parser converts LAMMPS one-based IDs.

## Two-Phase Execution

### Phase 1: Validation (1 000 steps)
1. Generate inputs for 1 000 steps with `thermo_freq = 100`.
2. Run fastMD and LAMMPS.
3. Parse both logs for potential energy and temperature at each thermo output.
4. Compute average potential energy and average temperature over the last 500 steps.
5. Check tolerances:
   - `|T_avg_fastMD - T_avg_lammps| / T_target < 0.05` (within 5% of target temperature)
   - `|PE_avg_fastMD - PE_avg_lammps| / |PE_avg_lammps| < 1e-2`
6. If checks fail, abort and print the energy/temperature traces for debugging.

### Phase 2: Benchmark (1 000 000 steps)
1. Generate inputs for 1 000 000 steps with `dump_freq = 0` and `thermo_freq = 1000` for both codes to minimize I/O overhead.
2. Run fastMD, measure wall time with `time.perf_counter()`.
3. Run LAMMPS, measure wall time the same way.
4. Compute `ns_per_day = (nsteps * dt) / (wall_time_s / 86400)`.
5. Compute `speedup = lammps_time / fastmd_time`.

## Error Handling
- Non-zero exit code from either simulator aborts the benchmark and surfaces stderr.
- Validation failure aborts before the benchmark phase.
- GPU/CUDA errors in fastMD are fatal and terminate the process; the runner detects this via exit code.

## Output Format

`benchmark_report.json`:
```json
{
  "timestamp": "2026-04-25T14:30:00",
  "gpu": "NVIDIA A100",
  "system": "input_000.data",
  "n_atoms": 30000,
  "nsteps": 1000000,
  "fastMD": {
    "wall_time_s": 45.2,
    "ns_per_day": 41.3
  },
  "lammps": {
    "wall_time_s": 62.1,
    "ns_per_day": 30.1
  },
  "speedup": 1.37,
  "validation": {
    "passed": true,
    "rel_pe_diff": 8.1e-3,
    "rel_temp_diff": 0.02
  }
}
```

## Testing
- The validation phase itself serves as the correctness test.
- A manual smoke test: run the harness on the 64-atom `tests/fixtures/lj_64.xyz` (without bonds/angles) to verify the runner works end-to-end before running the 30k-atom benchmark.

## Files Changed / Added
- `src/io/lammps_data.hpp` (new)
- `src/io/lammps_data.cpp` (new)
- `src/main.cu` (modified: load bonds/angles from data file)
- `benchmarks/generate_lammps_input.py` (new)
- `benchmarks/run.py` (new)
- `benchmarks/report.py` (new)
- `benchmarks/benchmark.json` (new)
