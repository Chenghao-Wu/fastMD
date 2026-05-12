# CG Water RDF Validation Design

## Goal

Validate fastMD's CG water model by comparing the O-O radial distribution function g(r) against a LAMMPS reference (`1water_oo_rdf.dat`), using visual overlay as the success criterion.

## Deliverable

A Jupyter notebook at `data/CG_water_md/validate_rdf.ipynb` that reads the fastMD binary trajectory, computes the RDF, and plots it overlaid with the reference.

## Simulation setup

Change `dump_freq` from `0` to `150` in `fastmd_nvt.conf` to output ~1000 frames over the 150000-step run (matching the reference's 1001 frames).

```
dump_freq 150
```

Run: `./build/fastmd data/CG_water_md/fastmd_nvt.conf` produces `data/CG_water_md/traj.bin`.

## Notebook cells

1. **Config note** — document the `dump_freq 150` change
2. **Binary reader** — parse fastMD trajectory header (magic/natoms/ntypes) and per-frame records (step/natoms/box_L/positions)
3. **RDF computation** — pairwise distance histogram with minimum-image convention, normalized to g(r)
4. **Reference loader** — read the two-column `1water_oo_rdf.dat`, skipping `#` comment lines
5. **Overlay plot** — matplotlib plot of both g(r) curves

## RDF algorithm

- Bin width `dr = 0.003 nm`, max distance `r_max = box_L / 2 ≈ 1.545 nm` (~515 bins)
- Compute all N×N pairwise distances per frame using minimum-image convention for PBC
- Accumulate histogram over all frames
- Normalize: `g(r) = count(r) / (4π × r² × dr × ρ × N_frames × N/2)`
- Process one frame at a time to keep memory low

## Error handling

- Missing `traj.bin`: show message with the exact command to run
- Incomplete file: verify expected size from header
- g(r)=0 at small r: expected due to excluded volume, matches reference
- Bin alignment: use same `dr=0.003 nm` as reference for direct overlay
