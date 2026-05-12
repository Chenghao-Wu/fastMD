# CG Water RDF Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate fastMD's CG water model by computing the O-O RDF from a simulation trajectory and comparing against the LAMMPS reference.

**Architecture:** A Jupyter notebook at `data/CG_water_md/validate_rdf.ipynb` reads the binary trajectory, computes the pairwise RDF histogram with minimum-image PBC, and plots g(r) overlaid with the reference. The config change (`dump_freq 150`) enables trajectory output.

**Tech Stack:** Python 3, NumPy, Matplotlib, Jupyter (ipynb), fastMD binary trajectory format

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `data/CG_water_md/fastmd_nvt.conf` | Modify | `dump_freq 0` → `dump_freq 150` |
| `data/CG_water_md/validate_rdf.ipynb` | Create | Notebook: read traj, compute RDF, plot overlay |
| `data/CG_water_md/traj.bin` | Generated | Binary trajectory (output of simulation run) |

---

### Task 1: Enable trajectory output in config

**Files:**
- Modify: `data/CG_water_md/fastmd_nvt.conf:7`

- [ ] **Step 1: Change dump_freq from 0 to 150**

```
dump_freq 150
```

This outputs every 150 steps, yielding ~1000 frames over the 150000-step run, matching the reference.

- [ ] **Step 2: Verify the edit**

Run: `grep dump_freq data/CG_water_md/fastmd_nvt.conf`
Expected: `dump_freq 150`

- [ ] **Step 3: Commit**

```bash
git add data/CG_water_md/fastmd_nvt.conf
git commit -m "feat: enable trajectory output (dump_freq 150) for CG water RDF validation"
```

---

### Task 2: Build and run the simulation

**Files:**
- Generated: `data/CG_water_md/traj.bin`

- [ ] **Step 1: Build fastMD**

Run: `cmake -B build -S . && cmake --build build -j$(nproc)`
Expected: successful compilation, binary at `build/fastmd`

- [ ] **Step 2: Run the simulation**

Run: `./build/fastmd data/CG_water_md/fastmd_nvt.conf`
Expected: completes all 150000 steps, generates `data/CG_water_md/traj.bin`

- [ ] **Step 3: Verify trajectory file**

Run: `ls -lh data/CG_water_md/traj.bin`
Expected: file exists, ~12 MB (12 + 1000 frames × 12016 bytes ≈ 12,016,012 bytes)

- [ ] **Step 4: Verify trajectory header**

```python
python3 -c "
import struct
with open('data/CG_water_md/traj.bin', 'rb') as f:
    magic = struct.unpack('i', f.read(4))[0]
    natoms = struct.unpack('i', f.read(4))[0]
    ntypes = struct.unpack('i', f.read(4))[0]
print(f'magic={hex(magic)}, natoms={natoms}, ntypes={ntypes}')
"
```
Expected: `magic=0x4d444247, natoms=1000, ntypes=1`

- [ ] **Step 5: Verify frame count**

```python
python3 -c "
import struct, os
frame_size = 8 + 4 + 4 + 1000 * 12
header_size = 12
file_size = os.path.getsize('data/CG_water_md/traj.bin')
n_frames = (file_size - header_size) // frame_size
rem = (file_size - header_size) % frame_size
print(f'{n_frames} frames, remainder={rem}')
"
```
Expected: `1000 frames, remainder=0`

---

### Task 3: Create the Jupyter notebook — header and binary reader

**Files:**
- Create: `data/CG_water_md/validate_rdf.ipynb`

- [ ] **Step 1: Create the notebook with cell 1 (markdown — config note) and cell 2 (binary reader)**

The notebook structure (create with `nbformat` or manually via the IDE):

**Cell 1 (markdown):**

```markdown
# CG Water O-O RDF Validation

This notebook validates fastMD's CG water model by comparing the
O-O radial distribution function g(r) against a LAMMPS reference.

**Prerequisite:** `dump_freq 150` set in `fastmd_nvt.conf` before simulation.
The simulation produces `traj.bin` in this directory.
```

**Cell 2 (code) — Binary trajectory reader:**

```python
import struct
import os
import numpy as np

def read_trajectory(path):
    """Read fastMD binary trajectory. Returns dict with natoms, ntypes, frames."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the simulation first:\n"
            f"  ./build/fastmd data/CG_water_md/fastmd_nvt.conf"
        )

    with open(path, 'rb') as f:
        magic = struct.unpack('i', f.read(4))[0]
        if magic != 0x4d444247:
            raise ValueError(f"Bad magic: {hex(magic)}, expected 0x4d444247")
        natoms = struct.unpack('i', f.read(4))[0]
        ntypes = struct.unpack('i', f.read(4))[0]

    header_size = 12
    frame_size = 8 + 4 + 4 + natoms * 12  # step(int64) + natoms(int32) + box_L(float32) + positions(float3)
    file_size = os.path.getsize(path)
    data_size = file_size - header_size

    if data_size % frame_size != 0:
        raise ValueError(
            f"Trajectory appears incomplete: data_size={data_size}, "
            f"frame_size={frame_size}, remainder={data_size % frame_size}"
        )

    n_frames = data_size // frame_size
    frames = []

    with open(path, 'rb') as f:
        f.seek(header_size)
        for _ in range(n_frames):
            raw = f.read(frame_size)
            step = struct.unpack('q', raw[0:8])[0]
            n = struct.unpack('i', raw[8:12])[0]
            box_L = struct.unpack('f', raw[12:16])[0]
            pos = np.frombuffer(raw[16:16 + n * 12], dtype=np.float32).reshape(n, 3)
            frames.append({'step': step, 'box_L': box_L, 'positions': pos})

    print(f"Loaded {n_frames} frames, natoms={natoms}, ntypes={ntypes}")
    return {'natoms': natoms, 'ntypes': ntypes, 'frames': frames}

# Check if trajectory exists and load first frame to verify
traj_path = 'traj.bin'
file_size = os.path.getsize(traj_path) if os.path.exists(traj_path) else 0
print(f"traj.bin: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

data = read_trajectory(traj_path)
box_L = data['frames'][0]['box_L']
natoms = data['natoms']
print(f"box_L = {box_L:.4f} nm, r_max = {box_L/2:.4f} nm")
```

- [ ] **Step 2: Run the notebook up to cell 2**

Expected output:
```
traj.bin: 12,016,012 bytes (11.5 MB)
Loaded 1000 frames, natoms=1000, ntypes=1
box_L = 3.0899 nm, r_max = 1.5449 nm
```

- [ ] **Step 3: Commit**

```bash
git add data/CG_water_md/validate_rdf.ipynb
git commit -m "feat: add RDF validation notebook with binary trajectory reader"
```

---

### Task 4: Add RDF computation cell

**Files:**
- Modify: `data/CG_water_md/validate_rdf.ipynb`

- [ ] **Step 1: Add cell 3 (code) — RDF computation**

```python
def compute_rdf(frames, dr, r_max):
    """Compute g(r) from trajectory frames with minimum-image PBC."""
    nbins = int(r_max / dr)
    total_hist = np.zeros(nbins, dtype=np.float64)
    n_frames = len(frames)

    for fi, frame in enumerate(frames):
        pos = frame['positions']
        box_L = frame['box_L']
        n = len(pos)

        for i in range(n):
            delta = pos - pos[i]
            # Minimum image convention
            delta -= box_L * np.round(delta / box_L)
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            # Only upper triangle: j > i
            mask = np.arange(n) > i
            dist = dist[mask]
            # Bin distances
            idx = (dist / dr).astype(np.int32)
            valid = idx < nbins
            np.add.at(total_hist, idx[valid], 1)

        if (fi + 1) % 100 == 0:
            print(f"  processed {fi + 1}/{n_frames} frames")

    # Normalize to g(r)
    r = (np.arange(nbins) + 0.5) * dr
    V = box_L ** 3
    rho = natoms / V
    # Shell volume: 4π r² dr
    shell_vol = 4.0 * np.pi * r * r * dr
    # Normalization: total pairs counted = N_frames * N * (N-1) / 2
    norm = rho * n_frames * natoms * (natoms - 1) / 2.0
    gr = total_hist / (shell_vol * norm)

    return r, gr

# Match reference binning: dr=0.003 nm, r_max = box_L/2
dr = 0.003
r_max = box_L / 2.0
print(f"Computing RDF: dr={dr} nm, r_max={r_max:.4f} nm, nbins={int(r_max/dr)}")
print(f"Reference used 501 bins from ~0 to {0.0015 + 500*0.003:.4f} nm")

r_bins, gr = compute_rdf(data['frames'], dr, r_max)
print("RDF computation complete")
```

- [ ] **Step 2: Run the notebook up to cell 3**

Expected: prints progress every 100 frames, then "RDF computation complete"

- [ ] **Step 3: Commit**

```bash
git add data/CG_water_md/validate_rdf.ipynb
git commit -m "feat: add RDF computation cell to validation notebook"
```

---

### Task 5: Add reference loader and overlay plot

**Files:**
- Modify: `data/CG_water_md/validate_rdf.ipynb`

- [ ] **Step 1: Add cell 4 (code) — load reference RDF**

```python
def load_reference_rdf(path):
    """Load reference RDF file (two-column: r(nm) g(r)), skip comments."""
    r_vals, gr_vals = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                r_vals.append(float(parts[0]))
                gr_vals.append(float(parts[1]))
    return np.array(r_vals), np.array(gr_vals)

ref_path = '1water_oo_rdf.dat'
ref_r, ref_gr = load_reference_rdf(ref_path)
print(f"Loaded reference: {len(ref_r)} points, r in [{ref_r[0]:.4f}, {ref_r[-1]:.4f}] nm")
```

- [ ] **Step 2: Add cell 5 (code) — overlay plot**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ref_r, ref_gr, 'k-', linewidth=1.5, label='LAMMPS reference', alpha=0.8)
ax.plot(r_bins, gr, 'r-', linewidth=1.5, label='fastMD', alpha=0.8)

ax.set_xlabel('r (nm)')
ax.set_ylabel('g(r)')
ax.set_title('CG Water O-O RDF: fastMD vs LAMMPS')
ax.legend()
ax.set_xlim(0, r_max)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('rdf_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved to rdf_comparison.png")
plt.show()
```

- [ ] **Step 3: Run the full notebook**

Expected: plot appears inline with both curves overlaid. The fastMD g(r) should show the same peak structure as the LAMMPS reference (excluded volume region at small r, first peak, subsequent solvation shells). The plot is also saved as `rdf_comparison.png`.

- [ ] **Step 4: Commit**

```bash
git add data/CG_water_md/validate_rdf.ipynb
git commit -m "feat: add reference loader and overlay plot to RDF validation notebook"
```

---

## Verification

After all tasks, run the full notebook end-to-end and verify:

1. `traj.bin` loads without errors — 1000 frames, 1000 atoms
2. RDF computation finishes without error
3. Plot shows fastMD g(r) overlaid with LAMMPS reference
4. Visual inspection: peaks align in position and approximate height, excluded volume region at small r is correct
