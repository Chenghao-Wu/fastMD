#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Bonded Ring Integration Tests ==="
echo ""

# Ensure data symlink exists
if [ ! -e data ]; then
    ln -sf /home/zhenghaowu/fastMD/data/bonded_ring_n40m30M25.data data
    echo "Created data symlink"
fi

# Step 1: Run 10-step simulations for force test
echo "--- Running 10-step simulations for force test ---"

python3 << 'PYEOF'
import utils
utils.set_work_dir()

print("Generating fastMD config for 10 steps...")
fastmd_config = utils.write_fastmd_config(10)
lammps_input = utils.write_lammps_rerun_input(10, "log.lammps_10")

print("Running fastMD (10 steps)...")
result = utils.run_fastmd(fastmd_config)
if result.returncode != 0:
    print("ERROR: fastMD 10-step run failed")
    print(result.stderr)
    exit(1)

print("Running LAMMPS (10 steps)...")
result = utils.run_lammps(lammps_input)
if result.returncode != 0:
    print("ERROR: LAMMPS 10-step run failed")
    print(result.stderr)
    exit(1)
print("10-step simulations complete.")
PYEOF

echo ""

# Step 2: Run 5000-step simulations for thermo and drift tests
echo "--- Running 5000-step simulations for thermo and drift tests ---"

python3 << 'PYEOF'
import utils
utils.set_work_dir()

print("Generating fastMD config for 5000 steps...")
fastmd_config = utils.write_fastmd_config(5000)
lammps_input = utils.write_lammps_rerun_input(5000, "log.lammps")

print("Running fastMD (5000 steps)...")
result = utils.run_fastmd(fastmd_config)
if result.returncode != 0:
    print("ERROR: fastMD 5000-step run failed")
    print(result.stderr)
    exit(1)

print("Running LAMMPS (5000 steps)...")
result = utils.run_lammps(lammps_input)
if result.returncode != 0:
    print("ERROR: LAMMPS 5000-step run failed")
    print(result.stderr)
    exit(1)
print("5000-step simulations complete.")
PYEOF

echo ""

# Step 3: Extract LAMMPS thermo from log files
echo "--- Extracting LAMMPS thermo data ---"
python3 << 'PYEOF'
import os

def extract_from_log(log_file, output_file):
    """Extract thermo table from LAMMPS log file."""
    if not os.path.exists(log_file):
        print(f"ERROR: {log_file} not found")
        return

    thermo_lines = []
    in_thermo = False

    with open(log_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Step") and ("KinEng" in stripped or "kineng" in stripped.lower()):
                thermo_lines.append("# step  KE  PE  T  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz")
                in_thermo = True
                continue
            if in_thermo:
                if stripped and stripped.split()[0].isdigit():
                    parts = stripped.split()
                    if len(parts) >= 10:
                        thermo_lines.append("  ".join(parts[:10]))
                elif stripped.startswith("Loop time"):
                    break

    with open(output_file, "w") as f:
        f.write("\n".join(thermo_lines) + "\n")

    print(f"Extracted {len(thermo_lines)-1} thermo lines to {output_file}")

extract_from_log("log.lammps_10", "thermo_lammps_10.dat")
extract_from_log("log.lammps", "thermo_lammps.dat")
PYEOF

echo ""

# Step 4: Run tests
echo "--- Running test: test_forces.py ---"
python3 test_forces.py
echo ""

echo "--- Running test: test_thermo.py ---"
python3 test_thermo.py
echo ""

echo "--- Running test: test_energy_drift.py ---"
python3 test_energy_drift.py
echo ""

echo "=== All tests passed ==="
