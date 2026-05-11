#!/bin/bash
# Full CG water table potential validation pipeline.
#
# Phases:
# 1. Static force comparison (fastMD vs LAMMPS, run 0)
# 2. Run full fastMD NVT trajectory
# 3. Compare time-averaged thermo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
FASTMD_BIN="${BUILD_DIR}/fastMD"
DATA_DIR="${PROJECT_DIR}/data/CG_water"
MD_DIR="${PROJECT_DIR}/data/CG_water_md"

echo "=== CG Water Table Validation ==="
echo ""

# Check prerequisites
if [ ! -f "$FASTMD_BIN" ]; then
    echo "Building fastMD..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd "$PROJECT_DIR"
fi

# Phase 1: Static force check (requires lmp in PATH)
echo "--- Phase 1: Static Force Check ---"
if command -v lmp &> /dev/null; then
    python3 "${SCRIPT_DIR}/static_force_check.py" \
        --lammps "$DATA_DIR" \
        --fastmd-conf "${MD_DIR}/fastmd_static.conf" \
        --fastmd-bin "$FASTMD_BIN"
    echo "Phase 1 PASSED"
else
    echo "WARNING: lmp (LAMMPS) not found in PATH; skipping Phase 1 static force comparison"
fi
echo ""

# Phase 2: Run fastMD NVT trajectory
echo "--- Phase 2: Run fastMD NVT Trajectory ---"
cd "$MD_DIR"
"$FASTMD_BIN" fastmd_nvt.conf
cd "$PROJECT_DIR"
echo "fastMD NVT trajectory complete"
echo ""

# Phase 3: Thermo comparison
echo "--- Phase 3: Thermo Comparison ---"
python3 "${SCRIPT_DIR}/compare_thermo.py" \
    --lammps "${DATA_DIR}/log.lammps" \
    --fastmd "${MD_DIR}/thermo_fastmd.dat" \
    --equilibration-skip 50.0
echo "Phase 3 PASSED"
echo ""

echo "=== All validation phases passed ==="
