#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NH NPT Integration Tests ==="

# Ensure data symlink
if [ ! -e data ]; then
    : "${BONDED_RING_DATA:?set BONDED_RING_DATA to bonded_ring_n40m30M25.data}"
    ln -sf "$BONDED_RING_DATA" data
    echo "Created data symlink"
fi

echo ""
echo "--- Test: NPT thermo vs LAMMPS ---"
python3 test_npt_thermo.py

echo ""
echo "=== All NH NPT tests passed ==="
