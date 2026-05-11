#!/usr/bin/env python3
"""Compare initial-state forces and PE between LAMMPS and fastMD.

Runs LAMMPS with run 0 to dump initial forces, runs fastMD with
forces_dump_file, then compares in MD units.

Usage:
    tools/static_force_check.py \
      --lammps data/CG_water/ \
      --fastmd-conf data/CG_water_md/fastmd_static.conf \
      --fastmd-bin build/fastmd
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

FORCE_CONV = 41.84      # kcal/(mol*A) -> kJ/(mol*nm)
ENERGY_CONV = 4.184     # kcal/mol -> kJ/mol


def run_lammps_static(data_dir):
    """Run LAMMPS with run 0, dump forces and PE."""
    lammps_contents = f"""units real
atom_style full
boundary p p p
read_data {os.path.join(data_dir, 'system.data')}

pair_style table linear 500
pair_modify shift yes
pair_coeff 1 1 {os.path.join(data_dir, 'pair_table.txt')} PAIR_0 15.0000

special_bonds lj 0.0 0.0 0.0

velocity all create 300 12345 dist gaussian

fix 1 all nve
fix 2 all langevin 300 300 100.0 12345

timestep 2
thermo_style custom step temp press pe ke etotal vol
thermo 1
dump d1 all custom 1 forces_lammps.dump id fx fy fz

run 0
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'in.lmp')
        with open(input_path, 'w') as f:
            f.write(lammps_contents)

        result = subprocess.run(
            ['lmp', '-in', input_path],
            cwd=tmpdir,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("LAMMPS run failed:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        # Parse forces dump
        forces_dump = os.path.join(tmpdir, 'forces_lammps.dump')
        forces = {}
        with open(forces_dump, 'r') as f:
            in_data = False
            for line in f:
                line = line.strip()
                if line.startswith('ITEM: ATOMS'):
                    in_data = True
                    continue
                if in_data and line:
                    parts = line.split()
                    if len(parts) >= 5:
                        atom_id = int(parts[0])
                        fx = float(parts[1])
                        fy = float(parts[2])
                        fz = float(parts[3])
                        forces[atom_id] = (fx, fy, fz)

        # Parse PE from thermo output
        pe = None
        thermo_file = os.path.join(tmpdir, 'log.lammps')
        with open(thermo_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[0].isdigit():
                    try:
                        pe = float(parts[3])
                    except ValueError:
                        continue

        if pe is None:
            print("Error: could not parse LAMMPS PE", file=sys.stderr)
            sys.exit(1)

        if not forces:
            print("Error: no forces parsed from LAMMPS dump", file=sys.stderr)
            sys.exit(1)

        return forces, pe


def run_fastmd_static(conf_path, fastmd_bin):
    """Run fastMD with forces_dump_file config, parse output."""
    conf_dir = os.path.dirname(os.path.abspath(conf_path))
    conf_basename = os.path.basename(conf_path)

    result = subprocess.run(
        [os.path.abspath(fastmd_bin), conf_basename],
        cwd=conf_dir,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("fastMD run failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Parse forces_fastmd.dat
    forces_file = os.path.join(conf_dir, 'forces_fastmd.dat')
    forces = {}
    with open(forces_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                atom_id = int(parts[0])
                fx = float(parts[1])
                fy = float(parts[2])
                fz = float(parts[3])
                pe = float(parts[4])
                forces[atom_id] = (fx, fy, fz, pe)

    # Parse PE from thermo
    thermo_file = os.path.join(conf_dir, 'thermo_fastmd_static.dat')
    pe = None
    with open(thermo_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                pe = float(parts[2])
                break

    if pe is None:
        print("Error: could not parse fastMD PE", file=sys.stderr)
        sys.exit(1)

    return forces, pe


def main():
    parser = argparse.ArgumentParser(
        description='Compare initial-state forces between LAMMPS and fastMD')
    parser.add_argument('--lammps', required=True,
                        help='Directory containing LAMMPS input data')
    parser.add_argument('--fastmd-conf', required=True,
                        help='Path to fastMD static config')
    parser.add_argument('--fastmd-bin', default='build/fastmd',
                        help='Path to fastMD binary')
    args = parser.parse_args()

    print("Running LAMMPS static (run 0)...")
    lmp_forces, lmp_pe = run_lammps_static(args.lammps)

    # Convert LAMMPS forces and PE to MD units
    lmp_forces_md = {}
    for atom_id, (fx, fy, fz) in lmp_forces.items():
        lmp_forces_md[atom_id - 1] = (fx * FORCE_CONV, fy * FORCE_CONV, fz * FORCE_CONV)
    lmp_pe_md = lmp_pe * ENERGY_CONV

    print("Running fastMD static...")
    fmd_forces, fmd_pe = run_fastmd_static(args.fastmd_conf, args.fastmd_bin)

    # Verify atom ID sets match
    if set(lmp_forces_md.keys()) != set(fmd_forces.keys()):
        print("Error: atom ID sets do not match", file=sys.stderr)
        print(f"  LAMMPS IDs: {sorted(lmp_forces_md.keys())[:10]}... ({len(lmp_forces_md)})")
        print(f"  fastMD IDs: {sorted(fmd_forces.keys())[:10]}... ({len(fmd_forces)})")
        sys.exit(1)

    natoms = len(lmp_forces_md)

    # Per-component max absolute error
    max_abs_fx = 0.0
    max_abs_fy = 0.0
    max_abs_fz = 0.0
    rms_sum = 0.0

    for i in range(natoms):
        lfx, lfy, lfz = lmp_forces_md[i]
        ffx, ffy, ffz, _ = fmd_forces[i]
        dfx = abs(lfx - ffx)
        dfy = abs(lfy - ffy)
        dfz = abs(lfz - ffz)
        max_abs_fx = max(max_abs_fx, dfx)
        max_abs_fy = max(max_abs_fy, dfy)
        max_abs_fz = max(max_abs_fz, dfz)
        rms_sum += dfx*dfx + dfy*dfy + dfz*dfz

    rms = math.sqrt(rms_sum / (3 * natoms))
    max_abs = max(max_abs_fx, max_abs_fy, max_abs_fz)
    pe_rel = abs(fmd_pe - lmp_pe_md) / abs(lmp_pe_md)

    print(f"\n  Atoms: {natoms}")
    print(f"  Max |dfx|:     {max_abs_fx:.6e} kJ/(mol*nm)")
    print(f"  Max |dfy|:     {max_abs_fy:.6e} kJ/(mol*nm)")
    print(f"  Max |dfz|:     {max_abs_fz:.6e} kJ/(mol*nm)")
    print(f"  Max abs force: {max_abs:.6e} kJ/(mol*nm)")
    print(f"  RMS force:     {rms:.6e} kJ/(mol*nm)")
    print(f"  PE LAMMPS MD:  {lmp_pe_md:.6f} kJ/mol")
    print(f"  PE fastMD:     {fmd_pe:.6f} kJ/mol")
    print(f"  PE rel error:  {pe_rel:.6e}")

    passed = True
    if max_abs >= 1e-3:
        print(f"  FAIL: max abs force {max_abs:.3e} >= 1e-3 kJ/(mol*nm)")
        passed = False
    else:
        print(f"  PASS: max abs force {max_abs:.3e} < 1e-3 kJ/(mol*nm)")

    if pe_rel >= 1e-4:
        print(f"  FAIL: PE rel error {pe_rel:.3e} >= 1e-4")
        passed = False
    else:
        print(f"  PASS: PE rel error {pe_rel:.3e} < 1e-4")

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
