#!/usr/bin/env python3
"""Convert LAMMPS real-units dataset to HOOMD MD reduced units.

Usage:
    tools/convert_lammps_units.py \
      --data data/CG_water/system.data \
      --table data/CG_water/pair_table.txt \
      --input data/CG_water/input.lmp \
      --src real --dst md \
      --out data/CG_water_md/
"""

import argparse
import os
import sys

# Conversion factors: LAMMPS real -> HOOMD md
# length:     A -> nm            x 0.1
# energy:     kcal/mol -> kJ/mol x 4.184
# mass:       g/mol = amu        x 1.0
# force:      kcal/(mol*A) -> kJ/(mol*nm)  x 41.84
# time:       fs -> ps           x 0.001
# temperature: K -> kJ/mol       x 0.00831446  (kB)
# pressure:   atm -> kJ/(mol*nm^3) x 0.061027

LENGTH = 0.1
ENERGY = 4.184
FORCE = 41.84
TIME = 0.001
KB = 0.00831446
PRESSURE = 0.061027


def is_section_header(line):
    """Check if line is a LAMMPS data file section header."""
    if not line:
        return False
    return line[0].isupper() and not line.startswith('#')


def convert_data_file(src_path, dst_path):
    """Convert LAMMPS data file: scale positions and box, preserve masses."""
    with open(src_path, 'r') as f:
        lines = f.readlines()

    out = []
    in_masses = False
    in_atoms = False

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines
        if not stripped:
            out.append(line)
            continue

        # Preserve comments
        if stripped.startswith('#'):
            out.append(line)
            continue

        # Box bounds: "0.0 30.899002 xlo xhi"
        parts = stripped.split()
        if len(parts) == 4 and parts[2] in ('xlo', 'ylo', 'zlo'):
            lo = float(parts[0]) * LENGTH
            hi = float(parts[1]) * LENGTH
            out.append(f"{lo:.6f} {hi:.6f} {parts[2]} {parts[3]}\n")
            continue

        # Track section headers
        if is_section_header(stripped):
            in_masses = (stripped == "Masses")
            # "Atoms # full" starts with Atoms
            in_atoms = stripped.startswith("Atoms")
            out.append(line)
            continue

        # Masses section: preserve mass values (conversion factor is 1.0)
        if in_masses:
            # Mass lines: "type_id mass_value"
            out.append(line)
            continue

        # Atoms section: scale positions (fields 4,5,6 for full style)
        if in_atoms:
            parts = stripped.split()
            # atom_style full: id mol type charge x y z [ix iy iz]
            if len(parts) >= 7:
                parts[4] = f"{float(parts[4]) * LENGTH:.6f}"
                parts[5] = f"{float(parts[5]) * LENGTH:.6f}"
                parts[6] = f"{float(parts[6]) * LENGTH:.6f}"
                out.append(' '.join(parts) + '\n')
            else:
                out.append(line)
            continue

        # Everything else: pass through
        out.append(line)

    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    with open(dst_path, 'w') as f:
        f.writelines(out)
    print(f"  Converted data file: {src_path} -> {dst_path}")


def convert_table_file(src_path, dst_path):
    """Convert pair table: r * LENGTH, U * ENERGY, F * FORCE."""
    with open(src_path, 'r') as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append('\n')
            continue
        if stripped.startswith('#') or stripped.startswith('PAIR_') or stripped.startswith('N '):
            out.append(line)
            continue

        parts = stripped.split()
        if len(parts) >= 4:
            idx = parts[0]
            r = float(parts[1]) * LENGTH
            u = float(parts[2]) * ENERGY
            f = float(parts[3]) * FORCE
            out.append(f"{idx} {r:.10e} {u:.10e} {f:.10e}\n")
        else:
            out.append(line)

    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    with open(dst_path, 'w') as f:
        f.writelines(out)
    print(f"  Converted table file: {src_path} -> {dst_path}")


def parse_lammps_input(path):
    """Extract key parameters from a LAMMPS input script."""
    params = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'timestep':
                params['dt'] = float(parts[1])
            elif parts[0] == 'run':
                params['nsteps'] = int(parts[1])
            elif parts[0] == 'thermo':
                params['thermo_freq'] = int(parts[1])
            elif parts[0] == 'fix' and 'langevin' in line:
                params['T_start'] = float(parts[4])
                params['T_stop'] = float(parts[5])
                params['Tdamp'] = float(parts[6])
                if len(parts) >= 8:
                    params['seed'] = int(parts[7])
    return params


def generate_fastmd_configs(out_dir, params, table_filename, data_filename):
    """Generate fastmd_static.conf and fastmd_nvt.conf."""
    natoms = 1000
    ntypes = 1
    rc = 15.0 * LENGTH  # 1.5 nm
    skin = 0.5 * LENGTH  # 0.05 nm
    dt = params['dt'] * TIME
    nsteps = params['nsteps']
    thermo_freq = params['thermo_freq']
    T_start = params['T_start']
    T_stop = params['T_stop']
    Tdamp = params['Tdamp'] * TIME
    seed = params.get('seed', 12345)
    kT = T_start * KB

    static_conf = f"""natoms {natoms}
ntypes {ntypes}
rc {rc:.6f}
skin {skin:.6f}
dt {dt:.6f}
nsteps 1
dump_freq 0
thermo 1 1 thermo_fastmd_static.dat
nvt_langevin {kT:.6f} {kT:.6f} {Tdamp:.6f} {seed}
table 0 0 {table_filename} PAIR_0
lammps_data_file {data_filename}
forces_dump_file forces_fastmd.dat
"""

    nvt_conf = f"""natoms {natoms}
ntypes {ntypes}
rc {rc:.6f}
skin {skin:.6f}
dt {dt:.6f}
nsteps {nsteps}
dump_freq 0
thermo 1 {thermo_freq} thermo_fastmd.dat
nvt_langevin {kT:.6f} {kT:.6f} {Tdamp:.6f} {seed}
table 0 0 {table_filename} PAIR_0
lammps_data_file {data_filename}
"""

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'fastmd_static.conf'), 'w') as f:
        f.write(static_conf)
    print(f"  Generated config: {out_dir}/fastmd_static.conf")
    with open(os.path.join(out_dir, 'fastmd_nvt.conf'), 'w') as f:
        f.write(nvt_conf)
    print(f"  Generated config: {out_dir}/fastmd_nvt.conf")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAMMPS real-units dataset to HOOMD MD reduced units')
    parser.add_argument('--data', required=True, help='Path to LAMMPS data file')
    parser.add_argument('--table', required=True, help='Path to pair table file')
    parser.add_argument('--input', required=True, help='Path to LAMMPS input script')
    parser.add_argument('--src', default='real', help='Source unit system (only "real" supported)')
    parser.add_argument('--dst', default='md', help='Destination unit system (only "md" supported)')
    parser.add_argument('--out', required=True, help='Output directory for converted files')
    args = parser.parse_args()

    if args.src != 'real' or args.dst != 'md':
        print("Error: only --src real --dst md is supported in this version", file=sys.stderr)
        sys.exit(1)

    params = parse_lammps_input(args.input)

    data_basename = os.path.basename(args.data)
    table_basename = os.path.basename(args.table)
    out_data = os.path.join(args.out, data_basename)
    out_table = os.path.join(args.out, table_basename)

    convert_data_file(args.data, out_data)
    convert_table_file(args.table, out_table)

    generate_fastmd_configs(args.out, params, table_basename, data_basename)

    print(f"\nDone. Converted dataset in {args.out}/")


if __name__ == '__main__':
    main()
