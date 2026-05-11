#!/usr/bin/env python3
"""Compare time-averaged thermo output between LAMMPS and fastMD.

Usage:
    tools/compare_thermo.py \\
      --lammps data/CG_water/log.lammps \\
      --fastmd thermo_fastmd.dat \\
      --equilibration-skip 50.0    # ps
"""

import argparse
import math
import sys

KB = 0.00831446       # K -> kJ/mol
ENERGY_CONV = 4.184   # kcal/mol -> kJ/mol
PRESSURE_CONV = 0.061027  # atm -> kJ/(mol*nm^3)


def parse_lammps_thermo(path, skip_ps):
    """Parse LAMMPS log.lammps, return (t_ps, T_md, PE_md, P_md) arrays."""
    t_ps, T_md, PE_md, P_md = [], [], [], []

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            if len(parts) < 7:
                continue
            try:
                step = int(parts[0])
                temp_K = float(parts[1])
                press_atm = float(parts[2])
                pe_kcal = float(parts[3])
            except ValueError:
                continue

            t = step * 0.002  # 2 fs timestep -> ps
            T_md_val = temp_K * KB
            PE_md_val = pe_kcal * ENERGY_CONV
            P_md_val = press_atm * PRESSURE_CONV

            if t >= skip_ps:
                t_ps.append(t)
                T_md.append(T_md_val)
                PE_md.append(PE_md_val)
                P_md.append(P_md_val)

    return t_ps, T_md, PE_md, P_md


def parse_fastmd_thermo(path, skip_ps):
    """Parse fastMD thermo output, return (t_ps, T, PE, P) arrays."""
    t_ps, T_vals, PE_vals, P_vals = [], [], [], []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                step = int(parts[0])
                pe = float(parts[2])
                temp = float(parts[3])
                pxx = float(parts[4])
                pyy = float(parts[5])
                pzz = float(parts[6])
            except ValueError:
                continue

            t = step * 0.002  # dt=0.002 ps
            P_scalar = (pxx + pyy + pzz) / 3.0

            if t >= skip_ps:
                t_ps.append(t)
                T_vals.append(temp)
                PE_vals.append(pe)
                P_vals.append(P_scalar)

    return t_ps, T_vals, PE_vals, P_vals


def compute_stats(values):
    """Return (mean, sem) for a list of values."""
    n = len(values)
    if n < 2:
        return float('nan'), float('nan')
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    sem = math.sqrt(variance / n)
    return mean, sem


def relative_error(a, b):
    if abs(b) < 1e-15:
        return float('inf')
    return abs(a - b) / abs(b)


def main():
    parser = argparse.ArgumentParser(
        description='Compare time-averaged thermo between LAMMPS and fastMD')
    parser.add_argument('--lammps', required=True, help='Path to LAMMPS log file')
    parser.add_argument('--fastmd', required=True, help='Path to fastMD thermo file')
    parser.add_argument('--equilibration-skip', type=float, default=50.0,
                        help='Equilibration time to skip in ps (default: 50.0)')
    args = parser.parse_args()

    print(f"Parsing LAMMPS thermo: {args.lammps}")
    t_lmp, T_lmp, PE_lmp, P_lmp = parse_lammps_thermo(args.lammps, args.equilibration_skip)

    print(f"Parsing fastMD thermo: {args.fastmd}")
    t_fmd, T_fmd, PE_fmd, P_fmd = parse_fastmd_thermo(args.fastmd, args.equilibration_skip)

    print(f"\n  Samples after {args.equilibration_skip} ps equilibration skip:")
    print(f"  LAMMPS: {len(T_lmp)} samples, t range [{min(t_lmp):.1f}, {max(t_lmp):.1f}] ps")
    print(f"  fastMD: {len(T_fmd)} samples, t range [{min(t_fmd):.1f}, {max(t_fmd):.1f}] ps")

    if len(T_lmp) < 2 or len(T_fmd) < 2:
        print("Error: insufficient data after equilibration skip", file=sys.stderr)
        sys.exit(1)

    lmp_T_mean, lmp_T_sem = compute_stats(T_lmp)
    lmp_PE_mean, lmp_PE_sem = compute_stats(PE_lmp)
    lmp_P_mean, lmp_P_sem = compute_stats(P_lmp)

    fmd_T_mean, fmd_T_sem = compute_stats(T_fmd)
    fmd_PE_mean, fmd_PE_sem = compute_stats(PE_fmd)
    fmd_P_mean, fmd_P_sem = compute_stats(P_fmd)

    T_rel = relative_error(fmd_T_mean, lmp_T_mean)
    PE_rel = relative_error(fmd_PE_mean, lmp_PE_mean)
    P_rel = relative_error(fmd_P_mean, lmp_P_mean)

    print(f"\n  {'Quantity':<12} {'LAMMPS':>14} {'fastMD':>14} {'Rel Error':>12} {'Tol':>10} {'Status'}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*12} {'-'*10} {'-'*6}")

    def report(name, lmp_val, fmd_val, rel_err, tol, unit):
        status = "PASS" if rel_err < tol else "FAIL"
        print(f"  {name:<12} {lmp_val:14.6f} {fmd_val:14.6f} {rel_err:12.3e} {tol:>10.1e} {status:>6}  {unit}")

    report("<T>", lmp_T_mean, fmd_T_mean, T_rel, 0.01, "kJ/mol")
    report("<PE>", lmp_PE_mean, fmd_PE_mean, PE_rel, 0.01, "kJ/mol")
    report("<P>", lmp_P_mean, fmd_P_mean, P_rel, 0.05, "kJ/(mol*nm^3)")

    print(f"\n  Standard errors of the mean:")
    print(f"  {'<T> SEM:':<12} LAMMPS={lmp_T_sem:.4e}  fastMD={fmd_T_sem:.4e} kJ/mol")
    print(f"  {'<PE> SEM:':<12} LAMMPS={lmp_PE_sem:.4e}  fastMD={fmd_PE_sem:.4e} kJ/mol")
    print(f"  {'<P> SEM:':<12} LAMMPS={lmp_P_sem:.4e}  fastMD={fmd_P_sem:.4e} kJ/(mol*nm^3)")

    passed = (T_rel < 0.01) and (PE_rel < 0.01) and (P_rel < 0.05)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
