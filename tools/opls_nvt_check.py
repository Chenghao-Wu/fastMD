#!/usr/bin/env python3
"""NVT dynamics validation (params-driven): LAMMPS vs fastMD.

LAMMPS equilibrates (velocity create + 2 ps NVT), writes data with
velocities; both engines then run 5 ps NVT from the same state.
Reads opls_params.json next to the data file (same schema as
opls_force_check.py). Compares average T and PE.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

KCAL2KJ = 4.184
A2NM = 0.1
KB = 0.00831446
LMP = os.environ.get('LAMMPS_BINARY', 'lmp')
FMD = os.environ.get('FASTMD_BINARY', 'fastmd')


def style_lines(pj):
    lines = ["bond_style harmonic"]
    for i, b in enumerate(pj['bond_types']):
        lines.append(f"bond_coeff {i+1} {b['K']} {b['r0']}")
    if pj.get('angle_types'):
        lines.append("angle_style harmonic")
        for i, a in enumerate(pj['angle_types']):
            lines.append(f"angle_coeff {i+1} {a['K']} {a['theta0']}")
    if pj.get('dihedral_types'):
        lines.append("dihedral_style opls")
        for i, d in enumerate(pj['dihedral_types']):
            v = d['V']
            lines.append(f"dihedral_coeff {i+1} {v[0]} {v[1]} {v[2]} {v[3]}")
    for i, t in enumerate(pj['atom_types']):
        lines.append(f"pair_coeff {i+1} {i+1} {t['eps']} {t['sig']}")
    return lines


def lammps_input(data, rc, acc, pj, eq_steps, prod_steps, temp, seed,
                 write_data=None, thermo=False, kspace='ewald'):
    # styles before read_data: write_data output embeds Coeffs sections,
    # which require the styles to be defined when read_data processes them
    lines = [
        "units real", "atom_style full", "boundary p p p",
        f"pair_style lj/cut/coul/long {rc}",
        "pair_modify mix geometric",
        "bond_style harmonic",
    ]
    if pj.get('angle_types'):
        lines.append("angle_style harmonic")
    if pj.get('dihedral_types'):
        lines.append("dihedral_style opls")
    lines += [
        f"read_data {data}",
        f"kspace_style {kspace} {acc}",
        "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5",
    ] + style_lines(pj)[1:] + [
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check no",
    ]
    if eq_steps > 0 and prod_steps == 0:
        lines += [
            f"velocity all create {temp} {seed} dist gaussian mom yes rot yes",
            f"fix 1 all nvt temp {temp} {temp} 100.0",
            "timestep 1.0",
            f"run {eq_steps}",
        ]
    else:
        lines += [
            f"fix 1 all nvt temp {temp} {temp} 100.0",
            "timestep 1.0",
        ]
        if thermo:
            lines += ["thermo 100", "thermo_style custom step temp pe"]
        lines.append(f"run {prod_steps if prod_steps > 0 else eq_steps}")
    if write_data:
        lines.append(f"write_data {write_data}")
    return '\n'.join(lines) + '\n'


def convert_nm(src, dst):
    out = []
    in_atoms = in_vel = False
    with open(src) as f:
        for line in f:
            s = line.strip()
            parts = s.split()
            if len(parts) == 4 and parts[2] in ('xlo', 'ylo', 'zlo'):
                out.append("%.6f %.6f %s %s\n" % (float(parts[0]) * A2NM,
                                                  float(parts[1]) * A2NM,
                                                  parts[2], parts[3]))
                continue
            if s and s[0].isupper():
                in_atoms = s.startswith('Atoms')
                in_vel = s.startswith('Velocities')
                out.append(line)
                continue
            if in_atoms and len(parts) >= 7:
                parts[4] = "%.6f" % (float(parts[4]) * A2NM)
                parts[5] = "%.6f" % (float(parts[5]) * A2NM)
                parts[6] = "%.6f" % (float(parts[6]) * A2NM)
                out.append(' '.join(parts) + '\n')
                continue
            if in_vel and len(parts) >= 4:
                parts[1] = "%.6f" % (float(parts[1]) * 100.0)
                parts[2] = "%.6f" % (float(parts[2]) * 100.0)
                parts[3] = "%.6f" % (float(parts[3]) * 100.0)
                out.append(' '.join(parts) + '\n')
                continue
            out.append(line)
    with open(dst, 'w') as f:
        f.writelines(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--temp', type=float, default=300.0)
    ap.add_argument('--rc', type=float, default=10.0)
    ap.add_argument('--ewald_acc', type=float, default=1e-4)
    ap.add_argument('--kmax', type=int, default=5)
    ap.add_argument('--eq', type=int, default=2000)
    ap.add_argument('--prod', type=int, default=5000)
    ap.add_argument('--gpu', action='store_true',
                    help='run LAMMPS with -sf gpu (GPU pair style; ewald '
                         'kspace stays on CPU, matching fastMD direct Ewald)')
    ap.add_argument('--pppm', action='store_true',
                    help='use pppm kspace instead of ewald (with --gpu this '
                         'is pppm/gpu, offloading kspace to the GPU too)')
    args = ap.parse_args()
    os.chdir(args.workdir)

    data_abs = os.path.abspath(args.data)
    with open(os.path.join(os.path.dirname(data_abs), 'opls_params.json')) as f:
        pj = json.load(f)

    natoms = 0
    with open(data_abs) as f:
        for line in f:
            if 'atoms' in line:
                natoms = int(line.split()[0])
                break

    print(f"[1/4] LAMMPS equilibration ({args.eq} steps) ...")
    lmp_cmd = [LMP] + (['-sf', 'gpu'] if args.gpu else [])
    ks = 'pppm' if args.pppm else 'ewald'
    r = subprocess.run(lmp_cmd + ['-in', '/dev/stdin', '-log', 'log.eq'],
                       input=lammps_input(data_abs, args.rc, args.ewald_acc, pj,
                                          args.eq, 0, args.temp, 12345,
                                          write_data='equil.data', kspace=ks),
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists('equil.data'):
        print(r.stdout[-2500:]); sys.exit("equilibration failed")

    print(f"[2/4] LAMMPS production ({args.prod} steps) ...")
    r = subprocess.run(lmp_cmd + ['-in', '/dev/stdin', '-log', 'log.prod'],
                       input=lammps_input('equil.data', args.rc, args.ewald_acc,
                                          pj, 0, args.prod, args.temp, 12345,
                                          thermo=True, kspace=ks),
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:]); sys.exit("production failed")

    print("[3/4] fastMD production ...")
    convert_nm('equil.data', 'equil_nm.data')
    gewald = None
    with open('log.eq') as f:
        for line in f:
            if 'G vector (1/distance)' in line:
                gewald = float(line.split('=')[1]) * 10.0
                break
    assert gewald
    kT = args.temp * KB
    lines = [
        f"natoms {natoms}",
        f"ntypes {len(pj['atom_types'])}",
        f"rc {args.rc * A2NM}",
        "skin 0.05",
        "dt 0.001",
        f"nsteps {args.prod}",
        "dump_freq 0",
        "thermo 1 100 thermo_nvt.dat",
        f"nvt_nh {kT:.5f} {kT:.5f} 0.1",
    ]
    for i, t in enumerate(pj['atom_types']):
        lines.append(f"lj_type {i} {t['eps'] * KCAL2KJ} {t['sig'] * A2NM}")
    lines.append(f"coulomb ewald {gewald:.6f} {args.kmax}")
    for i, b in enumerate(pj['bond_types']):
        lines.append(f"bond_harmonic {i} {b['K'] * KCAL2KJ * 100.0} {b['r0'] * A2NM}")
    for i, a in enumerate(pj.get('angle_types', [])):
        lines.append(f"angle_type {i} {a['K'] * KCAL2KJ} {a['theta0']}")
    for i, d in enumerate(pj.get('dihedral_types', [])):
        v = d['V']
        lines.append(f"dihedral_opls {i} {v[0]*KCAL2KJ} {v[1]*KCAL2KJ} "
                     f"{v[2]*KCAL2KJ} {v[3]*KCAL2KJ}")
    lines.append("lammps_data_file equil_nm.data")
    with open('nvt.conf', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    t0 = time.time()
    r = subprocess.run([FMD, 'nvt.conf'], capture_output=True, text=True)
    fmd_wall = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2500:], r.stderr[-1000:]); sys.exit("fastMD failed")

    print("[4/4] comparing ...")
    lmp_loop = None
    with open('log.prod') as f:
        for line in f:
            if line.strip().startswith('Loop time'):
                lmp_loop = float(line.split()[3])
    lt, lpe = [], []
    with open('log.prod') as f:
        in_t = False
        for line in f:
            if line.strip().startswith('Step'):
                in_t = True
                continue
            if in_t:
                p = line.split()
                if len(p) == 3 and p[0].isdigit():
                    lt.append(float(p[1]))
                    lpe.append(float(p[2]))
                elif line.strip().startswith('Loop'):
                    break
    ft, fpe = [], []
    with open('thermo_nvt.dat') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            p = s.split()
            ft.append(float(p[3]) / KB)
            fpe.append(float(p[2]) * natoms)

    n = min(len(lt), len(ft))
    lt, ft, lpe, fpe = lt[-n:], ft[-n:], lpe[-n:], fpe[-n:]
    avgT_l = sum(lt) / len(lt)
    avgT_f = sum(ft) / len(ft)
    avgPE_l = sum(lpe) / len(lpe) * KCAL2KJ
    avgPE_f = sum(fpe) / len(fpe)

    print(f"\n=== NVT {args.temp} K, {args.prod} steps ===")
    print("T:  LAMMPS %9.2f K   fastMD %9.2f K   (rel diff %.3f%%)" %
          (avgT_l, avgT_f, 100 * abs(avgT_l - avgT_f) / avgT_l))
    print("PE: LAMMPS %10.2f kJ   fastMD %10.2f kJ   (rel diff %.3f%%)" %
          (avgPE_l, avgPE_f, 100 * abs(avgPE_l - avgPE_f) / abs(avgPE_l)))
    if lmp_loop is not None:
        print("Wall: LAMMPS%s %.2f s   fastMD %.2f s   (%.2fx %s)" %
              ('-GPU' if args.gpu else '', lmp_loop, fmd_wall,
               lmp_loop / fmd_wall if fmd_wall > 0 else float('nan'),
               'fastMD faster' if fmd_wall < lmp_loop else 'LAMMPS faster'))
    ok = (abs(avgT_l - avgT_f) / avgT_l < 0.03 and
          abs(avgPE_l - avgPE_f) / abs(avgPE_l) < 0.03)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
