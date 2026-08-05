#!/usr/bin/env python3
"""Validate fastMD OPLS against LAMMPS (lj/cut/coul/long + dihedral_style opls).

Fully params-driven: reads opls_params.json next to the data file.
Schema:
  atom_types:     [{"name","eps","sig"} ...]        # index+1 = LAMMPS type
  bond_types:     [{"K","r0"} ...]                  # harmonic, kcal/mol/A^2
  angle_types:    [{"K","theta0"} ...]              # harmonic, kcal/mol/rad^2, deg
  dihedral_types: [{"V":[V1,V2,V3,V4]} ...]         # OPLS Fourier, kcal/mol

Runs a static (run 0) force/energy comparison:
  1. LAMMPS with pair lj/cut/coul/long, kspace ewald, special_bonds 0 0 0.5.
  2. fastMD with coulomb ewald (alpha matched to LAMMPS gewald), geometric
     mixing, forces_dump_file.
  3. Compare per-atom forces (kJ/(mol nm)) and total PE.

Units: LAMMPS real = kcal/mol, A. fastMD md = kJ/mol, nm.
"""
import argparse
import json
import math
import os
import subprocess
import sys

KCALA2KJNM = 41.84
KCAL2KJ = 4.184
A2NM = 0.1
COUL_MD = 138.9354576  # kJ*nm/(mol e^2)


def build_lammps_in(data, rc, acc, pj, kspace='ewald'):
    lines = [
        "units real", "atom_style full", "boundary p p p",
        f"read_data {data}",
        f"pair_style lj/cut/coul/long {rc}",
        "pair_modify mix geometric",
        f"kspace_style {kspace} {acc}",
        "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5",
        "bond_style harmonic",
    ]
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
    lines += [
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check no",
        "compute peratom all pe/atom pair bond angle dihedral kspace",
        "dump 1 all custom 1 forces.lmp id fx fy fz",
        "run 0",
        'print "ETOT: $(pe)"',
    ]
    return '\n'.join(lines) + '\n'


def build_fastmd_conf(natoms, data_nm, rc_nm, alpha_nm, kmax, pj):
    lines = [
        f"natoms {natoms}",
        f"ntypes {len(pj['atom_types'])}",
        f"rc {rc_nm}",
        "skin 0.05",
        "dt 0.0001",
        "nsteps 1",
        "dump_freq 0",
        "thermo 1 1 thermo_opls.dat",
        "nvt_langevin 2.49477 2.49477 0.1 42",
    ]
    for i, t in enumerate(pj['atom_types']):
        lines.append(f"lj_type {i} {t['eps'] * KCAL2KJ} {t['sig'] * A2NM}")
    lines.append(f"coulomb ewald {alpha_nm} {kmax}")
    for i, b in enumerate(pj['bond_types']):
        lines.append(f"bond_harmonic {i} {b['K'] * KCAL2KJ * 100.0} {b['r0'] * A2NM}")
    for i, a in enumerate(pj.get('angle_types', [])):
        lines.append(f"angle_type {i} {a['K'] * KCAL2KJ} {a['theta0']}")
    for i, d in enumerate(pj.get('dihedral_types', [])):
        v = d['V']
        lines.append(f"dihedral_opls {i} {v[0]*KCAL2KJ} {v[1]*KCAL2KJ} "
                     f"{v[2]*KCAL2KJ} {v[3]*KCAL2KJ}")
    lines += [
        f"lammps_data_file {data_nm}",
        "forces_dump_file forces_fastmd.dat",
    ]
    return '\n'.join(lines) + '\n'


def convert_data_nm(src, dst):
    out = []
    in_atoms = False
    with open(src) as f:
        for line in f:
            stripped = line.strip()
            parts = stripped.split()
            if len(parts) == 4 and parts[2] in ('xlo', 'ylo', 'zlo'):
                out.append("%.6f %.6f %s %s\n" % (float(parts[0]) * A2NM,
                                                  float(parts[1]) * A2NM,
                                                  parts[2], parts[3]))
                continue
            if stripped and stripped[0].isupper():
                in_atoms = stripped.startswith('Atoms')
                out.append(line)
                continue
            if in_atoms and len(parts) >= 7:
                parts[4] = "%.6f" % (float(parts[4]) * A2NM)
                parts[5] = "%.6f" % (float(parts[5]) * A2NM)
                parts[6] = "%.6f" % (float(parts[6]) * A2NM)
                out.append(' '.join(parts) + '\n')
                continue
            out.append(line)
    with open(dst, 'w') as f:
        f.writelines(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--lmp', default='/home/zhenghaowu/lammps/build/lmp')
    ap.add_argument('--fastmd', default='/home/zhenghaowu/fastMD/build/fastmd')
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--rc', type=float, default=10.0)
    ap.add_argument('--ewald_acc', type=float, default=1e-4)
    ap.add_argument('--kmax', type=int, default=5)
    ap.add_argument('--gpu', action='store_true',
                    help='run LAMMPS with -sf gpu (GPU pair style; ewald '
                         'kspace stays on CPU, matching fastMD direct Ewald)')
    ap.add_argument('--pppm', action='store_true',
                    help='use pppm kspace instead of ewald (with --gpu this '
                         'is pppm/gpu, offloading kspace to the GPU too)')
    args = ap.parse_args()

    data_abs = os.path.abspath(args.data)
    params_file = os.path.join(os.path.dirname(data_abs), 'opls_params.json')
    with open(params_file) as f:
        pj = json.load(f)

    natoms = 0
    with open(data_abs) as f:
        for line in f:
            if 'atoms' in line:
                natoms = int(line.split()[0])
                break

    with open(os.path.join(args.workdir, 'in.opls'), 'w') as f:
        f.write(build_lammps_in(data_abs, args.rc, args.ewald_acc, pj,
                                kspace='pppm' if args.pppm else 'ewald'))

    data_nm = os.path.join(args.workdir, 'system_nm.data')
    convert_data_nm(data_abs, data_nm)

    # --- run LAMMPS first to learn its gewald ---
    lmp_cmd = [args.lmp]
    if args.gpu:
        lmp_cmd += ['-sf', 'gpu']
    lmp_cmd += ['-in', 'in.opls', '-log', 'log.opls']
    r = subprocess.run(lmp_cmd, cwd=args.workdir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-2000:])
        sys.exit("LAMMPS run failed")

    alpha_A = None
    with open(os.path.join(args.workdir, 'log.opls')) as f:
        for line in f:
            if 'G vector (1/distance)' in line:
                alpha_A = float(line.split('=')[1])
                break
    assert alpha_A, "no gewald found in LAMMPS log"
    alpha_nm = alpha_A * 10.0
    print("LAMMPS gewald = %.6f 1/A (%.6f 1/nm)" % (alpha_A, alpha_nm))

    conf_path = os.path.join(args.workdir, 'fastmd_opls.conf')
    with open(conf_path, 'w') as f:
        f.write(build_fastmd_conf(natoms, data_nm, args.rc * A2NM,
                                  alpha_nm, args.kmax, pj))

    r = subprocess.run([args.fastmd, os.path.basename(conf_path)],
                       cwd=args.workdir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-2000:])
        sys.exit("fastMD run failed")

    pe_lmp = None
    with open(os.path.join(args.workdir, 'log.opls')) as f:
        for line in f:
            if line.startswith('ETOT:'):
                pe_lmp = float(line.split()[1])

    lmp_f = {}
    with open(os.path.join(args.workdir, 'forces.lmp')) as f:
        in_atoms = False
        for line in f:
            parts = line.split()
            if parts and parts[0] == 'ITEM:':
                in_atoms = (len(parts) >= 2 and parts[1] == 'ATOMS')
                continue
            if in_atoms and len(parts) == 4:
                i = int(parts[0]) - 1
                lmp_f[i] = [float(parts[1]) * KCALA2KJNM,
                            float(parts[2]) * KCALA2KJNM,
                            float(parts[3]) * KCALA2KJNM]

    fmd_f = {}
    pe_fmd = 0.0
    with open(os.path.join(args.workdir, 'forces_fastmd.dat')) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            i = int(p[0])
            fmd_f[i] = [float(p[1]), float(p[2]), float(p[3])]
            pe_fmd += float(p[4])

    qsum2 = 0.0
    with open(data_nm) as f:
        in_a = False
        for line in f:
            s = line.strip()
            if s.startswith('Atoms'):
                in_a = True
                continue
            if in_a and (s.startswith('Bonds') or s.startswith('Velocities')):
                in_a = False
            if in_a:
                p = s.split()
                if len(p) >= 4 and p[0].isdigit():
                    qv = float(p[3])
                    qsum2 += qv * qv
    self_off = -COUL_MD * alpha_nm / math.sqrt(math.pi) * qsum2
    pe_fmd_thermo = pe_fmd + self_off

    max_err = 0.0
    max_i = -1
    rms = 0.0
    for i in sorted(lmp_f):
        for d in range(3):
            e = abs(lmp_f[i][d] - fmd_f[i][d])
            rms += e * e
            if e > max_err:
                max_err = e
                max_i = i
    rms = math.sqrt(rms / (3 * len(lmp_f)))

    pe_lmp_kj = pe_lmp * KCAL2KJ if pe_lmp is not None else None

    print("\n=== OPLS validation ===")
    print("atoms: %d   alpha: %.4f 1/A   rc: %.2f A" %
          (natoms, alpha_A, args.rc))
    print("max |dF| = %.5f kJ/(mol nm)  (atom %d)   RMS |dF| = %.5f" %
          (max_err, max_i, rms))
    if pe_lmp_kj is not None:
        print("PE total: LAMMPS = %.3f kJ/mol   fastMD = %.3f   diff = %.4f" %
              (pe_lmp_kj, pe_fmd_thermo, pe_lmp_kj - pe_fmd_thermo))

    ok = (max_err < 25.0 and rms < 5.0 and
          pe_lmp_kj is not None and
          abs(pe_lmp_kj - pe_fmd_thermo) < 0.01 * abs(pe_lmp_kj) + 1.0)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
