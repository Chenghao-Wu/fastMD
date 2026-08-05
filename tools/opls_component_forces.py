#!/usr/bin/env python3
"""Bisect per-component forces: LAMMPS vs fastMD on the butane box.

Compares f_bonds, f_angle, f_dih, f_pair(lj+coul) as force differences
between cumulative-style runs.
"""
import math
import os
import subprocess

KCALA2KJNM = 41.84
LMP = '/home/zhenghaowu/lammps/build/lmp'
FMD = '/home/zhenghaowu/fastMD/build/fastmd'
WD = '/home/zhenghaowu/fastMD/data/OPLS_butane'

LMP_BASE = """units real
atom_style full
boundary p p p
read_data {data}
special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5
bond_style harmonic
bond_coeff 1 340.0 1.090
bond_coeff 2 268.0 1.529
{angle}
{dih}
{pair}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check no
dump 1 all custom 1 {dump} id fx fy fz
run 0
"""
ANGLE = """angle_style harmonic
angle_coeff 1 33.0 107.8
angle_coeff 2 58.35 112.7
angle_coeff 3 37.5 110.7"""
DIH = """dihedral_style opls
dihedral_coeff 1 1.300 -0.050 0.600 0.0
dihedral_coeff 2 0.0 0.0 0.318 0.0"""
PAIR = """pair_style lj/cut/coul/long 10.0
pair_modify mix geometric
kspace_style ewald 1e-4
pair_coeff 1 1 0.066 3.50
pair_coeff 2 2 0.030 2.50"""
NOPAIR = """pair_style zero 10.0
pair_coeff * *"""

def run_lmp(name, angle='', dih='', pair=''):
    with open(f'{WD}/{name}.in', 'w') as f:
        f.write(LMP_BASE.format(data=f'{WD}/butane.data',
                                angle=angle, dih=dih, pair=pair,
                                dump=f'{name}.lmp'))
    r = subprocess.run([LMP, '-in', f'{name}.in', '-log', f'{name}.log'],
                       cwd=WD, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); raise SystemExit(name + ' failed')
    return read_lmp_forces(f'{WD}/{name}.lmp')

def read_lmp_forces(path):
    out = {}
    in_a = False
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and p[0] == 'ITEM:':
                in_a = len(p) >= 2 and p[1] == 'ATOMS'
                continue
            if in_a and len(p) == 4:
                out[int(p[0])] = [float(p[1]), float(p[2]), float(p[3])]
    return out

def read_fmd_forces(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.split()
            out[int(p[0]) + 1] = [float(p[1]) / KCALA2KJNM,
                                  float(p[2]) / KCALA2KJNM,
                                  float(p[3]) / KCALA2KJNM]
    return out

FMD_BASE = """natoms 896
ntypes 2
rc 1.0
skin 0.05
dt 0.0001
nsteps 1
dump_freq 0
thermo 1 1 {name}.thermo
nvt_langevin 2.49477 2.49477 0.1 42
lammps_data_file system_nm.data
forces_dump_file {name}.fmd
{bond}
{ang}
{dih}
{lj}
{coul}
"""
BOND = "bond_harmonic 0 142256 0.109\nbond_harmonic 1 112131.2 0.1529"
ANG = "angle_type 0 138.072 107.8\nangle_type 1 244.133 112.7\nangle_type 2 156.9 110.7"
FDIH = "dihedral_opls 0 5.4392 -0.2092 2.5104 0.0\ndihedral_opls 1 0.0 0.0 1.33051 0.0"
LJ = "lj_type 0 0.276144 0.350\nlj_type 1 0.12552 0.250"
COUL = "coulomb ewald 3.0349 5"

def run_fmd(name, bond='', ang='', dih='', lj='', coul=''):
    with open(f'{WD}/{name}.conf', 'w') as f:
        f.write(FMD_BASE.format(name=name, bond=bond, ang=ang,
                                dih=dih, lj=lj, coul=coul))
    r = subprocess.run([FMD, f'{name}.conf'], cwd=WD,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-500:]); raise SystemExit(name + ' failed')
    return read_fmd_forces(f'{WD}/{name}.fmd')

def diff(a, b):
    return {i: [a[i][d] - b[i][d] for d in range(3)] for i in a}

def cmp(name, a, b):
    errs = []
    for i in sorted(a):
        errs.append(math.sqrt(sum((a[i][d] - b[i][d]) ** 2 for d in range(3))))
    errs.sort()
    med = errs[len(errs) // 2]
    p90 = errs[int(len(errs) * 0.9)]
    print('%-10s  median=%8.4f  p90=%8.4f  max=%8.4f kcal/(mol A)'
          % (name, med, p90, errs[-1]))

L0 = run_lmp('lb0', pair=NOPAIR)
L1 = run_lmp('lb1', angle=ANGLE, pair=NOPAIR)
L2 = run_lmp('lb2', angle=ANGLE, dih=DIH, pair=NOPAIR)
L3 = run_lmp('lb3', angle=ANGLE, dih=DIH, pair=PAIR)

F0 = run_fmd('fb0', bond=BOND)
F1 = run_fmd('fb1', bond=BOND, ang=ANG)
F2 = run_fmd('fb2', bond=BOND, ang=ANG, dih=FDIH)
F3 = run_fmd('fb3', bond=BOND, ang=ANG, dih=FDIH, lj=LJ, coul=COUL)

cmp('bonds', L0, F0)
cmp('angles', diff(L1, L0), diff(F1, F0))
cmp('dihed', diff(L2, L1), diff(F2, F1))
cmp('pair', diff(L3, L2), diff(F3, F2))
cmp('total', L3, F3)
