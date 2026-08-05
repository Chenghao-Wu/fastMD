#!/usr/bin/env python3
"""Generate an OPLS-AA liquid butane box as a LAMMPS data file.

Uses RDKit for the molecular geometry (MMFF-optimized conformer) and
packmol to pack molecules into a box at liquid density without overlaps.

OPLS-AA parameters (kcal/mol, Angstrom):
  bond:    CT-CT  K=268.0 r0=1.529 ;  CT-HC K=340.0 r0=1.090
  angle:   CT-CT-CT K=58.35 t0=112.7 ;  CT-CT-HC K=37.5 t0=110.7
           HC-CT-HC K=33.0 t0=107.8
  dihedral CT-CT-CT-CT: V1=1.300 V2=-0.050 V3=0.600 V4=0.0
           X-CT-CT-X (H-containing): V1=0.0 V2=0.0 V3=0.318 V4=0.0
  LJ:  CT eps=0.066 sig=3.50 ;  HC eps=0.030 sig=2.50  (geometric mixing)
  charges: CT(CH3) -0.18, CT(CH2) -0.12, HC +0.06

Outputs: butane.data (LAMMPS data file, real units / Angstrom)
         opls_params.json (force field parameters for the validator)
"""
import json
import math
import os
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

N_MOL = 640
DENSITY = 0.60  # g/cm^3
M_BUTANE = 58.12
PACKMOL_TOL = 2.0  # Angstrom minimum intermolecular distance

# --- build butane with RDKit ---
mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
if mol.GetNumAtoms() != 14:
    sys.exit("expected 14 atoms (4 C + 10 H)")
params = AllChem.ETKDGv3()
params.randomSeed = 42
if AllChem.EmbedMolecule(mol, params) != 0:
    sys.exit("RDKit embed failed")
AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
conf = mol.GetConformer()

# atom ordering: carbons first (chain order), then their hydrogens
bonds_mol = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
             for b in mol.GetBonds()]
carbons = sorted([a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"],
                 key=lambda c: -len([b for b in bonds_mol if c in b and
                                     mol.GetAtomWithIdx(b[0] if b[1] == c else b[1]).GetSymbol() == "C"]))
# order carbons along the chain
def chain_order(carbons, bonds):
    adj = {c: [] for c in carbons}
    for a, b in bonds:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    ends = [c for c in carbons if len(adj[c]) == 1]
    order = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    return order

c_order = chain_order(carbons, bonds_mol)
h_of = {}
for c in c_order:
    hs = sorted([b[0] if b[1] == c else b[1] for b in bonds_mol
                 if (b[0] == c or b[1] == c) and
                 mol.GetAtomWithIdx(b[0] if b[1] == c else b[1]).GetSymbol() == "H"])
    h_of[c] = hs

# new index: C0 C1 C2 C3 then H(C0)... H(C1)... H(C2)... H(C3)...
new_order = list(c_order)
for c in c_order:
    new_order += h_of[c]
old_to_new = {old: new for new, old in enumerate(new_order)}
assert len(new_order) == 14

coords = []
for old in new_order:
    p = conf.GetAtomPosition(old)
    coords.append([p.x, p.y, p.z])

# local topology in new indexing
loc_bonds = [(old_to_new[a], old_to_new[b]) for a, b in bonds_mol]
loc_bonds = [(min(a, b), max(a, b)) for a, b in loc_bonds]

# angles from bond graph
adj = {i: [] for i in range(14)}
for a, b in loc_bonds:
    adj[a].append(b)
    adj[b].append(a)
loc_angles = set()
for j in range(14):
    for i in adj[j]:
        for k in adj[j]:
            if i < k:
                loc_angles.add((i, j, k))
loc_angles = sorted(loc_angles)

# dihedrals: all simple paths i-j-k-l
loc_dih = set()
for j, k in loc_bonds:
    for i in adj[j]:
        if i == k:
            continue
        for l in adj[k]:
            if l == j or l == i:
                continue
            loc_dih.add((i, j, k, l))
            loc_dih.add((l, k, j, i))
loc_dih = sorted(loc_dih)

def is_c(i):
    return i < 4

# type assignment
def bond_type(a, b):
    return 2 if (is_c(a) and is_c(b)) else 1

def angle_type(i, j, k):
    return 2 if (is_c(i) and is_c(k)) else (1 if (not is_c(i) and not is_c(k)) else 3)

def dih_type(i, j, k, l):
    return 1 if (is_c(i) and is_c(j) and is_c(k) and is_c(l)) else 2

# charges: C0/C3 are CH3 (-0.18), C1/C2 are CH2 (-0.12), H +0.06
def charge(i):
    if i in (0, 3):
        return -0.18
    if i in (1, 2):
        return -0.12
    return 0.06

# --- write butane template PDB for packmol ---
def write_pdb(path, coords):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            el = "C" if is_c(i) else "H"
            f.write("HETATM%5d %-3s BUT %4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                    % (i + 1, el, 1, x, y, z, el))
        f.write("END\n")

write_pdb("butane_template.pdb", coords)

# --- packmol box ---
V_cm3 = N_MOL * M_BUTANE / (DENSITY * 6.022e23)
L_A = (V_cm3 * 1e24) ** (1.0 / 3.0)

with open("packmol.inp", "w") as f:
    f.write("tolerance %f\n" % PACKMOL_TOL)
    f.write("filetype pdb\n")
    f.write("output butane_box.pdb\n")
    # periodic packing: packmol checks minimum-image distances and keeps
    # all atoms inside the box, so no PBC-wrap clashes
    f.write("pbc 0. 0. 0. %f %f %f\n" % (L_A, L_A, L_A))
    f.write("structure butane_template.pdb\n")
    f.write("  number %d\n" % N_MOL)
    f.write("  inside box 0. 0. 0. %f %f %f\n" % (L_A, L_A, L_A))
    f.write("end structure\n")

r = subprocess.run(["packmol"], stdin=open("packmol.inp"),
                   capture_output=True, text=True)
if not os.path.exists("butane_box.pdb"):
    print(r.stdout[-2000:])
    sys.exit("packmol failed")

# --- read packmol output ---
mol_coords = []  # per molecule: list of 14 xyz
with open("butane_box.pdb") as f:
    cur = []
    for line in f:
        if line.startswith("HETATM") or line.startswith("ATOM"):
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            cur.append([x, y, z])
            if len(cur) == 14:
                mol_coords.append(cur)
                cur = []
assert len(mol_coords) == N_MOL, len(mol_coords)

# --- write LAMMPS data file ---
with open("butane.data", "w") as f:
    f.write("OPLS-AA butane box, %d molecules (RDKit+packmol)\n\n" % N_MOL)
    f.write("%d atoms\n" % (N_MOL * 14))
    f.write("%d bonds\n" % (N_MOL * len(loc_bonds)))
    f.write("%d angles\n" % (N_MOL * len(loc_angles)))
    f.write("%d dihedrals\n" % (N_MOL * len(loc_dih)))
    f.write("2 atom types\n2 bond types\n3 angle types\n2 dihedral types\n\n")
    f.write("0.0 %.6f xlo xhi\n" % L_A)
    f.write("0.0 %.6f ylo yhi\n" % L_A)
    f.write("0.0 %.6f zlo zhi\n\n" % L_A)
    f.write("Masses\n\n1 12.011\n2 1.008\n\n")
    f.write("Atoms # full\n\n")
    aid = 0
    for m, mcs in enumerate(mol_coords):
        for i, (x, y, z) in enumerate(mcs):
            aid += 1
            t = 1 if is_c(i) else 2
            f.write("%d %d %d %.6f %.6f %.6f %.6f\n"
                    % (aid, m + 1, t, charge(i), x, y, z))
    f.write("\nBonds\n\n")
    bid = 0
    for m in range(N_MOL):
        o = m * 14
        for (a, b) in loc_bonds:
            bid += 1
            f.write("%d %d %d %d\n" % (bid, bond_type(a, b), o + a + 1, o + b + 1))
    f.write("\nAngles\n\n")
    for m in range(N_MOL):
        o = m * 14
        for (i, j, k) in loc_angles:
            bid += 1
            f.write("%d %d %d %d %d\n"
                    % (bid, angle_type(i, j, k), o + i + 1, o + j + 1, o + k + 1))
    f.write("\nDihedrals\n\n")
    did = 0
    for m in range(N_MOL):
        o = m * 14
        for (i, j, k, l) in loc_dih:
            did += 1
            f.write("%d %d %d %d %d %d\n"
                    % (did, dih_type(i, j, k, l), o + i + 1, o + j + 1, o + k + 1, o + l + 1))

# --- params json (single source of truth for the validator) ---
params_out = {
    "units": "real",
    "atom_types": [
        {"name": "CT", "eps": 0.066, "sig": 3.50},
        {"name": "HC", "eps": 0.030, "sig": 2.50},
    ],
    "bond_types": [
        {"K": 340.0, "r0": 1.090, "desc": "CT-HC"},
        {"K": 268.0, "r0": 1.529, "desc": "CT-CT"},
    ],
    "angle_types": [
        {"K": 33.0, "theta0": 107.8, "desc": "HC-CT-HC"},
        {"K": 58.35, "theta0": 112.7, "desc": "CT-CT-CT"},
        {"K": 37.5, "theta0": 110.7, "desc": "CT-CT-HC"},
    ],
    "dihedral_types": [
        {"V": [1.300, -0.050, 0.600, 0.0], "desc": "CT-CT-CT-CT"},
        {"V": [0.0, 0.0, 0.318, 0.0], "desc": "X-CT-CT-X"},
    ],
}
with open("opls_params.json", "w") as f:
    json.dump(params_out, f, indent=2)

print("butane.data: %d atoms, %d bonds, %d angles, %d dihedrals, L=%.3f A"
      % (N_MOL * 14, N_MOL * len(loc_bonds), N_MOL * len(loc_angles),
         N_MOL * len(loc_dih), L_A))
