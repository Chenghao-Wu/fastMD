#!/usr/bin/env python3
"""Generate an OPLS-AA liquid methanol box as a LAMMPS data file.

RDKit for the molecular geometry (MMFF-optimized), packmol with periodic
boundaries for packing.

OPLS-AA methanol (kcal/mol, Angstrom):
  types:  CT(1) OH(2) HO(3) HC(4)
  charges: CT +0.145, OH -0.683, HO +0.418, HC +0.040
  LJ:  CT 0.066/3.50 ;  OH 0.170/3.12 ;  HO 0.0/0.0 ;  HC 0.030/2.50
  bonds:  CT-OH 320.0/1.410 ;  CT-HC 340.0/1.090 ;  OH-HO 553.0/0.945
  angles: HC-CT-OH 35.0/109.5 ;  HC-CT-HC 33.0/107.8 ;  CT-OH-HO 55.0/108.5
  dihedral HC-CT-OH-HO: V1=0 V2=0 V3=0.450 V4=0

Outputs: methanol.data, opls_params.json
"""
import json
import math
import os
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

N_MOL = 125
DENSITY = 0.791  # g/cm^3
M_METHANOL = 32.04
PACKMOL_TOL = 2.0

# --- build methanol with RDKit ---
mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
if mol.GetNumAtoms() != 6:
    sys.exit("expected 6 atoms")
params = AllChem.ETKDGv3()
params.randomSeed = 42
if AllChem.EmbedMolecule(mol, params) != 0:
    sys.exit("RDKit embed failed")
AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
conf = mol.GetConformer()

# atom ordering: C(0), O(1), H-hydroxyl(2), H3(3), H4(4), H5(5)
c_idx = o_idx = None
for a in mol.GetAtoms():
    if a.GetSymbol() == "C":
        c_idx = a.GetIdx()
    elif a.GetSymbol() == "O":
        o_idx = a.GetIdx()
bonds_mol = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

def neighbors(i):
    return [b[0] if b[1] == i else b[1] for b in bonds_mol if i in b]

ho_idx = [n for n in neighbors(o_idx) if mol.GetAtomWithIdx(n).GetSymbol() == "H"][0]
hc_idx = sorted(n for n in neighbors(c_idx) if mol.GetAtomWithIdx(n).GetSymbol() == "H")

new_order = [c_idx, o_idx, ho_idx] + hc_idx
old_to_new = {old: new for new, old in enumerate(new_order)}
assert len(new_order) == 6

coords = []
for old in new_order:
    p = conf.GetAtomPosition(old)
    coords.append([p.x, p.y, p.z])

# local topology in new indexing: 0=CT, 1=OH, 2=HO, 3..5=HC
loc_bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)]
# bond types: 1=CT-OH, 2=CT-HC, 3=OH-HO
def bond_type(a, b):
    s = {a, b}
    if s == {0, 1}: return 1
    if s == {1, 2}: return 3
    return 2

loc_angles = [(3, 0, 1), (4, 0, 1), (5, 0, 1),   # HC-CT-OH  (type 1)
              (3, 0, 4), (3, 0, 5), (4, 0, 5),   # HC-CT-HC  (type 2)
              (0, 1, 2)]                          # CT-OH-HO  (type 3)
def angle_type(i, j, k):
    if j == 1: return 3
    if i < 3 and k == 1 or k < 3 and i == 1: return 1
    return 2

loc_dih = [(3, 0, 1, 2), (4, 0, 1, 2), (5, 0, 1, 2)]  # all type 1

CHARGES = {0: 0.145, 1: -0.683, 2: 0.418, 3: 0.040, 4: 0.040, 5: 0.040}

# --- packmol ---
def write_pdb(path, coords):
    els = ["C", "O", "H", "H", "H", "H"]
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            f.write("HETATM%5d %-3s MET %4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                    % (i + 1, els[i], 1, x, y, z, els[i]))
        f.write("END\n")

write_pdb("methanol_template.pdb", coords)

V_cm3 = N_MOL * M_METHANOL / (DENSITY * 6.022e23)
L_A = (V_cm3 * 1e24) ** (1.0 / 3.0)

with open("packmol.inp", "w") as f:
    f.write("tolerance %f\n" % PACKMOL_TOL)
    f.write("filetype pdb\n")
    f.write("output methanol_box.pdb\n")
    f.write("pbc 0. 0. 0. %f %f %f\n" % (L_A, L_A, L_A))
    f.write("structure methanol_template.pdb\n")
    f.write("  number %d\n" % N_MOL)
    f.write("  inside box 0. 0. 0. %f %f %f\n" % (L_A, L_A, L_A))
    f.write("end structure\n")

r = subprocess.run(["packmol"], stdin=open("packmol.inp"),
                   capture_output=True, text=True)
if not os.path.exists("methanol_box.pdb"):
    print(r.stdout[-2000:])
    sys.exit("packmol failed")

mol_coords = []
with open("methanol_box.pdb") as f:
    cur = []
    for line in f:
        if line.startswith("HETATM") or line.startswith("ATOM"):
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            cur.append([x, y, z])
            if len(cur) == 6:
                mol_coords.append(cur)
                cur = []
assert len(mol_coords) == N_MOL, len(mol_coords)

# --- LAMMPS data file ---
with open("methanol.data", "w") as f:
    f.write("OPLS-AA methanol box, %d molecules (RDKit+packmol)\n\n" % N_MOL)
    f.write("%d atoms\n" % (N_MOL * 6))
    f.write("%d bonds\n" % (N_MOL * len(loc_bonds)))
    f.write("%d angles\n" % (N_MOL * len(loc_angles)))
    f.write("%d dihedrals\n" % (N_MOL * len(loc_dih)))
    f.write("4 atom types\n3 bond types\n3 angle types\n1 dihedral types\n\n")
    f.write("0.0 %.6f xlo xhi\n" % L_A)
    f.write("0.0 %.6f ylo yhi\n" % L_A)
    f.write("0.0 %.6f zlo zhi\n\n" % L_A)
    f.write("Masses\n\n1 12.011\n2 15.999\n3 1.008\n4 1.008\n\n")
    f.write("Atoms # full\n\n")
    aid = 0
    LTYPE = [1, 2, 3, 4, 4, 4]  # local idx -> LAMMPS type: CT OH HO HC HC HC
    for m, mcs in enumerate(mol_coords):
        for i, (x, y, z) in enumerate(mcs):
            aid += 1
            f.write("%d %d %d %.6f %.6f %.6f %.6f\n"
                    % (aid, m + 1, LTYPE[i], CHARGES[i], x, y, z))
    f.write("\nBonds\n\n")
    bid = 0
    for m in range(N_MOL):
        o = m * 6
        for (a, b) in loc_bonds:
            bid += 1
            f.write("%d %d %d %d\n" % (bid, bond_type(a, b), o + a + 1, o + b + 1))
    f.write("\nAngles\n\n")
    for m in range(N_MOL):
        o = m * 6
        for (i, j, k) in loc_angles:
            bid += 1
            f.write("%d %d %d %d %d\n"
                    % (bid, angle_type(i, j, k), o + i + 1, o + j + 1, o + k + 1))
    f.write("\nDihedrals\n\n")
    did = 0
    for m in range(N_MOL):
        o = m * 6
        for (i, j, k, l) in loc_dih:
            did += 1
            f.write("%d 1 %d %d %d %d\n"
                    % (did, o + i + 1, o + j + 1, o + k + 1, o + l + 1))

params_out = {
    "units": "real",
    "atom_types": [
        {"name": "CT", "eps": 0.066, "sig": 3.50},
        {"name": "OH", "eps": 0.170, "sig": 3.12},
        {"name": "HO", "eps": 0.0, "sig": 0.0},
        {"name": "HC", "eps": 0.030, "sig": 2.50},
    ],
    "bond_types": [
        {"K": 320.0, "r0": 1.410, "desc": "CT-OH"},
        {"K": 340.0, "r0": 1.090, "desc": "CT-HC"},
        {"K": 553.0, "r0": 0.945, "desc": "OH-HO"},
    ],
    "angle_types": [
        {"K": 35.0, "theta0": 109.5, "desc": "HC-CT-OH"},
        {"K": 33.0, "theta0": 107.8, "desc": "HC-CT-HC"},
        {"K": 55.0, "theta0": 108.5, "desc": "CT-OH-HO"},
    ],
    "dihedral_types": [
        {"V": [0.0, 0.0, 0.450, 0.0], "desc": "HC-CT-OH-HO"},
    ],
}
with open("opls_params.json", "w") as f:
    json.dump(params_out, f, indent=2)

print("methanol.data: %d atoms, %d bonds, %d angles, %d dihedrals, L=%.3f A"
      % (N_MOL * 6, N_MOL * len(loc_bonds), N_MOL * len(loc_angles),
         N_MOL * len(loc_dih), L_A))
