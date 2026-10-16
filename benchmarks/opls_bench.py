#!/usr/bin/env python3
"""Electrostatics benchmark: OPLS-AA liquid, LAMMPS KOKKOS (PPPM) vs fastMD
(direct Ewald and PME), with accuracy measured against a tight reference.

Steps (all in --workdir):
  1. LAMMPS (CPU) equilibrates the data file (NVT, Ewald) -> equil.data
     (optionally replicated n x n x n first).
  2. Reference forces: LAMMPS run 0 with kspace ewald 1e-8.
  3. Accuracy (k-space only): LAMMPS pppm <acc> vs LAMMPS ewald 1e-8 (same
     real-space code); fastMD Ewald / PME (alpha matched to LAMMPS pppm) vs
     fastMD converged Ewald. fastMD uses the smallest kmax / grid whose
     k-space error does not exceed LAMMPS'.
  4. Speed: --steps NVT steps, LAMMPS KOKKOS (pppm/kk; best of the
     neigh full/half x newton on/off settings) loop time vs fastMD loop time
     for each solver.

Usage:
  python benchmarks/opls_bench.py --data data/OPLS_butane_x10/butane.data \
      --lammps lmp_kk --fastmd build/fastmd --workdir bench_opls [--replicate 2]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from opls_force_check import convert_data_nm  # noqa: E402

KCAL2KJ = 4.184
KCALA2KJNM = 41.84
A2NM = 0.1
KB = 0.00831446


def styles(pj):
    lines = ["bond_style harmonic"] + [f"bond_coeff {i+1} {b['K']} {b['r0']}"
                                       for i, b in enumerate(pj["bond_types"])]
    if pj.get("angle_types"):
        lines += ["angle_style harmonic"] + [f"angle_coeff {i+1} {a['K']} {a['theta0']}"
                                             for i, a in enumerate(pj["angle_types"])]
    if pj.get("dihedral_types"):
        lines += ["dihedral_style opls"] + [
            f"dihedral_coeff {i+1} {d['V'][0]} {d['V'][1]} {d['V'][2]} {d['V'][3]}"
            for i, d in enumerate(pj["dihedral_types"])]
    lines += [f"pair_coeff {i+1} {i+1} {t['eps']} {t['sig']}" for i, t in enumerate(pj["atom_types"])]
    return lines


def lammps_in(data, pj, rc, kspace, body):
    head = ["units real", "atom_style full", "boundary p p p",
            f"pair_style lj/cut/coul/long {rc}", "pair_modify mix geometric",
            "bond_style harmonic"]
    if pj.get("angle_types"):
        head.append("angle_style harmonic")
    if pj.get("dihedral_types"):
        head.append("dihedral_style opls")
    head += [f"read_data {data}", f"kspace_style {kspace}",
             "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5"] + styles(pj)[1:]
    head += ["neighbor 2.0 bin", "neigh_modify delay 0 every 1 check yes"]
    return "\n".join(head + body) + "\n"


def run(cmd, inp=None, cwd=None):
    r = subprocess.run(cmd, input=inp, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def lmp_forces(path):
    f = {}
    in_atoms = False
    for line in open(path):
        p = line.split()
        if p and p[0] == "ITEM:":
            in_atoms = len(p) >= 2 and p[1] == "ATOMS"
            continue
        if in_atoms and len(p) == 4:
            f[int(p[0])] = [float(x) * KCALA2KJNM for x in p[1:]]
    return f


def fmd_forces(path):
    f = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        f[int(p[0]) + 1] = [float(x) for x in p[1:4]]
    return f


def rel_rms(f, ref):
    num = sum((f[i][d] - ref[i][d]) ** 2 for i in ref for d in range(3))
    den = sum(ref[i][d] ** 2 for i in ref for d in range(3))
    return math.sqrt(num / den)


def fastmd_conf(natoms, pj, rc_nm, coulomb, data_nm, nsteps, extra):
    kT = 300.0 * KB
    lines = [f"natoms {natoms}", f"ntypes {len(pj['atom_types'])}", f"rc {rc_nm}", "skin 0.2",
             "dt 0.001", f"nsteps {nsteps}", "thermo 1 100 thermo_fastmd.dat",
             f"nvt_nh {kT:.5f} {kT:.5f} 0.1"]
    lines += [f"lj_type {i} {t['eps'] * KCAL2KJ} {t['sig'] * A2NM}" for i, t in enumerate(pj["atom_types"])]
    lines.append(coulomb)
    lines += [f"bond_harmonic {i} {b['K'] * KCAL2KJ * 100.0} {b['r0'] * A2NM}"
              for i, b in enumerate(pj["bond_types"])]
    lines += [f"angle_type {i} {a['K'] * KCAL2KJ} {a['theta0']}" for i, a in enumerate(pj.get("angle_types", []))]
    lines += [f"dihedral_opls {i} " + " ".join(str(v * KCAL2KJ) for v in d["V"])
              for i, d in enumerate(pj.get("dihedral_types", []))]
    lines.append(f"lammps_data_file {data_nm}")
    return "\n".join(lines + extra) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lammps", default=os.environ.get("LAMMPS_BINARY", "lmp"))
    ap.add_argument("--fastmd", default=os.environ.get("FASTMD_BINARY", "fastmd"))
    ap.add_argument("--workdir", default="bench_opls")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--rc", type=float, default=10.0, help="cutoff (Angstrom)")
    ap.add_argument("--acc", type=float, default=1e-5, help="LAMMPS pppm relative accuracy")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--equil", type=int, default=5000)
    args = ap.parse_args()

    data = os.path.abspath(args.data)
    pj = json.load(open(os.path.join(os.path.dirname(data), "opls_params.json")))
    lmp = os.path.abspath(args.lammps) if os.path.exists(args.lammps) else args.lammps
    fmd = os.path.abspath(args.fastmd) if os.path.exists(args.fastmd) else args.fastmd
    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)
    gpu = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "off", "neigh", "full"]

    if not os.path.exists("equil.data"):
        print(f"[1] equilibrating ({args.equil} steps, replicate {args.replicate})")
        body = [f"replicate {args.replicate} {args.replicate} {args.replicate}"] if args.replicate > 1 else []
        body += ["velocity all create 300.0 4928 mom yes rot yes dist gaussian",
                 "fix 1 all nvt temp 300.0 300.0 100.0", "timestep 1.0", "thermo 1000",
                 f"run {args.equil}", "write_data equil.data nocoeff"]
        run([lmp] + gpu + ["-in", "/dev/stdin", "-log", "log.equil", "-screen", "none"],
            lammps_in(data, pj, args.rc, f"pppm {args.acc}", body))
    natoms = int(next(l.split()[0] for l in open("equil.data") if l.split()[1:2] == ["atoms"]))
    convert_data_nm("equil.data", "equil_nm.data")
    L_nm = next(float(l.split()[1]) - float(l.split()[0]) for l in open("equil_nm.data")
                if l.split()[2:3] == ["xlo"])

    print(f"[2] reference forces ({natoms} atoms, ewald 1e-8)")
    dump = ["dump 1 all custom 1 {} id fx fy fz", "dump_modify 1 sort id", "run 0"]
    run([lmp, "-in", "/dev/stdin", "-log", "log.ref", "-screen", "none"],
        lammps_in("equil.data", pj, args.rc, "ewald 1e-8",
                  [dump[0].format("f_ref.lmp"), dump[1], dump[2]]))
    ref = lmp_forces("f_ref.lmp")

    print(f"[3] k-space accuracy: LAMMPS pppm {args.acc}")
    run([lmp, "-in", "/dev/stdin", "-log", "log.pppm", "-screen", "none"],
        lammps_in("equil.data", pj, args.rc, f"pppm {args.acc}",
                  [dump[0].format("f_pppm.lmp"), dump[1], dump[2]]))
    log = open("log.pppm").read()
    g_A = float(re.search(r"G vector \(1/distance\)\s*=\s*([0-9.eE+-]+)", log).group(1))
    grid = re.search(r"grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)", log).groups()
    alpha_nm = g_A * 10.0
    err_pppm = rel_rms(lmp_forces("f_pppm.lmp"), ref)
    results = {"system": os.path.basename(data), "replicate": args.replicate, "natoms": natoms,
               "lammps_pppm": {"accuracy": args.acc, "grid": [int(g) for g in grid],
                               "alpha_per_A": g_A, "kspace_rel_rms_force_error": err_pppm}}
    print(f"    LAMMPS pppm grid {grid}, alpha {g_A:.4f}/A: k-space rel RMS force error "
          f"{err_pppm:.2e} (vs LAMMPS ewald 1e-8, same real-space code)")

    # fastMD k-space error against fastMD's own converged Ewald (same alpha
    # and real-space code), so only the reciprocal approximation differs.
    def fmd_force_run(tag, coul):
        open(f"f_{tag}.conf", "w").write(fastmd_conf(natoms, pj, args.rc * A2NM, coul,
                                                      "equil_nm.data", 1,
                                                      [f"forces_dump_file f_{tag}.dat"]))
        run([fmd, f"f_{tag}.conf"])
        return fmd_forces(f"f_{tag}.dat")
    kmax_ref = math.ceil(alpha_nm * L_nm * math.sqrt(23.0) / math.pi)   # exp(-k^2..) ~ 1e-10
    solvers = {}
    if kmax_ref <= 64:
        fref = fmd_force_run("ewald_ref", f"coulomb ewald {alpha_nm:.6f} {kmax_ref}")
        results["fastmd_total_rel_rms_force_error_vs_lammps"] = rel_rms(fref, ref)
        print(f"    fastMD converged Ewald (kmax {kmax_ref}) vs LAMMPS reference: total rel RMS "
              f"{rel_rms(fref, ref):.2e} (real-space erfc / fp32 floor)")
        for kmax in range(2, kmax_ref + 1):
            err = rel_rms(fmd_force_run(f"ewald{kmax}", f"coulomb ewald {alpha_nm:.6f} {kmax}"), fref)
            if err <= err_pppm:
                solvers["ewald"] = (f"coulomb ewald {alpha_nm:.6f} {kmax}", err)
                break
    else:
        fref = None
        print(f"    direct Ewald reference needs kmax {kmax_ref} > 64: PME errors vs LAMMPS reference")
    for K in range(8, 400):
        if not _friendly(K):
            continue
        f = fmd_force_run(f"pme{K}", f"coulomb pme {alpha_nm:.6f} {K} 5")
        err = rel_rms(f, fref) if fref else rel_rms(f, ref)
        if err <= err_pppm:
            solvers["pme"] = (f"coulomb pme {alpha_nm:.6f} {K} 5", err)
            break
    for name, (coul, err) in solvers.items():
        results[f"fastmd_{name}"] = {"coulomb": coul, "kspace_rel_rms_force_error": err}
        print(f"    fastMD {coul}: k-space rel RMS force error {err:.2e}")

    print(f"[4] speed: {args.steps} NVT steps")
    # Best of the KOKKOS neighbor-list / newton settings for LAMMPS.
    kk = {}
    for nl, newton in (("full", "off"), ("half", "on"), ("half", "off")):
        run([lmp, "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", nl, "newton", newton,
             "-in", "/dev/stdin", "-log", "log.kk", "-screen", "none"],
            lammps_in("equil.data", pj, args.rc, f"pppm {args.acc}",
                      ["fix 1 all nvt temp 300.0 300.0 100.0", "timestep 1.0", "thermo 100",
                       f"run {args.steps}"]))
        loop = float(re.search(r"Loop time of\s*([0-9.eE+-]+)", open("log.kk").read()).group(1))
        kk[f"neigh {nl} newton {newton}"] = args.steps / loop
        print(f"    LAMMPS KOKKOS pppm (neigh {nl}, newton {newton}): {args.steps / loop:.0f} steps/s")
    results["lammps_pppm"]["steps_per_s_by_setting"] = kk
    results["lammps_pppm"]["steps_per_s"] = max(kk.values())
    for name, (coul, _) in solvers.items():
        open(f"run_{name}.conf", "w").write(fastmd_conf(natoms, pj, args.rc * A2NM, coul, "equil_nm.data",
                                                         args.steps, []))
        stdout = run([fmd, f"run_{name}.conf"])
        sps = float(re.search(r"Steps/s:\s*([0-9.]+)", stdout).group(1))
        results[f"fastmd_{name}"]["steps_per_s"] = sps
        print(f"    fastMD {name}: {sps:.0f} steps/s")
    json.dump(results, open("opls_bench.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


def _friendly(n):
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


if __name__ == "__main__":
    main()
