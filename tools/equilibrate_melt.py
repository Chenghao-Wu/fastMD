#!/usr/bin/env python3
"""Push off and equilibrate a generated Kremer-Grest melt with fastMD.

Auhl-style fast push-off (J. Chem. Phys. 119, 12718 (2003)) in stages,
each a fastMD run (Langevin) that starts from the previous stage's restart:

  1. soft      pair_soft: A [1 + cos(pi r / rc)], A ramped 0 -> --soft-amax,
               rc = 2^(1/6), dt --soft-dt
  2. cap<F>    LJ with lj_force_cap F for F in --caps, dt --cap-dt
  3. equil     full LJ (--rc, optionally shifted) for --equil-steps, dt --dt

Bonds are FENE (k 30, R0 1.5, WCA eps = sigma = 1); with --kbend > 0 the
cosine bending potential K (1 + cos theta) acts on the angles of the data
file. The internal distances <R^2(n)>/n of the input and of every stage
are printed (and written to <workdir>/internal_distances.txt) so that
chain-scale distortions from the push-off can be checked.

Example:
    python tools/make_kg_chains.py --nchains 500 --chain-length 50 --kbend 1.5 -o raw.data
    python tools/equilibrate_melt.py raw.data --kbend 1.5 --equil-steps 200000 --workdir eq
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))


def read_data(path):
    """Unwrapped positions, molecule ids (atom-id order) and box edge lengths
    (Lx, Ly, Lz) of a LAMMPS data file written by fastMD or the tools."""
    with open(path) as f:
        lines = f.read().splitlines()
    L = np.zeros(3)
    atoms = []
    section = None
    for line in lines[1:]:
        tok = line.split("#")[0].split()
        if not tok:
            continue
        if tok[0][0].isalpha():
            section = tok[0]
            continue
        if section is None and len(tok) == 4 and tok[2] in ("xlo", "ylo", "zlo"):
            L["xyz".index(tok[2][0])] = float(tok[1]) - float(tok[0])
        elif section == "Atoms":
            atoms.append(tok)
    atoms.sort(key=lambda t: int(t[0]))
    ncol = len(atoms[0])
    # molecular: id mol type x y z [ix iy iz]; full: id mol type q x y z [ix iy iz]
    xc = 3 if ncol in (6, 9) else 4
    x = np.array([[float(t[xc]), float(t[xc + 1]), float(t[xc + 2])] for t in atoms])
    if ncol in (9, 10):
        img = np.array([[int(t[xc + 3]), int(t[xc + 4]), int(t[xc + 5])] for t in atoms])
        x = x + img * L
    mol = np.array([int(t[1]) for t in atoms])
    return x, mol, L


def internal(path):
    from fastmd.analysis import internal_distances
    x, mol, _ = read_data(path)
    return internal_distances(x, mol)


def fastmd_binary(arg):
    if arg:
        return os.path.abspath(arg)
    try:
        from fastmd import get_binary_path
        return get_binary_path()
    except Exception:
        return "fastmd"


def run_stage(fmd, name, data, args, pair_lines, rc, dt, steps, workdir):
    angles = [f"angle_cosine 0 {args.kbend}"] if args.kbend > 0 else []
    conf = "\n".join([
        "ntypes 1", f"rc {rc}", "skin 0.3", f"dt {dt}", f"nsteps {steps}",
        f"nvt_langevin {args.temp} {args.temp} {args.tdamp} {args.seed}",
        f"thermo 1 {max(1, steps // 20)} thermo_{name}.dat",
        "bond_type 0 30.0 1.5 1.0 1.0", *angles, *pair_lines,
        f"lammps_data_file {os.path.abspath(data)}", f"restart 0 {name}.data"]) + "\n"
    path = os.path.join(workdir, f"{name}.conf")
    with open(path, "w") as f:
        f.write(conf)
    r = subprocess.run([fmd, f"{name}.conf"], cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"stage {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    out = os.path.join(workdir, f"{name}_final.data")
    if not os.path.exists(out):
        sys.exit(f"stage {name}: {out} not written")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="generated melt (tools/make_kg_chains.py)")
    ap.add_argument("--fastmd", help="fastmd binary (default: package binary or PATH)")
    ap.add_argument("--workdir", default="equilibrate")
    ap.add_argument("--kbend", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tdamp", type=float, default=1.0, help="Langevin damping time (1/gamma)")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--soft-steps", type=int, default=20000)
    ap.add_argument("--soft-amax", type=float, default=100.0)
    ap.add_argument("--soft-dt", type=float, default=0.005)
    ap.add_argument("--caps", type=float, nargs="*", default=[10.0, 100.0, 1000.0])
    ap.add_argument("--cap-steps", type=int, default=5000)
    ap.add_argument("--cap-dt", type=float, default=0.005)
    ap.add_argument("--rc", type=float, default=2.0 ** (1.0 / 6.0), help="final LJ cutoff")
    ap.add_argument("--shift", action="store_true", help="energy-shifted LJ in the final stage")
    ap.add_argument("--equil-steps", type=int, default=100000)
    ap.add_argument("--dt", type=float, default=0.01)
    args = ap.parse_args()

    fmd = fastmd_binary(args.fastmd)
    os.makedirs(args.workdir, exist_ok=True)
    report = []
    n, r = internal(args.data)
    report.append(("input", n, r))
    wca = 2.0 ** (1.0 / 6.0)
    data = run_stage(fmd, "soft", args.data, args, [f"pair_soft 0 {args.soft_amax}"], wca,
                     args.soft_dt, args.soft_steps, args.workdir)
    report.append(("soft", *internal(data)))
    for F in args.caps:
        name = f"cap{int(F)}"
        data = run_stage(fmd, name, data, args, ["lj 0 0 1.0 1.0", f"lj_force_cap {F}"], wca,
                         args.cap_dt, args.cap_steps, args.workdir)
        report.append((name, *internal(data)))
    final_pair = ["lj 0 0 1.0 1.0"] + (["pair_shift yes"] if args.shift else [])
    data = run_stage(fmd, "equil", data, args, final_pair, args.rc, args.dt, args.equil_steps,
                     args.workdir)
    report.append(("equil", *internal(data)))

    pick = [k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500) if k <= len(report[0][1])]
    lines = ["# <R^2(n)>/n at n = " + " ".join(str(k) for k in pick)]
    for name, _, rr in report:
        lines.append(f"{name:8s} " + " ".join(f"{rr[k - 1]:.3f}" for k in pick))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.workdir, "internal_distances.txt"), "w") as f:
        f.write(text + "\n")
    print(f"equilibrated melt: {data}")


if __name__ == "__main__":
    main()
