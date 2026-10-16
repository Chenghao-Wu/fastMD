#!/usr/bin/env python3
"""Build a free-standing film from an equilibrated bulk melt with fastMD.

Follows the no-cutting route: chains are never cut.

  1. unwrap   the bulk melt is unwrapped along the film normal (z) and put
              in a box that is taller by 2 x --vacuum (x, y stay periodic);
              written as film_raw.data
  2. compress repulsive LJ 9-3 walls (cutoff at the potential minimum)
              move from just outside the melt to the target planes, about
              the film centre, --thickness apart (default: the bulk box
              height, i.e. bulk density) plus --wall-gap per side
  3. anneal   walls fixed at the target planes
  4. release  walls removed: free surfaces; `film z` tracks the film
              reference plane

Each stage is a fastMD Langevin run starting from the previous restart.
The density profile of the final film (fastmd.analysis.density_profile)
and its thickness between the half-density dividing surfaces are printed
and written to <workdir>/film_profile.txt.

Example:
    python tools/make_kg_chains.py --nchains 400 --chain-length 50 --kbend 1.5 \\
        --aspect 3 -o raw.data
    python tools/equilibrate_melt.py raw.data --kbend 1.5 --rc 2.5 --workdir bulk
    python tools/build_film.py bulk/equil_final.data --kbend 1.5 --workdir film
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)
from equilibrate_melt import fastmd_binary, read_data  # noqa: E402

WCA93 = (2.0 / 5.0) ** (1.0 / 6.0)   # minimum of the LJ 9-3 wall potential (sigma = 1)


def unwrap_into_tall_box(src, dst, vacuum):
    """Rewrite a LAMMPS data file with the chains unwrapped along z and a
    taller box. Returns (zlo_new_box_height, original height, z positions)."""
    lines = open(src).read().splitlines()
    zlo = zhi = None
    head_end = None
    for k, line in enumerate(lines):
        tok = line.split()
        if len(tok) == 4 and tok[2] == "zlo":
            zlo, zhi = float(tok[0]), float(tok[1])
        if line.strip().startswith("Atoms"):
            head_end = k
            break
    if zlo is None or head_end is None:
        sys.exit(f"{src}: no box or Atoms section")
    Lz = zhi - zlo
    # Atoms section lines
    k = head_end + 1
    while k < len(lines) and not lines[k].split():
        k += 1
    start = k
    while k < len(lines) and lines[k].split() and not lines[k].split()[0][0].isalpha():
        k += 1
    stop = k
    rows = [lines[j].split() for j in range(start, stop)]
    ncol = len(rows[0])
    if ncol not in (9, 10):
        sys.exit(f"{src}: Atoms lines need image flags (molecular or full style)")
    zc = 5 if ncol == 9 else 6        # z column; image flags follow x y z
    izc = zc + 3
    zu = np.array([float(r[zc]) - zlo + int(r[izc]) * Lz for r in rows])
    shift = vacuum - zu.min()
    height = zu.max() - zu.min() + 2.0 * vacuum
    for r, z in zip(rows, zu):
        r[zc] = f"{z + shift:.8f}"
        r[izc] = "0"
    out = lines[:]
    for j, r in zip(range(start, stop), rows):
        out[j] = " ".join(r)
    for j in range(head_end):
        tok = out[j].split()
        if len(tok) == 4 and tok[2] == "zlo":
            out[j] = f"0 {height:.8f} zlo zhi"
    open(dst, "w").write("\n".join(out) + "\n")
    return height, Lz, zu + shift


def run_stage(fmd, name, data, args, extra, steps, workdir):
    angles = [f"angle_cosine 0 {args.kbend}"] if args.kbend > 0 else []
    pair = ["lj 0 0 1.0 1.0"] + (["pair_shift yes"] if args.shift else [])
    conf = "\n".join([
        "ntypes 1", f"rc {args.rc}", "skin 0.3", f"dt {args.dt}", f"nsteps {steps}",
        f"nvt_langevin {args.temp} {args.temp} {args.tdamp} {args.seed}",
        f"thermo 1 {max(1, steps // 20)} thermo_{name}.dat",
        "bond_type 0 30.0 1.5 1.0 1.0", *angles, *pair, *extra,
        f"lammps_data_file {os.path.abspath(data)}", f"restart 0 {name}.data"]) + "\n"
    open(os.path.join(workdir, f"{name}.conf"), "w").write(conf)
    r = subprocess.run([fmd, f"{name}.conf"], cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"stage {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return os.path.join(workdir, f"{name}_final.data")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="equilibrated bulk melt (restart data file)")
    ap.add_argument("--fastmd")
    ap.add_argument("--workdir", default="film")
    ap.add_argument("--vacuum", type=float, default=10.0, help="vacuum per side (sigma)")
    ap.add_argument("--thickness", type=float, help="target thickness (default: bulk box height)")
    ap.add_argument("--wall-gap", type=float, default=0.5, help="extra wall offset per side")
    ap.add_argument("--kbend", type=float, default=0.0)
    ap.add_argument("--rc", type=float, default=2.5, help="LJ cutoff (attractive for free surfaces)")
    ap.add_argument("--shift", action="store_true", help="energy-shifted LJ")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tdamp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--compress-steps", type=int, default=20000)
    ap.add_argument("--anneal-steps", type=int, default=20000)
    ap.add_argument("--release-steps", type=int, default=50000)
    args = ap.parse_args()

    fmd = fastmd_binary(args.fastmd)
    os.makedirs(args.workdir, exist_ok=True)
    raw = os.path.join(args.workdir, "film_raw.data")
    height, Lz_bulk, z = unwrap_into_tall_box(args.data, raw, args.vacuum)
    h = args.thickness if args.thickness else Lz_bulk
    centre = z.mean()   # monodisperse KG melt: equal masses
    lo_target = centre - 0.5 * h - args.wall_gap
    hi_target = centre + 0.5 * h + args.wall_gap
    lo_start = max(z.min() - 0.5, 0.0)
    hi_start = min(z.max() + 0.5, height)
    if lo_target < lo_start or hi_target > hi_start:
        print(f"note: target planes lie outside the unwrapped melt; walls only confine")
        lo_target, hi_target = min(lo_target, lo_start), max(hi_target, hi_start)
    walls = lambda lo0, lo1, hi0, hi1: [f"wall zlo lj93 1.0 1.0 {WCA93:.6f} {lo0:.6f} {lo1:.6f}",
                                        f"wall zhi lj93 1.0 1.0 {WCA93:.6f} {hi0:.6f} {hi1:.6f}"]
    print(f"unwrapped melt spans {z.max() - z.min():.2f} (bulk box height {Lz_bulk:.2f}); "
          f"tall box {height:.2f}; walls {lo_start:.2f}->{lo_target:.2f}, {hi_start:.2f}->{hi_target:.2f}")
    data = run_stage(fmd, "compress", raw, args, walls(lo_start, lo_target, hi_start, hi_target),
                     args.compress_steps, args.workdir)
    data = run_stage(fmd, "anneal", data, args, walls(lo_target, lo_target, hi_target, hi_target),
                     args.anneal_steps, args.workdir)
    data = run_stage(fmd, "release", data, args, ["film z"], args.release_steps, args.workdir)

    from fastmd.analysis import density_profile, film_thickness
    x, mol, L = read_data(data)
    zc, rho = density_profile(x, axis=2, area=L[0] * L[1], bin_width=0.5)
    thick, lo, hi, rho_bulk = film_thickness(zc, rho)
    lines = [f"# film {data}", f"# thickness {thick:.4f} (dividing surfaces {lo:.3f} {hi:.3f} "
             f"about the centre), plateau density {rho_bulk:.4f}", "# z-z_c  rho"]
    lines += [f"{a:.3f} {b:.5f}" for a, b in zip(zc, rho)]
    open(os.path.join(args.workdir, "film_profile.txt"), "w").write("\n".join(lines) + "\n")
    print(lines[1][2:])
    print(f"film: {data}")


if __name__ == "__main__":
    main()
