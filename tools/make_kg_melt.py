#!/usr/bin/env python3
"""Generate an overlap-free Kremer-Grest bead-spring melt as a LAMMPS data file.

Beads are placed on a simple-cubic lattice (n^3 >= N sites filling the box)
traced by a boustrophedon ("snake") path, so consecutive sites are always
lattice neighbours. Chains occupy consecutive path segments, with the spare
sites spread evenly between chains; every bond therefore has length
a = L/n (~1.0-1.1 sigma), well inside the FENE range, and no two beads are
closer than a. A small random jitter breaks lattice symmetry.

The structure is artificial: equilibrate (e.g. a few 1e4 Langevin steps)
before measuring anything. Output uses ``Atoms # molecular`` with image
flags, a ``Velocities`` section (Maxwell-Boltzmann at --temp, zero total
momentum), ``Bonds`` and (unless --no-angles) ``Angles``.

Example (30,000 beads, 300 chains of 100, density 0.85):
    python tools/make_kg_melt.py --nchains 300 --chain-length 100 \
        --density 0.85 -o kg_30k.data
"""
import argparse
import math
import random


def snake_sites(n):
    """Yield (ix, iy, iz) for an n^3 lattice so consecutive sites are adjacent."""
    row = 0
    for iz in range(n):
        ys = range(n) if iz % 2 == 0 else range(n - 1, -1, -1)
        for iy in ys:
            xs = range(n) if row % 2 == 0 else range(n - 1, -1, -1)
            for ix in xs:
                yield ix, iy, iz
            row += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nchains", type=int, required=True)
    ap.add_argument("--chain-length", type=int, required=True)
    ap.add_argument("--density", type=float, default=0.85)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="uniform random displacement amplitude (sigma)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-angles", action="store_true")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    natoms = args.nchains * args.chain_length
    box = (natoms / args.density) ** (1.0 / 3.0)
    n = math.ceil(natoms ** (1.0 / 3.0))
    a = box / n                       # lattice spacing (fills the box exactly)
    if a < 0.9:
        raise SystemExit(f"lattice spacing {a:.3f} too small; lower the density")

    path = [((ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a)
            for ix, iy, iz in snake_sites(n)]
    # Spread the n^3 - natoms vacancies evenly between chains (never inside a
    # chain, which would stretch a bond to 2a) so the melt has no void slab.
    sites = []
    for c in range(args.nchains):
        start = (c * len(path)) // args.nchains
        sites.extend(path[start:start + args.chain_length])

    pos, img = [], []
    for x, y, z in sites:
        p = [x + rng.uniform(-args.jitter, args.jitter),
             y + rng.uniform(-args.jitter, args.jitter),
             z + rng.uniform(-args.jitter, args.jitter)]
        im = [0, 0, 0]
        for d in range(3):
            while p[d] < 0.0:
                p[d] += box
                im[d] -= 1
            while p[d] >= box:
                p[d] -= box
                im[d] += 1
        pos.append(p)
        img.append(im)

    sd = math.sqrt(args.temp)
    vel = [[rng.gauss(0.0, sd) for _ in range(3)] for _ in range(natoms)]
    for d in range(3):
        mean = sum(v[d] for v in vel) / natoms
        for v in vel:
            v[d] -= mean

    bonds, angles = [], []
    for c in range(args.nchains):
        base = c * args.chain_length
        for k in range(args.chain_length - 1):
            bonds.append((base + k, base + k + 1))
        if not args.no_angles:
            for k in range(args.chain_length - 2):
                angles.append((base + k, base + k + 1, base + k + 2))

    with open(args.output, "w") as f:
        f.write(f"Kremer-Grest melt: {args.nchains} chains x {args.chain_length} beads, "
                f"density {args.density} (make_kg_melt.py seed {args.seed})\n\n")
        f.write(f"{natoms} atoms\n{len(bonds)} bonds\n")
        if angles:
            f.write(f"{len(angles)} angles\n")
        f.write("1 atom types\n1 bond types\n")
        if angles:
            f.write("1 angle types\n")
        f.write(f"\n0.0 {box:.10f} xlo xhi\n0.0 {box:.10f} ylo yhi\n0.0 {box:.10f} zlo zhi\n")
        f.write("\nMasses\n\n1 1.0\n")
        f.write("\nAtoms # molecular\n\n")
        for i, (p, im) in enumerate(zip(pos, img)):
            mol = i // args.chain_length + 1
            f.write(f"{i + 1} {mol} 1 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                    f"{im[0]} {im[1]} {im[2]}\n")
        f.write("\nVelocities\n\n")
        for i, v in enumerate(vel):
            f.write(f"{i + 1} {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        f.write("\nBonds\n\n")
        for b, (i, j) in enumerate(bonds):
            f.write(f"{b + 1} 1 {i + 1} {j + 1}\n")
        if angles:
            f.write("\nAngles\n\n")
            for k, (i, j, l) in enumerate(angles):
                f.write(f"{k + 1} 1 {i + 1} {j + 1} {l + 1}\n")
    print(f"wrote {args.output}: {natoms} atoms, box {box:.4f}, lattice a={a:.4f}")


if __name__ == "__main__":
    main()
