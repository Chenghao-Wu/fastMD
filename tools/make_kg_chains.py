#!/usr/bin/env python3
"""Generate a Kremer-Grest melt of random-walk chains as a LAMMPS data file.

Chains are random walks with bond length --bond. Successive bond vectors
make an angle phi (phi = 0: straight) drawn from one of:

  * the Boltzmann distribution of the cosine bending potential
    E = K (1 + cos theta) = K (1 - cos phi) at temperature --temp
    (--kbend K; K = 0 gives a freely jointed chain), or
  * a freely rotating chain with fixed cos(phi) = (C - 1) / (C + 1), which
    has C_inf = C exactly (--cinf C), e.g. to match the measured
    <R^2(n)>/n of an equilibrated melt (Auhl et al., J. Chem. Phys. 119,
    12718 (2003)).

A step that would put a bead closer than --min13 to the bead two back is
redrawn. Chain starts are uniform in the box, so beads of different chains
overlap: push them apart with tools/equilibrate_melt.py (soft repulsion
ramp, then capped and full LJ), then equilibrate and check the internal
distances with fastmd.analysis.internal_distances.

Output: `Atoms # molecular` with image flags, `Velocities` (Maxwell-
Boltzmann at --temp, zero total momentum), `Bonds`, and `Angles` unless
--no-angles.

Example (2000 chains of 50 beads, semiflexible):
    python tools/make_kg_chains.py --nchains 2000 --chain-length 50 \\
        --kbend 1.5 -o melt.data
"""
import argparse
import math

import numpy as np


def sample_cos_phi(n, rng, kappa=None, cinf=None):
    if cinf is not None:
        return np.full(n, (cinf - 1.0) / (cinf + 1.0))
    u = rng.random(n)
    if kappa is None or kappa < 1e-8:
        return 2.0 * u - 1.0
    # p(x) ~ exp(kappa x) on [-1, 1], inverted CDF (stable for large kappa)
    return 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa


def rotate_bonds(prev, cos_phi, rng):
    """New unit bond vectors at angle phi to `prev` (unit), random azimuth."""
    n = len(prev)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi ** 2, 0.0, 1.0))
    psi = rng.random(n) * 2.0 * math.pi
    # orthonormal frame (prev, e1, e2)
    helper = np.where(np.abs(prev[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(prev, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(prev, e1)
    return (cos_phi[:, None] * prev + (sin_phi * np.cos(psi))[:, None] * e1
            + (sin_phi * np.sin(psi))[:, None] * e2)


def random_unit(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_chains(nchains, length, L, bond, rng, kappa=None, cinf=None, min13=1.0):
    """L: box edge lengths (3,) or a scalar for a cubic box."""
    L = np.broadcast_to(np.asarray(L, dtype=np.float64), (3,))
    pos = np.empty((nchains, length, 3))
    pos[:, 0] = rng.random((nchains, 3)) * L
    if length == 1:
        return pos
    b = random_unit(nchains, rng)
    pos[:, 1] = pos[:, 0] + bond * b
    for k in range(2, length):
        todo = np.arange(nchains)
        new_b = np.empty_like(b)
        for _ in range(1000):
            cand = rotate_bonds(b[todo], sample_cos_phi(len(todo), rng, kappa, cinf), rng)
            trial = pos[todo, k - 1] + bond * cand
            ok = np.linalg.norm(trial - pos[todo, k - 2], axis=1) >= min13
            new_b[todo[ok]] = cand[ok]
            todo = todo[~ok]
            if len(todo) == 0:
                break
        else:
            raise RuntimeError("could not satisfy --min13; lower it")
        b = new_b
        pos[:, k] = pos[:, k - 1] + bond * b
    return pos


def write_data(path, pos, L, rng, temp, angles=True):
    L = np.broadcast_to(np.asarray(L, dtype=np.float64), (3,))
    nchains, length, _ = pos.shape
    n = nchains * length
    flat = pos.reshape(n, 3)
    img = np.floor(flat / L).astype(np.int64)
    wrapped = flat - img * L
    vel = rng.normal(scale=math.sqrt(temp), size=(n, 3))
    vel -= vel.mean(axis=0)
    nbonds = nchains * (length - 1)
    nangles = nchains * max(length - 2, 0) if angles else 0
    with open(path, "w") as f:
        f.write("Kremer-Grest random-walk melt (tools/make_kg_chains.py)\n\n")
        f.write(f"{n} atoms\n{nbonds} bonds\n")
        if angles:
            f.write(f"{nangles} angles\n")
        f.write("1 atom types\n1 bond types\n")
        if angles:
            f.write("1 angle types\n")
        f.write(f"\n0 {L[0]:.10f} xlo xhi\n0 {L[1]:.10f} ylo yhi\n0 {L[2]:.10f} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\nAtoms # molecular\n\n")
        for c in range(nchains):
            for k in range(length):
                i = c * length + k
                x, y, z = wrapped[i]
                ix, iy, iz = img[i]
                f.write(f"{i + 1} {c + 1} 1 {x:.8f} {y:.8f} {z:.8f} {ix} {iy} {iz}\n")
        f.write("\nVelocities\n\n")
        for i in range(n):
            f.write(f"{i + 1} {vel[i, 0]:.8f} {vel[i, 1]:.8f} {vel[i, 2]:.8f}\n")
        f.write("\nBonds\n\n")
        b = 1
        for c in range(nchains):
            for k in range(length - 1):
                i = c * length + k + 1
                f.write(f"{b} 1 {i} {i + 1}\n")
                b += 1
        if angles and length > 2:
            f.write("\nAngles\n\n")
            a = 1
            for c in range(nchains):
                for k in range(length - 2):
                    i = c * length + k + 1
                    f.write(f"{a} 1 {i} {i + 1} {i + 2}\n")
                    a += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nchains", type=int, required=True)
    ap.add_argument("--chain-length", type=int, required=True)
    ap.add_argument("--density", type=float, default=0.85)
    ap.add_argument("--aspect", type=float, default=1.0,
                    help="Lx/Lz = Ly/Lz (> 1: laterally wide box, e.g. to build thin films)")
    ap.add_argument("--bond", type=float, default=0.97, help="bond length (default 0.97)")
    stiff = ap.add_mutually_exclusive_group()
    stiff.add_argument("--kbend", type=float, default=0.0,
                       help="K of the cosine bending potential (Boltzmann bond angles)")
    stiff.add_argument("--cinf", type=float, help="freely rotating chain with this C_inf")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--min13", type=float, default=1.0,
                    help="minimum distance between beads i and i+2")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-angles", action="store_true")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    if args.cinf is not None and args.cinf < 1.0:
        ap.error("--cinf must be >= 1")
    rng = np.random.default_rng(args.seed)
    n = args.nchains * args.chain_length
    lz = (n / (args.density * args.aspect ** 2)) ** (1.0 / 3.0)
    L = np.array([args.aspect * lz, args.aspect * lz, lz])
    pos = make_chains(args.nchains, args.chain_length, L, args.bond, rng,
                      kappa=args.kbend / args.temp if args.cinf is None else None,
                      cinf=args.cinf, min13=args.min13)
    write_data(args.output, pos, L, rng, args.temp, angles=not args.no_angles)
    r2 = ((pos[:, -1] - pos[:, 0]) ** 2).sum(axis=1).mean()
    print(f"wrote {args.output}: {n} beads, box {L[0]:.4f} x {L[1]:.4f} x {L[2]:.4f}, "
          f"<R^2>/((N-1) b^2) = {r2 / ((args.chain_length - 1) * args.bond ** 2):.3f}")


if __name__ == "__main__":
    main()
