"""Analysis of fastMD output (numpy only).

Trajectory correlations use multiple time origins: every frame whose step
is a multiple of ``origin_every`` is an origin, and every later frame
(up to ``max_lag`` steps) contributes at lag = step - origin step. With a
log-periodic trajectory (``traj <file> log <block>``) set ``origin_every``
to the block length: short lags then come from the log-spaced frames in
each block and long lags from frames of later blocks.

Optional ``labels(frame) -> int array`` assigns every object (bead or
chain) a group at the time origin, e.g. a depth bin; results are then
reported per group. This gives conditional correlations such as
P(t | z_cm(0) in bin) or MSD(t | z(0) in bin).

Chains are the molecules of the data file (``Trajectory.mol_ids``); the
chain functions expect monodisperse chains stored contiguously.
"""

import math
import re

import numpy as np

__all__ = [
    "chain_index", "internal_distances", "time_correlation", "msd", "end_to_end_acf",
    "rouse_mode_acf", "self_intermediate_scattering", "stress_relaxation", "storage_loss_moduli",
    "radius_of_gyration_tensor",
]


# --------------------------------------------------------------------------
# chains

def chain_index(mol_ids):
    """(nchains, N) atom indices of monodisperse, contiguously stored chains."""
    mol_ids = np.asarray(mol_ids)
    order = np.argsort(mol_ids, kind="stable")
    ids, counts = np.unique(mol_ids, return_counts=True)
    if len(set(counts.tolist())) != 1:
        raise ValueError("chains have different lengths")
    idx = order.reshape(len(ids), counts[0])
    return idx


def internal_distances(x, mol_ids):
    """<R^2(n)>/n over all chains and segment pairs |i - j| = n.

    x: unwrapped positions (natoms, 3). Returns (n, r2_over_n)."""
    idx = chain_index(mol_ids)
    c = np.asarray(x, dtype=np.float64)[idx]            # (nchains, N, 3)
    N = c.shape[1]
    n = np.arange(1, N)
    r2 = np.array([((c[:, k:] - c[:, :-k]) ** 2).sum(axis=2).mean() for k in n])
    return n, r2 / n


def radius_of_gyration_tensor(x, mol_ids):
    """Per-chain gyration tensors (nchains, 3, 3) and centres of mass (nchains, 3)."""
    idx = chain_index(mol_ids)
    c = np.asarray(x, dtype=np.float64)[idx]
    com = c.mean(axis=1)
    d = c - com[:, None, :]
    return np.einsum("cni,cnj->cij", d, d) / c.shape[1], com


# --------------------------------------------------------------------------
# multi-origin correlations

def time_correlation(traj, quantity, combine, origin_every=None, max_lag=None, labels=None,
                     frames=None):
    """Average combine(q(origin), q(t)) over objects and time origins.

    quantity(k) -> array (nobj, ...) for frame k; combine(q0, q1) -> (nobj,)
    values. Returns (lags in steps, mean values [ngroups, nlags],
    sample counts [ngroups, nlags], group ids)."""
    steps = traj.steps
    sel = np.arange(len(steps)) if frames is None else np.asarray(frames)
    if origin_every is None:
        origin_every = int(np.min(np.diff(steps[sel]))) if len(sel) > 1 else 1
    origins = [k for k in sel if steps[k] % origin_every == 0]
    sums, counts = {}, {}
    groups = set()
    for o in origins:
        q0 = quantity(o)
        lab = np.zeros(len(q0), dtype=np.int64) if labels is None else np.asarray(labels(o))
        groups.update(np.unique(lab).tolist())
        for k in sel[sel >= o]:
            lag = int(steps[k] - steps[o])
            if max_lag is not None and lag > max_lag:
                break
            v = np.asarray(combine(q0, quantity(k)), dtype=np.float64)
            s = np.bincount(lab, weights=v, minlength=lab.max() + 1)
            c = np.bincount(lab, minlength=lab.max() + 1)
            if lag not in sums:
                sums[lag] = np.zeros(0)
                counts[lag] = np.zeros(0)
            m = max(len(sums[lag]), len(s))
            sums[lag] = np.pad(sums[lag], (0, m - len(sums[lag]))) + np.pad(s, (0, m - len(s)))
            counts[lag] = np.pad(counts[lag], (0, m - len(counts[lag]))) + np.pad(c, (0, m - len(c)))
    if not sums:
        raise ValueError("no time origins (check origin_every)")
    lags = np.array(sorted(sums))
    gids = np.array(sorted(groups))
    G = gids.max() + 1
    S = np.zeros((G, len(lags)))
    C = np.zeros((G, len(lags)))
    for j, lag in enumerate(lags):
        S[:len(sums[lag]), j] = sums[lag]
        C[:len(counts[lag]), j] = counts[lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / C
    return lags, mean[gids], C[gids], gids


def msd(traj, com=False, **kw):
    """Mean-square displacement: of beads (g1) or, with com=True, of chain
    centres of mass (g3). Returns (lags, msd[groups, lags], counts, groups)."""
    if com:
        idx = chain_index(traj.mol_ids)
        quantity = lambda k: traj.unwrapped(k)[idx].mean(axis=1)
    else:
        quantity = traj.unwrapped
    return time_correlation(traj, quantity, lambda a, b: ((b - a) ** 2).sum(axis=1), **kw)


def end_to_end_acf(traj, **kw):
    """P(t) = <R(t).R(0)> / <R(0)^2> per group (R: chain end-to-end vector)."""
    idx = chain_index(traj.mol_ids)

    def ree(k):
        x = traj.unwrapped(k)
        return x[idx[:, -1]] - x[idx[:, 0]]
    lags, num, cnt, gids = time_correlation(traj, ree, lambda a, b: (a * b).sum(axis=1), **kw)
    j0 = np.nonzero(lags == 0)[0]
    if len(j0) == 0:
        raise ValueError("lag 0 missing")
    return lags, num / num[:, j0[0]][:, None], cnt, gids


def rouse_mode_acf(traj, modes=(1, 2, 4, 8), **kw):
    """Normalised autocorrelations <X_p(t).X_p(0)>/<X_p^2> of Rouse modes
    X_p = (1/N) sum_n r_n cos(p pi (n - 1/2) / N). Returns (lags,
    {p: acf[groups, lags]}, counts, groups)."""
    idx = chain_index(traj.mol_ids)
    N = idx.shape[1]
    n = np.arange(1, N + 1)
    basis = np.array([np.cos(p * math.pi * (n - 0.5) / N) / N for p in modes])   # (P, N)

    def modes_of(k):
        c = traj.unwrapped(k)[idx]                       # (nchains, N, 3)
        return np.einsum("pn,cnd->cpd", basis, c)        # (nchains, P, 3)
    out = {}
    lags = cnt = gids = None
    for j, p in enumerate(modes):
        lags, num, cnt, gids = time_correlation(
            traj, lambda k, j=j: modes_of(k)[:, j, :], lambda a, b: (a * b).sum(axis=1), **kw)
        j0 = np.nonzero(lags == 0)[0][0]
        out[p] = num / num[:, j0][:, None]
    return lags, out, cnt, gids


def self_intermediate_scattering(traj, q, directions=("x", "y", "z"), **kw):
    """F_s(q, t) = <cos(q e.(r(t) - r(0)))> averaged over beads, origins and
    the given unit directions e ('x', 'y', 'z' or 3-vectors). Use
    directions=('x', 'y') for in-plane and ('z',) for the film normal."""
    axes = {"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}
    E = np.array([axes[d] if isinstance(d, str) else d for d in directions], dtype=np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return time_correlation(traj, traj.unwrapped,
                            lambda a, b: np.cos(q * ((b - a) @ E.T)).mean(axis=1), **kw)


# --------------------------------------------------------------------------
# stress relaxation

def stress_relaxation(path, temperature, volume=None, components=("xy", "xz", "yz"),
                      fluctuations=True):
    """Shear relaxation modulus from a fastMD stress-correlation file.

    G(t) = V / (k_B T) <ds(t) ds(0)>, averaged over `components`, with
    ds the fluctuation about the run mean (fluctuations=True). `volume`
    defaults to the box volume recorded in the file; for a film pass the
    polymer volume A h (the stress in the file is normalised by the box
    volume, so it is rescaled accordingly). Returns (lag_steps, G)."""
    v_box = None
    means = None
    header = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"V = ([0-9.eE+-]+)", line)
            if m:
                v_box = float(m.group(1))
            if line.startswith("# channel means:"):
                means = np.array([float(x) for x in line.split(":")[1].split()])
            if line.startswith("# lag_steps"):
                header = line[1:].split()
    if header is None:
        raise ValueError(f"{path}: no '# lag_steps' header line")
    data = np.loadtxt(path, comments="#")
    names = {"xx": "C_xx", "xy": "C_xy", "xz": "C_xz", "yy": "C_yy", "yz": "C_yz", "zz": "C_zz"}
    chan = ["xx", "xy", "xz", "yy", "yz", "zz"]
    cols = [header.index(names[c]) for c in components]
    corr = data[:, cols]
    if fluctuations and means is not None:
        corr = corr - np.array([means[chan.index(c)] ** 2 for c in components])
    if v_box is None:
        raise ValueError(f"{path}: no volume in header")
    V = v_box if volume is None else volume
    # file stress s = S / V_box (S extensive); G = <dS dS> / (kT V)
    G = corr.mean(axis=1) * v_box ** 2 / (temperature * V)
    return data[:, 0].astype(np.int64), G


def storage_loss_moduli(t, G, omega):
    """G'(w) = w int G sin(wt) dt and G''(w) = w int G cos(wt) dt for a
    piecewise-linear G(t) on the sampled times, integrated exactly
    segment by segment (no FFT, works on log-spaced data). The data must
    start at t = 0 and decay within the window; frequencies below about
    2 pi / t_max are not supported by the data, and frequencies above
    about 1 / (sampling interval where G is still significant) are limited
    by the linear interpolation between samples. Returns (G', G'')."""
    t = np.asarray(t, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    if t[0] != 0.0:
        raise ValueError("G(t) must start at t = 0")
    w = np.asarray(omega, dtype=np.float64)[:, None]
    a, b = t[:-1][None, :], t[1:][None, :]
    ga, gb = G[:-1][None, :], G[1:][None, :]
    s = (gb - ga) / (b - a)
    c0 = ga - s * a                         # G = c0 + s t on [a, b]

    def int_sin(x):   # antiderivative of (c0 + s t) sin(w t)
        return -c0 * np.cos(w * x) / w + s * (np.sin(w * x) / w ** 2 - x * np.cos(w * x) / w)

    def int_cos(x):
        return c0 * np.sin(w * x) / w + s * (np.cos(w * x) / w ** 2 + x * np.sin(w * x) / w)
    Gp = w[:, 0] * (int_sin(b) - int_sin(a)).sum(axis=1)
    Gpp = w[:, 0] * (int_cos(b) - int_cos(a)).sum(axis=1)
    return Gp, Gpp
