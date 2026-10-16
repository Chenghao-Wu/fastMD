"""Tests of fastmd.analysis, fastmd.trajectory and tools/make_kg_chains.py on
synthetic data with known answers (numpy only, no GPU)."""
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from fastmd import analysis as A
from fastmd.trajectory import Trajectory, write_trajectory

SRC = os.environ.get("FASTMD_SOURCE_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def wrap(x, L):
    img = np.floor(x / L).astype(np.int64)
    return x - img * L, img


class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fastmd_analysis_")

    def traj(self, name, steps, X, L, mol=None):
        """Write unwrapped positions X[F, N, 3] as a trajectory and open it."""
        path = os.path.join(self.tmp, name)
        pos, img = zip(*(wrap(x, L) for x in X))
        write_trajectory(path, steps, [L, L, L, 0, 0, 0], np.array(pos), np.array(img),
                         mol_ids=mol)
        return Trajectory(path)

    def test_trajectory_unwrap_round_trip(self):
        rng = np.random.default_rng(1)
        X = rng.normal(scale=20.0, size=(3, 50, 3))
        t = self.traj("rt.fmd", [0, 5, 10], X, 7.0)
        for k in range(3):
            np.testing.assert_allclose(t.unwrapped(k), X[k], atol=2e-5 * 60)

    def test_msd_ballistic_and_labels(self):
        steps = np.arange(0, 21)
        v = np.array([[0.3, 0.0, 0.0]] * 10 + [[0.0, 0.1, 0.2]] * 10)
        X = np.array([1.0 + v * s for s in steps])
        t = self.traj("ball.fmd", steps, X, 5.0)
        lags, m, cnt, g = A.msd(t, origin_every=1, labels=lambda k: (np.arange(20) >= 10).astype(int))
        np.testing.assert_allclose(m[0], 0.09 * lags ** 2, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(m[1], 0.05 * lags ** 2, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(cnt[0], 10 * (21 - lags))

    def test_log_periodic_origins(self):
        steps = np.array([0, 1, 2, 4, 8, 10, 11, 12, 14, 18, 20])
        X = np.array([np.full((4, 3), 0.1 * s) for s in steps])
        t = self.traj("log.fmd", steps, X, 50.0)
        lags, m, cnt, _ = A.msd(t, origin_every=10)
        self.assertEqual(list(lags), [0, 1, 2, 4, 8, 10, 11, 12, 14, 18, 20])
        np.testing.assert_allclose(m[0], 3 * (0.1 * lags) ** 2, rtol=1e-4, atol=1e-7)
        self.assertEqual(cnt[0][list(lags).index(10)], 8)   # origins 0 and 10

    def test_end_to_end_and_rouse_rigid_rotation(self):
        # rigid straight chains rotating about z at angular speed w
        nch, N, w = 6, 12, 0.05
        steps = np.arange(0, 40, 2)
        base = np.stack([np.arange(N) * 0.97 - 5.0, np.zeros(N), np.zeros(N)], axis=1)
        X = []
        for s in steps:
            c, sn = math.cos(w * s), math.sin(w * s)
            R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
            X.append(np.concatenate([base @ R.T + [10.0 * k, 3.0, 3.0] for k in range(nch)]))
        mol = np.repeat(np.arange(1, nch + 1), N)
        t = self.traj("rot.fmd", steps, np.array(X), 30.0, mol)
        lags, P, _, _ = A.end_to_end_acf(t, origin_every=2)
        np.testing.assert_allclose(P[0], np.cos(w * lags), atol=1e-5)
        lags, modes, _, _ = A.rouse_mode_acf(t, modes=(1, 3), origin_every=2)
        np.testing.assert_allclose(modes[1][0], np.cos(w * lags), atol=1e-4)

    def test_self_intermediate_scattering(self):
        steps = np.arange(0, 11)
        X = np.array([np.tile([[2.0, 3.0, 4.0]], (8, 1)) + [0.2 * s, 0, 0] for s in steps])
        t = self.traj("fs.fmd", steps, X, 20.0)
        lags, fx, _, _ = A.self_intermediate_scattering(t, 7.0, ("x",), origin_every=1)
        lags, fz, _, _ = A.self_intermediate_scattering(t, 7.0, ("z",), origin_every=1)
        np.testing.assert_allclose(fx[0], np.cos(7.0 * 0.2 * lags), atol=1e-4)
        np.testing.assert_allclose(fz[0], 1.0)

    def test_internal_distances_straight_chain(self):
        N = 20
        x = np.stack([np.arange(N) * 0.97, np.zeros(N), np.zeros(N)], axis=1)
        n, r = A.internal_distances(x, np.ones(N, int))
        np.testing.assert_allclose(r, n * 0.97 ** 2)

    def test_storage_loss_single_maxwell_mode(self):
        tau, G0 = 3.0, 2.0
        t = np.concatenate([[0.0], np.logspace(-3, 2.3, 600)])
        w = np.logspace(-1, 1, 12)   # sampled finely enough for omega <= 10 here
        Gp, Gpp = A.storage_loss_moduli(t, G0 * np.exp(-t / tau), w)
        wt = w * tau
        np.testing.assert_allclose(Gp, G0 * wt ** 2 / (1 + wt ** 2), rtol=1e-3)
        np.testing.assert_allclose(Gpp, G0 * wt / (1 + wt ** 2), rtol=1e-3)

    def test_stress_relaxation_normalisation(self):
        path = os.path.join(self.tmp, "stress.dat")
        lags = np.array([0, 1, 2, 4])
        C = np.array([2.0, 1.5, 1.0, 0.5]) + 0.25   # means 0.5 for every channel
        with open(path, "w") as f:
            f.write("# multiple-tau correlator: p 16  m 2\n")
            f.write("# stress = (kinetic + virial) / V with V = 1000 (box volume); x\n")
            f.write("# channel means: 0.5 0.5 0.5 0.5 0.5 0.5\n")
            f.write("# lag_steps  C_xx  C_xy  C_xz  C_yy  C_yz  C_zz  count\n")
            for l, c in zip(lags, C):
                f.write(f"{l} {c} {c} {c} {c} {c} {c} 10\n")
        s, G = A.stress_relaxation(path, temperature=2.0)
        np.testing.assert_array_equal(s, lags)
        np.testing.assert_allclose(G, 1000 / 2.0 * (C - 0.25))
        _, Gf = A.stress_relaxation(path, temperature=2.0, volume=400.0)   # film: V_poly
        np.testing.assert_allclose(Gf, 1000 ** 2 / (2.0 * 400.0) * (C - 0.25))


class ChainGeneratorTest(unittest.TestCase):
    def generate(self, *extra):
        out = os.path.join(tempfile.mkdtemp(), "c.data")
        subprocess.run([sys.executable, os.path.join(SRC, "tools", "make_kg_chains.py"),
                        "--nchains", "400", "--chain-length", "60", "-o", out, *extra],
                       check=True, capture_output=True)
        sys.path.insert(0, os.path.join(SRC, "tools"))
        from equilibrate_melt import read_data
        return read_data(out)

    def test_bonds_and_freely_rotating_cinf(self):
        x, mol, L = self.generate("--cinf", "2.5")
        idx = A.chain_index(mol)
        c = x[idx]
        b = np.linalg.norm(np.diff(c, axis=1), axis=2)
        np.testing.assert_allclose(b, 0.97, atol=1e-5)
        n, r = A.internal_distances(x, mol)
        # freely rotating chain: <R^2(n)>/n -> C_inf b^2 (finite-n correction ~ 1/n)
        self.assertAlmostEqual(r[-1] / 0.97 ** 2, 2.5, delta=0.15)

    def test_bending_boltzmann_angles(self):
        kappa = 1.5
        x, mol, _ = self.generate("--kbend", "1.5", "--min13", "0")
        c = x[A.chain_index(mol)]
        u = np.diff(c, axis=1)
        u /= np.linalg.norm(u, axis=2, keepdims=True)
        cos_phi = (u[:, 1:] * u[:, :-1]).sum(axis=2).mean()
        self.assertAlmostEqual(cos_phi, 1 / math.tanh(kappa) - 1 / kappa, delta=0.01)


if __name__ == "__main__":
    unittest.main()
