"""Tests of the stateful Python interface (fastmd.Simulation).

Needs a GPU and the fastmd._core extension on the path (ctest sets
PYTHONPATH to the build tree; FASTMD_SOURCE_DIR locates tools/).
Run directly with:  python -m unittest -v test_simulation_api
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import fastmd
from fastmd import analysis as A

SRC = os.environ.get("FASTMD_SOURCE_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CONFIG = """natoms {natoms}
ntypes 1
rc 2.5
skin 0.3
dt 0.005
nsteps 1000
thermo 0 100 thermo.dat
{ensemble}
lj 0 0 1.0 1.0
bond_type 0 30.0 1.5 1.0 1.0
lammps_data_file {data}
"""


class SimulationAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fastmd_api_")
        cls.cwd = os.getcwd()
        os.chdir(cls.tmp)
        cls.data = os.path.join(cls.tmp, "kg.data")
        subprocess.run([sys.executable, os.path.join(SRC, "tools", "make_kg_melt.py"),
                        "--nchains", "40", "--chain-length", "25", "--no-angles",
                        "-o", cls.data], check=True, capture_output=True)
        cls.natoms = 1000

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def config(self, ensemble="nvt_langevin 1.0 1.0 1.0 7"):
        return CONFIG.format(natoms=self.natoms, ensemble=ensemble, data=self.data)

    def make(self, **kw):
        return fastmd.Simulation.from_text(self.config(**kw))

    def test_construction_and_shapes(self):
        path = os.path.join(self.tmp, "run.conf")
        with open(path, "w") as f:
            f.write(self.config())
        sim = fastmd.Simulation(path)
        self.assertEqual(sim.natoms, self.natoms)
        self.assertEqual(sim.ntypes, 1)
        self.assertEqual(sim.step, 0)
        self.assertAlmostEqual(sim.dt, 0.005)
        for name in ("positions", "velocities", "forces", "images"):
            self.assertEqual(getattr(sim, name).shape, (self.natoms, 3), name)
        self.assertEqual(sim.types.shape, (self.natoms,))
        # 1-D arrays are real copies (not broadcast views of one value)
        e = sim.per_atom_energy
        self.assertEqual(e.shape, (self.natoms,))
        self.assertGreater(np.ptp(e), 0.0)
        self.assertAlmostEqual(e.astype(np.float64).sum(), sim.thermo().potential_energy,
                               delta=1e-4 * abs(sim.thermo().potential_energy))
        x = sim.positions
        self.assertTrue(np.all(x >= 0.0) and np.all(x < sim.box_length))
        # same system from text and from file (threads per atom are autotuned
        # by timing, so the force summation order may differ)
        np.testing.assert_allclose(sim.forces, self.make().forces, rtol=1e-4, atol=1e-4)

    def test_set_positions_round_trip(self):
        sim = self.make()
        sim.run(50)
        x, im, f = sim.positions, sim.images, sim.forces
        sim.set_positions(x, im)
        np.testing.assert_array_equal(sim.positions, x)
        np.testing.assert_array_equal(sim.images, im)
        # same forces up to the summation order of the rebuilt neighbor list
        np.testing.assert_allclose(sim.forces, f, rtol=1e-4, atol=1e-4)
        # unwrapped coordinates are preserved when wrapping shifts images
        L = sim.box_length
        unwrapped = x + im * L
        sim.set_positions(unwrapped)
        np.testing.assert_allclose(sim.positions + sim.images * L, unwrapped, atol=1e-4)

    def test_forces_are_energy_gradient(self):
        # Central differences of the total energy (sum of per-atom energies).
        # LJ is truncated without an energy shift, so E jumps where a pair
        # crosses rc: displacements that move any pair across rc are skipped.
        sim = self.make()
        x0 = sim.positions
        im0 = sim.images
        f0 = sim.forces
        L = sim.box_length
        rc, h = 2.5, 1e-3

        def crosses_rc(i, d):
            r = []
            for s in (+1, -1):
                xi = x0[i].astype(np.float64)
                xi[d] += s * h
                dr = x0.astype(np.float64) - xi
                dr -= L * np.round(dr / L)
                r.append(np.sqrt((dr ** 2).sum(axis=1)))
            return np.any((r[0] - rc) * (r[1] - rc) <= 0.0)

        rng = np.random.default_rng(3)
        checked = 0
        for i in rng.permutation(self.natoms):
            if checked == 12:
                break
            for d in range(3):
                if checked == 12 or crosses_rc(i, d):
                    continue
                e, step = [], []
                for s in (+1, -1):
                    x = x0.copy()
                    x[i, d] = np.float32(x0[i, d] + s * h)
                    step.append(float(x[i, d]) - float(x0[i, d]))   # exact float32 step
                    sim.set_positions(x, im0)
                    e.append(sim.per_atom_energy.astype(np.float64).sum())
                fd = -(e[0] - e[1]) / (step[0] - step[1])
                self.assertAlmostEqual(fd, f0[i, d], delta=2e-2 * max(1.0, abs(f0[i, d])),
                                       msg=f"atom {i} dim {d}")
                checked += 1
        self.assertEqual(checked, 12)
        sim.set_positions(x0, im0)
        np.testing.assert_allclose(sim.forces, f0, rtol=1e-4, atol=1e-4)

    def test_run_is_resumable(self):
        a, b = self.make(), self.make()
        a.run(100)
        a.run(100)
        b.run(200)
        self.assertEqual(a.step, 200)
        L = a.box_length
        np.testing.assert_allclose(a.positions + a.images * L, b.positions + b.images * L,
                                   atol=1e-3)

    def test_velocities_and_thermo(self):
        sim = self.make(ensemble="nvt_nh 1.0 1.0 0.5")
        sim.run(20)
        v = sim.velocities.astype(np.float64)
        t = sim.thermo()
        # totals (not per atom); unit masses
        self.assertAlmostEqual(t.kinetic_energy, 0.5 * (v ** 2).sum(),
                               delta=1e-4 * t.kinetic_energy)
        self.assertAlmostEqual(t.temperature, 2.0 * t.kinetic_energy / (3 * self.natoms),
                               delta=1e-4 * t.temperature)
        self.assertAlmostEqual(t.potential_energy, sim.per_atom_energy.astype(np.float64).sum(),
                               delta=1e-4 * abs(t.potential_energy))
        P = t.pressure
        self.assertEqual(P.shape, (3, 3))
        np.testing.assert_allclose(P, P.T)
        sim.velocities = np.zeros((self.natoms, 3), np.float32)
        self.assertEqual(sim.thermo().kinetic_energy, 0.0)
        sim.run(10)   # thermostat restarts from rest without blowing up
        self.assertTrue(np.all(np.isfinite(sim.positions)))

    def test_trajectory_round_trip(self):
        from fastmd.trajectory import Trajectory
        path = os.path.join(self.tmp, "api.fmd")
        sim = fastmd.Simulation.from_text(self.config() + f"traj {path} linear 25\n")
        sim.run(100)
        x, im, L = sim.positions, sim.images, sim.box_length
        del sim
        traj = Trajectory(path)
        self.assertEqual(list(traj.steps), [0, 25, 50, 75, 100])
        self.assertEqual(traj.natoms, self.natoms)
        np.testing.assert_array_equal(traj.positions(4), x)
        np.testing.assert_array_equal(traj.images(4), im)
        self.assertEqual(traj.box(4)[0], np.float64(np.float32(L)))
        self.assertEqual(len(set(traj.mol_ids)), 40)
        # unwrapped coordinates are continuous between frames
        for k in range(1, len(traj)):
            self.assertLess(np.abs(traj.unwrapped(k) - traj.unwrapped(k - 1)).max(), 2.0)

    def test_pushoff_pipeline(self):
        """Random-walk chains with overlaps -> soft push-off -> capped LJ -> LJ."""
        work = os.path.join(self.tmp, "pushoff")
        raw = os.path.join(self.tmp, "raw.data")
        subprocess.run([sys.executable, os.path.join(SRC, "tools", "make_kg_chains.py"),
                        "--nchains", "40", "--chain-length", "25", "--kbend", "1.5", "-o", raw],
                       check=True, capture_output=True)
        fmd = os.environ.get("FASTMD_BINARY")
        if not fmd:
            self.skipTest("FASTMD_BINARY not set")
        r = subprocess.run([sys.executable, os.path.join(SRC, "tools", "equilibrate_melt.py"), raw,
                            "--fastmd", fmd, "--kbend", "1.5", "--soft-steps", "2000",
                            "--cap-steps", "500", "--equil-steps", "2000", "--workdir", work],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        sim = fastmd.Simulation.from_text(
            "ntypes 1\nrc 1.122462\nskin 0.3\ndt 0.005\nnsteps 10\nnve\nlj 0 0 1 1\n"
            "bond_type 0 30 1.5 1 1\nangle_cosine 0 1.5\n"
            f"lammps_data_file {os.path.join(work, 'equil_final.data')}\n")
        x = sim.positions
        L = sim.box_length
        d = x[:, None, :] - x[None, :, :]
        d -= L * np.round(d / L)
        r2 = (d ** 2).sum(axis=2) + np.eye(len(x)) * 100
        self.assertGreater(np.sqrt(r2.min()), 0.7)   # overlaps removed
        self.assertTrue(np.isfinite(sim.forces).all())

    def test_film_pipeline(self):
        """Generated chains -> push-off -> bulk -> unwrap, compress, anneal, release."""
        fmd = os.environ.get("FASTMD_BINARY")
        if not fmd:
            self.skipTest("FASTMD_BINARY not set")
        work = os.path.join(self.tmp, "filmpipe")
        os.makedirs(work, exist_ok=True)
        tools = os.path.join(SRC, "tools")
        raw = os.path.join(work, "raw.data")

        def run(*cmd):
            r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return r.stdout
        run(os.path.join(tools, "make_kg_chains.py"), "--nchains", "80", "--chain-length", "25",
            "--kbend", "1.5", "--aspect", "2.0", "-o", raw)
        run(os.path.join(tools, "equilibrate_melt.py"), raw, "--fastmd", fmd, "--kbend", "1.5",
            "--rc", "2.5", "--soft-steps", "2000", "--cap-steps", "500", "--equil-steps", "4000",
            "--workdir", os.path.join(work, "bulk"))
        out = run(os.path.join(tools, "build_film.py"), os.path.join(work, "bulk", "equil_final.data"),
                  "--fastmd", fmd, "--kbend", "1.5", "--compress-steps", "4000", "--anneal-steps", "1000",
                  "--release-steps", "3000", "--workdir", os.path.join(work, "film"))
        sys.path.insert(0, tools)
        from equilibrate_melt import read_data
        x, mol, L = read_data(os.path.join(work, "film", "release_final.data"))
        xb, _, Lb = read_data(os.path.join(work, "bulk", "equil_final.data"))
        idx = A.chain_index(mol)
        bonds = np.linalg.norm(np.diff(x[idx], axis=1), axis=2)
        self.assertLess(bonds.max(), 1.5)                      # no chain was cut
        z, rho = A.density_profile(x, axis=2, area=L[0] * L[1], bin_width=0.5)
        t, lo, hi, plateau = A.film_thickness(z, rho)
        self.assertAlmostEqual(t, Lb[2], delta=0.25 * Lb[2])   # about the bulk-density thickness
        self.assertGreater(L[2] - (x[:, 2].max() - x[:, 2].min()), 10.0)   # vacuum remains
        self.assertIn("thickness", out)

    def test_step_shear(self):
        sim = self.make()
        sim.run(100)
        before = sim.unwrapped_positions
        Ly = sim.box[1]
        sim.apply_step_shear(0.1)
        self.assertAlmostEqual(sim.box[3], 0.1 * Ly, places=4)
        after = sim.unwrapped_positions
        np.testing.assert_allclose(after[:, 0], before[:, 0] + 0.1 * before[:, 1], atol=2e-4)
        np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=2e-4)
        with self.assertRaises(ValueError):
            sim.box_length          # not cubic any more
        sim.set_tilt(0.0)           # non-affine: unwrapped positions unchanged
        np.testing.assert_allclose(sim.unwrapped_positions, after, atol=2e-4)

    def test_friction_profile_api(self):
        sim = fastmd.Simulation.from_text(self.config() + "film z\n")
        z = np.linspace(-20, 20, 9)
        sim.set_friction_profile(z, np.where(z < 0, 10.0, 1.0))
        sim.run(50)
        sim.set_friction_profile_analytic(1.5, 2.0, 10.0)
        sim.run(50)
        sim.clear_friction_profile()
        sim.run(50)
        self.assertTrue(np.isfinite(sim.positions).all())
        with self.assertRaises(ValueError):
            sim.set_friction_profile(z[::-1], np.ones_like(z))       # z must increase
        with self.assertRaises(ValueError):
            sim.set_friction_profile(z, -np.ones_like(z))            # factors >= 0
        nofilm = self.make()
        with self.assertRaises(ValueError):
            nofilm.set_friction_profile(z, np.ones_like(z))          # needs `film z`

    def test_layer_stress(self):
        sim = fastmd.Simulation.from_text(self.config() + "film z\nlayer_stress 5 3.0 10 /dev/null\n")
        sim.run(100)
        ls = sim.layer_stress()
        self.assertEqual(ls.shape, (5, 6))
        self.assertEqual(sim.nlayers, 5)
        V = np.prod(sim.box[:3])
        P = sim.thermo().pressure
        sums = ls.sum(axis=0)
        np.testing.assert_allclose(sums[[0, 3, 5, 1, 2, 4]],
                                   V * P[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                                   rtol=1e-4, atol=1e-2)

    def test_minimize(self):
        sim = self.make()
        v0 = sim.velocities
        e0 = sim.thermo().potential_energy
        r = sim.minimize(ftol=1e-2, max_iter=50000)
        self.assertTrue(r["converged"])
        self.assertLess(r["fmax"], 1e-2)
        self.assertLess(r["potential_energy"], e0)
        np.testing.assert_array_equal(sim.velocities, v0)   # restored

    def test_dyn_corr_matches_trajectory_analysis(self):
        """On-the-fly GPU correlations equal fastmd.analysis on a trajectory of the same run."""
        from fastmd.trajectory import Trajectory
        tp = os.path.join(self.tmp, "dyn.fmd")
        dp = os.path.join(self.tmp, "dyn.txt")
        sim = fastmd.Simulation.from_text(
            self.config() + f"traj {tp} linear 1\ndyn_corr {dp} 20 5 q 6.5 modes 1 3\n")
        sim.run(100)
        d = sim.dyn_corr()
        del sim
        traj = Trajectory(tp)
        lags = d["lags"]
        steps = traj.steps
        origins = [k for k in range(len(steps)) if steps[k] % 20 == 0]
        X = np.array([traj.unwrapped(k) for k in range(len(steps))])          # (F, N, 3)
        idx = A.chain_index(traj.mol_ids)
        C = X[:, idx]                                                          # (F, nc, Nc, 3)
        N = idx.shape[1]
        n = np.arange(1, N + 1)
        basis = {p: np.cos(p * np.pi * (n - 0.5) / N) / N for p in (1, 3)}
        checked = 0
        for j, lag in enumerate(lags):
            pairs = [(o, o + lag) for o in origins if o + lag < len(steps)]
            if not pairs:
                continue
            dr = np.concatenate([X[b] - X[a] for a, b in pairs])
            np.testing.assert_allclose(d["msd_par"][0, j], (dr[:, :2] ** 2).sum(1).mean(), rtol=2e-4, atol=1e-7)
            np.testing.assert_allclose(d["msd_perp"][0, j], (dr[:, 2] ** 2).mean(), rtol=2e-4, atol=1e-7)
            np.testing.assert_allclose(d["fs_par"][0, j], np.cos(6.5 * dr[:, :2]).mean(), atol=2e-5)
            np.testing.assert_allclose(d["fs_perp"][0, j], np.cos(6.5 * dr[:, 2]).mean(), atol=2e-5)
            self.assertEqual(d["n_bead"][0, j], len(dr))
            com = np.concatenate([C[b].mean(1) - C[a].mean(1) for a, b in pairs])
            np.testing.assert_allclose(d["com_par"][0, j] + d["com_perp"][0, j], (com ** 2).sum(1).mean(),
                                       rtol=2e-4, atol=1e-7)
            R = lambda f: C[f][:, -1] - C[f][:, 0]
            num = sum((R(b) * R(a)).sum() for a, b in pairs)
            den = sum((R(a) ** 2).sum() for a, b in pairs)
            np.testing.assert_allclose(d["P_t"][0, j], num / den, rtol=1e-4)
            for p in (1, 3):
                Xp = lambda f: np.einsum("n,cnd->cd", basis[p], C[f])
                num = sum((Xp(b) * Xp(a)).sum() for a, b in pairs)
                den = sum((Xp(a) ** 2).sum() for a, b in pairs)
                np.testing.assert_allclose(d[f"rouse_{p}"][0, j], num / den, rtol=1e-3, atol=1e-5)
            checked += 1
        self.assertGreater(checked, 15)

    def test_dyn_corr_film_bins_residence(self):
        dp = os.path.join(self.tmp, "dynfilm.txt")
        sim = fastmd.Simulation.from_text(self.config() + f"film z\ndyn_corr {dp} 50 4 bins 4 3.0\n")
        sim.run(400)
        d = sim.dyn_corr()
        nb, nr = d["n_bead"], d["n_resident"]
        self.assertEqual(nb.shape[0], 4)
        np.testing.assert_array_equal(nb[:, 0], nr[:, 0])          # lag 0: everyone resident
        self.assertTrue((nr <= nb + 1e-9).all())
        frac = nr.sum(0) / nb.sum(0)
        self.assertTrue(np.all(np.diff(frac[np.isfinite(frac)]) <= 1e-12))   # survival decreases
        self.assertLess(frac[-1], 1.0)

    def test_errors(self):
        sim = self.make()
        with self.assertRaises(ValueError):
            sim.positions = np.zeros((self.natoms - 1, 3), np.float32)
        with self.assertRaises(ValueError):
            sim.set_positions(np.full((self.natoms, 3), np.nan, np.float32))
        with self.assertRaises(RuntimeError) as ctx:
            fastmd.Simulation.from_text(self.config().replace("rc 2.5", "rc -1"))
        self.assertIn("rc", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
