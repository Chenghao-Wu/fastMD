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
