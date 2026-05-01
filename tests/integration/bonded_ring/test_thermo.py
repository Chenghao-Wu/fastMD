"""Test thermodynamic averages between fastMD and LAMMPS over 5000 steps."""
import sys
import os
import numpy as np
import utils

utils.set_work_dir()

THERMO_REL_TOL = 0.02  # 2%
EQUIL_FRAC = 0.1  # discard first 10% of steps
NSTEPS = 5000


def main():
    if not os.path.exists("thermo_fastmd.dat") or not os.path.exists("thermo_lammps.dat"):
        print("Thermo files not found. Run run_tests.sh to generate them first.")
        sys.exit(1)

    fastmd_thermo = utils.parse_thermo("thermo_fastmd.dat")
    lammps_thermo = utils.parse_thermo("thermo_lammps.dat")

    # Normalize LAMMPS KE and PE to per-atom
    natoms = 30000
    lammps_thermo["KE"] = lammps_thermo["KE"] / natoms
    lammps_thermo["PE"] = lammps_thermo["PE"] / natoms

    # Discard equilibration
    n_total = len(fastmd_thermo["step"])
    equil = int(n_total * EQUIL_FRAC)
    print(f"Discarding first {equil} of {n_total} steps for equilibration")

    quantities = ["T", "KE", "PE", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    all_passed = True

    for q in quantities:
        fm = fastmd_thermo[q][equil:]
        lm = lammps_thermo[q][equil:]

        fm_mean = float(np.mean(fm))
        lm_mean = float(np.mean(lm))

        rel = utils.relative_diff(
            np.array([fm_mean]), np.array([lm_mean])
        )[0]

        status = "PASS" if rel < THERMO_REL_TOL else "FAIL"
        if rel >= THERMO_REL_TOL:
            all_passed = False

        print(
            f"  {q:4s}  fastMD={fm_mean:10.4f}  LAMMPS={lm_mean:10.4f}  "
            f"rel_diff={rel*100:.4f}%  [{status}]"
        )

    if not all_passed:
        print(f"FAIL: some quantities exceed {THERMO_REL_TOL*100}% relative tolerance")
        sys.exit(1)

    print("PASS: thermo averages test")


if __name__ == "__main__":
    main()
