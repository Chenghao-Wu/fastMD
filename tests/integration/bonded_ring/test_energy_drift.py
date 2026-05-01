"""Test energy drift comparison between fastMD and LAMMPS."""
import sys
import os
import numpy as np
import utils

utils.set_work_dir()

MAX_ABS_DRIFT = 1e-5
DRIFT_RATIO_MIN = 0.5
DRIFT_RATIO_MAX = 2.0
DT = 0.001
NSTEPS = 5000


def main():
    if not os.path.exists("thermo_fastmd.dat") or not os.path.exists("thermo_lammps.dat"):
        print("Thermo files not found. Run run_tests.sh to generate them first.")
        sys.exit(1)

    fastmd_thermo = utils.parse_thermo("thermo_fastmd.dat")
    lammps_thermo = utils.parse_thermo("thermo_lammps.dat")

    # Both fastMD and LAMMPS output per-atom KE and PE — no normalization needed

    fm_total_e = utils.total_energy_per_atom(fastmd_thermo)
    lm_total_e = utils.total_energy_per_atom(lammps_thermo)

    fm_drift = utils.energy_drift(fm_total_e, DT)
    lm_drift = utils.energy_drift(lm_total_e, DT)

    print(f"fastMD  energy drift: {fm_drift:.6e} (energy/time)")
    print(f"LAMMPS  energy drift: {lm_drift:.6e} (energy/time)")

    # Check absolute drift
    fm_ok = abs(fm_drift) < MAX_ABS_DRIFT
    lm_ok = abs(lm_drift) < MAX_ABS_DRIFT

    if not fm_ok:
        print(f"FAIL: fastMD drift |{fm_drift:.6e}| exceeds {MAX_ABS_DRIFT}")
    if not lm_ok:
        print(f"FAIL: LAMMPS drift |{lm_drift:.6e}| exceeds {MAX_ABS_DRIFT}")

    if not fm_ok or not lm_ok:
        sys.exit(1)

    # Check drift ratio (both should be similarly small)
    denom = max(abs(lm_drift), 1e-15)
    drift_ratio = abs(fm_drift / denom)
    print(f"Drift ratio (fastMD/LAMMPS): {drift_ratio:.4f}")

    if drift_ratio < DRIFT_RATIO_MIN or drift_ratio > DRIFT_RATIO_MAX:
        print(
            f"FAIL: drift ratio {drift_ratio:.4f} outside "
            f"[{DRIFT_RATIO_MIN}, {DRIFT_RATIO_MAX}]"
        )
        sys.exit(1)

    print("PASS: energy drift test")


if __name__ == "__main__":
    main()
