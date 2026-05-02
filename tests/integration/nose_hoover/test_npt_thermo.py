"""Test NH NPT temperature and pressure against LAMMPS."""
import os, sys
import numpy as np
import utils

def test_npt():
    utils.set_work_dir()

    nsteps = 500
    T_start, T_stop, Tdamp = 1.0, 1.0, 1.0
    P_start, P_stop, Pdamp = 0.0, 0.0, 1.0

    print(f"Running fastMD NPT ({nsteps} steps)...")
    fastmd_config = utils.write_fastmd_config(
        nsteps, T_start, T_stop, Tdamp, P_start, P_stop, Pdamp)
    result = utils.run_fastmd(fastmd_config)
    if result.returncode != 0:
        print("ERROR: fastMD failed")
        print(result.stderr)
        sys.exit(1)

    print(f"Running LAMMPS NPT ({nsteps} steps)...")
    lammps_input = utils.write_lammps_input(
        nsteps, T_start, T_stop, Tdamp, P_start, P_stop, Pdamp)
    result = utils.run_lammps(lammps_input)
    if result.returncode != 0:
        print("ERROR: LAMMPS failed")
        print(result.stderr)
        sys.exit(1)

    print("Extracting LAMMPS thermo...")
    n = utils.extract_lammps_thermo("log.lammps", "thermo_lammps.dat")
    print(f"  Extracted {n} lines")

    fastmd_thermo = utils.parse_thermo("thermo_fastmd.dat")
    lammps_thermo = utils.parse_thermo("thermo_lammps.dat")

    # Compare temperature in the second half of the run
    half = len(fastmd_thermo["T"]) // 2
    T_fastmd = np.mean(fastmd_thermo["T"][half:])
    T_lammps = np.mean(lammps_thermo["T"][half:])
    print(f"  Mean T: fastMD={T_fastmd:.4f}  LAMMPS={T_lammps:.4f}")

    if abs(T_fastmd - T_lammps) > 0.02:
        print(f"FAIL: Temperature mismatch: {abs(T_fastmd - T_lammps):.4f} > 0.02")
        sys.exit(1)

    # Compare KE
    KE_fastmd = np.mean(fastmd_thermo["KE"][half:])
    KE_lammps = np.mean(lammps_thermo["KE"][half:])
    print(f"  Mean KE: fastMD={KE_fastmd:.2f}  LAMMPS={KE_lammps:.2f}")

    # Compare pressure (diagonal average)
    P_fastmd = np.mean((fastmd_thermo["Pxx"][half:]
                       + fastmd_thermo["Pyy"][half:]
                       + fastmd_thermo["Pzz"][half:]) / 3.0)
    P_lammps = np.mean((lammps_thermo["Pxx"][half:]
                       + lammps_thermo["Pyy"][half:]
                       + lammps_thermo["Pzz"][half:]) / 3.0)
    print(f"  Mean P: fastMD={P_fastmd:.4f}  LAMMPS={P_lammps:.4f}")

    if abs(P_fastmd - P_lammps) > 1.0:
        print(f"FAIL: Pressure mismatch: {abs(P_fastmd - P_lammps):.4f} > 1.0")
        sys.exit(1)

    print("PASS: NH NPT thermo matches LAMMPS")

if __name__ == "__main__":
    test_npt()
