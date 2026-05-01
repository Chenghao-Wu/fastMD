"""Test force/PE agreement between fastMD and LAMMPS on first 10 steps."""
import sys
import os
import numpy as np
import utils

utils.set_work_dir()

PE_REL_TOL = 0.01  # 1%
PXX_REL_TOL = 0.01  # 1%


def main():
    if not os.path.exists("thermo_fastmd_10.dat") or not os.path.exists("thermo_lammps_10.dat"):
        print("Thermo files not found. Run run_tests.sh to generate them first.")
        sys.exit(1)

    fastmd_thermo = utils.parse_thermo("thermo_fastmd_10.dat")
    lammps_thermo = utils.parse_thermo("thermo_lammps_10.dat")

    # Both fastMD and LAMMPS output per-atom KE and PE — no normalization needed

    # Compare PE at each step
    pe_rel_diff = utils.relative_diff(fastmd_thermo["PE"], lammps_thermo["PE"])
    max_pe_diff = float(np.max(pe_rel_diff))
    print(f"Max PE relative difference: {max_pe_diff*100:.4f}%")

    if max_pe_diff > PE_REL_TOL:
        print(f"FAIL: PE relative difference {max_pe_diff*100:.4f}% exceeds {PE_REL_TOL*100}%")
        print("PE values (fastMD vs LAMMPS):")
        n_steps = min(len(fastmd_thermo["PE"]), len(lammps_thermo["PE"]))
        for i in range(n_steps):
            print(f"  step {i+1}: {fastmd_thermo['PE'][i]:.4f} vs {lammps_thermo['PE'][i]:.4f}")
        sys.exit(1)

    # Compare Pxx at each step
    pxx_rel_diff = utils.relative_diff(fastmd_thermo["Pxx"], lammps_thermo["Pxx"])
    max_pxx_diff = float(np.max(pxx_rel_diff))
    print(f"Max Pxx relative difference: {max_pxx_diff*100:.4f}%")

    if max_pxx_diff > PXX_REL_TOL:
        print(f"FAIL: Pxx relative difference {max_pxx_diff*100:.4f}% exceeds {PXX_REL_TOL*100}%")
        sys.exit(1)

    print("PASS: forces test")


if __name__ == "__main__":
    main()
