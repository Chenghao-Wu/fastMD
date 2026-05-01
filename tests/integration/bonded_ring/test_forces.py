"""Test force/PE agreement between fastMD and LAMMPS on first 10 steps."""
import sys
import os
import numpy as np
import utils

utils.set_work_dir()

PE_REL_TOL = 0.01  # 1%
PXX_REL_TOL = 0.01  # 1%

NSTEPS = 10


def main():
    # Generate configs
    fastmd_config = utils.write_fastmd_config(NSTEPS)
    lammps_input = utils.write_lammps_input(NSTEPS)

    # Run fastMD
    print(f"Running fastMD for {NSTEPS} steps...")
    fastmd_result = utils.run_fastmd(fastmd_config)
    if fastmd_result.returncode != 0:
        print("FAIL: fastMD crashed")
        print(fastmd_result.stderr)
        sys.exit(1)

    # Run LAMMPS
    print(f"Running LAMMPS for {NSTEPS} steps...")
    lammps_result = utils.run_lammps(lammps_input)
    if lammps_result.returncode != 0:
        print("FAIL: LAMMPS crashed")
        print(lammps_result.stderr)
        sys.exit(1)

    # Extract thermo from LAMMPS log
    extract_lammps_thermo(lammps_result.stdout, "thermo_lammps_10.dat")

    # Parse both outputs
    fastmd_thermo = utils.parse_thermo("thermo_fastmd.dat")
    lammps_thermo = utils.parse_thermo("thermo_lammps_10.dat")

    # Normalize LAMMPS KE and PE to per-atom
    natoms = 30000
    lammps_thermo["KE"] = lammps_thermo["KE"] / natoms
    lammps_thermo["PE"] = lammps_thermo["PE"] / natoms

    # Compare PE at each step
    pe_rel_diff = utils.relative_diff(fastmd_thermo["PE"], lammps_thermo["PE"])
    max_pe_diff = float(np.max(pe_rel_diff))
    print(f"Max PE relative difference: {max_pe_diff*100:.4f}%")

    if max_pe_diff > PE_REL_TOL:
        print(f"FAIL: PE relative difference {max_pe_diff*100:.4f}% exceeds {PE_REL_TOL*100}%")
        print("PE values (fastMD vs LAMMPS):")
        for i in range(min(NSTEPS, len(fastmd_thermo["PE"]))):
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


def extract_lammps_thermo(stdout: str, output_path: str):
    """Extract thermo table from LAMMPS stdout into a file."""
    lines = stdout.splitlines()
    in_thermo = False
    thermo_lines = []

    for line in lines:
        if line.strip().startswith("Step ") and "ke" in line.lower():
            thermo_lines.append("# step  KE  PE  T  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz")
            in_thermo = True
            continue
        if in_thermo:
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                thermo_lines.append(stripped)
            elif thermo_lines and line.strip() == "":
                pass
            elif thermo_lines and not line.strip().startswith(("Loop", "WARNING", "ERROR")):
                pass

    # Fallback: parse the log file written by LAMMPS
    if len(thermo_lines) <= 1:
        log_path = os.path.join(os.path.dirname(__file__), "log.lammps")
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Step ") and "ke" in line.lower():
                        thermo_lines = [
                            "# step  KE  PE  T  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"
                        ]
                        continue
                    if thermo_lines and line and line.split()[0].isdigit():
                        parts = line.split()
                        if len(parts) >= 10:
                            thermo_lines.append("  ".join(parts[:10]))

    with open(os.path.join(os.path.dirname(__file__), output_path), "w") as f:
        f.write("\n".join(thermo_lines) + "\n")


if __name__ == "__main__":
    main()
