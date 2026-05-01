"""Shared utilities for bonded ring integration tests."""
import subprocess
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTMD_BIN = "/home/zhenghaowu/fastMD/build/fastMD"
LAMMPS_BIN = "/home/zhenghaowu/lammps/build/lmp"


def run_fastmd(config_path: str) -> subprocess.CompletedProcess:
    """Run fastMD with the given config, return completed process."""
    result = subprocess.run(
        [FASTMD_BIN, config_path],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if result.returncode != 0:
        print(f"fastMD stderr:\n{result.stderr}")
    return result


def run_lammps(input_path: str) -> subprocess.CompletedProcess:
    """Run LAMMPS with the given input script."""
    result = subprocess.run(
        [LAMMPS_BIN, "-in", input_path],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if result.returncode != 0:
        print(f"LAMMPS stderr:\n{result.stderr}")
    return result


def parse_thermo(filepath: str) -> dict:
    """Parse thermo data file into dict of numpy arrays.

    Expected header format (fastMD and LAMMPS both produce this):
        # step  KE  PE  T  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz

    Returns dict with keys: step, KE, PE, T, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz
    """
    path = os.path.join(SCRIPT_DIR, filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thermo file not found: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    columns = ["step", "KE", "PE", "T", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    if data.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns, got {data.shape[1]} in {filepath}"
        )

    return {col: data[:, i] for i, col in enumerate(columns)}


def total_energy_per_atom(thermo: dict) -> np.ndarray:
    """Return total energy per atom (KE + PE) at each step."""
    return thermo["KE"] + thermo["PE"]


def energy_drift(total_e: np.ndarray, dt: float) -> float:
    """Linear regression slope of total energy vs time.

    Args:
        total_e: total energy per atom at each step
        dt: timestep

    Returns:
        slope: drift rate (energy/time) in LJ units
    """
    time = np.arange(len(total_e)) * dt
    slope, _ = np.polyfit(time, total_e, 1)
    return float(slope)


def write_fastmd_config(nsteps: int) -> str:
    """Fill the template with nsteps, return path to written config."""
    template_path = os.path.join(SCRIPT_DIR, "fastmd_template.conf")
    with open(template_path) as f:
        content = f.read()

    content = content.replace("__NSTEPS__", str(nsteps))

    config_path = os.path.join(SCRIPT_DIR, f"fastmd_{nsteps}steps.conf")
    with open(config_path, "w") as f:
        f.write(content)

    return config_path


def write_lammps_input(nsteps: int) -> str:
    """Generate LAMMPS input with parameters matching fastMD config.

    Returns path to the written LAMMPS input file.
    """

    lammps_input = f"""units lj
atom_style molecular
newton off

pair_style lj/cut 2.5
bond_style fene

read_data data

pair_coeff 1 1 1.0 1.0
bond_coeff 1 30.0 1.5 1.0 1.0

special_bonds lj 0.0 1.0 1.0

neighbor 2.8 bin
neigh_modify delay 0 every 1 check yes

timestep 0.001

fix 1 all nve
fix 2 all langevin 1.0 1.0 1.0 42

thermo 1
thermo_style custom step ke pe temp pxx pyy pzz pxy pxz pyz
thermo_modify format float %.4f

run {nsteps}
"""

    input_path = os.path.join(SCRIPT_DIR, f"lammps_{nsteps}steps.in")
    with open(input_path, "w") as f:
        f.write(lammps_input)

    return input_path


def write_lammps_rerun_input(nsteps: int, log_path: str) -> str:
    """Generate LAMMPS input file that redirects output to a log file.

    Returns path to the written input file.
    """
    lammps_input = f"""units lj
atom_style molecular
newton off

pair_style lj/cut 2.5
bond_style fene

read_data data

pair_coeff 1 1 1.0 1.0
bond_coeff 1 30.0 1.5 1.0 1.0

special_bonds lj 0.0 1.0 1.0

neighbor 2.8 bin
neigh_modify delay 0 every 1 check yes

timestep 0.001

fix 1 all nve
fix 2 all langevin 1.0 1.0 1.0 42

thermo 1
thermo_style custom step ke pe temp pxx pyy pzz pxy pxz pyz
thermo_modify format float %.4f

log {log_path}

run {nsteps}
"""

    input_path = os.path.join(SCRIPT_DIR, f"lammps_{nsteps}steps.in")
    with open(input_path, "w") as f:
        f.write(lammps_input)

    return input_path


def relative_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise relative difference: |a-b| / max(|a|, |b|), with safe divide."""
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-12, 1.0, denom)
    return np.abs(a - b) / denom


def set_work_dir():
    """Change to the script directory so relative paths resolve correctly."""
    os.chdir(SCRIPT_DIR)
