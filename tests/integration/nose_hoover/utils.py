"""Shared utilities for NH NPT integration tests."""
import subprocess
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTMD_BIN = os.path.join(SCRIPT_DIR, "../../../build/fastMD")
LAMMPS_BIN = "/home/zhenghaowu/lammps/build/lmp"

def run_fastmd(config_path: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [FASTMD_BIN, config_path],
        cwd=SCRIPT_DIR,
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        print(f"fastMD stderr:\n{result.stderr}")
    return result

def run_lammps(input_path: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [LAMMPS_BIN, "-in", input_path],
        cwd=SCRIPT_DIR,
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        print(f"LAMMPS stderr:\n{result.stderr}")
    return result

def parse_thermo(filepath: str) -> dict:
    path = os.path.join(SCRIPT_DIR, filepath)
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    columns = ["step", "KE", "PE", "T", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
    if data.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} columns, got {data.shape[1]}")
    return {col: data[:, i] for i, col in enumerate(columns)}

def write_fastmd_config(nsteps: int, T_start=1.0, T_stop=1.0, Tdamp=1.0,
                        P_start=0.0, P_stop=0.0, Pdamp=1.0) -> str:
    template_path = os.path.join(SCRIPT_DIR, "fastmd_template.conf")
    with open(template_path) as f:
        content = f.read()
    content = content.replace("__NSTEPS__", str(nsteps))
    content = content.replace("__TSTART__", str(T_start))
    content = content.replace("__TSTOP__", str(T_stop))
    content = content.replace("__TDAMP__", str(Tdamp))
    content = content.replace("__PSTART__", str(P_start))
    content = content.replace("__PSTOP__", str(P_stop))
    content = content.replace("__PDAMP__", str(Pdamp))
    config_path = os.path.join(SCRIPT_DIR, f"fastmd_{nsteps}steps.conf")
    with open(config_path, "w") as f:
        f.write(content)
    return config_path

def write_lammps_input(nsteps: int, T_start=1.0, T_stop=1.0, Tdamp=1.0,
                       P_start=0.0, P_stop=0.0, Pdamp=1.0,
                       log_path="log.lammps") -> str:
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

fix 1 all npt temp {T_start} {T_stop} {Tdamp} iso {P_start} {P_stop} {Pdamp}

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

def extract_lammps_thermo(log_file: str, output_file: str) -> int:
    """Extract thermo table from LAMMPS log, return number of lines."""
    log_path = os.path.join(SCRIPT_DIR, log_file)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    thermo_lines = []
    in_thermo = False
    with open(log_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Step") and "KinEng" in stripped:
                thermo_lines.append("# step  KE  PE  T  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz")
                in_thermo = True
                continue
            if in_thermo:
                if stripped and stripped.split()[0].isdigit() and "CPU" not in stripped:
                    parts = stripped.split()
                    if len(parts) >= 10:
                        thermo_lines.append("  ".join(parts[:10]))
                elif stripped.startswith("Loop time"):
                    break
    out_path = os.path.join(SCRIPT_DIR, output_file)
    with open(out_path, "w") as f:
        f.write("\n".join(thermo_lines) + "\n")
    return len(thermo_lines) - 1

def set_work_dir():
    os.chdir(SCRIPT_DIR)
