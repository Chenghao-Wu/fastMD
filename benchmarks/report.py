#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def write_report(cfg: dict, ensemble: str, data_file: str, lammps_mode: str,
                 fastmd_loop: float, fastmd_wall: float,
                 lammps_loop: float, lammps_wall: float,
                 validation: dict, out_path=None):
    if out_path is None:
        out_path = f"benchmark_report_{ensemble}_{lammps_mode}.json"
    nsteps = cfg["nsteps_benchmark"]
    sim_ns = nsteps * cfg["dt"]  # reduced (LJ) time units, reported as "tau"

    def entry(loop, wall):
        return {
            "loop_time_s": round(loop, 3),
            "process_time_s": round(wall, 3),
            "steps_per_s": round(nsteps / loop, 1),
            "tau_per_day": round(sim_ns / (loop / 86400.0), 1),
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": get_gpu_name(),
        "ensemble": ensemble,
        "system": data_file,
        "n_atoms": cfg["natoms"],
        "nsteps": nsteps,
        "rc": cfg["rc"],
        "skin": cfg["skin"],
        "fastMD": entry(fastmd_loop, fastmd_wall),
        "lammps": dict(entry(lammps_loop, lammps_wall), mode=lammps_mode),
        "speedup_loop": round(lammps_loop / fastmd_loop, 2),
        "validation": validation,
    }
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report written to {out_path}")
