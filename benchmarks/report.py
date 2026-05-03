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


def write_report(
    cfg: dict,
    ensemble: str,
    fastmd_time: float,
    lammps_time: float,
    ns_per_day_f: float,
    ns_per_day_l: float,
    speedup: float,
    validation: dict,
    out_path: str = None,
):
    if out_path is None:
        out_path = f"benchmark_report_{ensemble}.json"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": get_gpu_name(),
        "ensemble": ensemble,
        "system": cfg["data_file"],
        "n_atoms": cfg["natoms"],
        "nsteps": cfg["nsteps_benchmark"],
        "fastMD": {
            "wall_time_s": round(fastmd_time, 2),
            "ns_per_day": round(ns_per_day_f, 2),
        },
        "lammps": {
            "wall_time_s": round(lammps_time, 2),
            "ns_per_day": round(ns_per_day_l, 2),
        },
        "speedup": round(speedup, 2),
        "validation": {
            "passed": validation["passed"],
            "rel_temp_diff": validation["rel_temp_diff"],
            "rel_pe_diff": validation["rel_pe_diff"],
        },
    }
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report written to {out_path}")
