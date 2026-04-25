#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

from report import write_report


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_fastmd(cfg: dict, phase: str) -> tuple:
    subprocess.run(
        [sys.executable, "benchmarks/generate_lammps_input.py",
         "benchmarks/benchmark.json", phase],
        check=True,
        cwd="/home/zhenghaowu/fastMD",
    )
    cmd = [cfg["fastmd_binary"], "fastmd_benchmark.conf"]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_time = time.perf_counter() - start
    return wall_time, result.stdout


def run_lammps(cfg: dict, phase: str) -> tuple:
    subprocess.run(
        [sys.executable, "benchmarks/generate_lammps_input.py",
         "benchmarks/benchmark.json", phase],
        check=True,
        cwd="/home/zhenghaowu/fastMD",
    )
    cmd = [cfg["lammps_binary"], "-in", "lammps_benchmark.in", "-log", "log.lammps"]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_time = time.perf_counter() - start
    return wall_time, Path("log.lammps").read_text()


def parse_fastmd_thermo(stdout: str):
    temps = []
    pes = []
    for line in stdout.splitlines():
        if line.startswith("Step "):
            parts = line.split()
            for i, p in enumerate(parts):
                if p.startswith("T="):
                    temps.append(float(p[2:]))
                elif p.startswith("PE="):
                    pes.append(float(p[3:]))
    return temps, pes


def parse_lammps_thermo(log: str):
    temps = []
    pes = []
    in_data = False
    for line in log.splitlines():
        if line.startswith("Step"):
            in_data = True
            continue
        if in_data:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    temps.append(float(parts[1]))
                    pes.append(float(parts[2]))
                except ValueError:
                    in_data = False
    return temps, pes


def validate(temps_f, pes_f, temps_l, pes_l, cfg: dict):
    n = cfg.get("validation_window", 500)
    if len(temps_f) < n or len(temps_l) < n:
        raise RuntimeError(f"Not enough thermo outputs: fastMD={len(temps_f)}, LAMMPS={len(temps_l)}")

    avg_t_f = sum(temps_f[-n:]) / n
    avg_t_l = sum(temps_l[-n:]) / n
    avg_pe_f = sum(pes_f[-n:]) / n
    avg_pe_l = sum(pes_l[-n:]) / n

    rel_temp_diff = abs(avg_t_f - avg_t_l) / cfg["temperature"]
    rel_pe_diff = abs(avg_pe_f - avg_pe_l) / abs(avg_pe_l) if avg_pe_l != 0 else abs(avg_pe_f - avg_pe_l)

    print(f"Validation averages (last {n} steps):")
    print(f"  T  fastMD={avg_t_f:.4f} LAMMPS={avg_t_l:.4f} rel_diff={rel_temp_diff:.4f}")
    print(f"  PE fastMD={avg_pe_f:.4f} LAMMPS={avg_pe_l:.4f} rel_diff={rel_pe_diff:.4f}")

    passed = rel_temp_diff < 0.05 and rel_pe_diff < 1e-2
    return passed, rel_temp_diff, rel_pe_diff


def main():
    cfg = load_config("benchmarks/benchmark.json")

    # Phase 1: Validation
    print("=== Validation phase (1k steps) ===")
    t_f, out_f = run_fastmd(cfg, "validation")
    t_l, out_l = run_lammps(cfg, "validation")

    temps_f, pes_f = parse_fastmd_thermo(out_f)
    temps_l, pes_l = parse_lammps_thermo(out_l)

    passed, rel_temp_diff, rel_pe_diff = validate(temps_f, pes_f, temps_l, pes_l, cfg)
    if not passed:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("Validation passed.")

    # Phase 2: Benchmark
    print("\n=== Benchmark phase (1M steps) ===")
    t_f, _ = run_fastmd(cfg, "benchmark")
    print(f"fastMD wall time: {t_f:.2f} s")
    t_l, _ = run_lammps(cfg, "benchmark")
    print(f"LAMMPS wall time: {t_l:.2f} s")

    ns_per_day_f = (cfg["nsteps_benchmark"] * cfg["dt"]) / (t_f / 86400.0)
    ns_per_day_l = (cfg["nsteps_benchmark"] * cfg["dt"]) / (t_l / 86400.0)
    speedup = t_l / t_f

    print(f"fastMD: {ns_per_day_f:.2f} ns/day")
    print(f"LAMMPS: {ns_per_day_l:.2f} ns/day")
    print(f"Speedup: {speedup:.2f}x")

    write_report(
        cfg=cfg,
        fastmd_time=t_f,
        lammps_time=t_l,
        ns_per_day_f=ns_per_day_f,
        ns_per_day_l=ns_per_day_l,
        speedup=speedup,
        validation={"passed": passed, "rel_temp_diff": rel_temp_diff, "rel_pe_diff": rel_pe_diff},
    )


if __name__ == "__main__":
    main()
