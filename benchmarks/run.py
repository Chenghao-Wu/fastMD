#!/usr/bin/env python3
"""Validate and time fastMD against LAMMPS on the same bead-spring system.

Usage (from anywhere):
    python benchmarks/run.py --fastmd build/fastmd --lammps /path/to/lmp \
        --data kg30k.data [--lammps-mode kokkos|gpu|cpu] [--workdir bench_out]

Paths may also come from $FASTMD_BINARY, $LAMMPS_BINARY and $FASTMD_BENCH_DATA.
Timings compare the MD loop only ("Wall time" for fastMD, "Loop time" for
LAMMPS), so process start-up, CUDA context creation and input parsing are
excluded from both; total process wall time is reported alongside.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_lammps_input import (LAMMPS_MODES, lammps_command, load_config,  # noqa: E402
                                   write_fastmd_config, write_lammps_input)
from report import write_report  # noqa: E402


def data_natoms(path: str) -> int:
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1] == "atoms":
                return int(parts[0])
    raise RuntimeError(f"no 'N atoms' header in {path}")


def run_fastmd(args, cfg, ensemble, phase, workdir: Path):
    nsteps, thermo_freq = phase_steps(cfg, phase)
    conf = workdir / f"fastmd_{phase}.conf"
    write_fastmd_config(cfg, ensemble, conf, nsteps, thermo_freq, args.data)
    start = time.perf_counter()
    result = subprocess.run([args.fastmd, conf.name], capture_output=True, text=True,
                            cwd=workdir)
    wall = time.perf_counter() - start
    if result.returncode != 0:
        sys.exit(f"fastMD failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    m = re.search(r"Wall time:\s*([0-9.]+)\s*s", result.stdout)
    loop = float(m.group(1)) if m else wall
    return loop, wall, (workdir / "thermo_fastmd.dat").read_text()


def run_lammps(args, cfg, ensemble, phase, workdir: Path):
    nsteps, thermo_freq = phase_steps(cfg, phase)
    inp = workdir / f"lammps_{phase}.in"
    log = workdir / f"log_{phase}.lammps"
    write_lammps_input(cfg, ensemble, inp, nsteps, thermo_freq, args.data, args.lammps_mode)
    cmd = lammps_command(args.lammps, inp.name, log.name, args.lammps_mode)
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    wall = time.perf_counter() - start
    if result.returncode != 0:
        sys.exit(f"LAMMPS failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    text = log.read_text()
    m = re.search(r"Loop time of\s*([0-9.eE+-]+)", text)
    loop = float(m.group(1)) if m else wall
    return loop, wall, text


def phase_steps(cfg, phase):
    if phase == "validation":
        return cfg["nsteps_validation"], cfg["thermo_freq_validation"]
    return cfg["nsteps_benchmark"], cfg["thermo_freq_benchmark"]


def parse_fastmd_thermo(text: str):
    """thermo file columns: step ke pe T pxx pyy pzz pxy pxz pyz (per atom)."""
    temps, pes = [], []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        pes.append(float(parts[2]))
        temps.append(float(parts[3]))
    return temps, pes


def parse_lammps_thermo(log: str):
    temps, pes = [], []
    in_data = False
    for line in log.splitlines():
        stripped = line.strip()
        if stripped.startswith("Step") and "Temp" in stripped:
            in_data = True
            continue
        if in_data:
            parts = stripped.split()
            try:
                int(parts[0])
                temps.append(float(parts[1]))
                pes.append(float(parts[2]))
            except (ValueError, IndexError):
                in_data = False
    return temps, pes


def validate(temps_f, pes_f, temps_l, pes_l, ensemble: dict, cfg: dict):
    n = cfg.get("validation_window", 5)
    if len(temps_f) < n or len(temps_l) < n:
        raise RuntimeError(f"Not enough thermo outputs: fastMD={len(temps_f)}, LAMMPS={len(temps_l)}")

    avg_t_f = sum(temps_f[-n:]) / n
    avg_t_l = sum(temps_l[-n:]) / n
    avg_pe_f = sum(pes_f[-n:]) / n
    avg_pe_l = sum(pes_l[-n:]) / n

    rel_temp_diff = abs(avg_t_f - avg_t_l) / ensemble["T_start"]
    rel_pe_diff = abs(avg_pe_f - avg_pe_l) / abs(avg_pe_l) if avg_pe_l != 0 else abs(avg_pe_f - avg_pe_l)

    print(f"Validation averages (last {n} samples):")
    print(f"  T  fastMD={avg_t_f:.4f} LAMMPS={avg_t_l:.4f} rel_diff={rel_temp_diff:.4f}")
    print(f"  PE fastMD={avg_pe_f:.4f} LAMMPS={avg_pe_l:.4f} rel_diff={rel_pe_diff:.4f}")

    passed = rel_temp_diff < 0.05 and rel_pe_diff < 1e-2
    return passed, rel_temp_diff, rel_pe_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "benchmark.json"))
    ap.add_argument("--fastmd", default=os.environ.get("FASTMD_BINARY"))
    ap.add_argument("--lammps", default=os.environ.get("LAMMPS_BINARY"))
    ap.add_argument("--data", default=os.environ.get("FASTMD_BENCH_DATA"))
    ap.add_argument("--lammps-mode", choices=LAMMPS_MODES, default="kokkos")
    ap.add_argument("--workdir", default="bench_out")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    for name in ("fastmd", "lammps", "data"):
        if not getattr(args, name):
            ap.error(f"--{name} is required (or set the corresponding environment variable)")
    args.fastmd = str(Path(args.fastmd).resolve())
    args.lammps = str(Path(args.lammps).resolve())
    args.data = str(Path(args.data).resolve())

    cfg = load_config(args.config)
    cfg["natoms"] = data_natoms(args.data)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    for ensemble in cfg["ensembles"]:
        etype = ensemble["type"]
        print(f"\n{'=' * 60}\n=== {etype}  ({cfg['natoms']} atoms, LAMMPS {args.lammps_mode})\n{'=' * 60}")

        passed, rel_t, rel_pe = None, None, None
        if not args.skip_validation:
            print(f"\n--- Validation phase ({cfg['nsteps_validation']} steps) ---")
            _, _, out_f = run_fastmd(args, cfg, ensemble, "validation", workdir)
            _, _, out_l = run_lammps(args, cfg, ensemble, "validation", workdir)
            temps_f, pes_f = parse_fastmd_thermo(out_f)
            temps_l, pes_l = parse_lammps_thermo(out_l)
            passed, rel_t, rel_pe = validate(temps_f, pes_f, temps_l, pes_l, ensemble, cfg)
            print(f"Validation {'passed' if passed else 'FAILED'} for {etype}.")

        print(f"\n--- Benchmark phase ({cfg['nsteps_benchmark']} steps) ---")
        loop_f, wall_f, _ = run_fastmd(args, cfg, ensemble, "benchmark", workdir)
        loop_l, wall_l, _ = run_lammps(args, cfg, ensemble, "benchmark", workdir)
        nsteps = cfg["nsteps_benchmark"]
        print(f"fastMD: loop {loop_f:.2f} s ({nsteps / loop_f:.0f} steps/s), process {wall_f:.2f} s")
        print(f"LAMMPS: loop {loop_l:.2f} s ({nsteps / loop_l:.0f} steps/s), process {wall_l:.2f} s")
        print(f"Speedup (loop time): {loop_l / loop_f:.2f}x")

        write_report(
            cfg=cfg, ensemble=etype, data_file=args.data, lammps_mode=args.lammps_mode,
            fastmd_loop=loop_f, fastmd_wall=wall_f,
            lammps_loop=loop_l, lammps_wall=wall_l,
            validation={"passed": passed, "rel_temp_diff": rel_t, "rel_pe_diff": rel_pe},
            out_path=str(workdir / f"benchmark_report_{etype}_{args.lammps_mode}.json"),
        )
        if passed is False:
            sys.exit(f"VALIDATION FAILED for {etype}")

    print("\n=== All benchmarks complete ===")


if __name__ == "__main__":
    main()
