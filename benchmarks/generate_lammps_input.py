#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_fastmd_config(cfg: dict, out_path: str, nsteps: int, thermo_freq: int):
    lines = [
        f"natoms {cfg['natoms']}",
        f"box_L {cfg['box_L']}",
        f"ntypes {cfg['ntypes']}",
        f"rc {cfg['rc']}",
        f"skin {cfg['skin']}",
        f"dt {cfg['dt']}",
        f"temperature {cfg['temperature']}",
        f"gamma {cfg['gamma']}",
        f"nsteps {nsteps}",
        f"dump_freq {cfg['dump_freq']}",
        f"thermo_freq {thermo_freq}",
        f"seed {cfg['seed']}",
    ]
    for ti, tj, eps, sig in cfg["lj_params"]:
        lines.append(f"lj {ti} {tj} {eps} {sig}")
    for t, k, R0, eps, sig in cfg["bond_params"]:
        lines.append(f"bond_type {t} {k} {R0} {eps} {sig}")
    for t, k_theta, theta0_deg in cfg["angle_params"]:
        lines.append(f"angle_type {t} {k_theta} {theta0_deg}")
    lines.append(f"lammps_data_file {cfg['data_file']}")

    Path(out_path).write_text("\n".join(lines) + "\n")


def write_lammps_input(cfg: dict, out_path: str, nsteps: int, thermo_freq: int):
    lj = cfg["lj_params"][0]
    bond = cfg["bond_params"][0]
    angle = cfg["angle_params"][0]
    neighbor_dist = cfg["rc"] + cfg["skin"]

    lines = [
        "units lj",
        "atom_style molecular",
        "newton off",
        "",
        "package gpu 1",
        "suffix gpu",
        "",
        f"read_data {cfg['data_file']}",
        "",
        f"pair_style lj/cut {cfg['rc']}",
        f"pair_coeff {lj[0]+1} {lj[1]+1} {lj[2]} {lj[3]}",
        "",
        "bond_style fene",
        f"bond_coeff {bond[0]+1} {bond[1]} {bond[2]} {bond[3]} {bond[4]}",
        "",
        "angle_style harmonic",
        f"angle_coeff {angle[0]+1} {angle[1]} {angle[2]}",
        "",
        f"neighbor {neighbor_dist} bin",
        "neigh_modify delay 0 every 1 check yes",
        "",
        f"timestep {cfg['dt']}",
        f"fix 1 all nve",
        f"fix 2 all langevin {cfg['temperature']} {cfg['temperature']} {1.0/cfg['gamma']} {cfg['seed']}",
        "",
        f"thermo {thermo_freq}",
        "thermo_style custom step temp pe ke etotal",
        "",
        f"run {nsteps}",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_lammps_input.py <benchmark.json>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    phase = sys.argv[2] if len(sys.argv) > 2 else "benchmark"

    if phase == "validation":
        nsteps = cfg["nsteps_validation"]
        thermo_freq = cfg["thermo_freq_validation"]
    else:
        nsteps = cfg["nsteps_benchmark"]
        thermo_freq = cfg["thermo_freq_benchmark"]

    write_fastmd_config(cfg, "fastmd_benchmark.conf", nsteps, thermo_freq)
    write_lammps_input(cfg, "lammps_benchmark.in", nsteps, thermo_freq)
    print("Generated fastmd_benchmark.conf and lammps_benchmark.in")


if __name__ == "__main__":
    main()
