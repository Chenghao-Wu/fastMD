#!/usr/bin/env python3
"""Write matched fastMD / LAMMPS inputs for the bead-spring benchmark.

Both codes use the same interaction cutoff ``rc`` and neighbor-list skin
``skin``. Note that LAMMPS' ``neighbor <skin> bin`` takes the *skin*, not the
list radius: the list radius is rc + skin in both codes.

LAMMPS modes:
  kokkos  KOKKOS/CUDA: pair, bonds, angles, integrator and neighbor lists on
          the GPU (run with ``-k on g 1 -sf kk``; see ``lammps_command``).
  gpu     GPU package: pair + neighbor lists on the GPU, bonded terms and the
          integrator on the CPU (``-sf gpu -pk gpu 1``).
  cpu     plain serial CPU styles.
"""
import json
import sys
from pathlib import Path

LAMMPS_MODES = ("kokkos", "gpu", "cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_fastmd_config(cfg: dict, ensemble: dict, out_path, nsteps: int,
                        thermo_freq: int, data_file: str):
    lines = [
        f"natoms {cfg['natoms']}",
        f"ntypes {cfg['ntypes']}",
        f"rc {cfg['rc']}",
        f"skin {cfg['skin']}",
        f"dt {cfg['dt']}",
        f"nsteps {nsteps}",
        f"dump_freq {cfg['dump_freq']}",
        f"thermo 1 {thermo_freq} thermo_fastmd.dat",
    ]
    etype = ensemble["type"]
    if etype == "nvt_langevin":
        lines.append(f"nvt_langevin {ensemble['T_start']} {ensemble['T_stop']} "
                     f"{ensemble['Tdamp']} {ensemble['seed']}")
    elif etype == "nvt_nh":
        chain = ensemble.get("chain_length", 3)
        lines.append(f"nvt_nh {ensemble['T_start']} {ensemble['T_stop']} "
                     f"{ensemble['Tdamp']} {chain}")
    elif etype == "npt_nh":
        chain = ensemble.get("chain_length", 3)
        lines.append(f"npt_nh {ensemble['T_start']} {ensemble['T_stop']} {ensemble['Tdamp']} "
                     f"{ensemble['P_start']} {ensemble['P_stop']} {ensemble['Pdamp']} {chain}")
    else:
        raise ValueError(f"unknown ensemble type {etype}")
    for ti, tj, eps, sig in cfg["lj_params"]:
        lines.append(f"lj {ti} {tj} {eps} {sig}")
    for t, k, R0, eps, sig in cfg["bond_params"]:
        lines.append(f"bond_type {t} {k} {R0} {eps} {sig}")
    for t, k_theta, theta0_deg in cfg["angle_params"]:
        lines.append(f"angle_type {t} {k_theta} {theta0_deg}")
    lines.append(f"lammps_data_file {data_file}")

    Path(out_path).write_text("\n".join(lines) + "\n")


def write_lammps_input(cfg: dict, ensemble: dict, out_path, nsteps: int,
                       thermo_freq: int, data_file: str, mode: str = "kokkos"):
    if mode not in LAMMPS_MODES:
        raise ValueError(f"mode must be one of {LAMMPS_MODES}")
    lj = cfg["lj_params"][0]
    bond = cfg["bond_params"][0]
    angle = cfg["angle_params"][0]

    etype = ensemble["type"]
    if etype == "nvt_langevin":
        fix_lines = [
            "fix 1 all nve",
            f"fix 2 all langevin {ensemble['T_start']} {ensemble['T_stop']} "
            f"{ensemble['Tdamp']} {ensemble['seed']}",
        ]
    elif etype == "nvt_nh":
        fix_lines = [
            f"fix 1 all nvt temp {ensemble['T_start']} {ensemble['T_stop']} "
            f"{ensemble['Tdamp']} tchain {ensemble.get('chain_length', 3)}",
        ]
    elif etype == "npt_nh":
        fix_lines = [
            f"fix 1 all npt temp {ensemble['T_start']} {ensemble['T_stop']} {ensemble['Tdamp']} "
            f"iso {ensemble['P_start']} {ensemble['P_stop']} {ensemble['Pdamp']} "
            f"tchain {ensemble.get('chain_length', 3)}",
        ]
    else:
        raise ValueError(f"unknown ensemble type {etype}")

    # GPU package requires newton off; KOKKOS/CUDA performs best with full
    # lists and newton off (set on the command line); CPU uses newton on.
    newton = "on" if mode == "cpu" else "off"
    lines = [
        "units lj",
        "atom_style molecular",
        f"newton {newton}",
        "",
        f"pair_style lj/cut {cfg['rc']}",
        "bond_style fene",
        "angle_style harmonic",
        "",
        f"read_data {data_file}",
        "",
        f"pair_coeff {lj[0]+1} {lj[1]+1} {lj[2]} {lj[3]}",
        f"bond_coeff {bond[0]+1} {bond[1]} {bond[2]} {bond[3]} {bond[4]}",
        f"angle_coeff {angle[0]+1} {angle[1]} {angle[2]}",
        "special_bonds lj 0.0 1.0 1.0",
        "",
        # LAMMPS' argument is the skin; the list radius is rc + skin.
        f"neighbor {cfg['skin']} bin",
        "neigh_modify delay 0 every 1 check yes",
        "",
        f"timestep {cfg['dt']}",
    ]
    lines.extend(fix_lines)
    lines.extend([
        "",
        f"thermo {thermo_freq}",
        "thermo_style custom step temp pe ke etotal press",
        "",
        f"run {nsteps}",
    ])
    Path(out_path).write_text("\n".join(lines) + "\n")


def lammps_command(lammps_binary: str, input_file: str, log_file: str,
                   mode: str = "kokkos") -> list:
    cmd = [lammps_binary, "-in", input_file, "-log", log_file, "-screen", "none"]
    if mode == "kokkos":
        cmd += ["-k", "on", "g", "1", "-sf", "kk",
                "-pk", "kokkos", "newton", "off", "neigh", "full"]
    elif mode == "gpu":
        cmd += ["-sf", "gpu", "-pk", "gpu", "1"]
    return cmd


def main():
    if len(sys.argv) < 5:
        print("Usage: generate_lammps_input.py <benchmark.json> <phase> "
              "<ensemble_json> <data_file> [lammps_mode]")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    phase = sys.argv[2]
    ensemble = json.loads(sys.argv[3])
    data_file = sys.argv[4]
    mode = sys.argv[5] if len(sys.argv) > 5 else "kokkos"

    if phase == "validation":
        nsteps = cfg["nsteps_validation"]
        thermo_freq = cfg["thermo_freq_validation"]
    else:
        nsteps = cfg["nsteps_benchmark"]
        thermo_freq = cfg["thermo_freq_benchmark"]

    write_fastmd_config(cfg, ensemble, "fastmd_benchmark.conf", nsteps, thermo_freq, data_file)
    write_lammps_input(cfg, ensemble, "lammps_benchmark.in", nsteps, thermo_freq, data_file, mode)
    print("Generated fastmd_benchmark.conf and lammps_benchmark.in")


if __name__ == "__main__":
    main()
