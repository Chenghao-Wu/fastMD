"""fastMD — GPU-accelerated molecular dynamics simulator.

``Simulation`` is the stateful interface to a simulation resident on the GPU
(compiled extension ``fastmd._core``)::

    import fastmd
    sim = fastmd.Simulation("run.conf")
    sim.run(1000)
    x, f = sim.positions, sim.forces        # (natoms, 3) numpy arrays
    print(sim.thermo())

``get_binary_path()`` locates the command-line ``fastmd`` executable.
"""

import os
import shutil
from pathlib import Path

__all__ = ["Simulation", "Thermo", "get_binary_path"]

try:
    from ._core import Simulation, Thermo  # noqa: F401
except ImportError as _err:  # extension not built (e.g. CMake build without FASTMD_BUILD_PYTHON)
    _import_error = _err

    def __getattr__(name):
        if name in ("Simulation", "Thermo"):
            raise ImportError(
                f"fastmd._core is not available ({_import_error}); build with "
                "-DFASTMD_BUILD_PYTHON=ON or install the wheel (pip install .)"
            ) from _import_error
        raise AttributeError(name)


def get_binary_path() -> str:
    """Return the absolute path to the compiled ``fastmd`` executable.

    Search order: ``$FASTMD_BINARY``, the binary shipped inside this package
    (wheel install), then ``fastmd`` on ``PATH``.

    Raises:
        FileNotFoundError: If no executable can be found.
    """
    env = os.environ.get("FASTMD_BINARY")
    if env:
        if Path(env).is_file():
            return str(Path(env).resolve())
        raise FileNotFoundError(f"FASTMD_BINARY={env} does not exist")

    candidate = Path(__file__).parent / "_binary" / "fastmd"
    if candidate.is_file():
        return str(candidate)

    on_path = shutil.which("fastmd")
    if on_path:
        return on_path

    raise FileNotFoundError(
        "fastmd binary not found. Reinstall the package "
        "(pip install fast-molecular-dynamics), put `fastmd` on PATH, "
        "or set FASTMD_BINARY."
    )
