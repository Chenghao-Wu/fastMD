"""fastMD — GPU-accelerated molecular dynamics simulator.

Provides ``get_binary_path()`` to locate the compiled ``fastmd`` executable
installed alongside this package (or found on ``PATH``).
"""

import os
import shutil
from pathlib import Path

__all__ = ["get_binary_path"]


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
