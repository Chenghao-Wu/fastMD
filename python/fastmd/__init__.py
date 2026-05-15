"""fastMD — GPU-accelerated molecular dynamics simulator.

Provides ``get_binary_path()`` to locate the compiled fastmd binary
installed alongside this package.
"""

import sysconfig
from pathlib import Path


def get_binary_path() -> str:
    """Return the absolute path to the compiled fastmd executable.

    Returns:
        str: Path to the ``fastmd`` binary shipped with this package.

    Raises:
        FileNotFoundError: If the binary is missing (indicates a broken
            installation — reinstall the package).
    """
    binary_rel = Path("_binary") / "fastmd"

    # Non-editable install: binary is next to __init__.py
    candidate = Path(__file__).parent / binary_rel
    if candidate.is_file():
        return str(candidate)

    # Editable install: binary lands under platlib/python/fastmd/
    candidate = Path(sysconfig.get_path("platlib")) / "python" / "fastmd" / binary_rel
    if candidate.is_file():
        return str(candidate)

    raise FileNotFoundError(
        f"fastmd binary not found. "
        f"Reinstall the fastmd package: pip uninstall fastmd && pip install fastmd"
    )
