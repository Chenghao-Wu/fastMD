"""fastMD — GPU-accelerated molecular dynamics simulator.

Provides ``get_binary_path()`` to locate the compiled fastmd binary
installed alongside this package.
"""

from pathlib import Path


def get_binary_path() -> str:
    """Return the absolute path to the compiled fastmd executable.

    Returns:
        str: Path to the ``fastmd`` binary shipped with this package.

    Raises:
        FileNotFoundError: If the binary is missing (indicates a broken
            installation — reinstall the package).
    """
    binary = Path(__file__).parent / "_binary" / "fastmd"
    if not binary.is_file():
        raise FileNotFoundError(
            f"fastmd binary not found at {binary}. "
            f"Reinstall the fastmd package: pip uninstall fastmd && pip install fastmd"
        )
    return str(binary)
