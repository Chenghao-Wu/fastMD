"""Check CUDA toolkit availability before CMake configure."""
import os
import shutil
import sys


def main():
    if shutil.which("nvcc") is not None:
        return

    hints = []
    for candidate in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]:
        if os.path.isfile(candidate):
            hints.append(
                f"  Found {candidate} — add to PATH: "
                f"export PATH={os.path.dirname(candidate)}:$PATH"
            )

    msg = [
        "Error: CUDA Toolkit not found.",
        "fastmd requires CUDA >= 11.0 with nvcc on PATH.",
        "Install CUDA from: https://developer.nvidia.com/cuda-downloads",
    ]
    if hints:
        msg.append("")
        msg.extend(hints)

    sys.exit("\n".join(msg))


if __name__ == "__main__":
    main()
