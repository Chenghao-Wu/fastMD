"""Reader for fastMD trajectory files (format version 2, ``traj`` keyword).

Frames hold the step, the box (Lx, Ly, Lz, xy, xz, yz), wrapped positions
and periodic image counts, in the atom order of the input data file. The
file is memory-mapped, so frames are read on demand::

    from fastmd.trajectory import Trajectory
    traj = Trajectory("traj.fmd")
    for k in range(len(traj)):
        x = traj.unwrapped(k)          # (natoms, 3) float64
    chains = traj.mol_ids               # molecule id of every atom
"""

import numpy as np

MAGIC = b"FMDTRJ02"

__all__ = ["Trajectory", "unwrap"]


def unwrap(pos, images, box):
    """Unwrapped coordinates from wrapped positions, image counts and a box
    (Lx, Ly, Lz, xy, xz, yz)."""
    lx, ly, lz, xy, xz, yz = box
    img = images.astype(np.float64)
    out = pos.astype(np.float64)
    out[:, 0] += img[:, 0] * lx + img[:, 1] * xy + img[:, 2] * xz
    out[:, 1] += img[:, 1] * ly + img[:, 2] * yz
    out[:, 2] += img[:, 2] * lz
    return out


class Trajectory:
    """Memory-mapped fastMD trajectory."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(f"{self.path}: not a fastMD v2 trajectory (magic {magic!r})")
            version, natoms, flags, _ = np.frombuffer(f.read(16), dtype="<i4")
        if version != 2:
            raise ValueError(f"{self.path}: unsupported version {version}")
        self.natoms = int(natoms)
        self.has_images = bool(flags & 1)
        head = 24 + 8 * self.natoms
        meta = np.memmap(self.path, dtype="<i4", mode="r", offset=24, shape=(2, self.natoms))
        self.types = np.array(meta[0])
        self.mol_ids = np.array(meta[1])
        fields = [("step", "<i8"), ("box", "<f8", (6,)), ("pos", "<f4", (self.natoms, 3))]
        if self.has_images:
            fields.append(("image", "<i4", (self.natoms, 3)))
        self._dtype = np.dtype(fields)
        size = _file_size(self.path) - head
        nframes = size // self._dtype.itemsize   # a partially written last frame is ignored
        self._frames = np.memmap(self.path, dtype=self._dtype, mode="r", offset=head,
                                 shape=(nframes,)) if nframes else np.zeros(0, self._dtype)

    def __len__(self):
        return len(self._frames)

    @property
    def steps(self):
        return np.array(self._frames["step"])

    def box(self, k):
        return np.array(self._frames["box"][k])

    def positions(self, k):
        """Wrapped positions of frame k, (natoms, 3) float32."""
        return np.array(self._frames["pos"][k])

    def images(self, k):
        if not self.has_images:
            raise ValueError("trajectory has no image counts")
        return np.array(self._frames["image"][k])

    def unwrapped(self, k):
        """Unwrapped positions of frame k, (natoms, 3) float64."""
        return unwrap(self._frames["pos"][k], self._frames["image"][k], self._frames["box"][k])

    def frame_index(self, step):
        """Index of the frame written at `step` (KeyError if absent)."""
        idx = np.nonzero(self._frames["step"] == step)[0]
        if len(idx) == 0:
            raise KeyError(step)
        return int(idx[0])


def _file_size(path):
    import os
    return os.path.getsize(path)
