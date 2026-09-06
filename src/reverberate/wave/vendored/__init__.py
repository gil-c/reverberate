"""Files this project replaces inside the vendored PFFDTD checkout.

Only whole files live here, never diffs. A ``sed`` applied on a rented machine
is invisible in review, unrunnable in a test, and silently a no-op the day
upstream reformats the line it matches; a checked-in file is none of those.

**The pin is what makes a replacement safe.** Our copy was derived from one
exact upstream file, and dropping it onto a different one would be a merge
nobody performed. :data:`UPSTREAM_SHA256` records what it was derived from and
:func:`reverberate.wave.voxelise.ensure_patched` refuses to install over
anything else, so a bumped ``PFFDTD_COMMIT`` fails loudly at the next
voxelisation instead of quietly producing different geometry.

Only the voxeliser is replaced, and only locally: a rented machine receives the
four HDF5 files the CUDA engine reads and never runs this Python at all.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PATCHED_FILES", "UPSTREAM_COMMIT", "UPSTREAM_SHA256", "patched_path"]

#: The PFFDTD commit our copies were taken from. Must match the pin in
#: ``scripts/build_pffdtd.sh``.
UPSTREAM_COMMIT = "aa319f6c86517cb95aabfae8656277da62c3ead5"

#: Path inside the checkout -> sha256 of the upstream file we started from.
UPSTREAM_SHA256 = {
    "python/voxelizer/vox_scene.py": (
        "c37f3529e6b07ee322335c9ed0da8bac931c060dc708548ad7280df3437f445b"
    ),
    "python/voxelizer/vox_grid_base.py": (
        "ba5bcd3b8e3c01b7d337a8a5c14391e800cf68d4e8cd3cc44e2160ca19dc8e11"
    ),
    "python/common/myfuncs.py": (
        "7e8db50e1c815618f2679ec1089c48712e1e4a48aff38f573032a5dad43b8922"
    ),
}

#: Path inside the checkout -> the file here that replaces it.
PATCHED_FILES = {
    "python/voxelizer/vox_scene.py": "vox_scene.py",
    "python/voxelizer/vox_grid_base.py": "vox_grid_base.py",
    "python/common/myfuncs.py": "myfuncs.py",
}


def patched_path(relative: str) -> Path:
    """The replacement file for ``relative``, as an absolute path."""
    return Path(__file__).with_name(PATCHED_FILES[relative])
