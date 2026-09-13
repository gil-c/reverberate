"""Scripts that run on rented machines under PFFDTD's own interpreter.

Kept as files rather than strings in the driver so they can be read, linted
and run by hand. :func:`install` copies the directory to a machine; the
driver then names each tool by :func:`tool`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Where the tools live on a rented machine.
REMOTE_TOOLS = "/root/tools"

#: Where the package is copied for the child encoder's imports.
REMOTE_SRC = "/root/src"


def tool(name: str) -> str:
    """The remote path of one script of this directory."""
    return f"{REMOTE_TOOLS}/{name}.py"


def install(machine: Any) -> None:
    """Copy the tools and the package to ``machine``; idempotent."""
    from reverberate.wave.remote import run_on
    from reverberate.wave.remote_voxelise import rsync

    package = Path(__file__).resolve().parents[3]  # .../src/reverberate
    run_on(machine, f"mkdir -p {REMOTE_TOOLS} {REMOTE_SRC}", what="mkdir tools")
    rsync(machine, [str(Path(__file__).parent) + "/"], REMOTE_TOOLS + "/", download=False)
    rsync(machine, [str(package)], REMOTE_SRC + "/", download=False)
