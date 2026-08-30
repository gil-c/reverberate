"""The one setting that says where everything on disk lives.

Roadmap section 10: every stage writes only inside one data root, and that root
is configurable through a single setting so the pipeline can be pointed at a
rented machine. This module is that setting, and nothing else reads the
environment for a path.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DATA_ROOT_ENV", "data_root", "runs_dir"]

#: Environment variable overriding the data root. Unset means "data/ beside the
#: repository", which is what a checkout on a workstation wants.
DATA_ROOT_ENV = "REVERBERATE_DATA"


def data_root() -> Path:
    """Absolute path of the directory every stage writes under.

    Never creates it: a stage that writes calls :func:`runs_dir` or makes its
    own subdirectory, so a typo in the environment fails where it is used
    rather than silently scattering an empty tree.
    """
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def runs_dir() -> Path:
    """``<data root>/runs``, created if missing. Solver logs and artefacts."""
    path = data_root() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path
