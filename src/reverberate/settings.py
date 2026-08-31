"""The one setting that says where everything on disk lives.

Roadmap section 10: every stage writes only inside one data root, and that root
is configurable through a single setting so the pipeline can be pointed at a
rented machine. This module is that setting, and nothing else reads the
environment for a path.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DATA_ROOT_ENV",
    "PFFDTD_PYTHON_ENV",
    "data_root",
    "interim_dir",
    "pffdtd_python",
    "runs_dir",
]

#: Environment variable overriding the data root. Unset means "data/ beside the
#: repository", which is what a checkout on a workstation wants.
DATA_ROOT_ENV = "REVERBERATE_DATA"

#: Environment variable pointing at PFFDTD's ``python`` directory. The solver
#: is not a package and is not pip-installable, so the one thing that needs it
#: on a workstation -- fitting boundary impedances -- is told where it is
#: rather than guessing.
PFFDTD_PYTHON_ENV = "PFFDTD_PYTHON"


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


def interim_dir(*parts: str) -> Path:
    """``<data root>/interim/...``, created if missing.

    Derived data that a stage may regenerate from checked-in inputs, as
    opposed to ``runs/``, which records something that was measured once.
    """
    path = data_root().joinpath("interim", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pffdtd_python() -> Path:
    """PFFDTD's ``python`` directory, from the environment.

    Raises rather than defaulting: a wrong path here would fit impedances with
    some other copy of the routine than the pinned one, and the resulting
    coefficients would look perfectly plausible.
    """
    override = os.environ.get(PFFDTD_PYTHON_ENV)
    if not override:
        raise RuntimeError(
            f"{PFFDTD_PYTHON_ENV} is not set; point it at the python/ directory of a "
            "PFFDTD checkout at the pinned commit (see scripts/build_pffdtd.sh)"
        )
    path = Path(override).expanduser().resolve()
    if not (path / "materials" / "adm_funcs.py").is_file():
        raise RuntimeError(f"{path} does not look like PFFDTD's python/ directory")
    return path
