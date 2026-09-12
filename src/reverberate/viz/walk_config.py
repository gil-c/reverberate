"""Where the app is told what to serve: one ``walk.toml``, found from any checkout.

The app used to take everything on the command line, which every agent on
every worktree had to retype. Now it reads a TOML file, looked for in this
order: ``--config``, the ``REVERBERATE_WALK`` variable, ``walk.toml`` in the
working directory, then ``walk.toml`` at the root of the **main checkout**,
which a worktree finds through git's common directory. The main checkout's
copy is deliberately not versioned: it holds this machine's paths.

.. code-block:: toml

    hssd_root = "/Users/me/Developer/reverberate/data/raw/hssd-hab"
    data_root = "/Users/me/Developer/reverberate/data"   # caches and voices
    runs = "/Users/me/Developer/reverberate/data/runs"   # runs with a walk.json
    run = "synthetic_walk_mock"                            # opened first, optional
    scene = "hssd_0002"                                    # opened first, optional
    port = 8765
    open_browser = true
    measured_head = "/Users/me/Developer/reverberate/data/raw/hrtf/HRIR_L2702.sofa"

``data_root`` becomes ``REVERBERATE_DATA`` for the process when that variable
is not already set, so the caches land in one place whichever worktree runs.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from reverberate.settings import DATA_ROOT_ENV

__all__ = ["CONFIG_ENV", "CONFIG_NAME", "WalkConfig", "find_config", "load_config"]

CONFIG_NAME = "walk.toml"
CONFIG_ENV = "REVERBERATE_WALK"


@dataclass(frozen=True)
class WalkConfig:
    hssd_root: Path
    data_root: Path | None = None
    runs: Path | None = None
    run: str | None = None
    scene: str | None = None
    port: int = 8765
    open_browser: bool = True
    voices: Path | None = None
    measured_head: Path | None = None
    rebuild: bool = False
    source: Path | None = field(default=None, compare=False)

    def apply_environment(self) -> None:
        """Point the caches at ``data_root`` unless the environment already does."""
        if self.data_root is not None and not os.environ.get(DATA_ROOT_ENV):
            os.environ[DATA_ROOT_ENV] = str(self.data_root)


def main_checkout(start: Path | None = None) -> Path | None:
    """The main checkout of the repository ``start`` lies in, worktree or not."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=start or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(common).parent if common else None


def find_config(explicit: Path | None = None, start: Path | None = None) -> Path | None:
    """The first ``walk.toml`` in the order the module docstring gives.

    The main checkout is looked for from ``start`` (the working directory by
    default) and, failing that, from this file's own checkout, so the app
    finds its configuration wherever it is launched from.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get(CONFIG_ENV):
        candidates.append(Path(os.environ[CONFIG_ENV]))
    here = start or Path.cwd()
    candidates.append(here / CONFIG_NAME)
    for origin in (here, Path(__file__).resolve().parent):
        main = main_checkout(origin)
        if main is not None:
            candidates.append(main / CONFIG_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path) -> WalkConfig:
    """Read one ``walk.toml``; relative paths are relative to the file."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    base = Path(path).resolve().parent

    def as_path(key: str) -> Path | None:
        value = raw.get(key)
        if value in (None, ""):
            return None
        return (base / Path(str(value)).expanduser()).resolve()

    if "hssd_root" not in raw:
        raise ValueError(f"{path}: hssd_root is required")
    hssd_root = as_path("hssd_root")
    assert hssd_root is not None
    return WalkConfig(
        hssd_root=hssd_root,
        data_root=as_path("data_root"),
        runs=as_path("runs"),
        run=str(raw["run"]) if raw.get("run") else None,
        scene=str(raw["scene"]) if raw.get("scene") else None,
        port=int(raw.get("port", 8765)),
        open_browser=bool(raw.get("open_browser", True)),
        voices=as_path("voices"),
        measured_head=as_path("measured_head"),
        rebuild=bool(raw.get("rebuild", False)),
        source=Path(path),
    )
