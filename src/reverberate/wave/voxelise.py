"""Voxelising off the rented machine, once per scene and grid step.

Roadmap section 11: of the B0 session, 52.3 minutes went to voxelisation and
22.4 to the solver, on a machine rented for its card. **About 80 per cent of the
bill bought CPU time on an idle GPU.** Voxelisation needs no GPU at all, so it
happens here, on whatever CPU is cheapest, and its result is cached.

The cache is content addressed. The key is a hash of everything the voxelisation
actually depends on: the scene's JSON bytes, the impedance files, the grid
parameters and the bounds. The roadmap says "keyed on scene and grid step only";
in practice ``sim_mats.h5`` is written by the same pass and does depend on the
materials, so they are in the key too. A key that ignored them would serve a
scene its neighbour's walls.

What an entry holds is the three scene-dependent files the engine reads,
``sim_consts.h5``, ``vox_out.h5`` and ``sim_mats.h5``, plus ``cart_grid.h5``,
which the engine never reads but :mod:`reverberate.wave.comms` cannot work
without, plus ``manifest.json`` with the timings, the sizes and the key's
inputs. It deliberately holds no ``comms_out.h5``: that file is per source and
receiver pair, and regenerating it for nothing is the point of the split.

PFFDTD runs in its own interpreter, located by ``PFFDTD_DIR`` and
``PFFDTD_PYTHON``, because it needs numpy below 2 and this project does not.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from reverberate.settings import data_root
from reverberate.wave.comms import ENGINE_FILES

__all__ = [
    "CACHE_FILES",
    "CacheEntry",
    "PFFDTD_DIR_ENV",
    "PFFDTD_PYTHON_ENV",
    "SceneSpec",
    "cache_root",
    "engine_inputs",
    "entry_for",
    "pffdtd_dir",
    "pffdtd_python",
    "voxelise",
]

#: Where PFFDTD is checked out, and which interpreter can import it.
PFFDTD_DIR_ENV = "PFFDTD_DIR"
PFFDTD_PYTHON_ENV = "PFFDTD_PYTHON"

#: What a published cache entry contains. ``cart_grid.h5`` is not shipped to the
#: engine but is required to place new sources and receivers, see
#: :func:`reverberate.wave.comms.transpose_order`.
CACHE_FILES = ("sim_consts.h5", "vox_out.h5", "sim_mats.h5", "cart_grid.h5")

_REPORT_MARKER = "@@REVERBERATE@@"


def pffdtd_dir() -> Path:
    """PFFDTD's checkout, from the environment. Never guessed."""
    value = os.environ.get(PFFDTD_DIR_ENV)
    if not value:
        raise RuntimeError(
            f"{PFFDTD_DIR_ENV} is unset. Point it at a checkout built by scripts/build_pffdtd.sh."
        )
    path = Path(value).expanduser().resolve()
    if not (path / "python" / "sim_setup.py").is_file():
        raise RuntimeError(f"{path} does not look like a PFFDTD checkout")
    return path


def pffdtd_python() -> str:
    """The interpreter that can import PFFDTD, defaulting to this one."""
    return os.environ.get(PFFDTD_PYTHON_ENV) or sys.executable


@dataclass(frozen=True)
class SceneSpec:
    """Everything a voxelisation depends on, and nothing else.

    Two specs with the same :attr:`key` describe the same grid and the same
    boundary nodes, whatever sources or receivers are later placed on them.
    """

    model_json: Path
    mat_folder: Path
    mat_files: dict[str, str]
    fmax: float
    ppw: float
    tc: float = 20.0
    rh: float = 50.0
    fcc: bool = False
    bmin: tuple[float, float, float] | None = None
    bmax: tuple[float, float, float] | None = None
    rot_az_el: tuple[float, float] = (0.0, 0.0)

    @property
    def key(self) -> str:
        """A hash over the scene's bytes, its materials' bytes and the grid."""
        digest = hashlib.sha256()
        digest.update(Path(self.model_json).read_bytes())
        for label in sorted(self.mat_files):
            digest.update(label.encode())
            digest.update((Path(self.mat_folder) / self.mat_files[label]).read_bytes())
        digest.update(
            json.dumps(
                {
                    "fmax": self.fmax,
                    "ppw": self.ppw,
                    "Tc": self.tc,
                    "rh": self.rh,
                    "fcc": self.fcc,
                    "bmin": list(self.bmin) if self.bmin else None,
                    "bmax": list(self.bmax) if self.bmax else None,
                    "rot_az_el": list(self.rot_az_el),
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()[:32]


@dataclass(frozen=True)
class CacheEntry:
    """A voxelisation on disk, ready for a source and receiver pair."""

    path: Path
    key: str
    manifest: dict[str, object] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether every file an entry promises is present."""
        return all((self.path / name).is_file() for name in (*CACHE_FILES, "manifest.json"))

    def sizes(self) -> dict[str, int]:
        """Bytes per file. W8 item 4 exists because these were once estimated."""
        return {
            name: (self.path / name).stat().st_size
            for name in CACHE_FILES
            if (self.path / name).is_file()
        }


def cache_root() -> Path:
    """``<data root>/cache/vox``, created if missing."""
    path = data_root() / "cache" / "vox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def entry_for(spec: SceneSpec) -> CacheEntry:
    """The cache entry for ``spec``, whether or not it has been computed."""
    key = spec.key
    path = cache_root() / key
    manifest_file = path / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    return CacheEntry(path=path, key=key, manifest=manifest)


def voxelise(spec: SceneSpec, *, nprocs: int | None = None, force: bool = False) -> CacheEntry:
    """Voxelise ``spec`` locally and publish it to the cache, or reuse it.

    Publication is atomic: the child writes into a sibling directory and the
    entry is renamed into place only once every file is there, so an interrupted
    run leaves no half voxelisation for a later run to trust.
    """
    entry = entry_for(spec)
    if entry.complete and not force:
        return entry

    root = cache_root()
    staging = root / f".{spec.key}.partial.{os.getpid()}"
    if staging.exists():
        _remove_tree(staging)
    staging.mkdir(parents=True)

    job = {
        "pffdtd_dir": str(pffdtd_dir()),
        "out_dir": str(staging),
        "model_json": str(Path(spec.model_json).resolve()),
        "mat_folder": str(Path(spec.mat_folder).resolve()),
        "mat_files": dict(spec.mat_files),
        "fmax": spec.fmax,
        "ppw": spec.ppw,
        "Tc": spec.tc,
        "rh": spec.rh,
        "fcc": spec.fcc,
        "bmin": list(spec.bmin) if spec.bmin else None,
        "bmax": list(spec.bmax) if spec.bmax else None,
        "rot_az_el": list(spec.rot_az_el),
        "nprocs": nprocs,
        "compress": None,
    }

    child = Path(__file__).with_name("_child_voxelise.py")
    started = time.time()
    completed = subprocess.run(
        [pffdtd_python(), str(child)],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        check=False,
    )
    wall_s = time.time() - started
    (staging / "voxelise.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"voxelisation failed, see {staging / 'voxelise.log'}\n"
            + completed.stderr.strip()[-2000:]
        )

    report = _parse_report(completed.stdout)
    report.update(
        {
            "key": spec.key,
            "model_json": str(spec.model_json),
            "model_sha256": hashlib.sha256(Path(spec.model_json).read_bytes()).hexdigest(),
            "fmax": spec.fmax,
            "ppw": spec.ppw,
            "Tc": spec.tc,
            "rh": spec.rh,
            "fcc": spec.fcc,
            "materials": dict(spec.mat_files),
            "wall_s": round(wall_s, 3),
            "pffdtd_dir": str(pffdtd_dir()),
            "pffdtd_commit": _git_commit(pffdtd_dir()),
            "file_bytes": {
                name: (staging / name).stat().st_size
                for name in CACHE_FILES
                if (staging / name).is_file()
            },
        }
    )
    (staging / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    missing = [name for name in CACHE_FILES if not (staging / name).is_file()]
    if missing:
        raise RuntimeError(f"voxelisation produced no {missing}, refusing to publish")

    destination = root / spec.key
    if destination.exists():
        _remove_tree(destination)
    staging.rename(destination)
    return entry_for(spec)


def engine_inputs(entry: CacheEntry, comms_out: Path) -> list[Path]:
    """The four files the engine reads, and no fifth.

    ``comms_out.h5`` comes from :func:`reverberate.wave.comms.write_comms`, the
    other three from the cache. ``cart_grid.h5`` is not among them; the engine
    has never read it, and shipping it would only pay for bandwidth.
    """
    paths = []
    for name in ENGINE_FILES:
        paths.append(comms_out if name == "comms_out.h5" else entry.path / name)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing engine inputs: {missing}")
    return paths


def _parse_report(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_REPORT_MARKER):
            return dict(json.loads(line[len(_REPORT_MARKER) :]))
    raise RuntimeError("the voxelisation child printed no report")


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
