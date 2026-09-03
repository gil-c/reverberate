"""Keep an assembled apartment on disk, so the viewer pays for it once.

Assembling one storey used to cost about 140 seconds, over 99 per cent of it
spent fitting an acoustic envelope to each of the 231 pieces of furniture. That
search is gone, so assembly is now dominated by parsing and exporting the
colliders instead. It is still repeated on every start unless it is kept,
because the served site is built into a ``TemporaryDirectory`` that goes away
with the process.

So an assembled scene is cached the way a voxelisation already is, and
deliberately in the same shape: ``<data root>/cache/<name>/<key>/`` holding the
artefacts plus a ``manifest.json`` of what produced them, written through a
staging directory and renamed into place, so an interrupted assembly leaves
nothing a later run would trust. See :mod:`reverberate.wave.voxelise`, whose
mechanism this mirrors.

**What the key covers.** The scene's own two source files, byte for byte, the
seed that pins the sampling, and the source of every module that shapes the
result. That last part is what makes the cache safe to keep across edits:
change how a material is chosen, or a collider placed, and the key changes
with it, so the old entry is neither served nor deleted. The HSSD assets
themselves are not hashed -- reading a hundred megabytes of glTF to decide
whether to skip two minutes of geometry would give a good part of the saving
back -- so a dataset edited in place is the one thing this cache would not
notice.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from reverberate.settings import data_root
from reverberate.viz.scene_manifest import write_manifest

__all__ = [
    "CACHE_NAME",
    "SOURCES",
    "SceneEntry",
    "cache_root",
    "code_digest",
    "ensure_scene",
    "entry_for",
    "scene_key",
]

#: Subdirectory of the data root's cache holding assembled scenes.
CACHE_NAME = "scenes"

#: The modules whose source decides what an assembled scene looks like. A file
#: listed here that stops existing is a mistake worth failing on, not a hash to
#: quietly skip: it would mean the key no longer covers the code it claims to.
SOURCES = (
    "acoustics.py",
    "geometry/apartment.py",
    "geometry/hssd_assets.py",
    "geometry/hssd_room.py",
    "geometry/materials.py",
    "geometry/sim_geometry.py",
    "viz/label_palette.py",
    "viz/room_surfaces.py",
    "viz/scene_manifest.py",
)


def cache_root() -> Path:
    """``<data root>/cache/scenes``, created if missing."""
    path = data_root() / "cache" / CACHE_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_digest() -> str:
    """A hash over the source of every module in :data:`SOURCES`."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def scene_key(hssd_root: Path, scene_id: str, seed: int = 0) -> str:
    """Everything an assembled scene depends on, and nothing else."""
    digest = hashlib.sha256()
    for path in (
        hssd_root / "scenes" / f"{scene_id}.scene_instance.json",
        hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json",
    ):
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            {
                "scene_id": scene_id,
                "seed": seed,
                "code": code_digest(),
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class SceneEntry:
    """An assembled scene on disk, ready to be served."""

    path: Path
    key: str
    manifest: dict[str, object]

    @property
    def complete(self) -> bool:
        """Whether the entry holds both the scene and the record of its build."""
        return (self.path / "manifest.json").is_file() and (self.path / "entry.json").is_file()

    def summary(self) -> str:
        """The assembly report, as :meth:`ManifestReport.summary` printed it."""
        return str(self.manifest.get("summary", ""))

    def storey(self) -> str:
        """The storey line of the report."""
        return str(self.manifest.get("storey", ""))


def entry_for(hssd_root: Path, scene_id: str, seed: int = 0) -> SceneEntry:
    """The cache entry for this scene, whether or not it has been assembled."""
    key = scene_key(hssd_root, scene_id, seed)
    path = cache_root() / key
    record = path / "entry.json"
    manifest = json.loads(record.read_text()) if record.is_file() else {}
    return SceneEntry(path=path, key=key, manifest=manifest)


def ensure_scene(
    hssd_root: Path,
    scene_id: str,
    seed: int = 0,
    force: bool = False,
) -> SceneEntry:
    """Assemble ``scene_id`` into the cache, or reuse what is already there.

    Publication is atomic: the assembly writes into a sibling staging directory
    and is renamed into place only once its record is written, so a run killed
    mid-assembly cannot leave a half scene that :attr:`SceneEntry.complete`
    would accept.
    """
    entry = entry_for(hssd_root, scene_id, seed)
    if entry.complete and not force:
        return entry

    root = cache_root()
    staging = root / f".{entry.key}.partial.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        report = write_manifest(hssd_root, scene_id, staging)
        record = {
            "scene_id": scene_id,
            "key": entry.key,
            "seed": seed,
            "summary": report.summary(),
            "storey": report.storey,
        }
        (staging / "entry.json").write_text(json.dumps(record, indent=2))
        # Another process may have finished the same scene while this one was
        # working. Its entry is byte for byte what this one would publish, so
        # the loser of the race discards its own work rather than replacing a
        # directory a server may already be serving files out of.
        if entry.path.exists():
            shutil.rmtree(staging)
        else:
            staging.rename(entry.path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return entry_for(hssd_root, scene_id, seed)
