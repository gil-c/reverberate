"""Tests for the driver that pays for the whole dataset once."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from reverberate.geometry import collider_cache
from reverberate.settings import DATA_ROOT_ENV
from reverberate.viz.assemble_dataset import (
    MIN_WARM_WORKERS,
    _pending,
    _warm,
    every_storey,
    every_template,
    merge_catalogue,
    scene_files,
)


def _dataset(tmp_path: Path, placements: dict[str, list[str]]) -> Path:
    """A dataset whose every placed template resolves to a render mesh."""
    (tmp_path / "scenes").mkdir(parents=True)
    for name, templates in placements.items():
        (tmp_path / "scenes" / f"{name}.scene_instance.json").write_text(
            json.dumps({"object_instances": [{"template_name": t} for t in templates]})
        )
        for template in templates:
            shard = tmp_path / "objects" / template[0]
            shard.mkdir(parents=True, exist_ok=True)
            (shard / f"{template}.glb").write_bytes(b"")
    return tmp_path


def test_every_storey_is_named_once_and_numbered_from_the_ground() -> None:
    rows = every_storey()
    assert len(rows) == 176, "168 scenes, six with two storeys and one with three"
    assert len({row.local for row in rows}) == 176
    by_scene: dict[str, list[str]] = {}
    for row in rows:
        by_scene.setdefault(row.scene_id, []).append(row.local)
    for names in by_scene.values():
        if len(names) == 1:
            assert re.fullmatch(r"hssd_\d{4}", names[0])
        else:
            base = names[0].rpartition("_")[0]
            assert sorted(names) == [f"{base}_{n}" for n in range(1, len(names) + 1)]


def test_the_shards_are_disjoint_and_cover_everything(tmp_path: Path) -> None:
    """Two machines given half the scenes would both carve the shared sofa."""
    root = _dataset(tmp_path, {"one": ["sofa", "lamp", "chair"], "two": ["sofa", "desk"]})
    first, second = every_template(root, 0, 2), every_template(root, 1, 2)
    assert not set(first) & set(second)
    assert sorted(first + second) == ["chair", "desk", "lamp", "sofa"]


def test_an_appledouble_file_is_not_a_scene(tmp_path: Path) -> None:
    root = _dataset(tmp_path, {"one": ["sofa"]})
    (root / "scenes" / "._one.scene_instance.json").write_bytes(b"\x00\x05\x16\x07Mac OS X\xa3")
    assert [p.name for p in scene_files(root)] == ["one.scene_instance.json"]
    assert every_template(root) == ["sofa"]


def test_a_template_the_dataset_cannot_resolve_is_not_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "data"))
    collider_cache.stamp.cache_clear()
    root = _dataset(tmp_path / "hssd", {"one": ["sofa"]})
    assert _pending(["sofa", "ghost"], root) == ["sofa"]


def test_a_broken_pool_falls_back_to_fewer_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container memory cap that ``free`` does not show kills workers outright."""
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "data"))
    collider_cache.stamp.cache_clear()
    root = _dataset(tmp_path / "hssd", {"one": ["a", "b"]})
    attempts: list[int] = []

    class _Breaking:
        def __init__(self, max_workers: int) -> None:
            attempts.append(max_workers)

        def __enter__(self) -> _Breaking:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def submit(self, *_: object) -> Future[object]:
            future: Future[object] = Future()
            future.set_exception(BrokenProcessPool("killed"))
            return future

    monkeypatch.setattr("reverberate.viz.assemble_dataset.ProcessPoolExecutor", _Breaking)
    _warm(argparse.Namespace(hssd_root=root, warm="0/1", workers=32))
    assert attempts[:4] == [32, 16, 8, 4]
    assert min(attempts) == MIN_WARM_WORKERS


def test_a_partial_run_adds_to_the_catalogue_and_never_shrinks_it() -> None:
    """Publishing one apartment once shrank the catalogue from 175 storeys to 1."""
    existing: list[dict[str, object]] = [
        {"local": "hssd_0002", "key": "old"},
        {"local": "hssd_0001_1", "key": "k"},
    ]
    fresh: list[dict[str, object]] = [
        {"local": "hssd_0002", "key": "new"},
        {"local": "hssd_0067", "key": "k67"},
    ]
    merged = merge_catalogue(existing, fresh)
    assert [r["local"] for r in merged] == ["hssd_0001_1", "hssd_0002", "hssd_0067"]
    assert {r["local"]: r["key"] for r in merged}["hssd_0002"] == "new"
