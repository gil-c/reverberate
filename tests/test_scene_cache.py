"""Tests for the assembled-scene cache.

Assembling a real apartment costs minutes, so these tests never do it: the
cache's job is to decide *whether* to assemble, and that decision is what is
checked here, with a counting stand-in for the assembly itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reverberate.settings import DATA_ROOT_ENV
from reverberate.viz import scene_cache
from reverberate.viz.scene_manifest import ManifestReport

SCENE = "102344022"


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the cache at a scratch data root, never the real one."""
    root = tmp_path / "data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    return root


def build_scene_files(root: Path, body: str = "{}") -> Path:
    """The two files the key is taken over. Their content is never parsed here."""
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    (root / "semantics" / "scenes").mkdir(parents=True, exist_ok=True)
    (root / "scenes" / f"{SCENE}.scene_instance.json").write_text(body)
    (root / "semantics" / "scenes" / f"{SCENE}.semantic_config.json").write_text("{}")
    return root


def counting_assembly(monkeypatch: pytest.MonkeyPatch, calls: list[Path]) -> None:
    """Stand in for ``write_manifest``, recording where it was asked to write."""

    def fake(hssd_root: Path, scene_id: str, target: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps({"title": scene_id}))
        report = ManifestReport(placed=1)
        report.storey = "one storey"
        return report

    monkeypatch.setattr(scene_cache, "write_manifest", fake)


def test_a_second_start_reuses_the_assembly_instead_of_repeating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: 140 seconds of geometry is paid for once."""
    hssd = build_scene_files(tmp_path / "hssd")
    calls: list[Path] = []
    counting_assembly(monkeypatch, calls)

    first = scene_cache.ensure_scene(hssd, SCENE)
    second = scene_cache.ensure_scene(hssd, SCENE)

    assert len(calls) == 1
    assert first.key == second.key
    assert second.complete
    assert second.summary() == ManifestReport(placed=1).summary()
    assert second.storey() == "one storey"


def test_force_reassembles_even_though_the_entry_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hssd = build_scene_files(tmp_path / "hssd")
    calls: list[Path] = []
    counting_assembly(monkeypatch, calls)

    scene_cache.ensure_scene(hssd, SCENE)
    scene_cache.ensure_scene(hssd, SCENE, force=True)

    assert len(calls) == 2


def test_an_edited_scene_is_not_served_its_old_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hssd = build_scene_files(tmp_path / "hssd")
    calls: list[Path] = []
    counting_assembly(monkeypatch, calls)

    before = scene_cache.ensure_scene(hssd, SCENE).key
    build_scene_files(hssd, body='{"moved": true}')
    after = scene_cache.ensure_scene(hssd, SCENE).key

    assert before != after
    assert len(calls) == 2


def test_changing_the_geometry_code_changes_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry must not outlive the code that produced it.

    Editing how an envelope is chosen changes what an assembly would return,
    and a cache that ignored that would serve a stale room for as long as the
    scene files stayed put -- silently, which is the worst way to be wrong.
    """
    hssd = build_scene_files(tmp_path / "hssd")
    before = scene_cache.scene_key(hssd, SCENE, 0.1)
    monkeypatch.setattr(scene_cache, "code_digest", lambda: "a different build")
    assert scene_cache.scene_key(hssd, SCENE, 0.1) != before


def test_the_detail_length_is_part_of_the_key(tmp_path: Path) -> None:
    """The viewer asks for the finest rung; a coarser one is a different scene."""
    hssd = build_scene_files(tmp_path / "hssd")
    assert scene_cache.scene_key(hssd, SCENE, 0.1) != scene_cache.scene_key(hssd, SCENE, 0.2)


def test_an_interrupted_assembly_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed run must not leave a half scene a later run would trust."""
    hssd = build_scene_files(tmp_path / "hssd")

    def explode(hssd_root: Path, scene_id: str, target: Path, *args: Any, **kwargs: Any) -> Any:
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text("{}")
        raise RuntimeError("killed mid assembly")

    monkeypatch.setattr(scene_cache, "write_manifest", explode)

    with pytest.raises(RuntimeError):
        scene_cache.ensure_scene(hssd, SCENE)

    entry = scene_cache.entry_for(hssd, SCENE, 0.1)
    assert not entry.complete
    # Nothing at all is left behind, staging directories included.
    assert list(scene_cache.cache_root().iterdir()) == []
