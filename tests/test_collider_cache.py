"""Tests for the per-template mesh pool.

The property under test is that the pool answers only for the code that made
it, and that what it hands back is what the uncached path would have produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from reverberate.geometry import collider_cache
from reverberate.geometry.carve import CarveResult
from reverberate.settings import DATA_ROOT_ENV


@pytest.fixture(autouse=True)
def _own_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    collider_cache.stamp.cache_clear()


def _result(mesh: trimesh.Trimesh) -> CarveResult:
    return CarveResult(mesh=mesh, carved=True, reason="", collider_volume=8.0, carved_volume=7.0)


def test_an_entry_comes_back_with_its_mesh_and_its_carve() -> None:
    box = trimesh.creation.box(extents=(1, 1, 1))
    collider_cache.store_entry("abc", box, merged=True, carve=_result(box))
    loaded = collider_cache.load_entry("abc")
    assert loaded is not None
    assert loaded.merged is True
    assert len(loaded.mesh.faces) == len(box.faces)
    assert loaded.carve.carved is True
    assert loaded.carve.collider_volume == pytest.approx(8.0)
    assert loaded.carve.shrink == pytest.approx(7.0 / 8.0)


def test_a_rule_change_does_not_serve_the_old_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stamp is in the filename, so two rules cannot be the same file.

    This is the failure the carve cache was already bitten by once: an entry
    written by a broken version answered for the fixed one, and the repair
    looked like it had not worked.
    """
    box = trimesh.creation.box(extents=(1, 1, 1))
    collider_cache.store_entry("abc", box, merged=True, carve=_result(box))
    before, _ = collider_cache.entry_paths("abc")

    collider_cache.stamp.cache_clear()
    monkeypatch.setattr(collider_cache, "SOURCES", ("geometry/carve.py",))
    after, _ = collider_cache.entry_paths("abc")

    assert before != after
    assert collider_cache.load_entry("abc") is None
    assert before.is_file(), "the old entry is kept, not deleted"


def test_the_stamp_names_a_file_rather_than_becoming_its_extension() -> None:
    """``with_suffix`` would eat the stamp and collapse every rule into one file."""
    mesh_file, record_file = collider_cache.entry_paths("abc")
    assert mesh_file.name == f"abc.{collider_cache.stamp()}.glb"
    assert record_file.name == f"abc.{collider_cache.stamp()}.json"


def test_a_truncated_entry_is_rebuilt_rather_than_read() -> None:
    """An interrupted copy left a zero-byte mesh beside a good record."""
    box = trimesh.creation.box(extents=(1, 1, 1))
    collider_cache.store_entry("abc", box, merged=True, carve=_result(box))
    mesh_file, _ = collider_cache.entry_paths("abc")
    mesh_file.write_bytes(b"")
    assert collider_cache.load_entry("abc") is None
