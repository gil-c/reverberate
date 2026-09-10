"""Tests for sharing an assembled apartment through the object store.

The property under test is the one the pools exist for: a scene carries only
what is its own, the meshes it shares are sent once for the whole dataset, and
a fetch either produces a whole apartment or none at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from reverberate.geometry import collider_cache
from reverberate.geometry.carve import CarveResult
from reverberate.settings import DATA_ROOT_ENV
from reverberate.store import MemoryStore
from reverberate.viz import scene_pool, scene_store
from reverberate.viz.scene_cache import SceneEntry, cache_root


@pytest.fixture(autouse=True)
def _own_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "data"))
    collider_cache.stamp.cache_clear()


def _build_entry(key: str, templates: list[str]) -> SceneEntry:
    """A cache entry shaped exactly as an assembly leaves one."""
    path = cache_root() / key
    path.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(1, 1, 1))
    instances = []
    for template in templates:
        render = scene_pool.render_pool() / f"{template}.glb"
        render.write_bytes(b"render " + template.encode())
        scene_pool.link_into(path, scene_pool.ASSET_DIR, f"{template}.glb", render)
        collider_cache.store_entry(
            template, box, merged=True, carve=CarveResult(mesh=box, carved=False)
        )
        mesh_file, _ = collider_cache.entry_paths(template)
        scene_pool.link_into(path, scene_pool.SIM_DIR, f"{template}.glb", mesh_file)
        instances.append(
            {
                "template": template,
                "render_url": f"{scene_pool.ASSET_DIR}/{template}.glb",
                "collider_url": f"{scene_pool.SIM_DIR}/{template}.glb",
            }
        )
    (path / "manifest.json").write_text(json.dumps({"local": "hssd_0001", "instances": instances}))
    (path / "shell_colour.glb").write_bytes(b"colour")
    (path / "shell_label.glb").write_bytes(b"label")
    (path / "entry.json").write_text(json.dumps({"key": key, "summary": "two pieces"}))
    return SceneEntry(path=path, key=key, manifest={"key": key, "summary": "two pieces"})


def test_a_published_scene_comes_back_whole_on_a_machine_without_the_dataset() -> None:
    store = MemoryStore()
    entry = _build_entry("aaa", ["sofa", "lamp"])
    scene_store.publish_entry(store, entry)

    # A second machine: same code, so the same key, but nothing on disk.
    import shutil

    shutil.rmtree(entry.path)
    shutil.rmtree(scene_pool.render_pool())
    shutil.rmtree(collider_cache.cache_root())

    fetched = scene_store.fetch_entry(store, "aaa")
    assert fetched is not None
    assert fetched.complete
    manifest = json.loads((fetched.path / "manifest.json").read_text())
    assert manifest["local"] == "hssd_0001"
    assert (fetched.path / "assets" / "sofa.glb").read_bytes() == b"render sofa"
    assert (fetched.path / "sim" / "lamp.glb").is_file()


def test_a_shared_mesh_is_sent_once_for_the_whole_dataset() -> None:
    store = MemoryStore()
    scene_store.publish_entry(store, _build_entry("aaa", ["sofa", "lamp"]))
    written = list(store.written)
    scene_store.publish_entry(store, _build_entry("bbb", ["sofa", "chair"]))
    added = [key for key in store.written[len(written) :] if "scene_pool" in key]
    assert not any("sofa" in key for key in added), "the sofa was sent twice"
    assert any("chair" in key for key in added)


def test_a_half_published_scene_is_not_served() -> None:
    """``entry.json`` is written last, so a scene without it is not a scene."""
    store = MemoryStore()
    entry = _build_entry("aaa", ["sofa"])
    scene_store.publish_entry(store, entry)
    del store.objects[f"reverberate/{scene_store.remote_prefix('aaa')}entry.json"]
    assert scene_store.fetch_entry(store, "aaa") is None


def test_a_machine_without_the_dataset_opens_a_storey_from_the_catalogue(tmp_path: Path) -> None:
    """The key hashes the scene's own files, so without HSSD it must be looked up.

    Fetching by key worked from the start; the viewer asks by name, and on a
    machine without the dataset it could compute no key and list no apartment.
    """
    from reverberate.viz import scene_cache

    store = MemoryStore()
    entry = _build_entry("aaa", ["sofa"])
    scene_store.publish_entry(store, entry)
    scene_store.publish_index(
        store, [{"local": "hssd_0001", "scene_id": "102343992", "storey_index": 0, "key": "aaa"}]
    )
    import shutil

    shutil.rmtree(entry.path)
    scene_cache._CATALOGUES.clear()

    opened = scene_cache.ensure_scene(tmp_path / "no-hssd", "102343992", storey=0, store=store)
    assert opened.complete
    assert opened.key == "aaa"
    again = scene_cache.ensure_scene(tmp_path / "no-hssd", "102343992", storey=0, store=store)
    assert again.path == opened.path, "a second open is served from disk"


def test_the_viewer_lists_the_catalogue_when_the_dataset_is_absent(tmp_path: Path) -> None:
    from reverberate.viz.serve_room import list_apartments

    store = MemoryStore()
    scene_store.publish_index(
        store,
        [
            {"local": "hssd_0002", "scene_id": "102344022", "storey_index": 0, "key": "k2"},
            {"local": "hssd_0001_2", "scene_id": "102343992", "storey_index": 1, "key": "k1"},
        ],
    )
    listed = list_apartments(tmp_path / "no-hssd", first="hssd_0002", store=store)
    assert [a["local"] for a in listed] == ["hssd_0002", "hssd_0001_2"]
    assert listed[1]["storey_index"] == "1"
