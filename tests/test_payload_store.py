"""Tests for sharing the audit payload.

The payload is derived and could always be rebuilt, so the thing worth pinning
is not that it travels but that a *partial* one never looks whole. W29's grid
went with the worktree that made it; a payload half-fetched and mistaken for a
whole one is the same failure with an extra step, because the reader would be
auditing a flat with rooms silently missing from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverberate.store import PREFIX, MemoryStore
from reverberate.viz.payload_store import (
    INDEX,
    fetch_payload,
    publish_payload,
    remote_prefix,
)


def write_payload(root: Path, rooms: int = 2) -> Path:
    """A payload shaped like `audit_view`'s, small enough to reason about."""
    root.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"total_nodes": 42, "rooms": []}
    listing: list[dict[str, object]] = []
    for index in range(rooms):
        name = f"room{index}"
        (root / name).mkdir()
        tiers = {}
        for tier in ("fine", "coarse"):
            for suffix, payload in (
                (f"{tier}.f32", b"corners" + bytes([index])),
                (f"{tier}_index.u32", b"index" + bytes([index])),
                (f"{tier}_label.i16", b"label" + bytes([index])),
            ):
                (root / name / suffix).write_bytes(payload)
            tiers[tier] = {
                "tiles": [
                    {
                        "corners_url": f"{tier}.f32",
                        "index_url": f"{tier}_index.u32",
                        "label_url": f"{tier}_label.i16",
                    }
                ]
            }
        listing.append({"name": name, "dir": name, **tiers})
    record["rooms"] = listing
    (root / INDEX).write_text(json.dumps(record))
    return root


def test_a_payload_survives_the_round_trip(tmp_path: Path) -> None:
    store = MemoryStore()
    source = write_payload(tmp_path / "out" / "voxels")

    publish_payload(store, "w34_audit_16k", source)
    landed = fetch_payload(store, "w34_audit_16k", tmp_path / "elsewhere" / "voxels")

    assert landed is not None
    for path in source.rglob("*"):
        if path.is_file():
            assert (landed / path.relative_to(source)).read_bytes() == path.read_bytes()


def test_a_run_the_store_has_never_heard_of_is_not_an_error(tmp_path: Path) -> None:
    """A checkout without the payload draws the triangles and says so; it does
    not fail to build a page."""
    assert fetch_payload(MemoryStore(), "never_published", tmp_path / "voxels") is None
    assert not (tmp_path / "voxels").exists()


def test_the_index_is_written_last(tmp_path: Path) -> None:
    """It is what `_link_audit` looks for, so a payload advertising itself
    before its tiles arrived would be read as complete."""
    store = MemoryStore()
    source = write_payload(tmp_path / "voxels")

    publish_payload(store, "run", source)

    keys = list(store.list(remote_prefix("run")))
    assert keys[-1].endswith(INDEX)


def test_a_fetch_that_fails_leaves_nothing_behind(tmp_path: Path) -> None:
    """Half a flat is worse than none: the reader would audit a scene with
    rooms missing and no way to tell."""
    store = MemoryStore()
    source = write_payload(tmp_path / "voxels")
    publish_payload(store, "run", source)
    # A tile the store forgot, which is what a pruned bucket looks like.
    del store.objects[f"{PREFIX}runs/run/voxels/room1/fine.f32"]

    destination = tmp_path / "pulled" / "voxels"
    with pytest.raises(Exception):  # noqa: B017 - any store error must clean up
        fetch_payload(store, "run", destination)

    assert not destination.exists()
    assert not list(destination.parent.glob(".*.fetch"))


def test_publishing_twice_does_not_re_send_the_tiles(tmp_path: Path) -> None:
    """A gigabyte and a half a run, and the roadmap puts a terabyte at 22 hours."""
    store = MemoryStore()
    source = write_payload(tmp_path / "voxels")
    publish_payload(store, "run", source)
    before = dict(store.objects)

    publish_payload(store, "run", source)

    assert store.objects.keys() == before.keys()


def test_it_refuses_to_publish_a_directory_with_no_index(tmp_path: Path) -> None:
    (tmp_path / "voxels").mkdir()
    with pytest.raises(ValueError, match="no rooms.json"):
        publish_payload(MemoryStore(), "run", tmp_path / "voxels")
