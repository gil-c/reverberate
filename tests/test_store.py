"""Tests for the object store and the cache sharing built on it.

Offline by construction, per the roadmap's second hard constraint: the fake is
:class:`reverberate.store.MemoryStore`, the same shape as the real client, so
these exercise the real call sequence rather than mocking out the code under
test. One marked test talks to the live bucket and is excluded from the default
run.

The properties worth constraining are the two the module exists for: **a
truncated upload must never become readable under its real key**, and **the
lookup order must be local, then remote, then compute**.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from reverberate import store as store_module
from reverberate.store import (
    PREFIX,
    STAGING,
    B2Store,
    MemoryStore,
    StoreError,
    digest_of_bytes,
    digest_of_file,
)
from reverberate.wave.vox_store import fetch_entry, publish_entry, remote_prefix
from reverberate.wave.voxelise import CACHE_FILES, CacheEntry


def test_put_and_get_round_trip() -> None:
    store = MemoryStore()
    digest = store.put_bytes("ir/abc/response.h5", b"payload")
    assert digest == digest_of_bytes(b"payload")
    assert store.get_bytes("ir/abc/response.h5") == b"payload"
    assert store.exists("ir/abc/response.h5")
    assert not store.exists("ir/abc/missing")


def test_every_written_key_sits_under_the_project_prefix() -> None:
    """The bucket is shared, so a key outside the prefix is a defect."""
    store = MemoryStore()
    store.put_bytes("vox/deadbeef/vox_out.h5", b"x")
    assert all(key.startswith(PREFIX) for key in store.written)


def test_staging_is_used_and_then_cleared() -> None:
    store = MemoryStore()
    store.put_bytes("ir/abc/response.h5", b"payload")
    assert any(key.startswith(STAGING) for key in store.written)
    assert not any(key.startswith(STAGING) for key in store.objects)


def test_ranged_read_returns_exactly_the_window() -> None:
    """How a clip is pulled out of a multi hundred megabyte zip shard."""
    store = MemoryStore()
    store.put_bytes("shard.zip", bytes(range(256)))
    assert store.get_range("shard.zip", 10, 4) == bytes([10, 11, 12, 13])
    with pytest.raises(StoreError):
        store.get_range("shard.zip", 254, 8)


def test_listing_hides_staging_and_strips_the_prefix() -> None:
    store = MemoryStore()
    store.put_bytes("vox/a/vox_out.h5", b"1")
    store.put_bytes("vox/b/vox_out.h5", b"2")
    store.objects[f"{STAGING}leftover"] = b"3"
    assert list(store.list("vox/")) == ["vox/a/vox_out.h5", "vox/b/vox_out.h5"]


def test_digest_of_file_matches_digest_of_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"a" * (2 << 20) + b"tail")
    assert digest_of_file(path) == digest_of_bytes(path.read_bytes())


class _TruncatingClient:
    """An S3 client that loses the tail of every upload.

    This is the failure the staging prefix exists for: without the size check a
    truncated object would sit under its real key and the cache would serve it
    for ever.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def put_object(self, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:  # noqa: N803
        self.objects[Key] = Body[:-1]

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return {"ContentLength": len(self.objects[Key]), "Metadata": {}}

    def copy_object(self, **kwargs: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("a short upload must never be promoted")

    def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        self.deleted.append(Key)
        self.objects.pop(Key, None)


def test_a_short_upload_is_never_promoted() -> None:
    client = _TruncatingClient()
    remote = B2Store(client=client, bucket="Clarify")
    with pytest.raises(StoreError, match="short upload"):
        remote.put_bytes("vox/a/vox_out.h5", b"the whole payload")
    assert client.deleted, "the truncated staging object must be removed"
    assert not any(key == f"{PREFIX}vox/a/vox_out.h5" for key in client.objects)


def _write_entry(root: Path, key: str) -> CacheEntry:
    path = root / key
    path.mkdir(parents=True)
    for index, name in enumerate(CACHE_FILES):
        (path / name).write_bytes(f"contents of {name}".encode() + bytes([index]))
    (path / "manifest.json").write_text(json.dumps({"key": key, "fmax": 4000.0}))
    return CacheEntry(path=path, key=key)


def test_a_cache_entry_survives_a_round_trip_through_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    root = tmp_path / "data" / "cache" / "vox"
    root.mkdir(parents=True)
    entry = _write_entry(root, "deadbeef")
    original = {name: (entry.path / name).read_bytes() for name in CACHE_FILES}

    store = MemoryStore()
    digests = publish_entry(store, entry)
    assert set(digests) == {*CACHE_FILES, "manifest.json"}
    assert store.exists(f"{remote_prefix('deadbeef')}vox_out.h5")

    for name in (*CACHE_FILES, "manifest.json"):
        (entry.path / name).unlink()
    entry.path.rmdir()

    fetched = fetch_entry(store, "deadbeef")
    assert fetched is not None
    assert fetched.complete
    assert fetched.manifest["fmax"] == 4000.0
    for name in CACHE_FILES:
        assert (fetched.path / name).read_bytes() == original[name]


def test_fetching_an_absent_entry_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    assert fetch_entry(MemoryStore(), "nothing-here") is None


def test_a_partially_published_entry_is_not_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half an entry in the store must read as no entry, not as a usable one."""
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    root = tmp_path / "data" / "cache" / "vox"
    root.mkdir(parents=True)
    entry = _write_entry(root, "halfway")
    store = MemoryStore()
    publish_entry(store, entry)
    del store.objects[f"{PREFIX}{remote_prefix('halfway')}sim_mats.h5"]
    assert fetch_entry(store, "halfway") is None


def test_publishing_an_incomplete_entry_is_refused(tmp_path: Path) -> None:
    entry = CacheEntry(path=tmp_path / "empty", key="empty")
    with pytest.raises(ValueError, match="incomplete"):
        publish_entry(MemoryStore(), entry)


def test_credentials_are_read_from_the_environment_and_named_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(store_module.BUCKET_ENV, raising=False)
    with pytest.raises(StoreError, match=store_module.BUCKET_ENV):
        B2Store(client=object())


@pytest.mark.slow
def test_the_live_bucket_answers() -> None:
    """Opt in, network. Excluded from ``make check`` by the ``slow`` marker.

    It exists because the sibling project's notes claim the S3 API rejects
    these credentials and only the native b2 backend works. That claim is
    stale, and an assertion is a better record of it than a paragraph.
    """
    from reverberate import auth

    auth.inject()
    if not os.environ.get(store_module.KEY_ID_ENV):
        pytest.skip("no B2 credentials in this environment")
    live = B2Store()
    assert any(True for _ in live.list("", shared=True))
