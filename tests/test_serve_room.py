"""Tests for the viewer server: how runs reach the selector, and how a field is read.

Assembling an apartment needs the HSSD dataset, so that path is exercised
elsewhere. What is worth pinning here is the join between a run and its
apartment, because getting it wrong makes a run invisible rather than broken,
and the two things the standard handler does not do: byte ranges and no cache.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path

from reverberate.viz.app_payload import WalkRun
from reverberate.viz.serve_room import _bind, _handler_for, _Server, attach_runs


def _run(name: str, scene_id: str) -> WalkRun:
    return WalkRun(name=name, path=Path(name), scene_id=scene_id, dwelling="", sources=[])


def test_an_apartment_is_told_about_its_own_runs_and_the_list_is_ordered() -> None:
    """HSSD ids look numeric and a plan may hold either type: a mismatch here
    silently hides every run. A run of a scene not in the dataset conjures no
    apartment. The first apartment is the one asked for, then those with a run."""
    apartments: list[dict[str, object]] = [
        {"local": "x", "scene_id": "999"},
        {"local": "hssd_0002", "scene_id": 102344022},
        {"local": "hssd_0007", "scene_id": "7"},
    ]
    attached = attach_runs(
        apartments, [_run("w20", "102344022"), _run("orphan", "not-a-scene")], first="hssd_0007"
    )

    assert [a["local"] for a in attached] == ["hssd_0007", "hssd_0002", "x"]
    assert [a["runs"] for a in attached] == [[], ["w20"], []]


def _serve(target: Path) -> _Server:
    builder = type("Builder", (), {"target": target, "ensure": lambda self, s: None})()
    server = _Server(("127.0.0.1", 0), _handler_for(builder))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_range_request_returns_exactly_those_bytes_and_nothing_is_cached(tmp_path: Path) -> None:
    """The page reads one cell of a field at a time straight out of the HDF5,
    so the server has to honour Range. And the site is rebuilt from the source
    tree on every start, so a browser must never keep a module: a cached one
    runs code that is no longer on disk and makes an edit look inert."""
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 4)
    server = _serve(tmp_path)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/blob.bin"
        request = urllib.request.Request(url, headers={"Range": "bytes=300-303"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 300-303/1024"
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == bytes([44, 45, 46, 47])
        request = urllib.request.Request(url, headers={"Range": "bytes=2000-"})
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 416
        else:
            raise AssertionError("a range past the end must be refused")
    finally:
        server.shutdown()
        server.server_close()


def test_a_taken_port_moves_the_server_to_the_next_free_one(tmp_path: Path) -> None:
    """Several sessions run a viewer at once; a port in use is not a reason to stop."""
    builder = type("Builder", (), {"target": tmp_path, "ensure": lambda self, s: None})()
    first = _Server(("127.0.0.1", 0), _handler_for(builder))
    taken = first.server_address[1]
    try:
        second = _bind(builder, taken, tries=3)
        try:
            assert second.server_address[1] == taken + 1
        finally:
            second.server_close()
    finally:
        first.server_close()
