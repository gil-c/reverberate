"""Tests for the viewer server, focused on how runs reach the selector.

Assembling an apartment needs the HSSD dataset, so that path is exercised
elsewhere. What is worth pinning here is the join between a rendered run and the
apartment it belongs to, because getting it wrong makes a run invisible rather
than broken, and an invisible feature fails quietly.
"""

from __future__ import annotations

import socketserver
from pathlib import Path

from reverberate.viz import serve_room
from reverberate.viz.app_payload import WalkRun
from reverberate.viz.serve_room import attach_runs


def _run(name: str, scene_id: str) -> WalkRun:
    return WalkRun(
        name=name, path=Path(name), scene_id=scene_id, dwelling="", room="bedroom", sources=[]
    )


def test_an_apartment_is_told_about_its_own_runs_only() -> None:
    apartments = [
        {"local": "hssd_0002", "scene_id": "102344022"},
        {"local": "x", "scene_id": "999"},
    ]

    attached = attach_runs(apartments, [_run("w20_first_listen", "102344022")])

    assert attached[0]["runs"] == ["w20_first_listen"]
    assert attached[1]["runs"] == []


def test_a_run_for_an_apartment_not_in_the_dataset_is_dropped_not_invented() -> None:
    """The selector offers apartments; a run cannot conjure one into the list."""
    attached = attach_runs(
        [{"local": "hssd_0007", "scene_id": "7"}], [_run("orphan", "not-a-scene")]
    )

    assert [a["local"] for a in attached] == ["hssd_0007"]
    assert attached[0]["runs"] == []


def test_scene_ids_compare_as_text_even_when_the_dataset_offers_numbers() -> None:
    """HSSD ids look numeric; a plan may hold either, and a type mismatch here
    silently hides every run."""
    attached = attach_runs(
        [{"local": "hssd_0002", "scene_id": 102344022}], [_run("w20", "102344022")]
    )

    assert attached[0]["runs"] == ["w20"]


def test_the_server_is_threaded_so_an_assembly_cannot_freeze_the_page() -> None:
    """Assembling an apartment takes seconds and must not block everything else.

    On a single threaded server the audio of the solver mode stops mid playback
    the moment someone switches apartment, because the WAV request queues behind
    the assembly. The builder's own lock still serialises the assembly itself.
    """
    from reverberate.viz import serve_room

    assert issubclass(serve_room._Server, socketserver.ThreadingTCPServer)
    assert serve_room._Server.daemon_threads is True
    assert serve_room._Server.allow_reuse_address is True


def test_the_server_forbids_caching_so_an_edit_cannot_be_invisible() -> None:
    """A cached ES module runs code that is no longer anywhere on disk.

    The site is rebuilt from the source tree on every start, so a browser
    holding the previous module reports the behaviour of an older revision and
    makes an edit look as though it did nothing.
    """
    source = Path(serve_room.__file__).read_text()

    assert '"Cache-Control", "no-store"' in source


def test_a_range_request_returns_exactly_those_bytes(tmp_path: Path) -> None:
    """The page reads one cell of a field at a time straight out of the HDF5,
    so the server has to honour Range, which the standard handler does not."""
    import threading
    import urllib.request

    from reverberate.viz.serve_room import _handler_for, _Server

    (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 4)
    builder = type("Builder", (), {"target": tmp_path, "ensure": lambda self, s: None})()
    server = _Server(("127.0.0.1", 0), _handler_for(builder))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/blob.bin", headers={"Range": "bytes=300-303"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 300-303/1024"
            assert response.read() == bytes([44, 45, 46, 47])
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/blob.bin", headers={"Range": "bytes=2000-"}
        )
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
    from reverberate.viz.serve_room import _bind, _handler_for, _Server

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
