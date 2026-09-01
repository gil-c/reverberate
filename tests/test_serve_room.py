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
from reverberate.viz.run_view import RunRef
from reverberate.viz.serve_room import attach_runs


def _run(name: str, scene_id: str) -> RunRef:
    return RunRef(name=name, scene_id=scene_id, room="bedroom.001", path=Path(name))


def test_an_apartment_is_told_about_its_own_runs_only() -> None:
    apartments = [{"id": "102344022", "label": "a"}, {"id": "999", "label": "b"}]

    attached = attach_runs(apartments, [_run("w20_first_listen", "102344022")])

    assert attached[0]["runs"] == ["w20_first_listen"]
    assert attached[1]["runs"] == []


def test_several_runs_of_one_apartment_are_all_offered() -> None:
    attached = attach_runs(
        [{"id": "7", "label": "a"}],
        [_run("first", "7"), _run("second", "7")],
    )

    assert attached[0]["runs"] == ["first", "second"]


def test_a_run_for_an_apartment_not_in_the_dataset_is_dropped_not_invented() -> None:
    """The selector offers apartments; a run cannot conjure one into the list."""
    attached = attach_runs([{"id": "7", "label": "a"}], [_run("orphan", "not-a-scene")])

    assert [a["id"] for a in attached] == ["7"]
    assert attached[0]["runs"] == []


def test_scene_ids_compare_as_text_even_when_the_dataset_offers_numbers() -> None:
    """HSSD ids look numeric; a plan may hold either, and a type mismatch here
    silently hides every run."""
    attached = attach_runs([{"id": 102344022, "label": "a"}], [_run("w20", "102344022")])

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
