"""The audit's sound: the engine's stems in a cache, mixed and served, and held to the engine.

What the owner hears in the page must be the output of
:class:`reverberate.render.engine.Engine`. So everything here is compared
with an engine made for the comparison, past the cache: a stem's chunks put
end to end are a one-shot render of the source, byte for byte; the bytes an
HTTP client receives for a chunk hash to the engine's samples; a mix is the
stems summed. The rest is the cache's own behaviour: what its key moves
with, that a second listen renders nothing, which chunk is rendered first.

The workers are threads here (``processes=False``): the same code path as a
worker process without a second's start; one slow test starts a process.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from reverberate.render.benchmark import density_pack
from reverberate.render.engine import Engine
from reverberate.render.output import open_signal
from reverberate.render.pack import read_pack, write_pack
from reverberate.viz import audit_stems
from reverberate.viz.audit_api import AuditService, Binary, pack_tracks
from reverberate.viz.audit_demo import demo_pack
from reverberate.viz.audit_dry import DrySources, dry_plan, dry_track
from reverberate.viz.audit_stems import AuditSettings, StemService, applies_directivity, stem_key
from reverberate.viz.scene_api import SceneError
from reverberate.viz.serve_room import _handler_for, _Server

#: Half a second, the engine's run, at 48 kHz.
CHUNK = 24000
SECONDS = 2.2  # four chunks and a short fifth: the scene does not end on a chunk


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two sources in the free field round a walking listener, active in turns."""
    path = tmp_path_factory.mktemp("pack") / "free.h5"
    write_pack(path, demo_pack(profile="free", duration_s=SECONDS, sources=2, seed=3))
    return path


@pytest.fixture
def service(tmp_path: Path) -> Iterator[StemService]:
    made = StemService(tmp_path / "cache", workers=2, processes=False)
    yield made
    made.close()


def _render_all(
    service: StemService, pack_path: Path, settings: AuditSettings | None = None
) -> Any:
    session = service.open(pack_path, settings)
    for index in range(session.chunks):
        assert service.wait(session, session.order, index, 30.0)
    return session


def _one_shot(session: Any, source_id: str) -> np.ndarray:
    """The engine's stem of a source, in one call, as frames: nothing of the cache."""
    engine = Engine(
        session.pack,
        {source_id: dry_track(session.plans[source_id])},
        settings=session.settings.render(),
    )
    return np.ascontiguousarray(engine.stem(source_id).T.astype("<f4"))


# -- the stem is the engine's -------------------------------------------------------


def test_the_chunks_of_a_stem_are_a_one_shot_render_byte_for_byte(
    service: StemService, pack_path: Path
) -> None:
    session = _render_all(service, pack_path)
    assert session.chunks == 5 and session.chunk_samples == CHUNK
    for name in session.order:
        stem = session.stems[name]
        chunks = [stem.read(index) for index in range(stem.chunks)]
        assert [len(c) for c in chunks] == [CHUNK] * 4 + [session.pack.header.samples - 4 * CHUNK]
        wanted = _one_shot(session, name)
        assert np.concatenate(chunks).tobytes() == wanted.tobytes()
        # Each chunk's digest was taken from the engine's samples before they were written.
        for index in range(stem.chunks):
            start, stop = stem.span(index)
            digest = hashlib.sha256(wanted[start:stop].tobytes()).hexdigest()
            assert stem.records[index]["sha256"] == digest


def test_a_complete_stem_is_a_scene_signal(service: StemService, pack_path: Path) -> None:
    session = _render_all(service, pack_path)
    stem = session.stems["s1"]
    for thread in service._threads:  # the header is written by the worker that finished it
        thread.join(timeout=0.01)
    deadline = threading.Event()
    for _ in range(100):
        if (stem.folder / "stem.json").is_file():
            break
        deadline.wait(0.02)
    signal = open_signal(stem.folder / "stem")
    assert signal.header["sources"] == ["s1"]
    assert signal.header["frames"] == session.pack.header.samples
    assert signal.header["sha256"] == hashlib.sha256(_one_shot(session, "s1").tobytes()).hexdigest()
    assert signal.header["recipe_sha256"] == session.pack.header.recipe_sha256


def test_a_silent_chunk_is_recorded_and_takes_no_disk(
    service: StemService, pack_path: Path
) -> None:
    session = _render_all(service, pack_path)
    late = session.stems["s2"]  # its first interval starts after the scene does
    first = session.plans["s2"].source["activity"][0]["start_s"]
    assert first >= 1.0
    assert late.records[0]["silent"] and late.records[1]["silent"]
    assert not late.read(0).any()
    assert late.records[0]["levels_db"] == [audit_stems.SILENCE_DB] * 10
    assert not late.records[int(first * 2) + 1]["silent"]
    # The stem's file is as long as the scene from the start; what was never written is
    # a hole, which a file system keeps as one when it is large enough to be worth it.
    assert late.data_path.stat().st_size == session.pack.header.samples * 64 * 4
    assert late.disk_bytes() <= late.data_path.stat().st_size


# -- the cache -----------------------------------------------------------------------


def test_a_second_listen_renders_nothing(
    tmp_path: Path, pack_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: list[tuple[str, int]] = []
    render = audit_stems.render_chunk

    def counted(task: Any, engines: Any) -> Any:
        rendered.append((task["source"], task["chunk"]))
        return render(task, engines)

    monkeypatch.setattr(audit_stems, "render_chunk", counted)
    first = StemService(tmp_path / "cache", workers=1, processes=False)
    session = first.open(pack_path)
    first.want(session, background=False, cursor=3, audible=["s2"])
    assert first.wait(session, ["s2"], 3, 30.0)
    first.close()
    done = sorted(rendered)
    kept = session.stems["s2"].read(3).tobytes()
    # The chunk under the cursor of the source that is heard is the first rendered.
    assert rendered[0] == ("s2", 3)
    # A line cut short by the stop is a chunk to render again, not an error.
    with open(session.stems["s2"].journal_path, "a") as handle:
        handle.write('{"chunk": 0, "sha2')

    rendered.clear()
    second = StemService(tmp_path / "cache", workers=1, processes=False)
    try:
        again = second.open(pack_path)
        assert (
            sorted(
                (name, int(i))
                for name in again.order
                for i in np.flatnonzero(again.stems[name].done)
            )
            == done
        )
        assert second.wait(again, ["s2"], 3, 0.0)
        assert again.stems["s2"].read(3).tobytes() == kept
        for index in range(again.chunks):
            assert second.wait(again, again.order, index, 30.0)
    finally:
        second.close()
    # Everything missing was rendered, and nothing that was there.
    assert not set(rendered) & set(done)
    assert len(rendered) + len(done) == 2 * again.chunks


def test_the_key_moves_with_what_a_stem_depends_on_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A voice whose model is traced on and a noise traced off: each switch moves one."""
    made = density_pack(duration_s=0.5, moving=False, sources=2)
    quiet = made.sources["s2"]
    made.sources["s2"] = type(quiet)(
        **{**quiet.__dict__, "directivity_model": "omni", "directivity_enabled": False}
    )
    path = tmp_path / "two.h5"
    write_pack(path, made)
    dry = DrySources()
    with read_pack(path) as pack:
        plans = {name: dry_plan(pack, name, dry) for name in pack.sources}

        def key(name: str, switch: bool | None, digest: str = "a" * 64) -> dict[str, Any]:
            return stem_key(digest, pack, name, AuditSettings(switch), plans[name])

        assert applies_directivity(pack, "s1", None) and not applies_directivity(pack, "s1", False)
        assert not applies_directivity(pack, "s2", None) and applies_directivity(pack, "s2", True)
        # The voice: as traced is on, so forcing it on is the same stem; off is another.
        assert key("s1", None) == key("s1", True) != key("s1", False)
        # The noise: as traced is off, so forcing it off is the same stem.
        assert key("s2", None) == key("s2", False) != key("s2", True)
        assert key("s1", None) != key("s2", None)
        assert key("s1", None) != key("s1", None, "b" * 64)  # another pack's bytes
        assert key("s1", None)["settings"]["workers"] == 1
        before = key("s1", None)
        monkeypatch.setattr(audit_stems, "engine_digest", lambda: "another engine")
        assert key("s1", None) != before

    service = StemService(tmp_path / "cache", workers=1, processes=False)
    try:
        traced = service.open(path, AuditSettings(None))
        omni = service.open(path, AuditSettings(False))
        # Two sessions, three stems: the noise's is one and the same.
        assert traced.stems["s2"] is omni.stems["s2"]
        assert traced.stems["s1"] is not omni.stems["s1"]
        assert traced.stems["s1"].folder.parent == omni.stems["s1"].folder.parent
    finally:
        service.close()


def test_a_pack_is_known_by_its_bytes(
    service: StemService, pack_path: Path, tmp_path: Path
) -> None:
    digest = service.pack_sha256(pack_path)
    assert digest == hashlib.sha256(pack_path.read_bytes()).hexdigest()
    copy = tmp_path / "elsewhere.h5"
    copy.write_bytes(pack_path.read_bytes())
    assert service.open(copy).stems["s1"] is service.open(pack_path).stems["s1"]
    assert json.loads((service.cache_root / "packs.json").read_text())  # hashed once, remembered


# -- what a source is fed --------------------------------------------------------------


def test_dry_audio_is_the_recipes_clip_or_a_placeholder_that_says_so(
    pack_path: Path, tmp_path: Path
) -> None:
    rate = 48000
    voices, clips = tmp_path / "voices", tmp_path / "clips"
    voices.mkdir()
    rng = np.random.default_rng(0)
    soundfile.write(voices / "p001.wav", 0.1 * rng.standard_normal(rate), rate, subtype="FLOAT")
    with read_pack(pack_path) as pack:
        bare = dry_plan(pack, "s1", DrySources())
        voiced = dry_plan(pack, "s1", DrySources(None, voices))
        noise = dry_plan(pack, "s3" if "s3" in pack.sources else "s2", DrySources(None, voices))
        assert bare.placeholder and bare.label == "placeholder: modulated noise"
        assert voiced.label == "placeholder: EARS p001, looped"
        assert bare.digest != voiced.digest
        assert noise.placeholder  # a far voice here; the label names what it is fed
        # A one second file under a seven second interval: it loops, at the level it has.
        track = dry_track(voiced)
        assert not track.silent(0, rate)
        heard = track.read(0, 3 * rate)
        assert np.allclose(heard[1000:2000], heard[rate + 1000 : rate + 2000])

        # The recipe's own clip, once a library holds it.
        entry = next(s for s in pack.recipe_json()["sources"] if s["id"] == "s1")
        clip = entry["activity"][0]["clip"]
        folder = clips / "ears"
        folder.mkdir(parents=True)
        seconds = entry["activity"][0]["end_s"] - entry["activity"][0]["start_s"]
        samples = 0.1 * rng.standard_normal(int(rate * seconds) + rate)
        soundfile.write(folder / f"{clip['name']}.wav", samples, rate, subtype="DOUBLE")
        recipe = pack.recipe_json()
        for source in recipe["sources"]:
            for interval in source["activity"]:
                interval["clip"]["library"] = "ears"
        pack.recipe = (json.dumps(recipe) + "\n").encode()
        mixed = dry_plan(pack, "s1", DrySources(clips, voices))
        assert mixed.label == "clips" and not mixed.placeholder
        assert mixed.digest not in (bare.digest, voiced.digest)
        # The other source's clip is not in the library: it keeps its placeholder.
        assert dry_plan(pack, "s2", DrySources(clips, voices)).placeholder
        first = dry_track(mixed).read(0, 1000)
        assert np.array_equal(first, samples[:1000])
        # A clip that is not the bytes the recipe pins is refused.
        for source in recipe["sources"]:
            for interval in source["activity"]:
                interval["clip"]["sha256"] = "0" * 64
        pack.recipe = (json.dumps(recipe) + "\n").encode()
        with pytest.raises(ValueError, match="not the clip the recipe pins"):
            dry_plan(pack, "s1", DrySources(clips, voices))


# -- the endpoints ------------------------------------------------------------------------


@pytest.fixture
def site(tmp_path: Path, pack_path: Path) -> Iterator[tuple[str, AuditService]]:
    """The server's own handler over an audit service, as a page reaches it."""
    stems = StemService(tmp_path / "cache", workers=2, processes=False)
    audit = AuditService(stems, folders=[pack_path.parent])
    builder = type("Builder", (), {"target": tmp_path, "ensure": lambda self, s: None})()
    builder.audit = audit
    server = _Server(("127.0.0.1", 0), _handler_for(builder))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/audit", audit
    finally:
        server.shutdown()
        server.server_close()
        stems.close()


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def test_the_bytes_the_page_receives_are_the_engines_samples(
    site: tuple[str, AuditService], pack_path: Path
) -> None:
    """The acceptance check of the lot: over HTTP, before any rotation or decode."""
    base, audit = site
    (found,) = _get(f"{base}/packs?recipe_sha256=nothing")
    assert found["name"] == "free" and found["matches"] is False
    assert [s["id"] for s in found["sources"]] == ["s1", "s2"]
    recipe = found["recipe_sha256"]
    assert _get(f"{base}/packs?recipe_sha256={recipe}")[0]["matches"] is True
    pack = found["id"]
    session = audit.stems.open(pack_path)
    for index in (0, 4):  # a chunk in which one source is silent, and the short last one
        for sources in (["s1"], ["s2"], ["s1", "s2"]):
            named = f"pack={pack}&index={index}&sources={','.join(sources)}"
            with urllib.request.urlopen(f"{base}/chunk?{named}&wait_ms=30000", timeout=30) as r:
                received = r.read()
                assert r.headers["Content-Type"] == "application/octet-stream"
                assert int(r.headers["X-Audit-Start"]) == index * CHUNK
                frames = int(r.headers["X-Audit-Frames"])
                assert set(json.loads(r.headers["X-Audit-Levels"])) == set(sources)
            assert len(received) == frames * 64 * 4
            start, stop = index * CHUNK, index * CHUNK + frames
            stems = [_one_shot(session, name)[start:stop] for name in sources]
            engine = (
                stems[0] if len(stems) == 1 else np.sum(stems, axis=0, dtype=float).astype("<f4")
            )
            digest = hashlib.sha256(received).hexdigest()
            assert digest == hashlib.sha256(engine.tobytes()).hexdigest()
            # And the endpoint the page compares with says the same, from the cache and past it.
            assert _get(f"{base}/checksum?{named}")["sha256"] == digest
            fresh = _get(f"{base}/checksum?{named}&fresh=1")
            assert fresh["sha256"] == digest and set(fresh["sources"]) == set(sources)


def test_a_mix_is_its_stems_summed_and_the_engines_mix_to_rounding(
    site: tuple[str, AuditService], pack_path: Path
) -> None:
    _, audit = site
    pack = audit.packs()[0]["id"]

    def chunk(sources: str) -> np.ndarray:
        answer = audit.handle(
            "GET", ["chunk"], {"pack": pack, "index": "3", "sources": sources, "wait_ms": "30000"}
        )
        assert isinstance(answer, Binary)
        return np.frombuffer(answer.payload, dtype="<f4").reshape(-1, 64)

    one, two, both = chunk("s1"), chunk("s2"), chunk("s2,s1")
    assert one.any() and two.any() and not np.array_equal(one, two)
    # A solo is the stem alone; both are their sum; no source is silence.
    assert np.array_equal(both, (one.astype(float) + two.astype(float)).astype("<f4"))
    assert not chunk("").any() and chunk("").shape == one.shape
    session = audit.stems.open(pack_path)
    engine = Engine(
        session.pack,
        {name: dry_track(session.plans[name]) for name in session.order},
        settings=session.settings.render(),
    )
    mixed = engine.render(3 * CHUNK, 4 * CHUNK).T
    assert np.max(np.abs(both - mixed)) < 1e-6 * np.max(np.abs(mixed))
    with pytest.raises(SceneError, match="holds no source"):
        chunk("s9")


def test_the_status_says_what_is_rendered_as_a_buffer_bar(
    site: tuple[str, AuditService],
) -> None:
    base, audit = site
    pack = audit.packs()[0]["id"]

    def status(**body: Any) -> Any:
        request = urllib.request.Request(
            f"{base}/status",
            data=json.dumps({"pack": pack, **body}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    first = status(cursor=2, audible=["s1"], background=False)
    assert first["chunks"] == 5 and first["chunk_samples"] == CHUNK
    assert first["samples"] == round(SECONDS / 0.05) * 2400 and first["channels"] == 64
    assert first["cursor"] == 2 and first["audible"] == ["s1"] and first["background"] is False
    assert first["sources"]["s1"]["placeholder"] is True
    session = next(iter(audit.stems._sessions.values()))
    for index in range(2, 5):
        assert audit.stems.wait(session, session.order, index, 30.0)
    later = status()
    # Without rendering ahead, only from the cursor on: chunks 0 and 1 stay unrendered.
    assert later["sources"]["s1"]["ready"] == [[2, 5]] == later["sources"]["s2"]["ready"]
    assert later["sources"]["s1"]["disk_bytes"] > 0
    # A chunk that is not rendered within the time asked: the page is told, and asks again.
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(f"{base}/chunk?pack={pack}&index=1&sources=s1,s2", timeout=30)
    assert refused.value.code == 503
    status(background=True)
    for index in range(2):
        assert audit.stems.wait(session, session.order, index, 30.0)
    assert status()["sources"]["s2"]["ready"] == [[0, 5]]
    with pytest.raises(urllib.error.HTTPError) as unknown:
        urllib.request.urlopen(f"{base}/chunk?pack=nope&index=0", timeout=30)
    assert unknown.value.code == 404


def test_a_pack_says_where_everything_is_and_where_the_sound_comes_from(
    tmp_path: Path, pack_path: Path
) -> None:
    dense = tmp_path / "dense.h5"
    write_pack(dense, density_pack(duration_s=0.5, moving=True, sources=1))
    stems = StemService(tmp_path / "cache", workers=1, processes=False)
    audit = AuditService(stems, packs=[pack_path, dense])
    try:
        free, tailed = audit.packs()
        assert free["directivity_switch"] is False and tailed["directivity_switch"] is True
        session = stems.open(pack_path)
        tracks = pack_tracks(session)
        assert tracks["duration_s"] == pytest.approx(SECONDS) and tracks["step_s"] == 0.1
        assert tracks["t"][-1] == pytest.approx(SECONDS) and len(tracks["t"]) == 23
        head = session.pack.listener
        assert tracks["listener"]["x"][2] == pytest.approx(head.position[4, 0], abs=1e-3)
        assert tracks["listener"]["yaw_deg"][2] == pytest.approx(head.orientation[4, 0], abs=1e-2)
        s2 = tracks["sources"][1]
        assert s2["activity"][0][0] >= 1.0 and s2["movement"] == []

        answer = audit.handle(
            "GET", ["arrivals"], {"pack": free["id"], "source": "s1", "from": "4", "to": "6"}
        )
        step = answer["steps"][0]
        source = session.pack.sources["s1"]
        # The free field: one arrival, from the source, as late as it is far.
        toward = source.position[4] - head.position[4]
        distance = float(np.linalg.norm(toward))
        assert step["listener"] == pytest.approx(head.position[4], abs=1e-3)
        assert step["direction"] == pytest.approx(toward / distance, abs=1e-3)
        assert step["delay_s"] == pytest.approx([distance / 343.2], abs=1e-5)
        assert step["gain"] == pytest.approx([1.0 / distance], abs=1e-5)
        assert step["order"] == [0] and step["tail"] is None and len(answer["steps"]) == 2

        dense_answer = audit.handle(
            "GET", ["arrivals"], {"pack": tailed["id"], "source": "s1", "from": "0", "to": "400"}
        )
        assert dense_answer["to"] == 11  # stopped by the scene's end
        grid = dense_answer["tail_grid"]
        count = grid["elevations"] * grid["azimuths"]
        directions = np.array(grid["directions"]).reshape(count, 3)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-3)
        energy = np.array(dense_answer["steps"][3]["tail"])
        assert energy.shape == (count,) and energy.max() == 1.0 and energy.min() >= 0.0
        assert len(dense_answer["steps"][3]["delay_s"]) >= 12
        with pytest.raises(SceneError, match="names one source"):
            audit.handle("GET", ["arrivals"], {"pack": free["id"], "from": "0", "to": "1"})
    finally:
        stems.close()


@pytest.mark.slow
def test_a_worker_process_renders_the_same_bytes(tmp_path: Path, pack_path: Path) -> None:
    service = StemService(tmp_path / "cache", workers=1, processes=True)
    try:
        session = _render_all(service, pack_path)
        for name in session.order:
            stem = session.stems[name]
            chunks = np.concatenate([stem.read(index) for index in range(stem.chunks)])
            assert chunks.tobytes() == _one_shot(session, name).tobytes()
    finally:
        service.close()
