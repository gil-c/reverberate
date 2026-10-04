"""Tests for the library of dry clips and for the generator that plays it.

No network and no audio from anywhere: the sibling project's library is a zip
built here and put in the fake store under its own keys, holding a "voice"
that is bursts of noise with silences between and a "noise" that is noise.
What matters is what the scene leans on: a clip built twice has one digest,
a file that is not the manifest's is refused, the level a clip is stored at
is the one the manifest states, and a talk spurt starts and ends where an
utterance does.
"""

from __future__ import annotations

import io
import json
import struct
import wave
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.scenes import (
    ClipEntry,
    ClipLibrary,
    canonical_bytes,
    generate,
    load_clip_library,
    validate,
)
from reverberate.scenes import clips as clip_library
from reverberate.scenes.__main__ import DEFAULT_MANIFEST, main
from reverberate.scenes.recipe import NOISE_SUBTYPES
from reverberate.store import MemoryStore
from scene_floor import SMALL, two_room_floor, two_room_layout

RATE = clip_library.RATE_HZ


# --------------------------------------------------------------------------
# a sibling library in the fake store
# --------------------------------------------------------------------------


def _wav(samples: np.ndarray, rate: int = RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _sentence(seed: int, seconds: float, level: float = 0.05) -> np.ndarray:
    """A burst of noise with 0.3 s of near silence on either side: one sentence."""
    rng = np.random.default_rng(seed)
    quiet = 1e-4 * rng.standard_normal(int(0.3 * RATE))
    return np.concatenate([quiet, level * rng.standard_normal(int(seconds * RATE)), quiet])


def _talk(seed: int, spells: list[float], pause: float = 0.5, level: float = 0.05) -> np.ndarray:
    """Bursts of noise of the lengths given, a near silence between: a monologue."""
    rng = np.random.default_rng(seed)
    quiet = 1e-4 * rng.standard_normal(int(pause * RATE))
    parts = [quiet]
    for seconds in spells:
        parts += [level * rng.standard_normal(int(seconds * RATE)), quiet]
    return np.concatenate(parts)


def _library_store() -> MemoryStore:
    """Two speakers of six sentences each and two noises, as the sibling project keeps them."""
    members: dict[str, bytes] = {}
    for speaker in ("p1", "p2"):
        for number in range(6):
            members[f"{speaker}/s{number}.wav"] = _wav(
                _sentence(hash((speaker, number)) % 1000, 1.0 + 0.4 * number)
            )
    rng = np.random.default_rng(7)
    members["hum.wav"] = _wav(0.02 * rng.standard_normal(6 * RATE) + 0.01)
    members["tune.wav"] = _wav(0.1 * rng.standard_normal(3 * 44100), rate=44100)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    blob = buffer.getvalue()
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            head = blob[info.header_offset : info.header_offset + 30]
            name_len, extra_len = struct.unpack("<HH", head[26:30])
            rows.append(
                {
                    "clip_id": info.filename,
                    "member_name": info.filename,
                    "shard_key": "library/shards/toy/a-0000.zip",
                    "payload_offset": info.header_offset + 30 + name_len + extra_len,
                    "payload_length": info.compress_size,
                    "crc32": info.CRC,
                    "extension": ".wav",
                }
            )
    store = MemoryStore()
    # The other project's keys: written into the fake, never through ``put_bytes``.
    store.objects["library/shards/toy/a-0000.zip"] = blob
    store.objects["library/catalog/shards/toy/a-0000.jsonl"] = "\n".join(
        json.dumps(row) for row in rows
    ).encode()
    return store


def _part(member: str, **more: Any) -> dict[str, Any]:
    return {"dataset": "toy", "shard": "a-0000", "member": member, **more}


SELECTION: dict[str, Any] = {
    "library": "toy_v1",
    "datasets": {"toy": {"licence": "CC0 1.0", "attribution": "made up for a test"}},
    "clips": [
        {
            "name": f"voice/{speaker}_{half}",
            "kind": "voice",
            "speaker": speaker,
            "process": {"gap_s": 0.4, "part_fade_s": 0.01},
            "parts": [_part(f"{speaker}/s{n}.wav") for n in numbers],
        }
        for speaker in ("p1", "p2")
        for half, numbers in (("a", (0, 1, 2)), ("b", (3, 4, 5)))
    ]
    + [
        {
            "name": "noise/hum",
            "kind": "appliance",
            "spl_1m_db": 50.0,
            "process": {"loop_crossfade_s": 0.5, "remove_dc": True},
            "parts": [_part("hum.wav", start_s=0.5)],
        },
        {
            "name": "noise/tune",
            "kind": "music",
            "spl_1m_db": 60.0,
            "process": {"fade_s": 0.2},
            "parts": [_part("tune.wav")],
        },
    ],
}


@pytest.fixture(scope="module")
def curated(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path, MemoryStore]:
    root = tmp_path_factory.mktemp("clips")
    store = _library_store()
    return clip_library.curate(SELECTION, store, root, jobs=1), root, store


# --------------------------------------------------------------------------
# the measurements
# --------------------------------------------------------------------------


def test_the_active_level_is_that_of_the_speech_and_not_of_its_silences() -> None:
    rng = np.random.default_rng(1)
    burst = 0.1 * rng.standard_normal(2 * RATE)
    signal = np.concatenate([burst, np.zeros(2 * RATE)] * 3)

    level, activity = clip_library.active_speech_level(signal)

    # The long term level is 3 dB under the bursts'. The active level is
    # theirs, spread over the 200 ms the standard keeps after each: 2.2 s for 2.
    assert level == pytest.approx(-20.0 - 10.0 * np.log10(1.1), abs=0.25)
    assert activity == pytest.approx(0.55, abs=0.03)
    assert clip_library.active_speech_level(np.zeros(RATE)) == (-np.inf, 0.0)


def test_utterances_end_in_the_pauses_and_a_long_one_is_cut_at_its_breath() -> None:
    rng = np.random.default_rng(2)

    def quiet(seconds: float) -> np.ndarray:
        return 1e-4 * rng.standard_normal(int(seconds * RATE))

    def loud(seconds: float) -> np.ndarray:
        return 0.05 * rng.standard_normal(int(seconds * RATE))

    # Two seconds, a pause; then fifteen with one breath of 0.14 s after nine.
    signal = np.concatenate(
        [quiet(0.5), loud(2.0), quiet(0.6), loud(9.0), quiet(0.14), loud(6.0), quiet(0.5)]
    )

    found = clip_library.find_utterances(signal)

    assert len(found) == 3
    assert all(b - a <= clip_library.MAX_UTTERANCE_S for a, b in found)
    assert all(found[k][1] <= found[k + 1][0] for k in range(2))
    for start, end in found:
        for edge in (start, end):
            at = int(round(edge * RATE))
            assert np.abs(signal[max(at - 48, 0) : at + 48]).max() < 1e-3
    assert found[1][1] == pytest.approx(12.17, abs=0.08)


# --------------------------------------------------------------------------
# curate, fetch, check
# --------------------------------------------------------------------------


def test_a_curated_library_is_at_its_stated_levels_and_passes_its_own_check(
    curated: tuple[dict[str, Any], Path, MemoryStore],
) -> None:
    manifest, root, _ = curated

    assert manifest["schema"] == clip_library.SCHEMA and manifest["library"] == "toy_v1"
    assert [clip["name"] for clip in manifest["clips"]] == sorted(
        clip["name"] for clip in manifest["clips"]
    )
    reports = clip_library.check(manifest, root)
    assert [report.failures for report in reports] == [()] * 6, clip_library.format_table(reports)
    by_name = {clip["name"]: clip for clip in manifest["clips"]}
    voice = by_name["voice/p1_a"]
    assert voice["level"]["active_dbfs"] == pytest.approx(clip_library.VOICE_ACTIVE_DBFS, abs=0.1)
    assert voice["level"]["spl_1m_db"] == 60.0 and voice["speaker"] == "p1"
    hum = by_name["noise/hum"]
    assert hum["level"]["rms_dbfs"] == pytest.approx(
        50.0 - clip_library.FULL_SCALE_SPL_1M_DB, abs=0.1
    )
    assert hum["loop"] and hum["subtype"] == "appliance" and hum["licence"] == "CC0 1.0"
    # Six seconds less the half second cut and the half second folded into the loop.
    assert hum["duration_s"] == 5.0
    # 44.1 kHz in, 48 kHz out.
    assert by_name["noise/tune"]["duration_s"] == pytest.approx(3.0, abs=0.002)


def test_a_sentence_clip_is_cut_between_its_sentences(
    curated: tuple[dict[str, Any], Path, MemoryStore],
) -> None:
    manifest, root, _ = curated
    clip = next(c for c in manifest["clips"] if c["name"] == "voice/p2_b")
    import soundfile

    samples, rate = soundfile.read(str(clip_library.clip_path(root, "toy_v1", clip["name"])))

    spoken = clip["utterances"]
    assert rate == RATE and len(spoken) == 3
    assert spoken[0][0] == 0.0 and spoken[-1][1] == clip["duration_s"]
    assert all(spoken[k][1] == spoken[k + 1][0] for k in range(2))
    for start, end in spoken:
        # Each holds one sentence, and is silent where it is cut.
        inside = samples[int(start * RATE) : int(end * RATE)]
        assert np.sqrt(np.mean(inside**2)) > 0.01
        for edge in (start, end):
            at = int(round(edge * RATE))
            # Between two sentences the silence is the clip's own; at the
            # clip's two ends it is the member's, faded.
            inner = 0.0 < edge < clip["duration_s"]
            assert np.abs(samples[max(at - 480, 0) : at + 480]).max() <= (0.0 if inner else 1e-3)
    # The members' own near silence went: 0.3 s either side, less the margins kept.
    lengths = [end - start for start, end in spoken]
    assert lengths == pytest.approx(
        [2.2 + 0.13 + 0.2, 2.6 + 0.13 + 0.4, 3.0 + 0.13 + 0.2], abs=0.03
    )


def test_fetch_rebuilds_the_same_bytes_and_refuses_any_other(
    curated: tuple[dict[str, Any], Path, MemoryStore], tmp_path: Path
) -> None:
    manifest, root, store = curated

    done = clip_library.fetch(manifest, tmp_path, store, jobs=2)

    assert set(done.values()) == {"fetched"}
    for clip in manifest["clips"]:
        ours = clip_library.clip_path(tmp_path, "toy_v1", clip["name"]).read_bytes()
        assert ours == clip_library.clip_path(root, "toy_v1", clip["name"]).read_bytes()
        assert len(ours) == clip["bytes"] == 44 + 2 * int(round(clip["duration_s"] * RATE))
    # A second fetch reads nothing; a file that was altered is built again.
    victim = clip_library.clip_path(tmp_path, "toy_v1", "noise/hum")
    victim.write_bytes(victim.read_bytes()[:-2] + b"\x01\x02")
    again = clip_library.fetch(manifest, tmp_path, MemoryStore(objects=dict(store.objects)), jobs=1)
    assert again["noise/hum"] == "fetched" and again["voice/p1_a"] == "kept"

    # A manifest that pins other bytes: nothing is written under its name.
    wrong = json.loads(json.dumps(manifest))
    wrong["clips"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="did not build to the manifest's digest"):
        clip_library.fetch(wrong, tmp_path / "other", store)
    assert not clip_library.clip_path(
        tmp_path / "other", "toy_v1", wrong["clips"][0]["name"]
    ).exists()


def test_check_fails_a_clip_that_is_not_the_manifests(
    curated: tuple[dict[str, Any], Path, MemoryStore], tmp_path: Path
) -> None:
    manifest, _, store = curated
    clip_library.fetch(manifest, tmp_path, store)
    path = clip_library.clip_path(tmp_path, "toy_v1", "noise/tune")
    rng = np.random.default_rng(3)
    path.write_bytes(_wav(np.clip(0.5 * rng.standard_normal(3 * RATE) + 0.2, -1, 1)))
    clip_library.clip_path(tmp_path, "toy_v1", "noise/hum").unlink()

    failures = {report.name: report.failures for report in clip_library.check(manifest, tmp_path)}

    assert failures["noise/hum"] == ("missing",)
    assert set(failures["noise/tune"]) >= {"digest", "clipped", "peak", "dc", "level"}
    assert failures["voice/p1_a"] == ()
    saved = tmp_path / "manifest.json"
    saved.write_text(json.dumps(manifest))
    assert main(["clips", "check", "--manifest", str(saved), "--root", str(tmp_path)]) == 1


# --------------------------------------------------------------------------
# the generator on a library
# --------------------------------------------------------------------------


def _toy_library() -> ClipLibrary:
    """Four speakers of three clips of twelve utterances, and noises of two kinds."""

    def digest(name: str) -> str:
        return (name.encode().hex() + "0" * 64)[:64]

    entries = []
    for speaker in ("a", "b", "c", "d"):
        for number in range(3):
            edges = np.round(
                np.cumsum([0.0] + [1.3 + 0.37 * ((k + number) % 5) for k in range(12)]), 3
            )
            spoken = tuple((float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True))
            name = f"voice/{speaker}{number}"
            entries.append(
                ClipEntry("toy_v1", name, digest(name), float(edges[-1]), "voice", speaker, spoken)
            )
    entries.append(ClipEntry("toy_v1", "noise/hum", digest("hum"), 7.5, "appliance", loop=True))
    entries.append(ClipEntry("toy_v1", "noise/fan", digest("fan"), 9.25, "appliance", loop=True))
    for number in range(3):
        name = f"noise/tune{number}"
        entries.append(ClipEntry("toy_v1", name, digest(name), 11.0 + number, "music"))
    return ClipLibrary(tuple(entries))


def _recipe(seed: int = 1, noises: int = 2) -> Any:
    from reverberate.scenes import placeholder_assets

    parameters = replace(SMALL, noise_count=(noises, noises), noise_steady_share=1.0)
    return generate(
        two_room_layout(), parameters, seed, assets=placeholder_assets(), clips=_toy_library()
    )


def test_a_talk_spurt_is_whole_utterances_read_on_through_the_speakers_clips() -> None:
    recipe = _recipe()
    library = {entry.name: entry for entry in _toy_library().entries}

    assert validate(recipe, two_room_floor()) == []
    assert recipe.generator is not None
    assert recipe.generator.parameters["clips"] == {"placeholder": False}
    speakers = []
    for source in recipe.sources:
        if source.kind == "noise":
            continue
        assert source.activity
        heard = []
        for interval in source.activity:
            entry = library[interval.clip.name]
            assert interval.clip.sha256 == entry.sha256 and interval.clip.library == "toy_v1"
            starts = [a for a, _ in entry.utterances]
            ends = [b for _, b in entry.utterances]
            first = starts.index(interval.clip_offset_s)
            stop = interval.clip_offset_s + interval.end_s - interval.start_s
            last = int(np.argmin(np.abs(np.asarray(ends) - stop)))
            assert ends[last] == pytest.approx(stop, abs=1e-6) and last >= first
            heard += [(entry.name, k) for k in range(first, last + 1)]
        # Nothing is said twice, and nothing is skipped: the clips are read on.
        assert len(set(heard)) == len(heard)
        assert heard == sorted(heard)
        speakers.append({library[i.clip.name].speaker for i in source.activity})
    assert all(len(who) == 1 for who in speakers)
    assert len({next(iter(who)) for who in speakers}) == 4


def test_a_noise_keeps_its_looping_clip_and_a_playlist_walks_on() -> None:
    recipe = _recipe()
    noises = [source for source in recipe.sources if source.kind == "noise"]
    library = {entry.name: entry for entry in _toy_library().entries}

    # The kinds are dealt from those the library holds: one of each before two of one.
    assert sorted(source.subtype or "" for source in noises) == ["appliance", "music"]
    for source in noises:
        pieces = source.activity
        assert pieces[0].start_s == 0.0 and pieces[-1].end_s == recipe.duration_s
        assert all(a.end_s == b.start_s for a, b in zip(pieces[:-1], pieces[1:], strict=True))
        names = [piece.clip.name for piece in pieces]
        for piece, following in zip(pieces[:-1], pieces[1:], strict=True):
            # Every piece but the last runs to its clip's end, and the next starts a clip.
            length = library[piece.clip.name].duration_s
            assert piece.clip_offset_s + piece.end_s - piece.start_s == pytest.approx(length)
            assert following.clip_offset_s == 0.0
        if source.subtype == "appliance":
            assert len(set(names)) == 1
        else:
            order = sorted(name for name in library if name.startswith("noise/tune"))
            at = order.index(names[0])
            assert names == [order[(at + k) % 3] for k in range(len(names))]

    # Two noises of one kind start on two clips.
    both = [s for s in _recipe(noises=3).sources if s.subtype == "appliance"]
    if len(both) == 2:
        assert both[0].activity[0].clip.name != both[1].activity[0].clip.name


def test_a_library_gives_the_same_recipe_twice_and_another_than_the_placeholder() -> None:
    from scene_floor import small_recipe

    assert canonical_bytes(_recipe(2)) == canonical_bytes(_recipe(2))
    assert canonical_bytes(_recipe(2)) != canonical_bytes(_recipe(3))
    placeholder = {a.clip.library for s in small_recipe(2).sources for a in s.activity}
    assert placeholder == {"placeholder"}


def test_the_generator_reads_a_manifest_and_the_bare_list(
    curated: tuple[dict[str, Any], Path, MemoryStore], tmp_path: Path
) -> None:
    manifest, _, _ = curated
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(manifest))

    library = load_clip_library(path)

    assert [len(shelf) for shelf in library.speakers()] == [2, 2]
    assert [entry.name for entry in library.noises("appliance")] == ["noise/hum"]
    assert library.noises("appliance")[0].loop and not library.noises("music")[0].loop
    assert len(library.speakers()[0][0].utterances) == 3
    bare = tmp_path / "bare.json"
    bare.write_text(
        json.dumps(
            [{"library": "x", "name": "v", "sha256": "ab" * 32, "duration_s": 4.0, "speaker": "s"}]
        )
    )
    assert load_clip_library(bare).speakers()[0][0].utterances == ()
    path.write_text(json.dumps({**manifest, "schema_version": 2}))
    with pytest.raises(ValueError, match="not a clip library"):
        load_clip_library(path)


def test_the_first_scenes_library_holds_what_the_first_scene_needs() -> None:
    """The manifest in the package, read as the generator reads it; no audio."""
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    library = load_clip_library(DEFAULT_MANIFEST)

    assert manifest["library"] == "clarify_v1" and manifest["sample_rate_hz"] == RATE
    speakers = library.speakers()
    assert len(speakers) >= 9
    # Three near voices of four minutes and six far ones of seven and a half.
    seconds = [sum(entry.duration_s for entry in shelf) for shelf in speakers]
    assert min(seconds[:3]) >= 270.0 and min(seconds[3:9]) >= 480.0
    assert all(library.noises(subtype) for subtype in NOISE_SUBTYPES)
    for clip in manifest["clips"]:
        assert len(clip["sha256"]) == 64 and clip["licence"] and clip["credit"]
        assert clip["level"]["peak_dbfs"] <= clip_library.HEADROOM_DBFS + 0.01
        spoken = clip.get("utterances", [])
        assert all(a < b for a, b in spoken)
        assert all(spoken[k][1] <= spoken[k + 1][0] for k in range(len(spoken) - 1))
        assert not spoken or spoken[-1][1] <= clip["duration_s"]
        if clip["kind"] == "voice":
            assert clip["level"]["active_dbfs"] == pytest.approx(
                clip_library.VOICE_ACTIVE_DBFS, abs=0.1
            )
