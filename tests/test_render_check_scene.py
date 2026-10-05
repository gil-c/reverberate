"""What the first whole scene taught the sound check: far sources, hidden ones, things that move.

Every reading here was wrong about a room before 2026-10-05, on a pack
whose sources stood rooms away and walked: a tone's level read as a band's,
a band's loudest arrival read as its first, a reflection's Doppler shift
read as a line of the engine's, a clip's loop read as a cut. Each test
holds the reading that replaced it on a signal whose answer is known, and
the fault it must still find.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from reverberate.render.check import measure
from reverberate.render.check.clips import ClipSource
from reverberate.render.check.report import _spread, check_pack
from reverberate.render.check.run import (
    COMBS_HZ,
    FAIL,
    INFO,
    LIMITS,
    PASS,
    SKIP,
    WARN,
    CheckSettings,
    Result,
    _field_range,
    probe_source,
)
from reverberate.render.pack import KIND_DIRECT, ScenePack, synthetic_free_field

FS = 48000.0
SETTINGS = CheckSettings(probe_seconds=1.5, workers=1)
C = 343.2


def one(results: list[Result], test: str) -> Result:
    found = [r for r in results if r.test == test]
    assert len(found) == 1, f"{test}: {[r.test for r in results]}"
    return found[0]


def two_paths(tones_hz: tuple[float, ...], seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """Tones heard by a head walking at 1 m/s from one wall to the other: two paths, 0.9 apart.

    One path shortens at the walking speed and the other lengthens, 1.3 ms
    longer to begin with: each tone beats at twice the speed over its
    wavelength, 14.6 Hz at 2.5 kHz.
    """
    time = np.arange(int(seconds * FS)) / FS
    phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, len(tones_hz))
    near, far = -time / C, 0.0013 + time / C
    heard = np.zeros(time.size)
    for tone, phase in zip(tones_hz, phases, strict=True):
        heard += np.sin(2.0 * np.pi * tone * (time - near) + phase)
        heard += 0.9 * np.sin(2.0 * np.pi * tone * (time - far) + phase)
    return heard


def test_a_comb_keeps_its_level_where_one_tone_walks_through_the_nulls_of_two_paths() -> None:
    for tones in COMBS_HZ:
        dry = measure.comb(48000, FS, tones, seed=3)
        levels = measure.frame_levels_db(dry, 480)
        assert np.ptp(levels) < 1e-6  # every 10 ms of it holds the same power
        assert np.sqrt(np.mean(dry**2)) == pytest.approx(0.1, rel=1e-3)
    # One tone between two walls: what the check read as a switch, and failed.
    tone = measure.level_step_db(two_paths((2500.0,)), FS)[0]
    assert tone > 10.0 and LIMITS["level_step"].judge(tone) == FAIL
    # The band it stands in, through the same two paths: no step.
    band = two_paths(COMBS_HZ[1])
    size, _ = measure.level_step_db(band, FS, envelope=False)
    assert size < LIMITS["level_step"].ok
    # And a switch of 3.5 dB on that band, half way, is still found where it is.
    switched = band.copy()
    switched[24000:] *= 10.0 ** (3.5 / 20.0)
    size, at = measure.level_step_db(switched, FS, envelope=False)
    assert size == pytest.approx(3.5, abs=0.4) and abs(at - 24000) <= 960
    assert LIMITS["level_step"].judge(size) == FAIL


def test_a_band_s_direct_sound_is_read_where_it_is_said_to_be_and_not_at_its_loudest() -> None:
    # A far pair under the crossover: the direct sound, a floor's reflection 1.5 ms after
    # and larger, and the loudest arrival 12 ms later, four times the direct.
    pulse = np.sinc(np.arange(-240, 241) * 2.0 * 1400.0 / FS) * np.hanning(481)
    response = np.zeros(9600)
    for at, size in ((2000, 0.25), (2072, 0.3), (2576, 1.0)):
        response[at - 240 : at + 241] += size * pulse
    assert measure.envelope_arrival(response) == pytest.approx(2576, abs=2)  # the old reading
    at, share = measure.arrival_near(response, 2003.0, 144)
    assert at == pytest.approx(2000, abs=8) and 0.2 < share < 0.4
    # On another clock, 10 ms off, nothing is there: the caller falls back and says so.
    assert measure.arrival_near(response, 1400.0, 144) == (-1, 0.0)
    assert measure.arrival_near(np.zeros(4800), 2000.0, 144) == (-1, 0.0)


def test_a_reflection_s_doppler_shift_is_not_a_line_of_the_engine_and_a_wobble_still_is() -> None:
    time = np.arange(int(4.0 * FS)) / FS
    tone = 0.3 * np.sin(2.0 * np.pi * 2500.0 * time)
    # A static source, a head at 1 m/s: a wall behind it sends the tone back 6 Hz up.
    shifted = tone + 0.1 * np.sin(2.0 * np.pi * 2506.0 * time)
    read = measure.sidebands_db(shifted, FS, 2500.0, 2.0)
    assert read.harmonic == 3 and read.level_db == pytest.approx(-9.5, abs=0.5)
    assert LIMITS["zipper"].judge(read.level_db) == FAIL  # what failed noise_4
    spread = 2500.0 * 1.0 / C  # 7.3 Hz
    assert measure.sidebands_db(shifted, FS, 2500.0, 2.0, beyond_hz=spread).level_db < -60.0
    # The step rate's lines lie outside the spread and are read as before.
    wobble = shifted * (1.0 + 0.05 * np.sin(2.0 * np.pi * 20.0 * time))
    found = measure.sidebands_db(wobble, FS, 2500.0, 20.0, beyond_hz=spread)
    assert found.level_db == pytest.approx(-32.0, abs=0.5)
    # With a spread that covers every line asked for, no line is the answer.
    assert measure.sidebands_db(shifted, FS, 2500.0, 2.0, beyond_hz=20.0).level_db == float("-inf")


def test_a_join_that_goes_on_is_not_a_step_and_a_cut_is() -> None:
    noise = measure.pink_noise(9600, FS, seed=2)
    assert measure.join_step(noise, 4800, FS) < 4.0  # any sample of a signal that goes on
    hum = 0.1 * np.sin(2.0 * np.pi * 100.0 * np.arange(9600) / FS)
    assert measure.join_step(hum, 4800, FS) == pytest.approx(1.0, abs=0.6)
    cut = np.concatenate([hum[:4800], hum[4920:]])  # a quarter period on
    assert measure.join_step(cut, 4800, FS) > 50.0
    assert measure.join_step(np.zeros(100), 50, FS) == 0.0


def hidden(pack: ScenePack, steps: slice = slice(None)) -> ScenePack:
    """``pack`` with its direct path called a bend round an edge at ``steps``: not seen."""
    source = pack.sources["s1"]
    kind = np.array(source.early.kind)
    offsets = np.asarray(source.early.offsets)
    rows = np.zeros(kind.size, dtype=bool)
    for k in np.arange(pack.header.steps)[steps]:
        rows[int(offsets[k]) : int(offsets[k + 1])] = True
    kind[rows & (kind == KIND_DIRECT)] = 2
    return replace(pack, sources={"s1": replace(source, early=replace(source.early, kind=kind))})


def test_where_the_listener_does_not_see_the_source_the_bands_are_not_aligned_on_anything() -> None:
    pack = synthetic_free_field(
        level="B", source=(1.2, 1.5, 1.6), listener_start=(0.0, 1.5, 0.0), duration_s=2.0
    )
    seen = probe_source(pack, pack.sources["s1"], 4, SETTINGS, None)[0]
    aligned = one(seen, "band_alignment")
    assert aligned.status == PASS and "within 3 ms" in aligned.note
    assert aligned.detail["low_direct_over_its_largest"] == pytest.approx(1.0, abs=0.01)
    away = hidden(pack)
    results = probe_source(away, away.sources["s1"], 4, SETTINGS, None)[0]
    told = one(results, "band_alignment")
    assert told.status == INFO and told.value is None and "no direct path" in told.note
    # What comes before the straight line could bring anything is still judged, and is the same.
    assert one(results, "pre_arrival_energy").value == pytest.approx(
        one(seen, "pre_arrival_energy").value, abs=0.01
    )
    assert not [r for r in results if r.test in ("direct_level", "direct_band_balance")]


def test_the_field_speaks_for_a_source_within_a_metre_of_its_own_and_gives_its_range() -> None:
    pack = synthetic_free_field(source=(0.0, 1.5, -3.0), listener_start=(0.0, 1.5, 0.0))
    own = probe_source(pack, pack.sources["s1"], 2, SETTINGS, None)[1]["response"]
    reference: dict[str, Any] = {
        "name": "itself",
        "scene_id": "synthetic",
        "positions": np.array([[5.0, 1.5, 0.0], [0.1, 1.5, 0.0]]),
        "response": lambda index: own * (index == 1),
        "source_position": np.array([0.3, 1.5, -3.0]),
    }
    near = CheckSettings(workers=1, reference=reference)
    judged = one(
        probe_source(pack, pack.sources["s1"], 2, near, None)[0], "late_spectrum_reference"
    )
    assert judged.status == PASS and "told" not in judged.note
    rooms_away = {**reference, "source_position": np.array([6.0, 1.5, 4.0])}
    far = CheckSettings(workers=1, reference=rooms_away)
    told = one(probe_source(pack, pack.sources["s1"], 2, far, None)[0], "late_spectrum_reference")
    assert told.status == INFO and "9.2 m from this one" in told.note
    assert told.value == pytest.approx(judged.value, abs=1e-9)  # the number is still given

    # The range: three points that decay in 0.3, 0.4 and 0.5 s, and one that is silent.
    def decay(index: int) -> np.ndarray:
        if index == 3:
            return np.zeros(48000)
        noise = np.random.default_rng(index).standard_normal(48000)
        t60 = (0.3, 0.4, 0.5)[index]
        return np.asarray(noise * 10.0 ** (-3.0 * np.arange(48000) / (t60 * FS)))

    field: dict[str, Any] = {"positions": np.zeros((4, 3)), "response": decay}
    least, most = _field_range(field, FS)
    middle = slice(1, 6)  # 250 Hz to 4 kHz
    assert least[middle] == pytest.approx(0.31, rel=0.2)  # the 5th percentile of the three
    assert most[middle] == pytest.approx(0.49, rel=0.2)
    assert field["t20_range"][0] is least  # made once


def test_what_the_sources_of_a_dwelling_span_is_told_and_one_probe_spans_nothing() -> None:
    bank = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    near = {"name": "a@1.00s", "bank_hz": bank, "t20_s": np.full(8, 0.40)}
    far = {
        "name": "b@2.00s",
        "bank_hz": bank,
        "t20_s": np.array([9, 0.5, 0.6, 0.4, 0.3, 0.4, 9, 9.0]),
    }
    told = _spread([near, far, {"name": "silent"}])
    assert told is not None and told.status == INFO
    assert told.value == pytest.approx(50.0)  # 0.6 s over 0.4 s at 500 Hz; 125 Hz is not read
    assert "500 Hz 0.40 to 0.60 s (median 0.50)" in told.note
    assert _spread([near]) is None and _spread([]) is None
    never = {"name": "c", "bank_hz": bank, "t20_s": np.full(8, np.nan)}
    assert _spread([never, never]) is None  # a pack without a tail decays nowhere


def test_a_law_of_distance_is_read_where_the_source_is_seen_and_a_hidden_doppler_is_told() -> None:
    walk = synthetic_free_field(
        source=(6.0, 1.5, 0.0),
        listener_start=(0.0, 1.5, 0.0),
        listener_end=(4.5, 1.5, 0.0),
        duration_s=3.0,
    )
    # The same walk, the source behind something for all of it: nothing to hold a law to.
    away = hidden(walk)
    results = check_pack(away, settings=SETTINGS, families=("continuity",))["results"]
    law = one(results, "level_distance")
    assert law.status == SKIP and "where the listener sees the source" in law.note
    assert one(results, "doppler").status == INFO
    # A band's level steps nowhere on a walk in free air, and says what it was read on.
    step = one(results, "level_step")
    assert step.status == PASS and "the comb from" in step.note
    assert set(step.detail["one_tone_alone_db"]) == {"400 Hz", "2500 Hz"}
    assert "are the movement's own" in one(results, "zipper").note


def test_a_loop_is_a_join_a_fragment_of_speech_is_told_and_the_level_is_read_on_the_window(
    tmp_path: Path,
) -> None:
    pack = synthetic_free_field(
        source=(6.0, 1.5, 0.0),
        listener_start=(0.0, 1.5, 0.0),
        listener_end=(4.5, 1.5, 0.0),
        duration_s=3.0,
    )
    # A hum of 2 s made to loop (whole periods), and the same cut a quarter period short.
    time = np.arange(96000) / FS
    hum = 10.0 ** (-26.0 / 20.0) * np.sqrt(2.0) * np.sin(2.0 * np.pi * 400.0 * time)
    (tmp_path / "made" / "noise").mkdir(parents=True)
    soundfile.write(str(tmp_path / "made" / "noise" / "hum.wav"), hum, 48000, subtype="FLOAT")

    def recipe(second_offset_s: float) -> dict[str, Any]:
        clip = {"library": "made", "name": "noise/hum"}
        return {
            "sources": [
                {
                    "id": "s1",
                    "activity": [
                        {"start_s": 0.5, "end_s": 1.5, "clip_offset_s": 1.0, "clip": clip},
                        {
                            "start_s": 1.5,
                            "end_s": 2.5,
                            "clip_offset_s": second_offset_s,
                            "clip": clip,
                        },
                    ],
                }
            ]
        }

    def checked(second_offset_s: float, window: tuple[float, float]) -> list[Result]:
        outcome = check_pack(
            pack,
            recipe=recipe(second_offset_s),
            clips=ClipSource(tmp_path),
            settings=SETTINGS,
            window_s=window,
            families=("mix",),
        )
        results: list[Result] = outcome["results"]
        return results

    looped = checked(0.0, (0.0, 2.9))
    assert one(looped, "interval_join").status == PASS
    assert one(looped, "interval_join").value == pytest.approx(1.0, abs=0.6)
    edge = one(looped, "interval_edge")
    assert edge.status == PASS and edge.detail["edges"] == 2  # the two real ends, faded
    assert one(checked(0.000625, (0.0, 2.9)), "interval_join").status == FAIL
    # The level is held against the distance the source has in the window, not in the scene:
    # the head is at 3.75 to 2.25 m while the second second sounds, 5.25 to 3.75 m in the first.
    late = one(checked(0.0, (1.5, 2.9)), "level_at_listener")
    assert late.detail["distance_m"] == pytest.approx(2.9, abs=0.2)
    assert late.value == pytest.approx(0.0, abs=0.7)
    early = one(checked(0.0, (0.0, 1.5)), "level_at_listener")
    assert early.detail["distance_m"] == pytest.approx(4.4, abs=0.2)
    assert early.value == pytest.approx(0.0, abs=0.7)
    # A voice of which the window holds a second is told, whatever it reads.
    voice = replace(pack, sources={"s1": replace(pack.sources["s1"], kind="near_voice")})
    outcome = check_pack(
        voice,
        recipe=recipe(0.0),
        clips=ClipSource(tmp_path),
        settings=SETTINGS,
        window_s=(0.0, 2.9),
        families=("mix",),
    )
    heard = one(outcome["results"], "audibility")
    assert heard.status == INFO and "speaks for 2.0 s of the window" in heard.note
    assert one(outcome["results"], "near_voice_seen").status == PASS
    # The same voice behind a wall for the scene's last second: said, with where and how near.
    walled = hidden(voice, slice(40, None))
    outcome = check_pack(
        walled,
        recipe=recipe(0.0),
        clips=ClipSource(tmp_path),
        settings=SETTINGS,
        window_s=(0.0, 2.9),
        families=("mix",),
    )
    unseen = one(outcome["results"], "near_voice_seen")
    assert unseen.status == WARN and unseen.value == pytest.approx(0.55, abs=0.06)
    assert unseen.detail["first_s"] == pytest.approx(2.0) and unseen.detail["nearest_m"] < 3.1
    assert heard.status != WARN
