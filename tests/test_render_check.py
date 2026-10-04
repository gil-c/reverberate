"""The sound check: every detector on a signal whose answer is known, then on a seeded fault.

A free field is the pack whose render is known in closed form, so every
check has its answer there: the arrival is the distance, the level is one
over it, the direction is the source's. Each fault a listener would reject
is then put in by hand (a click, a step of 6 dB, a source on the wrong
side, the two ears exchanged, the two bands 10 ms apart, a reflection
written twice) and the check that is there for it must fail. Nothing here
moves a limit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import soundfile
from scipy.signal import fftconvolve

from reverberate.render.__main__ import main
from reverberate.render.check import measure, reference
from reverberate.render.check.binaural import PageDecoder, head_matrix, page_decoder, rotation
from reverberate.render.check.clips import ClipSource, feed_of
from reverberate.render.check.report import check_pack
from reverberate.render.check.run import (
    FAIL,
    LIMITS,
    PASS,
    SKIP,
    CheckSettings,
    Result,
    duplicate_arrivals,
    probe_source,
)
from reverberate.render.dry import EDGE_FADE_S, DryTrack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import ScenePack, synthetic_free_field, write_pack
from reverberate.spatial.binaural import BinauralDecoder
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
from reverberate.spatial.sh import real_sh, rotate_yaw

FS = 48000.0
SETTINGS = CheckSettings(probe_seconds=1.5, workers=1)


def tone(hz: float, seconds: float = 1.0, amplitude: float = 0.3) -> np.ndarray:
    return np.asarray(amplitude * np.sin(2.0 * np.pi * hz * np.arange(int(seconds * FS)) / FS))


def one(results: list[Result], test: str) -> Result:
    found = [r for r in results if r.test == test]
    assert len(found) == 1, f"{test}: {[r.test for r in results]}"
    return found[0]


# --------------------------------------------------------------------------
# the detectors, each on a clean signal and on its fault
# --------------------------------------------------------------------------


def test_a_steady_tone_leaves_nothing_and_a_click_of_one_per_cent_is_found_where_it_is() -> None:
    clean = tone(2500.0)
    assert measure.tone_residual_db(clean, 2500.0, FS)[0] < -200.0
    clicked = clean.copy()
    clicked[20000] += 0.01 * 0.3
    size, at = measure.tone_residual_db(clicked, 2500.0, FS)
    assert -41.0 < size < -33.0 and abs(at - 20000) <= 1
    assert LIMITS["tone_click"].judge(size) == FAIL
    # What may move without being a fault: a crossfade over one step, from nothing to full.
    faded = clean * np.clip(np.arange(clean.size) / 2400.0, 0.0, 1.0) ** 2
    assert LIMITS["tone_click"].judge(measure.tone_residual_db(faded[2400:], 2500.0, FS)[0]) == PASS
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip((np.arange(clean.size) - 9600) / 2400.0, 0.0, 1.0))
    assert measure.tone_residual_db((clean * ramp)[9000:], 2500.0, FS)[0] < -55.0


def test_a_step_of_six_decibels_is_a_click_and_a_step_of_level() -> None:
    stepped = tone(400.0)
    stepped[24007:] *= 2.0
    assert LIMITS["tone_click"].judge(measure.tone_residual_db(stepped, 400.0, FS)[0]) == FAIL
    size, at = measure.level_step_db(stepped, FS)
    assert 5.5 < size < 6.5 and abs(at - 24007) <= 960
    assert LIMITS["level_step"].judge(size) == FAIL
    assert measure.level_step_db(tone(400.0)[4800:-4800], FS)[0] < 0.05


def test_noise_has_no_outlier_and_a_click_in_it_is_one() -> None:
    # The false alarm rate, measured: six seconds of noise, six seeds, none passes 6.5.
    worst = max(
        measure.difference_outlier(measure.pink_noise(48000, FS, seed=seed), FS)[0]
        for seed in range(6)
    )
    assert worst < LIMITS["noise_click"].ok
    noise = measure.pink_noise(48000, FS, seed=1)
    step = float(np.sqrt(np.mean(np.diff(noise) ** 2)))
    noise[30000:] += 12.0 * step  # a jump of twelve times the rms difference
    ratio, at = measure.difference_outlier(noise, FS)
    assert ratio > LIMITS["noise_click"].warn and at == 30000


def test_a_gain_that_moves_in_steps_draws_lines_at_the_step_rate() -> None:
    clean = tone(2500.0, 4.0)
    assert measure.sidebands_db(clean, FS, 2500.0, 20.0).level_db < -90.0
    # A gain that drifts by 10 per cent within a step and is put back at the next.
    drift = 1.0 + 0.1 * (np.arange(clean.size) % 2400) / 2400.0
    found = measure.sidebands_db(clean * drift, FS, 2500.0, 20.0)
    assert found.harmonic == 1 and found.prominence_db > 30.0
    assert -40.0 < found.level_db < -26.0
    # 5 per cent of modulation at 20 Hz: two lines at -32 dB.
    wobble = clean * (1.0 + 0.05 * np.sin(2.0 * np.pi * 20.0 * np.arange(clean.size) / FS))
    assert measure.sidebands_db(wobble, FS, 2500.0, 20.0).level_db == pytest.approx(-32.0, abs=0.3)


def test_the_spectrum_reads_a_hole_at_the_crossover_and_not_a_tilt() -> None:
    impulse = np.zeros(48000)
    impulse[100] = 1.0
    assert abs(measure.seam_deviation_db(measure.third_octave_db(impulse, FS))[0]) < 0.1
    freqs = np.fft.rfftfreq(48000, 1.0 / FS)
    tilted = np.fft.irfft(np.fft.rfft(impulse) * np.maximum(freqs, 20.0) ** 0.5, n=48000)
    assert abs(measure.seam_deviation_db(measure.third_octave_db(tilted, FS))[0]) < 0.3
    hole = np.where((freqs > 890.0) & (freqs < 1120.0), 0.5, 1.0)
    worst, bands = measure.seam_deviation_db(
        measure.third_octave_db(np.fft.irfft(np.fft.rfft(impulse) * hole, n=48000), FS)
    )
    assert worst == pytest.approx(-6.0, abs=0.3) and int(np.argmin(bands)) == 2
    assert LIMITS["seam_third_octaves"].judge(worst) != PASS


def test_the_decay_the_weighting_the_arrival_and_the_frequency_are_what_they_are_made() -> None:
    rng = np.random.default_rng(3)
    time = np.arange(int(1.2 * FS)) / FS
    decaying = rng.standard_normal(time.size) * 10.0 ** (-3.0 * time / 0.5)  # 60 dB in 0.5 s
    t20, t30 = measure.decay_times_s(decaying, FS, 0)
    np.testing.assert_allclose(t20[1:7], 0.5, rtol=0.12)
    np.testing.assert_allclose(t30[1:7], 0.5, rtol=0.12)
    # A weighting: nothing at 1 kHz, -19.1 dB at 100 Hz.
    for hz, wanted in ((1000.0, 0.0), (100.0, -19.1), (4000.0, 1.0)):
        got = measure.spl_db(measure.a_weighted(tone(hz), FS)) - measure.spl_db(tone(hz))
        assert got == pytest.approx(wanted, abs=0.15)
    assert measure.spl_db(tone(1000.0, amplitude=np.sqrt(2.0))) == pytest.approx(86.0, abs=0.01)
    # The first arrival, when a later one is louder.
    response = np.zeros(4800)
    response[1000], response[1500] = 0.6, -1.0
    assert measure.envelope_arrival(response) == 1000
    chirped = np.sin(2.0 * np.pi * 2507.0 * np.arange(48000) / FS)
    np.testing.assert_allclose(measure.instantaneous_hz(chirped, FS)[1:-1], 2507.0, atol=0.01)
    wave = real_sh(1, np.array([[0.6, -0.8, 0.0]]))[0][:, None] * tone(700.0, 0.01)[None, :]
    np.testing.assert_allclose(measure.direction_of(wave), [0.6, -0.8, 0.0], atol=1e-9)


# --------------------------------------------------------------------------
# the page's decode
# --------------------------------------------------------------------------


def fake_decoder(order: int = 2) -> BinauralDecoder:
    filters = np.random.default_rng(5).standard_normal((2, (order + 1) ** 2, 512))
    filters *= np.hanning(512)
    return BinauralDecoder(filters, FS, order, 256, 2000.0, True, "made up")


def test_the_head_s_rotation_is_the_library_s_yaw_and_puts_a_wave_where_the_head_sees_it() -> None:
    field = np.random.default_rng(0).standard_normal(16)
    turned = rotation(3, head_matrix(35.0)) @ field
    np.testing.assert_allclose(turned, rotate_yaw(field, -np.radians(35.0), 3), atol=1e-12)
    # A wave from the scene's front, the head pitched up 30 degrees and rolled: it comes
    # from below the nose.
    matrix = head_matrix(20.0, 30.0, 10.0)
    wave = real_sh(3, np.array([[1.0, 0.0, 0.0]]))[0]
    seen = real_sh(3, (np.array([1.0, 0.0, 0.0]) @ matrix)[None, :])[0]
    np.testing.assert_allclose(rotation(3, matrix) @ wave, seen, atol=1e-12)
    assert (np.array([1.0, 0.0, 0.0]) @ head_matrix(0.0, 30.0))[2] < 0.0


def test_the_block_decode_is_the_convolution_and_in_pieces_is_the_whole() -> None:
    decoder = fake_decoder()
    page = PageDecoder(decoder)
    signal = np.random.default_rng(1).standard_normal((9, 3000))
    still = page.decode(signal, lambda n: (0.0, 0.0, 0.0))
    direct = sum(
        np.stack([fftconvolve(signal[c], decoder.filters[e, c])[:3000] for e in range(2)])
        for c in range(9)
    )
    np.testing.assert_allclose(still, direct, atol=1e-5 * np.abs(direct).max())

    def pose(n: int) -> tuple[float, float, float]:  # a head that turns
        return (0.02 * n, 0.0, 0.0)

    whole = page.decode(signal[:, :2560], pose)
    first = page.decode(signal[:, :1536], pose)
    second = page.decode(signal[:, 1536:2560], pose, start=1536, before=signal[:, 1024:1536])
    np.testing.assert_allclose(np.concatenate([first, second], axis=1)[:, :1536], whole[:, :1536])
    # Past the join the first block fades from the head as it was: within a block's turn.
    assert np.abs(second - whole[:, 1536:]).max() < 0.2 * np.abs(whole).max()


# --------------------------------------------------------------------------
# a free field: every check has its answer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def head() -> BinauralDecoder:
    return page_decoder(None, order=7)


def resting() -> ScenePack:
    """Level B: the two sides of the crossover, a source 2 m ahead and to the right."""
    return synthetic_free_field(
        level="B", source=(1.2, 1.5, 1.6), listener_start=(0.0, 1.5, 0.0), duration_s=2.0
    )


@pytest.fixture(scope="module")
def at_rest(head: BinauralDecoder) -> list[Result]:
    pack = resting()
    return probe_source(pack, pack.sources["s1"], 4, SETTINGS, head)[0]


def test_at_rest_the_arrival_the_level_the_direction_and_the_ears_are_the_geometry_s(
    at_rest: list[Result],
) -> None:
    for test, limit in (
        ("arrival_time", 0.03),
        ("band_alignment", 0.2),
        ("direct_level", 0.3),
        ("direct_band_balance", 1.5),  # this pack's low band holds nothing under 160 Hz
        ("direction", 0.1),
    ):
        result = one(at_rest, test)
        assert result.status == PASS and result.value is not None
        assert abs(result.value) < limit, (test, result.value)
    assert one(at_rest, "seam_third_octaves:whole").status == PASS
    # This pack's low band is cut under 160 Hz by a zero phase filter of its own, which
    # rings 29 dB down before the arrival and round the response's end: the two checks
    # that read it are held on the whole band pack below.
    before = one(at_rest, "pre_arrival_energy").value
    assert before is not None and -32.0 < before < -27.0
    ears = one(at_rest, "binaural_left_right")
    assert ears.status == PASS
    assert ears.detail["left"]["itd_ms"] > 0.25 and ears.detail["right"]["itd_ms"] < -0.25
    # The source is at the scene's +z, which is the right of a head facing +x.
    assert ears.detail["scene"]["ild_db"] < -3.0 and ears.detail["scene"]["azimuth_deg"] < -50.0
    # A sphere is the same from the front and from behind: the check says it cannot tell.
    assert one(at_rest, "binaural_front_back").status in (SKIP, PASS)
    assert one(at_rest, "reverberation_plausible").status == SKIP


def test_a_whole_band_free_field_has_its_bank_s_ring_before_the_arrival_and_no_echo() -> None:
    pack = synthetic_free_field(source=(0.0, 1.5, -3.0), listener_start=(0.0, 1.5, 0.0))
    results = probe_source(pack, pack.sources["s1"], 2, SETTINGS, None)[0]
    for test in ("arrival_time", "direct_level", "direction", "late_echo"):
        assert one(results, test).status == PASS, (test, one(results, test).value)
    # With no crossover the octave bank is heard whole: it passes nothing under 88 Hz and
    # is zero phase, so 5 ms of it come before the arrival, 31 dB down. A pack with a low
    # band cuts that with its mask.
    assert one(results, "pre_arrival_energy").value == pytest.approx(-31.0, abs=1.5)


def seeded(pack: ScenePack, **early: Any) -> ScenePack:
    source = pack.sources["s1"]
    return replace(pack, sources={"s1": replace(source, early=replace(source.early, **early))})


def test_a_source_drawn_on_the_other_side_fails_the_direction(head: BinauralDecoder) -> None:
    pack = resting()
    mirrored = np.array(pack.sources["s1"].early.arrival) * np.array([1.0, 1.0, -1.0])
    wrong = seeded(pack, arrival=mirrored.astype(np.float32))
    results = probe_source(wrong, wrong.sources["s1"], 4, SETTINGS, None)[0]
    result = one(results, "direction")
    # 106 degrees over the crossover; under it the solved band still says where it is.
    assert result.status == FAIL and result.value == pytest.approx(106.3, abs=3.0)


def test_the_two_ears_exchanged_fail_left_and_right(head: BinauralDecoder) -> None:
    pack = resting()
    swapped = replace(head, filters=head.filters[::-1].copy())
    results = probe_source(pack, pack.sources["s1"], 4, SETTINGS, swapped)[0]
    assert one(results, "binaural_left_right").status == FAIL
    assert one(results, "direction").status == PASS  # the field itself is right


def test_the_two_bands_ten_milliseconds_apart_fail_and_so_does_a_level_9_db_off() -> None:
    pack = resting()
    # The fault of the first traced pack: the mirror on a clock 10.7 ms after the low band.
    late = replace(pack, mirror=replace(pack.mirror, lead_s=512 / FS))
    results = probe_source(late, late.sources["s1"], 4, SETTINGS, None)[0]
    apart = one(results, "band_alignment")
    assert apart.status == FAIL and apart.value == pytest.approx(-10.67, abs=0.3)
    before = one(results, "pre_arrival_energy")
    assert before.status == FAIL and before.value is not None and before.value > -16.0
    source = pack.sources["s1"]
    quiet = replace(
        pack,
        sources={
            "s1": replace(
                source, level=replace(source.level, high_gain_db=source.level.high_gain_db - 9.0)
            )
        },
    )
    results = probe_source(quiet, quiet.sources["s1"], 4, SETTINGS, None)[0]
    assert one(results, "direct_level").status == FAIL
    assert one(results, "direct_level").value == pytest.approx(-9.0, abs=0.3)
    assert one(results, "direct_band_balance").status == FAIL


def test_a_reflection_written_twice_is_named() -> None:
    pack = synthetic_free_field(duration_s=0.2)
    source = pack.sources["s1"]
    assert duplicate_arrivals(pack, source, range(4)).status == PASS
    early = source.early
    twice = replace(
        early,
        offsets=early.offsets * 2,
        path_id=np.repeat(early.path_id, 2)
        + np.tile(np.array([0, 1], np.uint64), early.order.size),
        **{
            name: np.repeat(getattr(early, name), 2, axis=0)
            for name in ("delay_s", "arrival", "departure", "gain", "order", "kind")
        },
    )
    result = duplicate_arrivals(pack, replace(source, early=twice), range(4))
    assert result.status == FAIL and result.value == 1.0  # the pair is 6 dB over the direct


# --------------------------------------------------------------------------
# moving, and the mix
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def walking() -> dict[str, Any]:
    """The listener walks at 1.5 m/s towards a source 6 m ahead; a noise plays in the middle."""
    pack = synthetic_free_field(
        source=(6.0, 1.5, 0.0),
        listener_start=(0.0, 1.5, 0.0),
        listener_end=(4.5, 1.5, 0.0),
        duration_s=3.0,
    )
    recipe = {
        "sources": [
            {
                "id": "s1",
                "activity": [
                    {
                        "start_s": 0.5,
                        "end_s": 1.5,
                        "clip_offset_s": 0.25,
                        "gain_db": -6.0,
                        "clip": {"library": "made", "name": "noise/hum"},
                    }
                ],
            }
        ]
    }
    return {"pack": pack, "recipe": recipe}


def test_a_walk_in_a_free_field_is_continuous_and_keeps_the_laws(walking: dict[str, Any]) -> None:
    outcome = check_pack(walking["pack"], settings=SETTINGS, families=("pack", "continuity"))
    results = outcome["results"]
    for test in ("tone_click", "noise_click", "zipper", "doppler", "level_distance"):
        assert one(results, test).status == PASS, (test, one(results, test).value)
    shift = one(results, "doppler")
    assert shift.detail["largest_shift_hz"] == pytest.approx(2500.0 * 1.5 / 343.2, rel=0.01)
    assert shift.value is not None and shift.value < 5.0
    slope = one(results, "level_distance").detail["slope_db_per_decade"]
    assert slope == pytest.approx(-20.0, abs=0.5)
    assert "the listener moves" in one(results, "tone_click").note


def test_the_mix_is_at_the_convention_s_level_and_silent_where_the_recipe_is(
    walking: dict[str, Any], tmp_path: Path
) -> None:
    hum = measure.pink_noise(96000, FS, seed=4) * 10.0 ** (-6.0 / 20.0)  # -26 dB re full scale
    (tmp_path / "made" / "noise").mkdir(parents=True)
    soundfile.write(str(tmp_path / "made" / "noise" / "hum.wav"), hum, 48000, subtype="FLOAT")
    outcome = check_pack(
        walking["pack"],
        recipe=walking["recipe"],
        clips=ClipSource(tmp_path),
        settings=SETTINGS,
        sphere_head=True,
        families=("mix",),
    )
    results = outcome["results"]
    level = one(results, "level_at_listener")
    # 60 dB SPL at 1 m, 6 dB under by the interval's gain, at 4.5 m or so: the free field's.
    assert level.status == PASS and level.value == pytest.approx(0.0, abs=0.5)
    assert level.detail["fed_at_1m_db"] == pytest.approx(54.0, abs=0.2)
    for test in ("audibility", "silence", "interval_edge", "binaural_peak"):
        assert one(results, test).status == PASS, test
    assert outcome["mix"]["mix_ears"].shape == (2, 3 * 48000)
    # The placeholder's stand-in is said, and the clip that is not there is noise.
    feed = feed_of(
        walking["pack"],
        "s1",
        {
            "sources": [
                {
                    "id": "s1",
                    "activity": [
                        {
                            "start_s": 0.0,
                            "end_s": 1.0,
                            "clip": {"library": "placeholder", "name": "voice_02"},
                        }
                    ],
                }
            ]
        },
        ClipSource(tmp_path),
    )
    assert feed.stand_in and "no clip to stand in" in feed.label


def test_an_interval_is_faded_at_its_ends_and_not_where_its_pieces_join() -> None:
    signal = np.ones(48000)

    def track(*spans: tuple[float, float]) -> DryTrack:
        activity = [
            {"start_s": a, "end_s": b, "clip_offset_s": a, "clip": {"name": "ones"}}
            for a, b in spans
        ]
        return DryTrack.from_recipe({"activity": activity}, lambda clip: (signal, FS), rate=FS)

    fade = int(EDGE_FADE_S * FS)
    whole = track((0.1, 0.5)).read(0, 48000)
    assert whole[4800] == 0.0 and whole[4800 + fade] == 1.0 and whole[24000 - 1] == 0.0
    assert np.all(np.diff(whole[4800 : 4800 + fade + 1]) >= 0.0)
    # A noise cut where it stands starts a whole sample high: that is the click.
    assert np.abs(np.diff(whole)).max() < 0.01
    np.testing.assert_array_equal(track((0.1, 0.3), (0.3, 0.5)).read(0, 48000), whole)


def test_the_command_writes_the_report_and_the_files_to_hear(
    walking: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = json.dumps(walking["recipe"]).encode()
    digest = hashlib.sha256(recipe).hexdigest()
    header = replace(walking["pack"].header, recipe_sha256=digest)
    pack = replace(walking["pack"], recipe=recipe, header=header)
    write_pack(tmp_path / "pack.h5", pack)
    out = tmp_path / "out"
    code = main(
        [
            "check",
            str(tmp_path / "pack.h5"),
            "--out",
            str(out),
            "--clips",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "none.json"),
            "--measured-head",
            str(tmp_path / "none.sofa"),
            "--reference",
            str(tmp_path / "none.h5"),
            "--probe-seconds",
            "1",
            "--workers",
            "1",
            "--window",
            "0",
            "2.5",
        ]
    )
    document = json.loads((out / "check.json").read_text())
    assert code == (1 if document["verdict"] == FAIL else 0)
    assert document["window_s"] == [0.0, 2.5] and document["sources"] == ["s1"]
    assert "sphere" in " ".join(document["notes"]) and "LIMITS" not in document
    assert {"tone_click", "direct_level", "level_at_listener"} <= {
        r["test"] for r in document["results"]
    }
    text = (out / "check.md").read_text()
    assert "## The limits and why" in text and "`band_alignment`" in text
    ears, rate = soundfile.read(str(out / "listen" / "mix.wav"))
    assert rate == 48000 and ears.shape == (int(2.5 * 48000), 2)
    assert "listen/mix.wav" in capsys.readouterr().out


def test_the_validated_field_is_read_at_the_nearest_point_of_the_same_scene() -> None:
    pack = synthetic_free_field(source=(0.0, 1.5, -3.0), listener_start=(0.0, 1.5, 0.0))
    own = probe_source(pack, pack.sources["s1"], 2, SETTINGS, None)[1]["response"]
    reference = {
        "name": "itself",
        "scene_id": "synthetic",
        "positions": np.array([[5.0, 1.5, 0.0], [0.1, 1.5, 0.0]]),
        "response": lambda index: own * (index == 1),
    }
    settings = CheckSettings(workers=1, reference=reference)
    results = probe_source(pack, pack.sources["s1"], 2, settings, None)[0]
    same = one(results, "late_spectrum_reference")
    assert same.status == PASS and same.value == pytest.approx(0.0, abs=1e-6)
    assert "0.10 m from the head" in same.note
    other = CheckSettings(workers=1, reference={**reference, "scene_id": "another dwelling"})
    results = probe_source(pack, pack.sources["s1"], 2, other, None)[0]
    assert not [r for r in results if r.test.endswith("_reference")]


def test_a_pack_on_the_field_s_own_source_is_held_to_it_sample_for_sample(tmp_path: Path) -> None:
    """``--reference-point``: the field is the pack's own render, then the same late and halved."""
    pack = resting()
    h = pack.header
    count = int(1.2 * FS)
    dry = DryTrack.from_array(np.array([1.0]), start_s=0.0, rate=FS)
    engine = Engine(pack, {"s1": dry}, settings=RenderSettings(directivity=False, workers=1))
    own = np.asarray(engine.stem("s1", 0, count)[0]) * FIELD_UNIT_AT_1M

    def field(name: str, response: np.ndarray, **attrs: Any) -> Path:
        path = tmp_path / name
        with h5py.File(path, "w") as handle:
            handle.attrs["scene_id"] = h.scene_id
            handle.attrs["sample_rate_hz"] = FS
            handle.attrs["source_position"] = np.asarray(pack.sources["s1"].position[0])
            handle.attrs["gain"] = 1.0
            for key, value in attrs.items():
                handle.attrs[key] = value
            handle["positions"] = np.array([[5.0, 1.5, 0.0], pack.listener.position[0]])
            handle["ir"] = np.stack([np.zeros_like(response), response])[:, None, :]
        return path

    same = reference.compare(pack, field("same.h5", own), workers=1)
    (found,) = same["points"]
    assert found["point"] == 1 and found["direct"] and found["head_off_the_point_m"] == 0.0
    heard = np.isfinite(found["images_level_db"])
    assert heard.sum() >= 15
    assert np.abs(found["images_level_db"][heard]).max() < 1e-3
    assert found["images_error_db"][heard].max() < -60.0
    assert np.abs(found["low_level_db"]).max() < 1e-3 and found["low_error_db"].max() < -60.0
    assert "s1 to lattice point 1" in reference.markdown(same)
    # The seeded fault: the field half a millisecond later and half as loud.
    late = np.zeros_like(own)
    late[24:] = 0.5 * own[:-24]
    fault = reference.compare(pack, field("late.h5", late), workers=1)["points"][0]
    np.testing.assert_allclose(fault["images_level_db"][heard], 6.02, atol=0.1)
    at_1k = int(np.argmin(np.abs(fault["third_octaves_hz"] - 1000.0)))
    assert fault["images_error_db"][at_1k] > 0.0
    assert fault["pack_arrival_ms"] == pytest.approx(fault["arrival_ms"] - 0.5, abs=0.05)
    # A field of another dwelling, or whose source is not the pack's, is refused.
    with pytest.raises(ValueError, match="scene"):
        reference.compare(pack, field("other.h5", own, scene_id="another"), workers=1)
    with pytest.raises(ValueError, match="stands on the field's source"):
        elsewhere = np.asarray(pack.sources["s1"].position[0]) + [1.0, 0.0, 0.0]
        reference.compare(pack, field("moved.h5", own, source_position=elsewhere), workers=1)


def test_a_placeholder_is_played_from_the_manifest_s_speaker_and_says_so(tmp_path: Path) -> None:
    clips = [
        {"name": f"voice/{speaker}_{n}", "kind": "voice", "speaker": speaker}
        for speaker in ("s01", "s02")
        for n in (1, 2)
    ] + [{"name": "noise/water_tap", "kind": "noise", "subtype": "water"}]
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps({"library": "made", "clips": clips}))
    for folder in ("voice", "noise"):
        (tmp_path / "made" / folder).mkdir(parents=True)
    for index, clip in enumerate(clips):
        samples = np.full(4800, 0.01 * (index + 1))
        soundfile.write(str(tmp_path / "made" / f"{clip['name']}.wav"), samples, 48000, "FLOAT")
    source = ClipSource(tmp_path, manifest)
    pack = synthetic_free_field(duration_s=1.0)

    def fed(name: str) -> np.ndarray:
        interval = {"start_s": 0.0, "end_s": 0.5, "clip_offset_s": 0.05}
        activity = [{**interval, "clip": {"library": "placeholder", "name": name}}]
        feed = feed_of(pack, "s1", {"sources": [{"id": "s1", "activity": activity}]}, source)
        assert feed.stand_in and name in feed.label and "of made" in feed.label
        return feed.track.read(0, 24000)

    # The second speaker's two clips end to end, from 50 ms in, and round again.
    voice = fed("voice_02")
    np.testing.assert_allclose(voice[[1000, 3000, 8000, 13000]], [0.03, 0.04, 0.03, 0.04], 1e-6)
    np.testing.assert_allclose(fed("noise_water")[1000:20000], 0.05, 1e-6)
