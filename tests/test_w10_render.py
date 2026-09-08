"""The render path, exercised end to end on a run that was never solved.

A rented run produces one file of over a gigabyte after three hours on a card.
Discovering a shape error or a missing key at that point costs the rental. So
the whole path from ``sim_outs.h5`` to the report runs here on a hand built run
of a few kilobytes, with a field whose direction is known so the answer can be
checked rather than merely produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.audio import Atmosphere
from reverberate.experiments.w10_ambisonic import COMMS_NAME
from reverberate.experiments.w10_render import (
    band_directions,
    binaural_measures,
    decoders,
    room_report,
    sphere_head,
    write_audio,
)
from reverberate.spatial.array import design_array
from reverberate.spatial.encode import Ambisonic, EncoderSettings
from reverberate.spatial.sh import channel_count, directions, real_sh
from reverberate.wave.comms import Grid

RATE = 48000.0
SOUND_SPEED = 343.2
STEP = 0.008171428571428571


def a_grid(nodes: int = 200) -> Grid:
    axis = np.arange(nodes) * STEP
    return Grid(
        h=STEP,
        Ts=STEP / SOUND_SPEED / np.sqrt(3.0),
        l2=1 / 3,
        fcc_flag=0,
        xv=axis,
        yv=axis,
        zv=axis,
    )


def a_solved_run(tmp_path: Path, *, samples: int = 2048) -> Path:
    """A run directory holding everything the render reads, and a plane wave in it.

    The pressures are a plane wave arriving from the scene direction of the
    source, sampled at the array's own nodes, so the direction of arrival the
    report measures has a right answer.
    """
    grid = a_grid()
    centre = np.full(3, 100 * STEP)
    source = centre + np.array([1.5, 0.0, 0.0])
    design = design_array(centre, grid, fit_order=6, outer_radius_m=0.16)
    run = tmp_path / "w10_test"
    (run / "comms").mkdir(parents=True)
    (run / "source0").mkdir(parents=True)
    np.save(run / "array_positions.npy", design.positions)

    extras = [[float(v) for v in centre + np.array([0.5, 0.0, 0.0])]]
    plan = {
        "run": run.name,
        "scene_id": "test",
        "room": "box",
        "cache_key": "0" * 32,
        "geometry_sha256": "1" * 64,
        "reference_run": "w29_16k",
        "reference_note": "synthetic",
        "sample_rate_hz": RATE,
        "sound_speed_m_s": SOUND_SPEED,
        "duration_s": samples / RATE,
        "samples": samples,
        "source": [float(v) for v in source],
        "centre": [float(v) for v in centre],
        "extra_receivers": extras,
        "clearance": {"boundary_nodes_in_ball": 0},
        "cost": {"grid_points": 1, "receivers": design.count + 1},
        "array": design.record(),
        "encoder": EncoderSettings().record(),
        "conditioning": {"effective_order": [7]},
    }
    (run / "plan.json").write_text(json.dumps(plan))

    # A plane wave from the source's own direction, delayed at each node by the
    # distance it sits along that direction, plus a decaying tail so the octave
    # band measures have something to measure.
    #
    # The delay is applied in the frequency domain rather than by moving a
    # sample. Whole sample delays quantise to 20.8 microseconds, which is four
    # per cent of the 466 microsecond spread across a 16 cm array, and a fixture
    # that coarse tests the rounding rather than the encoder.
    from reverberate.spatial.sh import scene_to_ambisonic

    unit = scene_to_ambisonic((source - centre)[None, :])[0]
    unit = unit / np.linalg.norm(unit)
    offsets = scene_to_ambisonic(design.offsets) @ unit
    rows = np.vstack([design.positions, np.asarray(extras, dtype=float)])
    rng = np.random.default_rng(0)
    time = np.arange(samples) / RATE
    frequency = np.fft.rfftfreq(samples, 1.0 / RATE)
    impulse = np.zeros(samples)
    impulse[0] = 1.0
    spectrum = np.fft.rfft(impulse)
    pressure = np.zeros((rows.shape[0], samples))
    for index in range(rows.shape[0]):
        along = offsets[index] if index < design.count else 0.0
        delay = 0.004 - along / SOUND_SPEED
        pressure[index] = np.fft.irfft(
            spectrum * np.exp(-2j * np.pi * frequency * delay), n=samples
        )
        pressure[index] += 0.001 * rng.standard_normal(samples) * np.exp(-time / 0.05)

    with h5py.File(run / "source0" / "sim_outs.h5", "w") as handle:
        handle.create_dataset("u_out", data=pressure)
    with h5py.File(run / "comms" / COMMS_NAME, "w") as handle:
        handle.create_dataset("out_alpha", data=np.ones((rows.shape[0], 1)))
        handle.create_dataset("diff", data=np.int8(0))
    with h5py.File(run / "source0" / "sim_consts.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(STEP))
        handle.create_dataset("SR", data=np.float64(RATE))
    return run


def test_the_report_reads_a_run_end_to_end_and_points_at_the_source(tmp_path: Path) -> None:
    """The whole path, on a run of a few kilobytes rather than of a gigabyte."""
    run = a_solved_run(tmp_path)
    report = room_report(
        run,
        EncoderSettings(order=3, fit_order=7, max_frequency_hz=4000.0),
        air=Atmosphere(),
        lowcut_hz=40.0,
        yaws_deg=(0.0, 90.0),
        filter_length=256,
    )
    assert report["binaural"] is True
    assert report["air_absorption"]["humidity_percent"] == 50.0
    assert report["air_absorption"]["standard"] == "ISO 9613-1"
    usable = [row for row in report["direction_of_arrival"] if row["usable"]]
    assert usable, "no band had enough cycles to place a direction"
    assert max(row["error_deg"] for row in usable) < 1.0
    assert len(report["extra_receivers"]) == 1
    assert report["omnidirectional"]["rt60_s"]
    assert set(report["binaural_decodes"]) == {"sphere_magls", "sphere_plain"}
    assert set(report["binaural_decodes"]["sphere_magls"]["measures"]) == {"yaw_0", "yaw_90"}


def test_skipping_air_absorption_is_recorded_as_an_omission(tmp_path: Path) -> None:
    """A response read without knowing this overstates its own treble by 38 per cent."""
    run = a_solved_run(tmp_path, samples=1024)
    report = room_report(
        run,
        EncoderSettings(order=1, fit_order=5, max_frequency_hz=4000.0),
        air=None,
        lowcut_hz=40.0,
        yaws_deg=(0.0,),
        filter_length=256,
    )
    assert report["air_absorption"]["applied"] is False
    assert "overstated" in report["air_absorption"]["why"]


def test_the_report_is_json_and_carries_no_infinities(tmp_path: Path) -> None:
    """A metric that could not be measured is null, never a number that looks like one."""
    run = a_solved_run(tmp_path, samples=1024)
    report = room_report(
        run,
        EncoderSettings(order=1, fit_order=5, max_frequency_hz=4000.0),
        air=None,
        lowcut_hz=40.0,
        yaws_deg=(0.0,),
        filter_length=256,
    )
    text = json.dumps(report)
    assert "Infinity" not in text
    assert "NaN" not in text


def test_both_decoders_are_built_and_named(tmp_path: Path) -> None:
    built = decoders(3, RATE, filter_length=256)
    assert built["sphere_magls"].cut_on_hz == 2000.0
    assert built["sphere_plain"].covariance_constrained is False
    assert "sphere" in built["sphere_magls"].head


def test_the_head_is_sampled_on_the_decoder_s_own_grid() -> None:
    head = sphere_head(RATE, 256)
    assert np.allclose(head.frequency_hz, np.fft.rfftfreq(256, 1.0 / RATE))
    assert head.weights is not None


def test_a_band_with_too_few_cycles_is_marked_unusable() -> None:
    """Calibrated on W10's rehearsal: under a period the intensity vector reverses."""
    order = 3
    signals = np.zeros((channel_count(order), 256))
    direction = directions(np.array(0.3), np.array(0.1))
    signals[:, 40] = real_sh(order, direction[None, :])[0]
    rows = band_directions(
        Ambisonic(signals, RATE, order, np.zeros(3)),
        direction,
        bands_hz=(125.0, 4000.0),
        window_s=0.001,
    )
    assert rows[0]["usable"] is False
    assert rows[1]["usable"] is True


def test_the_binaural_measures_report_the_delay_beside_what_a_sphere_predicts() -> None:
    rate = 48000.0
    brir = np.zeros((2, 512))
    brir[0, 100] = 1.0
    brir[1, 130] = 1.0
    measures = binaural_measures(brir, rate, np.radians(90.0))
    assert measures["itd_us"] == pytest.approx(30 / rate * 1e6, abs=25.0)
    assert measures["woodworth_itd_us"] > 500.0
    assert measures["ild_db"] == pytest.approx(0.0, abs=0.01)


def test_one_gain_covers_every_file_so_the_ears_stay_comparable(tmp_path: Path) -> None:
    """A gain per file would make a shadowed ear as loud as a near one.

    Interaural level difference is the cue this dataset exists to carry, and
    within one ambisonic file the direction is nothing but the ratios between
    the channels. So the peak is taken over everything written together.
    """
    pytest.importorskip("soundfile")
    rng = np.random.default_rng(5)
    ambisonic = Ambisonic(rng.standard_normal((16, 512)) * 0.1, RATE, 3, np.zeros(3))
    quiet = np.zeros((2, 256))
    quiet[0, 10] = 0.01
    loud = np.zeros((2, 256))
    loud[0, 10] = 1.0
    brirs = {"sphere": {0.0: loud, 90.0: quiet}}
    written = write_audio(ambisonic, brirs, np.array([1.0, 0.0, 0.0]), tmp_path / "audio")
    gains = {entry["write_gain"] for entry in written}
    assert len(gains) == 1
    peaks = {str(entry["path"]): float(entry["peak"]) for entry in written if "peak" in entry}
    assert peaks["binaural_sphere_yaw0.wav"] > 50.0 * peaks["binaural_sphere_yaw90.wav"]


def test_the_ambisonic_wav_is_written_beside_the_ears(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    rng = np.random.default_rng(6)
    ambisonic = Ambisonic(rng.standard_normal((16, 256)) * 0.1, RATE, 3, np.zeros(3))
    brir = np.zeros((2, 128))
    brir[:, 5] = 0.5
    written = write_audio(ambisonic, {"sphere": {0.0: brir}}, np.array([1.0]), tmp_path / "a")
    names = [entry["path"] for entry in written]
    assert "ambisonic_acn_sn3d.wav" in names
    samples, rate = soundfile.read(str(tmp_path / "a" / "ambisonic_acn_sn3d.wav"))
    assert samples.shape[1] == 16
    assert rate == int(RATE)


def test_the_report_hands_back_what_it_rendered_so_nothing_is_encoded_twice(
    tmp_path: Path,
) -> None:
    run = a_solved_run(tmp_path, samples=1024)
    rendered: dict[str, object] = {}
    room_report(
        run,
        EncoderSettings(order=1, fit_order=5, max_frequency_hz=4000.0),
        air=None,
        lowcut_hz=40.0,
        yaws_deg=(0.0, 90.0),
        filter_length=256,
        rendered=rendered,
    )
    assert isinstance(rendered["ambisonic"], Ambisonic)
    brirs = rendered["brirs"]
    assert isinstance(brirs, dict)
    assert set(brirs) == {"sphere_magls", "sphere_plain"}
    assert sorted(brirs["sphere_magls"]) == [0.0, 90.0]
