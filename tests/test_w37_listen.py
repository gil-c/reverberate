"""Tests for the synthetic tail listening test.

The page exists to let one decay be compared against another by ear, so the
tests are mostly about the things that would silently destroy that comparison:
a gain applied per file, a variant that is not actually different, or a picker
that says "source 1" when a listener needs to know which of three renderings
they are hearing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from reverberate.experiments.w37_listen import VARIANTS, build, shared_gain
from reverberate.response import Provenance, ResponseSet, write_raw

SOUND_SPEED = 343.2
SAMPLE_RATE = 48000.0
SCENE = "f" * 64


def _report() -> dict[str, Any]:
    return {
        "cost": {"grid_points": 4_607_993_520, "steps": 291_276},
        "cache_key": "0" * 32,
        "model_json": "models/scene.json",
        "room": {
            "per_class": [
                {"label": "wall", "area_m2": 100.0, "random_incidence_absorption": [0.15] * 7},
                {"label": "carpet", "area_m2": 25.0, "random_incidence_absorption": [0.45] * 7},
            ]
        },
        "omissions": ["an existing omission that must survive"],
    }


def _plan() -> dict[str, Any]:
    return {
        "scene_id": "102344022",
        "room": "bedroom.001",
        "placement": {
            "sources": [{"position": [0.0, 1.0, 0.0], "archetype": "speaker"}],
            "receivers": [{"position": [1.0, 1.0, 0.0]}, {"position": [2.0, 1.0, 0.0]}],
        },
    }


def _write_run(root: Path, name: str, *, fmax_hz: float, seconds: float, with_audio: bool) -> Path:
    run = root / name
    (run / "responses").mkdir(parents=True)
    rng = np.random.default_rng(20250101)
    samples = int(seconds * SAMPLE_RATE)
    times = np.arange(samples) / SAMPLE_RATE
    ir = rng.standard_normal((2, samples)) * 10.0 ** (-60.0 * times / (20.0 * 0.30))
    write_raw(
        ResponseSet(
            ir=ir,
            sample_rate_hz=SAMPLE_RATE,
            source_position=np.zeros(3),
            receiver_positions=np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
            provenance=Provenance(
                scene_sha256=SCENE,
                mats_hash="0" * 32,
                engine="cpu",
                band="wave",
                fmax_hz=fmax_hz,
                grid_step_m=343.2 / (fmax_hz * 10.5),
                points_per_wavelength=10.5,
                sound_speed_m_s=SOUND_SPEED,
                seed=20250101,
                run_id=name,
                notes="two bare points",
            ),
        ),
        run / "responses" / "source0.h5",
    )
    (run / "report.json").write_text(json.dumps(_report()))
    (run / "plan.json").write_text(json.dumps(_plan()))
    if with_audio:
        (run / "audio").mkdir()
        voice = rng.standard_normal(int(0.5 * SAMPLE_RATE)) * 0.1
        soundfile.write(
            str(run / "audio" / "dry_voice.wav"), voice, int(SAMPLE_RATE), subtype="FLOAT"
        )
    return run


def _pair(root: Path) -> tuple[Path, Path]:
    return (
        _write_run(root, "high", fmax_hz=16000.0, seconds=0.8, with_audio=True),
        _write_run(root, "mid", fmax_hz=4000.0, seconds=0.8, with_audio=False),
    )


def test_the_gain_is_shared_across_every_file(tmp_path: Path) -> None:
    """A gain per file would make a dead response as loud as a full one."""
    high, mid = _pair(tmp_path)
    report = build(high, mid, tmp_path / "out", window_ms=60.0)
    gains = {entry["write_gain"] for entry in report["audio"]}
    assert len(gains) == 1
    assert report["write_gain"] == pytest.approx(next(iter(gains)))


def test_shared_gain_is_set_by_the_loudest_block() -> None:
    quiet = np.full(10, 0.1)
    loud = np.full(10, 0.5)
    gain = shared_gain([quiet, loud], headroom_db=0.0)
    assert (loud * gain).max() == pytest.approx(1.0)
    assert (quiet * gain).max() == pytest.approx(0.2)


def test_a_silent_set_does_not_divide_by_zero() -> None:
    assert shared_gain([np.zeros(8)]) == 1.0


def test_three_variants_are_written_and_named(tmp_path: Path) -> None:
    high, mid = _pair(tmp_path)
    report = build(high, mid, tmp_path / "out", window_ms=60.0)
    assert [source["variant"] for source in report["sources"]] == [name for name, _ in VARIANTS]
    for source in report["sources"]:
        assert source["label"].startswith(source["variant"])
        assert (tmp_path / "out" / "responses" / f"source{source['source_index']}.h5").is_file()


def test_the_truncated_variant_really_is_empty_after_the_window(tmp_path: Path) -> None:
    """The control, and it must actually be a control."""
    high, mid = _pair(tmp_path)
    build(high, mid, tmp_path / "out", window_ms=60.0)
    from reverberate.response import read_raw

    truncated = read_raw(tmp_path / "out" / "responses" / "source1.h5").ir
    cut = int(0.060 * SAMPLE_RATE)
    assert np.all(truncated[:, cut:] == 0.0)
    assert np.any(truncated[:, :cut] != 0.0)


def test_the_synthetic_variant_keeps_the_computed_part_and_fills_the_rest(tmp_path: Path) -> None:
    high, mid = _pair(tmp_path)
    build(high, mid, tmp_path / "out", window_ms=60.0)
    from reverberate.response import read_raw

    reference = read_raw(tmp_path / "out" / "responses" / "source0.h5").ir
    synthetic = read_raw(tmp_path / "out" / "responses" / "source2.h5").ir
    intact = int(0.045 * SAMPLE_RATE)
    assert np.allclose(synthetic[:, :intact], reference[:, :intact], atol=1e-9)
    assert np.any(synthetic[:, int(0.2 * SAMPLE_RATE) :] != 0.0)


def test_one_audio_file_per_variant_and_receiver(tmp_path: Path) -> None:
    high, mid = _pair(tmp_path)
    report = build(high, mid, tmp_path / "out", window_ms=60.0)
    names = {entry["path"] for entry in report["audio"]}
    assert len(names) == len(VARIANTS) * 2
    assert "source2_receiver1_wet.wav" in names
    assert (tmp_path / "out" / "audio" / "dry_voice.wav").is_file()


def test_the_placement_lists_one_source_per_variant(tmp_path: Path) -> None:
    """The viewer lines its source list up with its samples."""
    high, mid = _pair(tmp_path)
    report = build(high, mid, tmp_path / "out", window_ms=60.0)
    archetypes = [entry["archetype"] for entry in report["placement"]["sources"]]
    assert archetypes == [name for name, _ in VARIANTS]


def test_the_saving_is_reported_with_its_rate(tmp_path: Path) -> None:
    high, mid = _pair(tmp_path)
    report = build(high, mid, tmp_path / "out", window_ms=60.0)
    cost = report["cost"]
    assert cost["usd_per_hour_per_card"] == 0.917
    assert cost["saving"] == pytest.approx(0.8 / 0.060, rel=0.02)


def test_the_existing_omissions_survive_and_the_new_ones_are_added(tmp_path: Path) -> None:
    high, mid = _pair(tmp_path)
    report = build(high, mid, tmp_path / "out", window_ms=60.0)
    assert "an existing omission that must survive" in report["omissions"]
    assert any("noise with a measured per-band decay" in text for text in report["omissions"])


def test_two_geometries_cannot_be_compared(tmp_path: Path) -> None:
    high, _ = _pair(tmp_path)
    other = _write_run(tmp_path, "other", fmax_hz=4000.0, seconds=0.8, with_audio=False)

    # Rewrite the mid run under a different scene digest.
    from reverberate.response import read_raw

    response = read_raw(other / "responses" / "source0.h5")
    write_raw(
        ResponseSet(
            ir=response.ir,
            sample_rate_hz=response.sample_rate_hz,
            source_position=response.source_position,
            receiver_positions=response.receiver_positions,
            provenance=Provenance(**{**vars(response.provenance), "scene_sha256": "a" * 64}),
        ),
        other / "responses" / "source0.h5",
    )
    with pytest.raises(ValueError, match="different geometries"):
        build(high, other, tmp_path / "out", window_ms=60.0)


def test_a_reference_without_rendered_audio_is_refused(tmp_path: Path) -> None:
    """Rather than choosing a different dry voice and quietly changing the test."""
    high = _write_run(tmp_path, "high", fmax_hz=16000.0, seconds=0.8, with_audio=False)
    mid = _write_run(tmp_path, "mid", fmax_hz=4000.0, seconds=0.8, with_audio=False)
    with pytest.raises(ValueError, match="render the reference run's audio first"):
        build(high, mid, tmp_path / "out", window_ms=60.0)
