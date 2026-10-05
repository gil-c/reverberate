"""The levelling scalar steadied along a walk: what moves is averaged, what rests is kept.

The first whole scene's sources breathed by 5 to 8 dB within a second
above the crossover while anything walked (2026-10-05): each pair of the
low band is levelled on its own and neighbours 8 cm apart differ by
decibels. These hold the averaging that replaces a step's own seam, and
the command that brings a traced pack to it in place and back; then the
other ways of writing the table (one number for the scene, the pair's own
at the crossover fading to that number above it), the table given to the
engine in place of the pack's, and the trace's own option.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.metrics import octave_filter_rows
from reverberate.render.__main__ import main
from reverberate.render.benchmark import density_pack
from reverberate.render.dry import group_kernels
from reverberate.render.engine import Engine
from reverberate.render.pack import read_pack, synthetic_free_field, write_pack
from reverberate.render.relevel import (
    STEADY_S,
    levelled,
    pack_tables,
    relevel_pack,
    scene_constant_db,
    steadied,
    taper,
    wander_db,
)
from reverberate.trace.level import step_levels

BANK = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def walk() -> tuple[np.ndarray, np.ndarray]:
    """A source at rest, a silence, then a walk whose pairs' seams are 3 dB apart at random."""
    rng = np.random.default_rng(5)
    gain = np.full(400, -3.25, dtype=np.float32)
    gain[200:] = (-3.0 + 0.02 * np.arange(200) + rng.uniform(-1.5, 1.5, 200)).astype(np.float32)
    audible = np.ones(400, dtype=bool)
    audible[150:200] = False
    gain[150:200] = 0.0
    return gain, audible


def test_a_walk_is_steadied_and_what_rests_keeps_its_scalar_to_the_bit() -> None:
    gain, audible = walk()
    made = steadied(gain, audible, 40)
    assert made.dtype == np.float32
    assert np.array_equal(made[:200], gain[:200])  # at rest, and where nothing is heard
    # The walk keeps its trend, 4 dB over its ten seconds, and loses what went up and down.
    trend = -3.0 + 0.02 * np.arange(200)
    assert np.abs(made[240:360] - trend[40:160]).max() < 0.45
    assert wander_db(gain, audible, 10)["p90"] > 2.0
    assert wander_db(made, audible, 10)["max"] < 0.6
    # Nothing is averaged across the silence: the walk's first step does not know the rest.
    assert made[200] == pytest.approx(float(np.mean(gain[200:215])), abs=0.5)
    assert abs(made[200] - gain[149]) < 1.5
    # No window at all is the table itself, and a run shorter than the window is its own mean.
    assert np.array_equal(steadied(gain, audible, 0), gain)
    short = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    assert steadied(short, np.ones(3, dtype=bool), 40) == pytest.approx(2.0, abs=0.05)
    assert wander_db(np.full(50, -3.0), np.ones(50, dtype=bool), 10) == {
        "p90": 0.0,
        "p99": 0.0,
        "max": 0.0,
    }


def test_a_pack_is_relevelled_in_place_once_and_put_back(tmp_path: Path) -> None:
    pack = synthetic_free_field(
        source=(6.0, 1.5, 0.0),
        listener_start=(0.0, 1.5, 0.0),
        listener_end=(4.5, 1.5, 0.0),
        duration_s=3.0,
    )
    target = write_pack(tmp_path / "pack.h5", pack)
    wobble = np.random.default_rng(1).uniform(-2.0, 2.0, pack.header.steps).astype(np.float32)
    with h5py.File(target, "r+") as f:
        f["sources/s1/level/high_gain_db"][...] = wobble
    report = relevel_pack(target)
    row = report["sources"]["s1"]
    assert report["seconds"] == STEADY_S
    assert row["within_half_a_second_before_db"]["max"] > 2.5
    assert row["within_half_a_second_after_db"]["max"] < 0.8
    with read_pack(target) as read:  # still a pack the engine reads
        steady = np.array(read.sources["s1"].level.high_gain_db)
    assert np.ptp(steady) < 1.0 and np.ptp(wobble) > 3.5
    with h5py.File(target, "r") as f:
        assert np.array_equal(f["sources/s1/level/high_gain_db_traced"][...], wobble)
        assert f["sources/s1/level"].attrs["steadied_s"] == STEADY_S
    # A second call starts from the trace's table, not from the steadied one.
    relevel_pack(target, seconds=STEADY_S)
    with h5py.File(target, "r") as f:
        assert np.array_equal(f["sources/s1/level/high_gain_db"][...], steady)
    # The command puts the trace's table back and leaves nothing behind.
    assert main(["relevel", str(target), "--undo"]) == 0
    with h5py.File(target, "r") as f:
        level = f["sources/s1/level"]
        assert np.array_equal(level["high_gain_db"][...], wobble)
        assert "high_gain_db_traced" not in level and "steadied_s" not in level.attrs


def test_the_command_says_what_it_did(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = write_pack(tmp_path / "pack.h5", synthetic_free_field(duration_s=1.0))
    assert main(["relevel", str(target), "--seconds", "1"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["seconds"] == 1.0 and said["sources"]["s1"]["largest_change_db"] == 0.0


def test_one_number_for_the_scene_moves_nothing_and_a_taper_keeps_the_join() -> None:
    gain, audible = walk()
    base = -5.2
    constant = scene_constant_db(np.array([0.4, 1.9, 3.4, 8.0]))
    assert constant == pytest.approx(2.65)  # the median: the pair behind two walls does not pull
    flat = levelled(gain, audible, seam="constant", base_db=base, constant_db=constant)
    assert flat.dtype == np.float32 and flat.shape == gain.shape
    assert np.all(flat[audible] == np.float32(base + constant))
    assert np.array_equal(flat[~audible], gain[~audible])  # what is not heard keeps what it had
    assert wander_db(flat, audible, 10) == {"p90": 0.0, "p99": 0.0, "max": 0.0}
    # The pair's own way is the trace's table, and the smooth one is the averaging above.
    assert np.array_equal(levelled(gain, audible, seam="pair", base_db=base), gain)
    assert np.array_equal(
        levelled(gain, audible, seam="smooth", base_db=base, half_steps=40),
        steadied(gain, audible, 40),
    )
    # Tapered: the step's own seam to 1 kHz, half of it in decibels at 2 kHz, none from 4 kHz.
    assert taper(BANK) == pytest.approx([1, 1, 1, 1, 0.5, 0, 0, 0])
    assert taper(BANK, (1000.0, 16000.0)) == pytest.approx([1, 1, 1, 1, 0.75, 0.5, 0.25, 0])
    bands = levelled(
        gain, audible, seam="tapered", base_db=base, constant_db=constant, bank_hz=BANK
    )
    assert bands.shape == (gain.size, len(BANK)) and bands.dtype == np.float32
    assert np.array_equal(bands[:, 3], gain)  # nothing is left at the join
    assert bands[audible, 4] == pytest.approx(0.5 * (gain[audible] + flat[audible]), abs=1e-5)
    assert np.array_equal(bands[:, 6], flat)  # and nothing moves from 4 kHz up
    assert wander_db(bands[:, 4], audible, 10)["max"] == pytest.approx(
        0.5 * wander_db(gain, audible, 10)["max"], abs=0.02
    )
    with pytest.raises(ValueError, match="one of pair, smooth, constant, tapered"):
        levelled(gain, audible, seam="structured", base_db=base)
    with pytest.raises(ValueError, match="the bank's centres"):
        levelled(gain, audible, seam="tapered", base_db=base)
    with pytest.raises(ValueError, match="up to a higher one"):
        taper(BANK, (4000.0, 1000.0))


def traced_file(tmp_path: Path) -> tuple[Path, np.ndarray, float]:
    """A small pack with a low band, written as a trace would: the gain plus each step's seam."""
    pack = density_pack(duration_s=0.2, moving=True, bins=30)
    target = write_pack(tmp_path / "pack.h5", pack)
    rng = np.random.default_rng(2)
    with h5py.File(target, "r+") as f:
        f["mirror"].attrs["alignment_gain"] = 0.5
        low = f["sources/s1/low"]
        seams = rng.uniform(0.0, 4.0, low["seam_db"].shape[0]).astype(np.float32)
        low["seam_db"][...] = seams
        level = f["sources/s1/level/high_gain_db"]
        traced = (20.0 * np.log10(0.5) + rng.uniform(0.0, 4.0, level.shape[0])).astype(np.float32)
        level[...] = traced
    return target, traced, float(np.median(seams))


def test_a_pack_takes_each_way_in_place_and_the_trace_s_table_comes_back_to_the_bit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target, traced, median = traced_file(tmp_path)
    before = target.read_bytes()
    # Nothing is written by a dry run, nor ever for a level a band.
    said = relevel_pack(target, seam="constant", dry_run=True)
    assert said["written"] is False and said["constant_db"] == pytest.approx(median, abs=1e-3)
    row = said["sources"]["s1"]
    assert row["within_a_second_after_db"]["max"] == 0.0
    assert row["left_at_the_join_db"]["largest"] > 1.0  # what a constant leaves, and says
    tapered = relevel_pack(target, seam="tapered")
    assert tapered["written"] is False and tapered["kept_of_the_pair"][3:6] == [1.0, 0.5, 0.0]
    assert tapered["sources"]["s1"]["left_at_the_join_db"]["largest"] == 0.0
    assert tapered["sources"]["s1"]["within_half_a_second_after_db"]["8000"]["max"] == 0.0
    assert target.read_bytes() == before
    # One number, the scene's or the one given; the way is recorded where a reader looks.
    assert main(["relevel", str(target), "--seam", "constant"]) == 0
    assert json.loads(capsys.readouterr().out)["written"] is True
    with h5py.File(target, "r") as f:
        level = f["sources/s1/level"]
        flat = np.float32(20.0 * np.log10(0.5) + median)
        assert np.all(level["high_gain_db"][...] == flat)
        assert np.array_equal(level["high_gain_db_traced"][...], traced)
        assert level.attrs["seam"] == "constant" and "steadied_s" not in level.attrs
        assert json.loads(f.attrs["provenance_json"])["seam"]["seam"] == "constant"
    relevel_pack(target, seam="constant", constant_db=2.6)
    with read_pack(target) as read:
        assert read.header.provenance["seam"]["constant_db"] == 2.6
        assert np.all(read.sources["s1"].level.high_gain_db == np.float32(-6.0206 + 2.6))
    # Another way starts from the trace's table, and the pair's own way is the way back.
    relevel_pack(target, seam="smooth")
    steps = traced.size
    with h5py.File(target, "r") as f:
        assert f["sources/s1/level"].attrs["seam"] == "smooth"
        assert np.array_equal(
            f["sources/s1/level/high_gain_db"][...],
            steadied(traced, np.ones(steps, dtype=bool), 40),
        )
    assert main(["relevel", str(target), "--seam", "pair"]) == 0
    with h5py.File(target, "r") as f:
        level = f["sources/s1/level"]
        assert np.array_equal(level["high_gain_db"][...], traced)
        assert "high_gain_db_traced" not in level and not set(level.attrs) & {"seam", "steadied_s"}
        assert "seam" not in json.loads(f.attrs["provenance_json"])


def test_a_table_given_to_the_engine_is_the_pack_with_that_table_written_in_it(
    tmp_path: Path,
) -> None:
    target, traced, _ = traced_file(tmp_path)
    steps = traced.size
    dry = {"s1": np.random.default_rng(4).standard_normal(int(0.2 * 48000))}
    tables, _ = pack_tables(target, seam="constant")
    with read_pack(target) as pack:
        own = Engine(pack, dry).stem("s1")
        given = Engine(pack, dry, high_gain_db=tables).stem("s1")
        # A level a band whose bands are all one is that scalar; the low band is untouched.
        wide = np.repeat(tables["s1"][:, None], len(BANK), axis=1)
        banded = Engine(pack, dry, high_gain_db={"s1": wide}).stem("s1")
        np.testing.assert_allclose(banded, given, atol=1e-12 * np.abs(given).max())
        # 6 dB more from 4 kHz up is 6 dB more there in both parts, and nothing at 1 kHz.
        up = wide.copy()
        up[:, 5:] += 6.0
        was, raised = (
            Engine(pack, dry, high_gain_db=tables),
            Engine(pack, dry, high_gain_db={"s1": up}),
        )
        for part in ("early", "tail"):
            read = [
                np.sum(
                    octave_filter_rows(
                        np.repeat(engine.stem("s1", parts=(part,))[0:1], 8, 0), 48000, np.arange(8)
                    )
                    ** 2,
                    axis=1,
                )
                for engine in (was, raised)
            ]
            change = 10.0 * np.log10(read[1] / read[0])
            assert change[3] == pytest.approx(0.0, abs=0.05), part
            assert change[6:] == pytest.approx(6.0, abs=0.05), part
        with pytest.raises(ValueError, match=rf"\[{steps}\] or \[{steps}, 8\]"):
            Engine(pack, dry, high_gain_db={"s1": np.zeros(3)}).stem("s1")
    assert not np.array_equal(own, given)
    relevel_pack(target, seam="constant")
    with read_pack(target) as pack:
        np.testing.assert_array_equal(Engine(pack, dry).stem("s1"), given)


def test_the_filters_that_share_a_signal_among_bands_add_to_it() -> None:
    kernels = group_kernels(48000.0, BANK, ((0, 1, 2, 3), (4,), (5, 6, 7)))
    total = np.sum(kernels, axis=0)
    assert total[total.size // 2] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(np.delete(total, total.size // 2)).max() < 1e-12
    freqs = np.fft.rfftfreq(8192, 1 / 48000.0)
    held = [np.abs(np.fft.rfft(k, 8192)) for k in kernels]
    for group, (centre, others) in enumerate([(1000, (2000, 4000)), (2000, (1000, 4000))]):
        assert held[group][np.argmin(np.abs(freqs - centre))] == pytest.approx(1.0, abs=0.02)
        for other in others:
            assert held[group][np.argmin(np.abs(freqs - other))] < 0.02
    alone = group_kernels(48000.0, BANK, (tuple(range(8)),))[0]
    assert alone[alone.size // 2] == 1.0 and np.count_nonzero(alone) == 1
    with pytest.raises(ValueError, match="every band of the bank once"):
        group_kernels(48000.0, BANK, ((0, 1), (1, 2)))


def test_the_trace_writes_the_table_the_way_it_is_told() -> None:
    steps = 6
    early: Any = SimpleNamespace(
        rows=lambda row: slice(0, 0),
        delay_s=np.zeros(0),
        listener=np.zeros((steps, 3)),
        source=np.tile([3.432, 0.0, 0.0], (steps, 1)),
        sound_speed_m_s=343.2,
    )
    pair = np.full((steps, 2, 2), -1, dtype=np.int32)
    pair[:, 0, 0] = np.arange(steps) % 3
    audible = np.ones(steps, dtype=bool)
    audible[5] = False
    asked: dict[str, Any] = {
        "audible": audible,
        "pair": pair,
        "position_weight": np.zeros(steps, dtype=np.float32),
        "cell": np.zeros((steps, 2), dtype=np.int32),
        "mode": np.ones(steps, dtype=np.uint8),
        "listener": np.zeros((steps, 3)),
        "cells": np.zeros((1, 3)),
        "seam_db_of": np.array([0.5, 2.0, 3.5]),
        "trail_s_of": np.array([0.011, 0.012, 0.013]),
        "early": early,
        "alignment_gain": 0.5,
    }
    own, onset = step_levels(**asked)
    assert own[:5] == pytest.approx(-6.0206 + np.array([0.5, 2.0, 3.5, 0.5, 2.0]), abs=1e-4)
    assert own[5] == 0.0 and onset[:3] == pytest.approx([0.021, 0.022, 0.023])
    flat, same = step_levels(**asked, seam="constant", constant_db=2.0)
    assert np.all(flat[:5] == np.float32(-6.0206 + 2.0)) and flat[5] == 0.0
    assert np.array_equal(same, onset)  # where the two are joined in time is the pairs' own
    smooth, _ = step_levels(**asked, seam="smooth", half_steps=40)
    assert np.array_equal(smooth, steadied(own, audible, 40))
    with pytest.raises(ValueError, match="a pack does not"):
        step_levels(**asked, seam="tapered")
