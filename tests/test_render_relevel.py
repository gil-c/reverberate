"""The levelling scalar steadied along a walk: what moves is averaged, what rests is kept.

The first whole scene's sources breathed by 5 to 8 dB within a second
above the crossover while anything walked (2026-10-05): each pair of the
low band is levelled on its own and neighbours 8 cm apart differ by
decibels. These hold the averaging that replaces a step's own seam, and
the command that brings a traced pack to it in place and back.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.render.__main__ import main
from reverberate.render.pack import read_pack, synthetic_free_field, write_pack
from reverberate.render.relevel import STEADY_S, relevel_pack, steadied, wander_db


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
