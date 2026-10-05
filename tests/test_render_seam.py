"""The tapered join and the gains: a pair's own seam at the crossover, one number above it.

The first whole scene's sources breathed by 2 to 5 dB within half a second
above the crossover while anything walked (2026-10-05): a pair's seam,
which makes the two bands meet at 1 kHz, was laid on everything above it.
These hold the level a band that replaces it (whole at 1 kHz, half at
2 kHz, none from 4 kHz), what the engine does with it, the command that
gives it to a traced pack in place, and the one that sets a source's gain.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.render.__main__ import main
from reverberate.render.benchmark import density_pack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.gains import set_gains
from reverberate.render.pack import ScenePack, read_pack, write_pack
from reverberate.render.seam import (
    TAPER,
    applied_gain_db,
    band_levels,
    group_kernels,
    motion_db,
    scene_constant_db,
    taper_of,
    taper_pack,
)

FS = 48000
BANK = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)
#: The alignment's gain plus the scene's constant, and a walk's seams round it.
LEVEL_DB = -3.3


def band_noise(low_hz: float, high_hz: float, seconds: float = 0.3) -> np.ndarray:
    n = int(round(seconds * FS))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    shape = np.clip((freqs - low_hz) / 50.0, 0, 1) * np.clip((high_hz - freqs) / 50.0, 0, 1)
    white = np.random.default_rng(3).standard_normal(n)
    return np.asarray(np.fft.irfft(np.fft.rfft(white) * shape, n) * np.hanning(n))


def walked(pack: ScenePack, scalar: np.ndarray, table: np.ndarray | None = None) -> ScenePack:
    """``pack`` with its source's scalar, and its level a band if one is given."""
    source = pack.sources["s1"]
    level = replace(source.level, high_gain_db=scalar, band_gain_db=table)
    return replace(pack, sources={"s1": replace(source, level=level)})


def scalar_of(pack: ScenePack, spread_db: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (LEVEL_DB + rng.uniform(-spread_db, spread_db, pack.header.steps)).astype(np.float32)


def test_the_taper_is_whole_at_the_crossover_half_an_octave_up_and_none_from_4_khz() -> None:
    shares = taper_of(BANK, 1000.0)
    assert shares.tolist() == [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    assert taper_of(BANK, 1000.0, (0.0,)).tolist() == [0.0] * 8
    assert taper_of(BANK, 2000.0, (1.0, 0.25)).tolist() == [1.0] * 5 + [0.25] * 3
    with pytest.raises(ValueError, match="shares between 0 and 1"):
        taper_of(BANK, 1000.0, (1.5,))
    # Each pair once, in the pack's single precision; nothing is 0 dB.
    assert scene_constant_db([np.array([0.5, 3.0]), np.array([1.9])]) == np.float32(1.9)
    assert scene_constant_db([]) == 0.0


def test_a_level_a_band_keeps_1_khz_to_the_bit_and_stands_still_from_4_khz() -> None:
    rng = np.random.default_rng(2)
    scalar = (LEVEL_DB + rng.uniform(-3.0, 3.0, 400)).astype(np.float32)
    audible = np.ones(400, dtype=bool)
    audible[100:140] = False
    scalar[100:140] = 0.0
    made = band_levels(scalar, audible, LEVEL_DB, taper_of(BANK, 1000.0))
    assert made.dtype == np.float32 and made.shape == (400, 8)
    for band in range(4):  # to the crossover's band: the trace's scalar, bit for bit
        assert np.array_equal(made[:, band], scalar)
    assert np.all(made[audible, 5:] == np.float32(LEVEL_DB))
    assert np.array_equal(made[~audible], np.zeros((40, 8), dtype=np.float32))
    np.testing.assert_allclose(
        made[audible, 4], LEVEL_DB + 0.5 * (scalar[audible] - LEVEL_DB), atol=1e-6
    )
    before = motion_db(scalar, audible, 10)
    assert before["p90"] > 4.0 and motion_db(made[:, 3], audible, 10) == before
    assert motion_db(made[:, 4], audible, 10)["p90"] == pytest.approx(0.5 * before["p90"], abs=0.01)
    assert motion_db(made[:, 5], audible, 10)["max"] == 0.0
    # Every share one is the scalar, every share zero the one number.
    ones = band_levels(scalar, audible, LEVEL_DB, np.ones(8))
    assert np.array_equal(ones, np.repeat(scalar[:, None], 8, axis=1))
    zeros = band_levels(scalar, audible, LEVEL_DB, np.zeros(8))
    assert np.all(zeros[audible] == np.float32(LEVEL_DB)) and np.all(zeros[~audible] == 0.0)


def test_the_filters_of_the_late_part_add_to_the_signal_and_the_gain_is_a_slope() -> None:
    groups = [(0, 1, 2, 3), (4,), (5, 6, 7)]
    kernels = group_kernels(FS, groups)
    total = np.sum(kernels, axis=0)
    centre = total.size // 2
    assert total[centre] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(np.delete(total, centre)).max() < 1e-12
    with pytest.raises(ValueError, match="every band of the bank once"):
        group_kernels(FS, [(0, 1), (3,)])
    # A seam of +3 dB: whole to 1 kHz, +1.5 dB at 2 kHz, nothing from 4 kHz, and between
    # two centres a slope, since the bank's bands cross in amplitude: no step anywhere.
    levels = 3.0 * taper_of(BANK, 1000.0)
    thirds = 1000.0 * 2.0 ** (np.arange(-3, 13) / 3.0)
    gain = applied_gain_db(thirds, levels, FS)
    assert gain[:4] == pytest.approx(3.0, abs=0.02)
    assert gain[6] == pytest.approx(1.5, abs=0.02)
    assert np.abs(gain[9:]).max() < 0.02
    assert np.all(np.diff(gain) < 0.01) and np.diff(gain).min() > -0.8
    fine = applied_gain_db(np.geomspace(700.0, 8000.0, 400), levels, FS)
    assert np.abs(np.diff(fine)).max() < 0.04


def test_the_engine_takes_the_level_a_band_and_the_scalar_when_told() -> None:
    pack = density_pack(duration_s=0.3, moving=True, bins=30)
    scalar = scalar_of(pack)
    audible = np.asarray(pack.sources["s1"].audible, dtype=bool)
    dry = {"s1": np.random.default_rng(0).standard_normal(int(0.3 * FS))}
    today = Engine(walked(pack, scalar), dry).render()
    assert np.abs(today).max() > 0.0
    # Every share one: today's render, bit for bit.
    ones = band_levels(scalar, audible, LEVEL_DB, np.ones(8))
    np.testing.assert_array_equal(Engine(walked(pack, scalar, ones), dry).render(), today)
    # The tapered table in the pack and the scalar asked for: today's render, bit for bit.
    tapered = band_levels(scalar, audible, LEVEL_DB, taper_of(BANK, 1000.0))
    held = walked(pack, scalar, tapered)
    broadband = Engine(held, dry, settings=RenderSettings(seam="broadband")).render()
    np.testing.assert_array_equal(broadband, today)
    # Every share zero: the pack whose scalar is the one number, bit for bit.
    zeros = band_levels(scalar, audible, LEVEL_DB, np.zeros(8))
    constant = np.where(audible, np.float32(LEVEL_DB), scalar).astype(np.float32)
    np.testing.assert_array_equal(
        Engine(walked(pack, scalar, zeros), dry).render(),
        Engine(walked(pack, constant), dry).render(),
    )
    # The default: the table is rendered, and it is not today's.
    made = Engine(held, dry).render()
    assert np.abs(made - today).max() > 1e-3 * np.abs(today).max()
    with pytest.raises(ValueError, match="a seam is one of"):
        Engine(held, dry, settings=RenderSettings(seam="narrow")).render()


@pytest.mark.parametrize("part", ["early", "tail"])
def test_each_part_lays_the_band_s_own_level_on_the_band(part: str) -> None:
    """Fails if the engine ignores the level a band: a seam of +3 dB, heard band by band."""
    pack = density_pack(duration_s=0.3, moving=True, bins=30)
    audible = np.asarray(pack.sources["s1"].audible, dtype=bool)
    scalar = np.full(pack.header.steps, LEVEL_DB + 3.0, dtype=np.float32)
    table = band_levels(scalar, audible, LEVEL_DB, taper_of(BANK, 1000.0))
    without, held = walked(pack, scalar), walked(pack, scalar, table)

    def lost_db(low_hz: float, high_hz: float) -> float:
        dry = {"s1": band_noise(low_hz, high_hz)}
        was = Engine(without, dry).stem("s1", parts=(part,))
        now = Engine(held, dry).stem("s1", parts=(part,))
        return float(10.0 * np.log10(np.sum(now**2) / np.sum(was**2)))

    assert lost_db(950.0, 1050.0) == pytest.approx(0.0, abs=0.05)  # the join: as traced
    assert lost_db(1950.0, 2050.0) == pytest.approx(-1.5, abs=0.1)
    assert lost_db(5000.0, 7000.0) == pytest.approx(-3.0, abs=0.05)  # the one number


def test_a_traced_pack_is_given_its_level_a_band_in_place_and_keeps_its_scalar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = density_pack(duration_s=0.3, moving=True, bins=30)
    scalar = scalar_of(pack)
    target = write_pack(tmp_path / "pack.h5", walked(pack, scalar))
    with h5py.File(target, "r") as f:
        provenance = f.attrs["provenance_json"]
        seams = f["sources/s1/low/seam_db"][...]
        base_db = 20.0 * np.log10(float(f["mirror"].attrs["alignment_gain"]))
    said = taper_pack(target, dry_run=True)
    assert said["written"] is False
    with h5py.File(target, "r") as f:
        assert "band_gain_db" not in f["sources/s1/level"]
    assert main(["seam", str(target)]) == 0
    said = json.loads(capsys.readouterr().out)
    constant = float(np.median(seams))
    assert said["seam"]["constant_db"] == pytest.approx(constant, abs=1e-4)
    assert said["seam"]["shares"] == [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    assert said["seam"]["constant"] == "the median of the pairs' seams"
    row = said["sources"]["s1"]
    assert row["within_half_a_second_db"]["1000"] == row["within_half_a_second_scalar_db"]
    assert row["within_half_a_second_db"]["4000"]["max"] == 0.0
    with read_pack(target, deep=True) as read:  # still a pack, and the engine's table
        level = read.sources["s1"].level
        assert np.array_equal(level.high_gain_db, scalar)
        table = np.array(level.band_gain_db)
        assert read.header.provenance["seam"] == said["seam"]
    audible = np.asarray(pack.sources["s1"].audible, dtype=bool)
    wanted = band_levels(scalar, audible, base_db + constant, taper_of(BANK, 1000.0))
    assert np.array_equal(table, wanted)
    # A given number, and another taper: written over, from the trace's scalar again.
    assert main(["seam", str(target), "--constant-db", "2.5", "--taper", "1", "0"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["seam"]["constant"] == "given" and said["seam"]["constant_db"] == 2.5
    with read_pack(target, deep=True) as read:
        table = np.array(read.sources["s1"].level.band_gain_db)
    assert np.array_equal(table[:, 3], scalar)
    assert np.all(table[audible, 4:] == np.float32(base_db + 2.5))
    # A copy keeps it, and --undo leaves the pack as the trace wrote it.
    with read_pack(target) as read:
        copied = write_pack(tmp_path / "copy.h5", read)
    with read_pack(copied) as read:
        assert np.array_equal(np.asarray(read.sources["s1"].level.band_gain_db), table)
    assert main(["seam", str(target), "--undo"]) == 0
    with h5py.File(target, "r") as f:
        assert "band_gain_db" not in f["sources/s1/level"]
        assert f.attrs["provenance_json"] == provenance
        assert np.array_equal(f["sources/s1/level/high_gain_db"][...], scalar)
    with read_pack(target, deep=True) as read:
        assert read.sources["s1"].level.band_gain_db is None


def test_a_table_of_another_shape_is_not_a_pack(tmp_path: Path) -> None:
    from reverberate.render.pack import PackError

    pack = density_pack(duration_s=0.3, moving=True, bins=30)
    scalar = scalar_of(pack)
    short = np.zeros((pack.header.steps, 7), dtype=np.float32)
    with pytest.raises(PackError, match="band_gain_db"):
        write_pack(tmp_path / "pack.h5", walked(pack, scalar, short))


def test_gains_are_set_in_place_said_in_the_provenance_and_put_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = density_pack(duration_s=0.3, moving=True, bins=30, sources=2)
    target = write_pack(tmp_path / "pack.h5", pack)

    def held() -> tuple[dict[str, dict[str, float]], str, bytes, str]:
        with h5py.File(target, "r") as f:
            attrs = {
                name: {key: float(group.attrs[key]) for key in group.attrs if "gain_db" in key}
                for name, group in f["sources"].items()
            }
            return (
                attrs,
                f.attrs["provenance_json"],
                f["recipe"][...].tobytes(),
                f.attrs["recipe_sha256"],
            )

    before = held()
    kind = pack.sources["s1"].kind
    assert main(["gains", str(target), "--set", "s1=-4.5", "--dry-run"]) == 0
    capsys.readouterr()
    assert held() == before
    assert main(["gains", str(target), "--set", "s1=-4.5"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["sources"]["s1"] == {"kind": kind, "traced_db": 0.0, "was_db": 0.0, "db": -4.5}
    assert said["gains"]["sources"] == {"s1": {"recipe_db": 0.0, "db": -4.5}}
    with read_pack(target, deep=True) as read:  # still a pack, and the engine's gain
        assert read.sources["s1"].gain_db == -4.5 and read.sources["s2"].gain_db == 0.0
        assert read.header.provenance["gains"]["sources"]["s1"]["recipe_db"] == 0.0
        dry = {"s1": np.random.default_rng(0).standard_normal(int(0.3 * FS))}
        quiet = Engine(read, dry).stem("s1")
    loud = Engine(pack, dry).stem("s1")
    # To the rounding of the fast engine's single precision.
    np.testing.assert_allclose(quiet, loud * 10.0 ** (-4.5 / 20.0), atol=1e-6 * np.abs(loud).max())
    # A kind is moved from what the trace wrote, so twice is once; a source named is not.
    for _ in range(2):
        set_gains(target, gains_db={"s1": -4.5}, add_db={kind: 12.0})
    attrs, _, recipe, digest = held()
    assert attrs["s1"] == {"gain_db": -4.5, "gain_db_traced": 0.0}
    assert attrs["s2"] == {"gain_db": 12.0, "gain_db_traced": 0.0}
    assert (recipe, digest) == before[2:]  # the recipe that was traced, and its digest
    with pytest.raises(KeyError, match="no source s9"):
        set_gains(target, gains_db={"s9": 1.0})
    with pytest.raises(KeyError, match="no source of kind"):
        set_gains(target, add_db={"thunder": 1.0})
    assert main(["gains", str(target), "--undo"]) == 0
    capsys.readouterr()
    assert held() == before
    with read_pack(target, deep=True) as read:
        assert read.sources["s2"].gain_db == 0.0


def test_the_default_taper_is_the_owner_s() -> None:
    assert TAPER == (1.0, 0.5, 0.0)
    assert RenderSettings().seam == "tapered"
