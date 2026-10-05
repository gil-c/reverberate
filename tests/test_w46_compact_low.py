"""W46 on data known in closed form: the scores, the ways a pair is kept, the line.

Nothing is read from the data root. The pairs are rooms of noise in a pair
cache made here, and the line is a handful of plane waves, whose order 7
coefficients at any point are exact (as in ``test_w44_interpolation``).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.accel.pairs import PairCache
from reverberate.experiments.w46_compact_low.__main__ import main
from reverberate.experiments.w46_compact_low.line import TODAY, Line, cutoffs, measure_pitch
from reverberate.experiments.w46_compact_low.pairs import (
    Context,
    _chains,
    find_pairs,
    measure_pairs,
    measure_rails,
    variants,
)
from reverberate.experiments.w46_compact_low.scoring import (
    BANDS_HZ,
    band_masks,
    head_weights,
    scores,
    stats,
)
from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic
from reverberate.spatial.translate import SOUND_SPEED_M_S

GRID = "0123456789abcdef0123456789abcdef"
BLOCK = "bins int16, a scale a channel and 100 Hz"


def room(seed: int, t60_s: float = 0.3) -> np.ndarray:
    """A pair in the cache's form: 64 channels of noise that arrive at 5 ms and decay."""
    rng = np.random.default_rng(seed)
    decay = np.zeros(4800)
    decay[20:] = 10.0 ** (-3.0 * np.arange(4780) / (t60_s * 4000.0))
    noise = rng.standard_normal((64, 4800)) * decay
    # Band limited as a solve to 1500 Hz is, and without its lowest octave.
    freqs = np.fft.rfftfreq(4800, 1 / 4000.0)
    shape = np.clip((1500.0 - freqs) / 100.0, 0, 1) * np.clip(freqs / 60.0, 0, 1)
    return np.asarray(np.fft.irfft(np.fft.rfft(noise, axis=-1) * shape, 4800, axis=-1), "f4")


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Twelve rail positions 8 cm apart heard at one cell, and one pair elsewhere."""
    root = tmp_path_factory.mktemp("w46") / "pairs"
    held = PairCache(root, GRID)
    for i in range(12):
        source = [2.0 + 0.08 * i, 1.7, 0.0]
        held.write(f"{i:064x}", room(i), {"source_m": source, "cell_m": [0.0, 1.7, 0.0]})
    held.write("f" * 64, room(99), {"source_m": [0.0, 1.7, 5.0], "cell_m": [1.0, 1.7, 1.0]})
    return root


def test_a_score_is_the_error_over_the_truth_in_every_third_octave() -> None:
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((64, 4800))
    found = scores(0.1 * truth, truth, 400, 7)
    assert found["early"].shape == found["whole"].shape == (len(BANDS_HZ),)
    assert np.allclose(found["early"], -20.0, atol=1e-6)
    assert np.allclose(found["whole"], -20.0, atol=1e-6)
    # The degrees add to the total, and a tenth more of the truth is 0.83 dB of level.
    assert np.allclose(10 * np.log10((10 ** (found["whole_degree"] / 10)).sum(axis=1)), -20.0)
    assert np.allclose(found["late_level"], 20 * np.log10(1.1), atol=1e-6)
    masks = band_masks(4800)
    assert masks.shape == (12, 2401) and masks.any(axis=1).all()
    weight = head_weights(7, np.array([0.0, 1000.0]))
    assert weight.shape == (64, 2) and weight[0, 0] == 1.0 and not weight[1:, 0].any()
    read = stats(np.array([[1.0, -5.0], [3.0, -1.0]]))
    assert read["worst"] == [3.0, -1.0]
    assert stats(np.array([[1.0], [3.0]]), low_is_bad=True)["worst"] == [1.0]


def test_every_way_gives_a_response_back_and_its_bytes() -> None:
    context = Context()
    x, onset = context.stored(room(3))
    assert x.shape == (64, 4800) and 40 < onset < 100, "5 ms and the lead's 10.7 ms"
    ways = variants(context)
    assert "bins int16, decay 60" in ways and "time float16" in ways
    errors, sizes = {}, {}
    for name in ("bins float32", "time float16", "bins int16, a scale a pair", BLOCK, "decay 60"):
        back, sizes[name] = ways[name](x)
        errors[name] = 10 * np.log10(np.sum((back - x) ** 2) / np.sum(x**2))
    assert errors["bins float32"] < -130.0 and sizes["bins float32"] == 64 * 1698 * 8
    assert errors[BLOCK] < errors["time float16"] < -60.0
    assert errors[BLOCK] < errors["bins int16, a scale a pair"]
    assert sizes["decay 60"] < 0.5 * sizes["bins float32"] and errors["decay 60"] < -50.0


def test_pairs_are_found_once_and_measured_where_a_head_hears_them(cache: Path) -> None:
    records = find_pairs([cache, cache], GRID)
    assert len(records) == 13 and records[0]["distance_m"] == pytest.approx(2.0)
    summary = measure_pairs(
        [cache], GRID, count=2, only=["bins float32", BLOCK], say=lambda message: None
    )
    assert summary["pairs"] == 2 and summary["pairs_found"] == 13
    exact, kept = summary["ways"]["bins float32"], summary["ways"][BLOCK]
    assert exact["factor"] == pytest.approx(1.41, abs=0.01)
    assert kept["factor"] == pytest.approx(2.83, abs=0.01)
    for distance in ("0.00", "0.10", "0.20"):
        assert max(exact[distance]["whole"]["worst"]) < -120.0
        assert max(kept[distance]["whole"]["worst"]) < -70.0
        assert max(kept[distance]["late_level"]["worst_abs"]) < 0.01
    # Sixteen bits with a scale per block stay under a decay of 60 dB in 0.3 s.
    assert kept["margin_under_decay_db"]["worst"][0] > 10.0
    assert len(summary["degree_over_channel_0_db"]["whole"]["median"]) == len(BANDS_HZ)


def test_a_rail_of_independent_rooms_has_its_whole_rank(cache: Path) -> None:
    places = np.array([[0.0, 0, 0], [0.08, 0, 0], [0.16, 0, 0], [3.0, 0, 0]])
    assert [len(chain) for chain in _chains(places)] == [3, 1]
    summary = measure_rails([cache], GRID, rails=2, say=lambda message: None)
    assert len(summary["rails"]) == 1 and summary["positions"] == 12
    rail = summary["rails"][0]
    assert rail["source_distance_m"] == [2.0, 2.88]
    # Twelve rooms that share nothing: a quarter of the rank holds a quarter of them.
    assert rail["as stored"]["rank_for_db"]["-20"] == 12
    assert -2.0 < rail["as stored"]["left_after_rank_db"]["3"] < -0.5


class PlaneWaves:
    """A few plane waves in scene coordinates: their order 7 coefficients at any point."""

    def __init__(self, count: int = 5, seed: int = 46) -> None:
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(count, 3))
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.amplitudes = rng.uniform(0.3, 1.0, size=count)
        self.delays_s = rng.uniform(0.004, 0.012, size=count)
        self.basis = real_sh(7, scene_to_ambisonic(self.directions))

    def coefficients(self, point: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        k = 2 * np.pi * freqs_hz / SOUND_SPEED_M_S
        phase = k[:, None] * (self.directions @ point)[None, :]
        phase = phase - 2 * np.pi * freqs_hz[:, None] * self.delays_s[None, :]
        return np.asarray((self.amplitudes[None, :] * np.exp(1j * phase)) @ self.basis)


@pytest.fixture(scope="module")
def line(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Thirty points 3 cm apart along z, 0.1 s at 8 kHz, the source 9 m away; and their plan."""
    folder = tmp_path_factory.mktemp("w46_line")
    field, samples, rate = PlaneWaves(), 800, 8000.0
    positions = np.array([[0.0, 1.7, 0.03 * i] for i in range(30)])
    freqs = np.fft.rfftfreq(samples, 1 / rate)
    shape = np.clip((1500.0 - freqs) / 200.0, 0, 1)
    with h5py.File(folder / "line.h5", "w") as f:
        f.attrs["sample_rate_hz"] = rate
        f.attrs["order"] = 7
        f.attrs["source_position"] = np.array([0.0, 1.7, -9.0])
        f["positions"] = positions
        ir = f.create_dataset("ir", (30, channel_count(7), samples), dtype="f4")
        for i, point in enumerate(positions):
            ir[i] = np.fft.irfft((field.coefficients(point, freqs) * shape[:, None]).T, samples)
    band = {"centres": positions.tolist(), "grid_step_m": 0.03}
    (folder / "plan.json").write_text(json.dumps({"bands": {"low": band, "mid": band}}))
    return folder / "line.h5", folder / "plan.json"


def test_the_cutoffs_are_named_and_today_is_none_of_them() -> None:
    rules = cutoffs(7)
    assert not rules[TODAY].any() and not rules["50 dB"][:4].any()
    assert np.allclose(rules["50 dB"][4:], [181.9, 288.1, 403.5, 524.9], atol=0.1)
    assert rules["k R = 1 n"][7] == pytest.approx(7 * SOUND_SPEED_M_S / (2 * np.pi * 0.30))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_on_plane_waves_the_rule_in_the_bessel_bound_holds_and_the_linear_one_does_not(
    line: tuple[Path, Path], tmp_path: Path
) -> None:
    field, plan = line
    command = ["degrees", "--field", str(field), "--plan", str(plan), "--cases", "2"]
    assert main([*command, "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary_degrees.json").read_text())
    assert len(summary["geometries"]) == 5 and summary["pass_hz"]["50 dB"][7] == 525.0
    moved = summary["geometries"]["one cell, 0.180 m"]["rule"]
    # A plane wave 0.18 m from its cell, on the head's sphere: the order's own error.
    today = max(moved[TODAY]["early"]["worst"])
    assert today < -25.0
    # What the rule at 50 dB changes is under the level it states; a degree let go from
    # k R = n takes the prediction with it.
    assert max(moved["50 dB"]["changed_early"]["worst"]) < -50.0
    assert abs(max(moved["50 dB"]["early"]["worst"]) - today) < 0.1
    assert max(moved["k R = 1 n"]["early"]["worst"]) > today + 10.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_two_cells_fused_hold_less_as_they_stand_further_apart(line: tuple[Path, Path]) -> None:
    read = Line(*line, say=lambda message: None)
    assert len(read.of_node) == 30 and read.samples == 400
    summary = measure_pitch(read, pitches=(4, 14), cases=2, say=lambda message: None)
    near = summary["pitches"]["4"]["source 2.5 to 100 m"]
    far = summary["pitches"]["14"]["source 2.5 to 100 m"]
    assert near["allowed_by_the_rule"] == near["cell_pairs"] == 2
    assert summary["pitches"]["14"]["pitch_m"] == 0.42
    assert max(near["early"]["worst"]) < max(far["early"]["worst"]) - 10.0
