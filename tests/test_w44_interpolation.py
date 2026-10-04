"""Translation and fusion against a field known in closed form, and the experiment on it.

The field is a handful of plane waves, whose order 7 coefficients at any
point are exact: a wave from ``s`` of amplitude ``a`` is ``a exp(i k s . x)``
at ``x`` and ``a Y_c(s) exp(i k s . x)`` in channel ``c`` of an expansion
centred there. Nothing is solved and nothing is read from the data root.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.experiments.w44_interpolation import fusion_weights, translation_weights
from reverberate.experiments.w44_interpolation.__main__ import main
from reverberate.experiments.w44_interpolation.leave_one_out import leave_one_out
from reverberate.experiments.w44_interpolation.line_channels import line_channels
from reverberate.experiments.w44_interpolation.scoring import (
    BANDS,
    band_energy,
    delayed,
    early_window,
    onset_of,
)
from reverberate.experiments.w44_interpolation.translation_failures import THIRDS
from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic
from reverberate.spatial.translate import SOUND_SPEED_M_S

ORDER = 7


class PlaneWaves:
    """A field of a few plane waves, in scene coordinates: its pressure and its coefficients."""

    def __init__(self, count: int = 5, seed: int = 44) -> None:
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(count, 3))
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.amplitudes = rng.uniform(0.3, 1.0, size=count)
        #: When each wave passes the origin, in seconds.
        self.delays_s = rng.uniform(0.004, 0.012, size=count)
        self.basis = real_sh(ORDER, scene_to_ambisonic(self.directions))  # [wave, channel]

    def waves(self, point: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """Each wave's complex amplitude at ``point``, ``[frequency, wave]``."""
        k = 2 * np.pi * freqs_hz / SOUND_SPEED_M_S
        phase = k[:, None] * (self.directions @ point)[None, :]
        phase = phase - 2 * np.pi * freqs_hz[:, None] * self.delays_s[None, :]
        return np.asarray(self.amplitudes[None, :] * np.exp(1j * phase))

    def pressure(self, point: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        return np.asarray(self.waves(point, freqs_hz).sum(axis=1))

    def coefficients(self, point: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """``[frequency, channel]`` of the order 7 expansion centred on ``point``."""
        return np.asarray(self.waves(point, freqs_hz) @ self.basis)


def frequency_of(kd: float, distance_m: float) -> np.ndarray:
    return np.array([kd * SOUND_SPEED_M_S / (2 * np.pi * distance_m)])


def error_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(estimate - truth) ** 2) / np.sum(np.abs(truth) ** 2)))


class TestTranslation:
    field = PlaneWaves()
    centre = np.array([0.3, 1.7, -0.2])
    offset = np.array([0.12, -0.05, 0.38])  # about 0.40 m, off every axis

    def translated(self, kd: float) -> tuple[np.ndarray, np.ndarray]:
        freqs = frequency_of(kd, float(np.linalg.norm(self.offset)))
        weights = translation_weights(self.offset, freqs, ORDER)
        estimate = np.einsum("fc,fc->f", weights, self.field.coefficients(self.centre, freqs))
        return estimate, self.field.pressure(self.centre + self.offset, freqs)

    def test_the_weights_are_a_filter_per_frequency_and_channel(self) -> None:
        weights = translation_weights(self.offset, np.array([100.0, 200.0, 400.0]), ORDER)
        assert weights.shape == (3, channel_count(ORDER)) and weights.dtype == np.complex64

    def test_at_the_centre_only_the_omni_channel_is_read(self) -> None:
        weights = translation_weights(self.offset * 1e-6, np.array([500.0]), ORDER)
        assert weights[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert np.abs(weights[0, 1:]).max() < 1e-5

    @pytest.mark.parametrize("kd", [0.5, 2.0, 4.0])
    def test_a_short_translation_reproduces_the_field(self, kd: float) -> None:
        estimate, truth = self.translated(kd)
        assert error_db(estimate, truth) < -40.0

    def test_the_translation_degrades_past_the_order(self) -> None:
        errors = [error_db(*self.translated(kd)) for kd in (4.0, 7.0, 10.0, 14.0, 20.0)]
        assert errors == sorted(errors), "the error grows with k d"
        assert errors[1] < -10.0, "still a prediction at k d = order"
        assert errors[-1] > -3.0, "no better than silence at k d = 20"

    def test_the_offset_is_from_the_centre_to_the_point(self) -> None:
        freqs = frequency_of(3.0, float(np.linalg.norm(self.offset)))
        backwards = translation_weights(-self.offset, freqs, ORDER)
        estimate = np.einsum("fc,fc->f", backwards, self.field.coefficients(self.centre, freqs))
        truth = self.field.pressure(self.centre + self.offset, freqs)
        assert error_db(estimate, truth) > -10.0


class TestFusion:
    field = PlaneWaves()
    target = np.array([0.3, 1.7, -0.2])
    step = np.array([0.0, 0.0, 0.40])

    def errors(self, kd: float) -> tuple[float, float, float]:
        """One neighbour translated, one fused, the two opposite neighbours fused: dB."""
        freqs = frequency_of(kd, 0.40)
        truth = self.field.pressure(self.target, freqs)
        below = self.field.coefficients(self.target - self.step, freqs)
        above = self.field.coefficients(self.target + self.step, freqs)
        translated = np.einsum("fc,fc->f", translation_weights(self.step, freqs, ORDER), below)
        # An offset is the target seen from the cell: +step from the cell below.
        one = fusion_weights(np.stack([self.step]), freqs, ORDER)
        pair = fusion_weights(np.stack([self.step, -self.step]), freqs, ORDER)
        fused_one = np.einsum("fc,fc->f", one[:, 0], below)
        fused = np.einsum("fc,fc->f", pair[:, 0], below) + np.einsum("fc,fc->f", pair[:, 1], above)
        return error_db(translated, truth), error_db(fused_one, truth), error_db(fused, truth)

    def test_the_operator_is_a_filter_per_frequency_neighbour_and_channel(self) -> None:
        operator = fusion_weights(
            np.stack([self.step, -self.step]), np.array([200.0, 400.0]), ORDER
        )
        assert operator.shape == (2, 2, channel_count(ORDER)) and operator.dtype == np.complex64

    def test_at_low_kd_the_fusion_reproduces_the_field(self) -> None:
        translated, one, pair = self.errors(2.0)
        assert translated < -40.0 and one < -40.0 and pair < -40.0

    @pytest.mark.parametrize("kd", [6.0, 7.0, 8.0, 9.0])
    def test_two_neighbours_fused_are_no_worse_than_one_translated(self, kd: float) -> None:
        translated, _, pair = self.errors(kd)
        assert pair <= translated

    def test_two_neighbours_hold_where_one_translation_has_failed(self) -> None:
        translated, _, pair = self.errors(9.0)
        assert translated > -10.0 and pair < -20.0

    def test_the_fusion_is_deterministic(self) -> None:
        freqs = np.array([300.0, 900.0])
        offsets = np.stack([-self.step, self.step])
        assert np.array_equal(
            fusion_weights(offsets, freqs, ORDER), fusion_weights(offsets, freqs, ORDER)
        )


class TestScoring:
    def test_the_onset_is_the_first_sample_over_a_tenth_of_the_peak(self) -> None:
        assert onset_of(np.array([0.0, 0.05, -0.2, 1.0, 0.3])) == 2

    def test_the_window_ends_on_a_fade_and_stops_at_the_response(self) -> None:
        win = early_window(20, 10, 4)
        assert win[:7].tolist() == [1.0] * 7 and win[10:].tolist() == [0.0] * 10
        assert np.all(np.diff(win[6:10]) < 0) and 0.0 < win[9] < 0.2
        assert early_window(8, 50, 4)[:4].tolist() == [1.0] * 4

    def test_a_tone_falls_in_its_own_octave(self) -> None:
        rate = 16000.0
        tone = np.sin(2 * np.pi * 1000.0 * np.arange(1600) / rate)
        energy = band_energy(tone, rate)
        assert int(np.argmax(energy)) == BANDS.index(1000)
        assert energy[BANDS.index(1000)] > 1e6 * energy[BANDS.index(250)]

    def test_a_delay_of_whole_samples_moves_the_response(self) -> None:
        pulse = np.zeros(64)
        pulse[10] = 1.0
        assert int(np.argmax(delayed(pulse, 5 / 8000.0, 8000.0))) == 15


RATE = 8000.0


def write_field(path: Path, positions: np.ndarray, cells: np.ndarray, samples: int) -> None:
    """The plane waves as a response field: order 7, every point exact, band limited."""
    field = PlaneWaves()
    freqs = np.fft.rfftfreq(samples, 1 / RATE)
    # A raised cosine down to the Nyquist frequency keeps each arrival short.
    shape = 0.5 * (1 + np.cos(np.pi * freqs / freqs[-1]))
    source = np.array([-1.0, 1.7, -1.5])
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate_hz"] = RATE
        f.attrs["order"] = ORDER
        f.attrs["grid_step_m"] = np.array([0.4, 0.0, 0.4])
        f.attrs["source_room"] = "living room"
        f.attrs["source_position"] = source
        f["positions"] = positions
        f["cell_index"] = cells.astype(np.int32)
        f["direct_path_m"] = np.linalg.norm(positions - source, axis=1)
        f["rooms"] = np.array([b"living room"] * len(positions))
        f["low_borrowed"] = np.zeros(len(positions), dtype=bool)
        ir = f.create_dataset("ir", (len(positions), channel_count(ORDER), samples), dtype="f4")
        for i, point in enumerate(positions):
            spectrum = field.coefficients(point, freqs) * shape[:, None]
            ir[i] = np.fft.irfft(spectrum.T, samples, axis=-1)


def write_lattice(path: Path, samples: int) -> Path:
    cells = np.array([(x, 0, z) for x in range(3) for z in range(3)])
    write_field(path, cells * np.array([0.4, 0.0, 0.4]) + np.array([0.0, 1.7, 0.0]), cells, samples)
    return path


@pytest.fixture(scope="module")
def lattice(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three by three points at 0.40 m; long enough for the late window, 150 ms on."""
    return write_lattice(tmp_path_factory.mktemp("w44") / "lattice.h5", 1600)


@pytest.fixture(scope="module")
def short_lattice(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same lattice, the early part only: a fifth of the frequencies to fuse."""
    return write_lattice(tmp_path_factory.mktemp("w44") / "short.h5", 256)


@pytest.fixture(scope="module")
def line(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Forty-five points every 2 cm along z."""
    cells = np.array([(0, 0, z) for z in range(45)])
    path = tmp_path_factory.mktemp("w44") / "line.h5"
    write_field(path, cells * np.array([0.0, 0.0, 0.02]) + np.array([0.0, 1.7, 0.0]), cells, 256)
    return path


def band(values: list[float], centre: int) -> float:
    return values[BANDS.index(centre)]


class TestOnASyntheticField:
    """What W44 measured on hssd_0076, where the truth is known.

    The lattice's bands over the sampling rate's reach hold no energy, so
    their error is 0 / 0; numpy says so and the tests read the bands under.
    """

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_leave_one_out_on_the_lattice(self, lattice: Path, tmp_path: Path) -> None:
        summary = leave_one_out(lattice, tmp_path, say=lambda m: None)
        assert summary["order"] == ORDER and summary["pitch_m"] == pytest.approx(0.4)
        # The middle point of each of three rows and three columns.
        assert summary["samples"] == 6 and "other_rooms" not in summary
        found = summary["source_room"]
        # k d = 7 at 0.40 m is 955 Hz: the translation holds under it and fails over it.
        for centre in (125, 250, 500):
            assert band(found["translate_early"]["worst"], centre) < -30.0
        assert band(found["translate_early"]["median"], 2000) > -6.0
        # A mean of two responses 0.80 m apart is no prediction at 500 Hz.
        assert band(found["linear_early"]["median"], 500) > -6.0
        assert band(found["translate_early"]["worst"], 500) < band(
            found["linear_early"]["median"], 500
        )
        # Two translations of half a pitch to one midpoint agree where each holds.
        assert band(found["midpoint_early"]["worst"], 500) < -30.0
        rows = json.loads((tmp_path / "rows.json").read_text())
        assert len(rows) == 6 and {row["axis"] for row in rows} == {"x", "z"}
        assert json.loads((tmp_path / "summary.json").read_text())["samples"] == 6

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_the_fusion_on_the_lattice_from_the_command_line(
        self, short_lattice: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["plane-wave", "--field", str(short_lattice), "--out", str(tmp_path)]) == 0
        found = json.loads((tmp_path / "summary_planewave.json").read_text())["source_room"]
        assert found["pair"]["samples"] == 6 and found["four"]["samples"] == 1
        # At 1 kHz, k d = 7.3: one neighbour has failed, two hold.
        assert band(found["pair"]["worst"], 1000) < -20.0
        assert band(found["pair"]["median"], 1000) < band(found["one"]["median"], 1000)
        assert band(found["four"]["worst"], 1000) < -20.0
        assert "operators in" in capsys.readouterr().out

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_the_error_against_the_spacing_on_the_line(self, line: Path, tmp_path: Path) -> None:
        assert main(["line-gaps", "--field", str(line), "--out", str(tmp_path)]) == 0
        summary = json.loads((tmp_path / "summary_gaps.json").read_text())
        assert summary["points"] == 45
        close, wide = summary["spacing_0.08"], summary["spacing_0.80"]
        # Translated 4 cm, the expansion is exact through 2 kHz; 40 cm, to under 1 kHz.
        assert band(close["translate"]["p90"], 2000) < -40.0
        assert band(wide["translate"]["p90"], 500) < -30.0
        assert band(wide["translate"]["median"], 2000) > -6.0
        # The fusion of the pair is no worse than one translation where that one strains.
        assert band(wide["fusion"]["median"], 1000) <= band(wide["translate"]["median"], 1000)
        # Each spacing costs every predictor that reads no direction.
        for name in ("nearest", "linear"):
            assert band(close[name]["median"], 1000) < band(wide[name]["median"], 1000)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_the_failures_are_counted_and_kept(self, lattice: Path, tmp_path: Path) -> None:
        assert main(["failures", "--field", str(lattice), "--out", str(tmp_path)]) == 0
        summary = json.loads((tmp_path / "summary_failures.json").read_text())
        # Twelve neighbouring pairs, each translated both ways; free field, none fails.
        assert summary["cases"] == 24 and summary["bad"] == 0
        assert summary["median_bad"] is None and summary["factors"]["dn"]["bad"] is None
        assert summary["median_good"][THIRDS.index(250)] < -40.0
        assert summary["by_cells_missing_round_target"]["0"]["cases"] == 4, "from the centre"
        assert np.load(tmp_path / "diag_rows.npy").shape == (24, 3)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_all_channels_on_the_line_between_the_arrays_centres(
        self, line: Path, tmp_path: Path
    ) -> None:
        assert main(["line-channels", "--field", str(line), "--out", str(tmp_path / "a")]) == 0
        summary = json.loads((tmp_path / "a" / "summary_channels.json").read_text())
        assert summary["centres"] == 45 and summary["columns"][-2:] == ["all channels", "channel 0"]
        near, wide = summary["distances"]["0.020"], summary["distances"]["0.240"]
        assert near["translate"]["far_cases"] == near["cases"], "the source is 1.8 m away"
        # On the head's sphere a translation of 2 cm is exact on every channel.
        assert max(near["translate"]["far_all_channels"]["worst"]) < -60.0
        assert max(near["translate"]["far_per_degree_worst_band"]["worst"]) < -60.0
        # At 0.24 m and 1 kHz, k d = 4.4: two cells hold more than one does.
        one = wide["translate"]["far_all_channels"]["median"][BANDS.index(1000)]
        two = wide["fuse"]["far_all_channels"]["median"][BANDS.index(1000)]
        assert two < one and two < -20.0
        shares = summary["by_share_of_source_distance"]["translate"]
        assert sum(row["cases"] for row in shares.values()) > 0
        # A plan whose arrays stand where the field says they do changes nothing.
        with h5py.File(line, "r") as f:
            centres = f["positions"][:].tolist()
        plan = tmp_path / "plan.json"
        band_record = {"centres": centres, "grid_step_m": 0.02}
        plan.write_text(json.dumps({"bands": {"low": band_record, "mid": band_record}}))
        again = line_channels(line, tmp_path / "b", plan=plan, steps=(1,), say=lambda m: None)
        assert again["distances"]["0.020"] == near
