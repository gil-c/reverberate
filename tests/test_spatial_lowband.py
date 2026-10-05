"""The stored form of the band under the crossover, against the join it is the low side of."""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.audio import Atmosphere
from reverberate.mirror.hybrid import Crossover, blend
from reverberate.spatial.lowband import (
    FIELD_UNIT_AT_1M,
    LOW_RATE_HZ,
    LOW_SAMPLES,
    decimate,
    delayed,
    from_stored,
    low_side,
    onset_s,
    pair_key,
    solve_fmax_hz,
    to_stored,
)

RATE = 48000.0
SAMPLES = 57600


def a_response(channels: int = 4, seed: int = 7, samples: int = SAMPLES) -> np.ndarray:
    """A direct sound at 20 ms and a decaying tail, the whole band, ``[channel, sample]``."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / RATE
    tail = rng.standard_normal((channels, samples)) * np.exp(-t / 0.15) * (t > 0.02)
    tail[:, 960] += 40.0
    return np.asarray(tail)


def band_limited(response: np.ndarray, top_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(response, axis=-1)
    spectrum[..., np.fft.rfftfreq(response.shape[-1], 1 / RATE) > top_hz] = 0.0
    return np.asarray(np.fft.irfft(spectrum, n=response.shape[-1], axis=-1))


def relative_db(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(np.sum((a - b) ** 2) / np.sum(b**2)))


class TestDecimation:
    def test_a_response_under_2_khz_comes_back_exactly(self) -> None:
        response = band_limited(a_response(), 1900.0)
        low = decimate(response, RATE)
        assert low.shape == (4, LOW_SAMPLES) and low.dtype == np.float32
        assert relative_db(from_stored(low), response) < -120.0

    def test_what_lies_over_2_khz_is_dropped_and_not_folded(self) -> None:
        response = a_response()
        under = band_limited(response, 1999.9)  # every bin short of the new Nyquist
        kept, same = decimate(response, RATE), decimate(under, RATE)
        assert np.abs(kept - same).max() < 1e-5 * np.abs(same).max()

    def test_the_engine_s_extra_samples_are_cut(self) -> None:
        """A field's low band is 57 603 samples, the engine running over its window."""
        longer = np.concatenate([band_limited(a_response(), 1500.0), np.zeros((4, 3))], axis=1)
        assert np.array_equal(decimate(longer, RATE), decimate(longer[:, :SAMPLES], RATE))

    def test_a_rate_the_window_does_not_divide_is_refused(self) -> None:
        with pytest.raises(ValueError, match="whole number"):
            decimate(np.zeros((1, 100)), 18204.7382)


class TestTheLowSide:
    crossover = Crossover()

    def test_it_is_the_low_part_of_the_join(self) -> None:
        """With nothing on the high side, ``blend`` returns the low side alone."""
        response = a_response()
        joined, _ = blend(response, np.zeros_like(response), RATE, self.crossover)
        assert relative_db(low_side(response, RATE, self.crossover), joined) < -200.0
        stored = to_stored(response, RATE, self.crossover)
        assert stored.shape == (4, LOW_SAMPLES) and stored.dtype == np.float32
        assert relative_db(from_stored(stored), joined) < -120.0

    def test_it_is_zero_above_the_ramp(self) -> None:
        stored = to_stored(a_response(), RATE, self.crossover)
        spectrum = np.abs(np.fft.rfft(stored, axis=-1))
        freqs = np.fft.rfftfreq(LOW_SAMPLES, 1 / LOW_RATE_HZ)
        assert spectrum[:, freqs > 1414.3].max() < 1e-6 * spectrum.max()
        assert spectrum[:, freqs < 700.0].max() > 0.1 * spectrum.max()

    def test_the_cache_form_gives_the_stored_form_the_field_s_response_gives(self) -> None:
        """A response already under 2 kHz: its masks taken at 4 kHz are those taken at 48."""
        response = band_limited(a_response(), 1700.0)
        direct = to_stored(response, RATE, self.crossover)
        cached = to_stored(decimate(response, RATE), LOW_RATE_HZ, self.crossover)
        assert relative_db(cached, direct) < -60.0

    def test_the_onset_is_read_at_48_khz_whatever_the_rate_kept(self) -> None:
        response = band_limited(a_response(), 1900.0)
        assert onset_s(response[0], RATE) == pytest.approx(0.02, abs=1e-4)
        assert onset_s(decimate(response, RATE)[0], LOW_RATE_HZ) == onset_s(response[0], RATE)
        with pytest.raises(ValueError, match="48 kHz or at 4 kHz"):
            onset_s(response[0], 44100.0)

    def test_the_air_takes_the_late_part_down_and_leaves_the_onset(self) -> None:
        response = decimate(band_limited(a_response(), 1700.0), RATE)
        dry = to_stored(response, LOW_RATE_HZ, self.crossover)
        wet = to_stored(response, LOW_RATE_HZ, self.crossover, atmosphere=Atmosphere())
        late, early = slice(2400, 4400), slice(60, 120)
        loss = 10 * np.log10(np.sum(wet[:, late] ** 2) / np.sum(dry[:, late] ** 2))
        assert -1.5 < loss < -0.05
        assert relative_db(wet[:, early], dry[:, early]) < -30.0


class TestThePairKey:
    encoder = {"order": 7, "fit_order": 10, "fmax_hz": 1768.0}

    def key(self, **changes: object) -> str:
        given: dict[str, object] = {
            "voxel_low_key": "grid",
            "source_m": [1.0, 1.7, 2.0],
            "cell_m": [3.0, 1.7, 4.0],
            "encoder": self.encoder,
            "solver": "engine/1",
        }
        given.update(changes)
        return pair_key(
            str(given["voxel_low_key"]),
            given["source_m"],
            given["cell_m"],
            encoder=dict(given["encoder"]),  # type: ignore[call-overload]
            solver=str(given["solver"]),
        )

    def test_it_is_64_hexadecimal_characters_and_the_same_twice(self) -> None:
        assert len(self.key()) == 64 and int(self.key(), 16) >= 0
        assert self.key() == self.key()

    def test_a_millimetre_changes_it_and_less_does_not(self) -> None:
        assert self.key(source_m=[1.001, 1.7, 2.0]) != self.key()
        assert self.key(cell_m=np.array([3.0, 1.7, 4.0004])) == self.key()

    def test_the_grid_the_encoder_and_the_solver_each_change_it(self) -> None:
        keys = {
            self.key(),
            self.key(voxel_low_key="other"),
            self.key(encoder={**self.encoder, "fmax_hz": 1000.0}),
            self.key(solver="engine/2"),
        }
        assert len(keys) == 4


class TestTheLead:
    """A pair on the geometric clock is brought to the pack's, the mirror's lead later."""

    def test_a_whole_number_of_samples_is_a_shift_and_nothing_wraps(self) -> None:
        response = decimate(band_limited(a_response(), 1500.0), RATE).astype(float)
        late = delayed(response, LOW_RATE_HZ, 16 / LOW_RATE_HZ)
        assert late.shape == response.shape
        peak = np.abs(response).max()
        assert np.abs(late[:, 16:-16] - response[:, :-32]).max() < 1e-9 * peak
        # What a circular shift would bring back to the start is the response's end.
        assert np.abs(late[:, :16]).max() < 1e-9 * peak
        assert np.abs(response[:, -16:]).max() > 1e-4 * peak
        # The new end is not a cut: the last 4 ms fall to zero.
        assert np.all(late[:, -1] == 0.0)
        assert np.all(np.abs(late[:, -16:]) <= np.abs(response[:, -32:-16]) + 1e-12)
        assert np.array_equal(delayed(response, LOW_RATE_HZ, 0.0), response)
        with pytest.raises(ValueError, match="never advanced"):
            delayed(response, LOW_RATE_HZ, -0.001)

    def test_the_lead_of_hssd_0076_is_not_a_whole_sample_at_4_khz_and_is_exact(self) -> None:
        lead = 512 / RATE  # 42.67 samples at 4 kHz
        response = band_limited(a_response(), 1500.0)
        late = delayed(decimate(response, RATE), LOW_RATE_HZ, lead)
        want = np.zeros_like(response)
        want[:, 512:] = response[:, :-512]
        # Measured -68 dB: the response is not periodic, and its two ends ring a little.
        assert relative_db(from_stored(late)[:, 600:-600], want[:, 600:-600]) < -60.0
        assert onset_s(late[0], LOW_RATE_HZ) == pytest.approx(onset_s(response[0], RATE) + lead)

    def test_the_field_s_unit_is_the_free_space_of_one_grid_step_at_8_khz(self) -> None:
        # 0.0258 to 0.0264 measured on the validated field of hssd_0076, 0.3 to 0.7 m from S1.
        assert pytest.approx(0.02625, abs=1e-5) == FIELD_UNIT_AT_1M


def test_a_low_only_solve_reaches_past_the_ramp_s_top() -> None:
    """1500 Hz for the crossover at 1 kHz, where a field's low band stops at 1000."""
    assert solve_fmax_hz() == pytest.approx(1500.0)
    assert solve_fmax_hz() > Crossover().band_hz()[1]
    assert solve_fmax_hz(Crossover(cutoff_hz=500.0)) == pytest.approx(750.0)
    assert solve_fmax_hz(headroom=1.25) == pytest.approx(1000.0 * np.sqrt(2.0) / 0.8)
