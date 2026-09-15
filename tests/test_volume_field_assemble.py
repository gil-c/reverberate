"""The levelling of a borrowed low band."""

from __future__ import annotations

import numpy as np

from reverberate.experiments.w40_volume_field.assemble import level_borrowed_low


class TestBorrowedLow:
    def test_a_borrowed_band_lands_at_the_ratio_the_bandwidths_predict(self) -> None:
        """The pair check refuses more than 3 dB from 20 log10(1000 / 4000);
        a neighbour's low band 5 dB off is scaled onto it exactly."""
        from reverberate import bands as band_split
        from reverberate.spatial.bands import calibration_bands_hz

        rng = np.random.default_rng(0)
        rate = 48000
        mid = rng.standard_normal((4, 19200)).astype(np.float32)
        low = (mid * 10 ** (-7.0 / 20)).astype(np.float32)  # 7 dB under, 12 predicted: 5 dB off
        levelled = level_borrowed_low(low, mid, rate, 1000.0, 4000.0)
        measured = band_split.level_ratio(
            levelled[0].astype(float),
            mid[0].astype(float),
            rate,
            calibration_hz=calibration_bands_hz(1000.0, rate),
        )
        assert abs(20 * np.log10(measured) - 20 * np.log10(1000 / 4000)) < 0.05
        assert levelled.dtype == np.float32 and levelled.shape == low.shape
