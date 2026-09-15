"""The grid budget of a plan: the pitch that fills a cap."""

from __future__ import annotations

from reverberate.experiments.w40_volume_field.plan import pitch_for


class TestGridBudget:
    def test_the_pitch_that_fills_a_cap_puts_the_cap_on_the_floor(self) -> None:
        pitch = pitch_for(85.6, 560)
        assert 0.35 <= pitch <= 0.45
        assert 85.6 / pitch**2 <= 560 * 1.05
