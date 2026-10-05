"""Where a pair's two bands are joined in pressure: on its direct sound, not on its loudest sample.

The first whole scene (2026-10-05) had its far pairs anchored 5 to 20 ms
after their direct sound, on whichever later arrival was the loudest, and
two neighbours on a rail anchored 10 to 45 ms apart. These are the rule
that replaces it, on responses whose arrivals are placed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.audio import Atmosphere
from reverberate.mirror.hybrid import Crossover
from reverberate.spatial.lowband import LOW_RATE_HZ, LOW_SAMPLES, low_side, onset_s
from reverberate.trace.level import pair_anchor_s, pair_low

LEAD_S = 512 / 48000.0
C = 343.2


def far_pair(direct: float = 0.5, later_ms: float = 12.0, metres: float = 6.0) -> np.ndarray:
    """A pair in the cache form, geometric clock: a direct sound and a louder arrival after it."""
    response = np.zeros((4, LOW_SAMPLES), dtype=np.float32)
    at = int(round(metres / C * LOW_RATE_HZ))
    response[:, at] = direct
    response[:, at + int(round(later_ms * 1e-3 * LOW_RATE_HZ))] = 1.0
    return response


def gain_at_1khz(stored: np.ndarray, cached: np.ndarray, at_s: float) -> float:
    """What the masks made of the arrival at ``at_s`` (pack clock) at 1 kHz, on channel 0."""
    centre = int(round(at_s * LOW_RATE_HZ))
    window = np.zeros(LOW_SAMPLES)
    window[centre - 12 : centre + 13] = np.hanning(25)
    tone = np.exp(-2j * np.pi * 1000.0 * np.arange(LOW_SAMPLES) / LOW_RATE_HZ)
    lead = int(round(LEAD_S * LOW_RATE_HZ))
    before = np.roll(np.asarray(cached[0], dtype=float), lead)
    return float(abs(np.sum(stored[0] * window * tone)) / abs(np.sum(before * window * tone)))


def test_the_anchor_is_the_direct_sound_where_there_is_one_and_the_loudest_where_not() -> None:
    cached = far_pair()
    straight = 6.0 / C
    aired = np.roll(np.asarray(cached[0], dtype=float), int(round(LEAD_S * LOW_RATE_HZ)))
    loudest = onset_s(aired, LOW_RATE_HZ)
    assert loudest == pytest.approx(straight + LEAD_S + 0.012, abs=3e-4)
    anchor = pair_anchor_s(aired, straight, lead_s=LEAD_S, sound_speed_m_s=C)
    assert anchor == pytest.approx(straight + LEAD_S, abs=3e-4)
    # In the shadow of a wall the response holds nothing at the straight line's time
    # (under a fifth of its loudest sample): the loudest sample, as before.
    shadowed = np.roll(np.asarray(far_pair(direct=0.1)[0], dtype=float), 43)
    assert pair_anchor_s(shadowed, straight, lead_s=LEAD_S, sound_speed_m_s=C) == pytest.approx(
        onset_s(shadowed, LOW_RATE_HZ)
    )
    # And nothing at all there: the same.
    alone = np.roll(np.asarray(far_pair(direct=0.0)[0], dtype=float), 43)
    assert pair_anchor_s(alone, straight, lead_s=LEAD_S, sound_speed_m_s=C) == pytest.approx(
        onset_s(alone, LOW_RATE_HZ)
    )


def test_a_far_pair_s_later_arrival_takes_the_power_mask_once_joined_on_the_direct() -> None:
    crossover = Crossover()
    still = Atmosphere(20.0, 50.0, 101.325)
    cached = far_pair(later_ms=30.0)
    straight = 6.0 / C
    kw = {"sound_speed_m_s": C, "lead_s": LEAD_S}
    old, old_onset, _ = pair_low(cached, crossover, still, **kw)
    new, new_onset, _ = pair_low(cached, crossover, still, straight_s=straight, **kw)
    assert old_onset == pytest.approx(straight + LEAD_S + 0.030, abs=3e-4)
    assert new_onset == pytest.approx(straight + LEAD_S, abs=3e-4)
    direct_s, later_s = straight + LEAD_S, straight + LEAD_S + 0.030
    # The direct sound is joined in pressure either way: half of it under the crossover at 1 kHz.
    assert gain_at_1khz(old, cached, direct_s) == pytest.approx(0.5, abs=0.03)
    assert gain_at_1khz(new, cached, direct_s) == pytest.approx(0.5, abs=0.03)
    # The arrival 30 ms later was inside the old window, in pressure; it is now in power.
    assert gain_at_1khz(old, cached, later_s) == pytest.approx(0.5, abs=0.03)
    assert gain_at_1khz(new, cached, later_s) == pytest.approx(np.sqrt(0.5), abs=0.03)
    # Without the straight line a pair is stored as every pack before was, to the bit.
    again = pair_low(cached, crossover, still, **kw)[0]
    assert np.array_equal(old, again)
    # And a near pair, whose loudest sample is its direct sound, does not change at all.
    near = far_pair(direct=1.0, later_ms=12.0, metres=1.0)
    near[:, int(round(1.0 / C * LOW_RATE_HZ)) + 48] = 0.4
    a = pair_low(near, crossover, still, **kw)
    b = pair_low(near, crossover, still, straight_s=1.0 / C, **kw)
    assert a[1] == pytest.approx(b[1], abs=1.5e-4)
    assert np.abs(a[0] - b[0]).max() < 2e-3 * np.abs(a[0]).max()


def test_the_low_side_is_anchored_where_it_is_told() -> None:
    crossover = Crossover()
    response = np.zeros((1, LOW_SAMPLES))
    response[0, 400] = 1.0
    response[0, 520] = 0.5
    told = low_side(response, LOW_RATE_HZ, crossover, onset=520 / LOW_RATE_HZ)
    own = low_side(response, LOW_RATE_HZ, crossover)
    late = low_side(response, LOW_RATE_HZ, crossover, onset=400 / LOW_RATE_HZ)
    assert np.allclose(own, late, atol=1e-12)  # the loudest sample is at 400
    assert np.abs(told - own).max() > 0.02  # the arrival at 520 changed masks
