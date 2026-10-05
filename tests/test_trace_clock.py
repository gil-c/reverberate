"""The clock and the scale of a trace, read where the mirror puts the direct sound.

Two whole scenes were stopped after their solves (2026-10-05) by a check
that read each pair's loudest sample, which at a far pair is a later
arrival. These hold the check to what it is for: a window of far pairs
alone passes, and a cache on another clock or another scale stops the
trace with a message that says which.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.audio import Atmosphere
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
from reverberate.trace.clock import CLOCK_S, SCALE_DB, read_direct, verdict
from reverberate.trace.engines import FreeFieldPairs
from reverberate.trace.level import pair_omni
from test_trace import assets, resting_recipe, traced

C = 343.2
LEAD = 512 / 48000.0


def far_pair(
    tmp: Path,
    distance_m: float,
    *,
    later_s: float = 0.007,
    later_gain: float = 1.5,
    shift_s: float = 0.0,
    gain: float = 1.0,
) -> dict[str, Any]:
    """A pair as the levelling records it: a direct sound, and a louder arrival after it.

    The direct sound is a monopole's in free air at ``distance_m``, on the
    field's scale; the later arrival is another's, ``later_s`` behind and
    ``later_gain`` times as loud, which is what a far pair of hssd_0076
    holds 5 to 20 ms after its direct sound. ``shift_s`` and ``gain`` are
    a cache on another clock and on another scale.
    """
    second = distance_m + later_s * C
    cells = np.array([[distance_m, 1.5, 0.0], [second, 1.5, 0.0]])
    pairs = FreeFieldPairs(
        np.array([[0.0, 1.5, 0.0]]), cells, tmp, sound_speed_m_s=C, gain=FIELD_UNIT_AT_1M
    )
    omni = pairs.response(0, 0)[:1] + later_gain * (second / distance_m) * pairs.response(0, 1)[:1]
    onset, aired = pair_omni(omni * gain, Atmosphere(), sound_speed_m_s=C, lead_s=LEAD + shift_s)
    first = distance_m / C
    return {
        "direct": True,
        "first_s": first,
        "onset_s": onset,
        **read_direct(aired, first, lead_s=LEAD, sound_speed_m_s=C),
    }


def test_a_far_pair_is_read_at_its_direct_sound_and_not_at_its_loudest_sample(
    tmp_path: Path,
) -> None:
    pair = far_pair(tmp_path, 6.0)
    # The loudest sample is the later arrival, 7 ms after the direct sound ...
    assert pair["onset_s"] - pair["first_s"] - LEAD == pytest.approx(0.007, abs=2e-4)
    # ... and the direct sound is where the mirror puts it, at 1/d on the field's scale.
    assert abs(pair["direct_trail_s"] - LEAD) < 5e-5
    assert abs(pair["direct_level_db"]) < 0.5
    assert pair["direct_share"] == pytest.approx(1 / 1.5, abs=0.05)
    # A floor's reflection 1.5 ms behind, and larger: the first peak is still the direct sound.
    near = far_pair(tmp_path, 8.0, later_s=0.0015, later_gain=1.3)
    assert abs(near["direct_trail_s"] - LEAD) < 1.5e-4


def test_a_window_of_far_pairs_alone_passes_where_the_loudest_sample_failed(
    tmp_path: Path,
) -> None:
    pairs = [far_pair(tmp_path, 4.0 + 0.5 * k, later_s=0.005 + 0.001 * k) for k in range(13)]
    # What stopped two scenes: no percentile of the loudest samples is the lead.
    old = np.array([p["onset_s"] - p["first_s"] for p in pairs])
    assert np.percentile(old, 10) - LEAD > 10 * CLOCK_S
    report, stopped = verdict(pairs, LEAD, sound_speed_m_s=C)
    assert stopped is None
    assert report["pairs"] == report["pairs_with_their_direct_sound_there"] == 13
    assert abs(report["off_s"]["first_quartile"]) < 1e-4
    assert abs(report["level_db"]["third_quartile"]) < 1.0
    assert set(report["by_distance"]) == {"4 to 8 m", "over 8 m"}
    # A pair the mirror gives no direct path is not read, and a run with none is not judged.
    assert verdict(
        [{"direct": False, "first_s": 0.01, "onset_s": 0.5}], LEAD, sound_speed_m_s=C
    ) == (
        {"pairs": 0},
        None,
    )


def said(message: str, unit: str, after: str) -> float:
    """The figure a message gives before ``unit`` and ``after``: ``1.00 ms after`` is 1.0."""
    found = re.search(rf"([0-9.]+) {unit} {after}", message)
    assert found is not None, message
    return float(found.group(1))


@pytest.mark.parametrize(
    ("fault", "words", "figure"),
    [
        ({"shift_s": 0.001}, ("not on one clock",), ("ms", "after the mirror's", 1.0, 0.05)),
        ({"shift_s": -0.001}, ("not on one clock",), ("ms", "before the mirror's", 1.0, 0.05)),
        ({"gain": 0.5}, ("not on one scale", "the clock is right"), ("dB", "under 1/d", 6.0, 0.4)),
        ({"gain": 2.0}, ("not on one scale",), ("dB", "over 1/d", 6.0, 0.4)),
        ({"gain": 0.1}, ("not on one scale",), ("dB", "under 1/d", 20.0, 0.4)),
        (
            {"shift_s": LEAD},
            ("not on one clock", "holds none", "a lead of its own"),
            ("ms", "late", 1e3 * LEAD + 5.0, 6.0),
        ),
    ],
    ids=["1 ms late", "1 ms early", "6 dB under", "6 dB over", "20 dB under", "its own lead"],
)
def test_a_cache_on_another_clock_or_scale_is_named_on_far_pairs_too(
    tmp_path: Path,
    fault: dict[str, float],
    words: tuple[str, ...],
    figure: tuple[str, str, float, float],
) -> None:
    pairs = [far_pair(tmp_path, 4.0 + 0.5 * k, **fault) for k in range(9)]
    _, stopped = verdict(pairs, LEAD, sound_speed_m_s=C)
    assert stopped is not None
    for word in words:
        assert word in stopped, stopped
    assert ("not on one clock" in stopped) != ("not on one scale" in stopped)
    unit, after, value, within = figure
    assert said(stopped, unit, after) == pytest.approx(value, abs=within)


def test_the_allowances_are_half_a_millisecond_and_three_decibels(tmp_path: Path) -> None:
    assert CLOCK_S == 0.5e-3 and SCALE_DB == 3.0
    inside = [far_pair(tmp_path, 3.0 + 0.5 * k, shift_s=0.0003, gain=1.25) for k in range(5)]
    assert verdict(inside, LEAD, sound_speed_m_s=C)[1] is None
    outside = [far_pair(tmp_path, 3.0 + 0.5 * k, shift_s=0.0007) for k in range(5)]
    stopped = str(verdict(outside, LEAD, sound_speed_m_s=C)[1])
    assert said(stopped, "ms", "after") == pytest.approx(0.7, abs=0.05)


@pytest.mark.parametrize(
    ("engine", "words", "figure"),
    [
        ({"lead_s": 0.001}, ("not on one clock",), ("ms", "after", 1.0, 0.05)),
        ({"gain": 0.5 * FIELD_UNIT_AT_1M}, ("not on one scale",), ("dB", "under", 6.0, 0.5)),
        (
            {"lead_s": assets().pack_lead_s},
            ("not on one clock", "holds none"),
            ("ms", "late", 1e3 * assets().pack_lead_s, 0.1),
        ),
    ],
    ids=["1 ms late", "6 dB under", "its own lead"],
)
def test_a_trace_whose_pairs_are_off_stops_before_its_pack_and_says_which(
    tmp_path: Path,
    engine: dict[str, Any],
    words: tuple[str, ...],
    figure: tuple[str, str, float, float],
) -> None:
    trace, _, _ = traced(tmp_path, resting_recipe(), rays=16, **engine)
    with pytest.raises(RuntimeError) as stopped:
        trace.run()
    for word in words:
        assert word in str(stopped.value), stopped.value
    unit, after, value, within = figure
    assert said(str(stopped.value), unit, after) == pytest.approx(value, abs=within)
    assert (tmp_path / "out" / "campaign.failed").is_file()
    assert not (tmp_path / "out" / "pack.h5").exists()
    # The report keeps what was read, the old statistic beside the new.
    level = trace.report["level"]
    assert "trail_s_at_the_first_decile" in level and level["direct_sound"]["pairs"] > 0
