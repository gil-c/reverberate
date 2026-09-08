"""Tests for the solve cost model.

Every expectation here is a figure the roadmap wrote down before this module
existed: the W29 anchor of 1211 s, and the five rows of W30's cost tables. The
model has one free constant and five independent chances to disagree with a
published number, which is what makes it a test rather than a restatement.

The other half of the file is about the two rules the signatures enforce. A
cost figure that does not carry its hourly rate is not a measurement, and a
room that needs three cards costs three times the rate for the same wall
clock. Both have burned this project before.
"""

from __future__ import annotations

import pytest

from reverberate.cost import (
    BYTES_PER_POINT,
    SECONDS_PER_POINT_STEP,
    cards_for,
    estimate,
    steps_for,
)

#: The grid's own rate at 16 kHz and 10.5 points per wavelength, read from the
#: voxelisation manifest of the bedroom entry rather than derived.
SOLVER_RATE_HZ = 291275.8114830544

#: W29's rate, billed rather than quoted.
RATE_USD_PER_HOUR = 0.917


def test_the_anchor_round_trips_to_its_measured_engine_time() -> None:
    """The one measurement the constant is fitted to, checked back.

    B0 and W29: ``bedroom_only`` at 16 kHz, 4 607 993 520 points, a 60 ms
    window, 1211 s of engine on one A100 80 GB.
    """
    anchor = estimate(
        4_607_993_520,
        0.060,
        SOLVER_RATE_HZ,
        usd_per_hour_per_card=RATE_USD_PER_HOUR,
    )
    assert anchor.gpu_seconds == pytest.approx(1211.0, rel=0.005)
    assert anchor.cards == 1


def test_throughput_is_the_roadmaps_66_5_billion_updates_per_second() -> None:
    assert pytest.approx(6.65e10, rel=0.005) == 1.0 / SECONDS_PER_POINT_STEP


@pytest.mark.parametrize(
    ("name", "grid_points", "duration_s", "gpu_hours", "usd", "usd_rel", "cards"),
    [
        # The first three rows are quoted to two figures. The last two carry a
        # tilde in the roadmap, so they are checked to the precision they were
        # actually stated at rather than to a sharpness nobody claimed.
        ("kitchen", 5.44e9, 1.2, 7.9, 7.3, 0.02, 1),
        ("bathroom.001", 2.53e9, 2.0, 6.2, 5.6, 0.02, 1),
        ("living room", 1.81e10, 1.2, 26.5, 24.0, 0.02, 3),
        ("living room, 30 ms control", 1.81e10, 0.030, 0.66, 0.6, 0.10, 3),
        ("living room, 15 ms ellipsoid", 5.57e9, 0.030, 0.20, 0.2, 0.10, 1),
    ],
)
def test_the_published_cost_tables_are_reproduced(
    name: str,
    grid_points: float,
    duration_s: float,
    gpu_hours: float,
    usd: float,
    usd_rel: float,
    cards: int,
) -> None:
    """Roadmap W30's own tables, every row."""
    cost = estimate(
        int(grid_points),
        duration_s,
        SOLVER_RATE_HZ,
        usd_per_hour_per_card=RATE_USD_PER_HOUR,
    )
    assert cost.gpu_hours == pytest.approx(gpu_hours, rel=0.02), name
    assert cost.usd == pytest.approx(usd, rel=usd_rel), name
    assert cost.cards == cards, name


def test_the_hourly_rate_cannot_be_omitted() -> None:
    """Hard constraint 10: a cost without a rate is not a measurement."""
    with pytest.raises(TypeError):
        estimate(1_000_000, 0.1, SOLVER_RATE_HZ)  # type: ignore[call-arg]


def test_cost_is_linear_in_the_window() -> None:
    """The duration lever, which is what the whole item turns on."""
    short = estimate(int(5.44e9), 0.060, SOLVER_RATE_HZ, usd_per_hour_per_card=1.0)
    long = estimate(int(5.44e9), 0.600, SOLVER_RATE_HZ, usd_per_hour_per_card=1.0)
    assert long.gpu_hours / short.gpu_hours == pytest.approx(10.0, rel=0.01)


def test_a_room_needing_three_cards_costs_three_times_the_rate() -> None:
    """Wall clock falls with cards; money does not."""
    cost = estimate(int(1.81e10), 0.030, SOLVER_RATE_HZ, usd_per_hour_per_card=1.0)
    assert cost.cards == 3
    assert cost.wall_clock_hours == pytest.approx(cost.gpu_hours / 3.0)
    assert cost.usd == pytest.approx(cost.gpu_hours)


def test_the_card_ceiling_matches_the_measured_capacity() -> None:
    """About 8.6e9 points fit an 80 GB card, and one more point does not."""
    usable = (80.0 * 1e9 - 2.13e9) / BYTES_PER_POINT
    assert usable == pytest.approx(8.7e9, rel=0.02)
    assert cards_for(int(usable * 0.99)) == 1
    assert cards_for(int(usable * 1.01)) == 2


def test_a_48_gb_card_does_not_hold_the_kitchen() -> None:
    """Why 'one card' is not enough detail to rent against."""
    assert cards_for(int(5.44e9), card_ram_gb=80.0) == 1
    assert cards_for(int(5.44e9), card_ram_gb=48.0) == 2


def test_steps_follow_the_solver_rate_and_not_the_delivery_rate() -> None:
    """One second is 291 276 steps at 16 kHz, not 48 000."""
    assert steps_for(1.0, SOLVER_RATE_HZ) == 291276
    assert steps_for(1.0, 48000.0) == 48000


def test_steps_round_up_so_a_window_is_never_short() -> None:
    assert steps_for(1.5e-5, 48000.0) == 1


def test_sources_multiply_but_receivers_are_absent() -> None:
    """Each source needs its own solve; receivers cost 64 bytes a step."""
    one = estimate(1_000_000, 0.1, 48000.0, usd_per_hour_per_card=1.0)
    four = estimate(1_000_000, 0.1, 48000.0, usd_per_hour_per_card=1.0, sources=4)
    assert four.point_steps == pytest.approx(4.0 * one.point_steps)


def test_the_summary_always_names_the_rate() -> None:
    text = estimate(
        int(5.44e9), 1.2, SOLVER_RATE_HZ, usd_per_hour_per_card=RATE_USD_PER_HOUR
    ).summary()
    assert "USD/h" in text
    assert "0.917" in text


def test_the_record_carries_the_constants_it_assumed() -> None:
    record = estimate(
        int(5.44e9), 1.2, SOLVER_RATE_HZ, usd_per_hour_per_card=RATE_USD_PER_HOUR
    ).record()
    assert record["seconds_per_point_step"] == SECONDS_PER_POINT_STEP
    assert record["bytes_per_point"] == BYTES_PER_POINT
    assert record["usd_per_hour_per_card"] == RATE_USD_PER_HOUR


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((0, 1.0, 48000.0), {"usd_per_hour_per_card": 1.0}),
        ((1000, -1.0, 48000.0), {"usd_per_hour_per_card": 1.0}),
        ((1000, 1.0, 0.0), {"usd_per_hour_per_card": 1.0}),
        ((1000, 1.0, 48000.0), {"usd_per_hour_per_card": 1.0, "sources": 0}),
        ((1000, 1.0, 48000.0), {"usd_per_hour_per_card": -1.0}),
    ],
)
def test_impossible_estimates_are_refused(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        estimate(*args, **kwargs)  # type: ignore[arg-type]
