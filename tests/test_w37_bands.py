"""Tests for the three-band cost sweep.

The point of the module is an asymmetry that is easy to state and easy to get
wrong: each band's price follows its own top frequency to the fourth power, so
the lower crossover prices only the low band and the upper one prices only the
mid band, while the high band does not move with either. Most of these tests
are that asymmetry, because a sweep that quietly coupled the bands would still
produce a plausible surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reverberate.experiments.w37_bands import build, price

FLAT = (23.575, 2.885, 18.439)
ROOM = (4.354, 2.824, 3.145)
RATE = 0.917


def _price(low: float = 1000.0, high: float = 4000.0) -> dict[str, float]:
    plans = price(
        FLAT,
        ROOM,
        low_crossover_hz=low,
        high_crossover_hz=high,
        usd_per_hour_per_card=RATE,
    )
    return {plan.name: plan.cost.usd for plan in plans}


def test_the_bands_run_on_the_domains_the_architecture_gives_them() -> None:
    plans = {
        plan.name: plan
        for plan in price(
            FLAT,
            ROOM,
            low_crossover_hz=1000.0,
            high_crossover_hz=4000.0,
            usd_per_hour_per_card=RATE,
        )
    }
    assert plans["low"].span_m == FLAT
    assert plans["mid"].span_m == FLAT
    assert plans["high"].span_m == ROOM
    assert plans["high"].fmax_hz == 16000.0


def test_the_upper_crossover_prices_only_the_mid_band() -> None:
    """The asymmetry the whole sweep turns on."""
    at_four = _price(high=4000.0)
    at_two = _price(high=2000.0)
    assert at_two["mid"] < at_four["mid"]
    assert at_two["low"] == pytest.approx(at_four["low"])
    assert at_two["high"] == pytest.approx(at_four["high"])


def test_the_lower_crossover_prices_only_the_low_band() -> None:
    at_one = _price(low=1000.0)
    at_seven = _price(low=700.0)
    assert at_seven["low"] < at_one["low"]
    assert at_seven["mid"] == pytest.approx(at_one["mid"])
    assert at_seven["high"] == pytest.approx(at_one["high"])


def test_a_band_costs_the_fourth_power_of_its_top_frequency() -> None:
    """Points go as the cube and steps as the first power.

    The measured ratio is 15.55 rather than 16, and the shortfall is the
    absorbing pad: PFFDTD adds 3.5 cells to every face, and on a coarse grid
    those cells are a larger fraction of the box than on a fine one. So the
    ratio approaches 16 from below, and a test asserting exactly 16 would be
    asserting that the pad does not exist.
    """
    at_four = _price(high=4000.0)["mid"]
    at_two = _price(high=2000.0)["mid"]
    ratio = at_four / at_two
    assert 15.0 < ratio < 16.0


def test_the_low_band_is_negligible_against_the_other_two() -> None:
    """Which is why the lower crossover has no money behind it."""
    costs = _price()
    assert costs["low"] < 0.01 * (costs["mid"] + costs["high"])


def test_the_mid_band_is_the_largest_single_line() -> None:
    costs = _price()
    assert costs["mid"] > costs["high"]
    assert costs["mid"] > costs["low"]


def test_crossovers_that_do_not_rise_are_refused() -> None:
    with pytest.raises(ValueError):
        price(
            FLAT,
            ROOM,
            low_crossover_hz=4000.0,
            high_crossover_hz=1000.0,
            usd_per_hour_per_card=RATE,
        )
    with pytest.raises(ValueError):
        price(
            FLAT,
            ROOM,
            low_crossover_hz=1000.0,
            high_crossover_hz=20000.0,
            usd_per_hour_per_card=RATE,
        )


def test_the_hourly_rate_cannot_be_omitted() -> None:
    with pytest.raises(TypeError):
        price(FLAT, ROOM, low_crossover_hz=1000.0, high_crossover_hz=4000.0)  # type: ignore[call-arg]


def test_the_report_names_what_it_does_not_measure(tmp_path: Path) -> None:
    """The saving is bought entirely with an unmeasured assumption."""
    report = build(FLAT, ROOM, tmp_path / "out", usd_per_hour_per_card=RATE)
    assert "W23" in report["unmeasured"]
    assert "do not choose a crossover" in report["unmeasured"]
    assert report["reference_run"] is None
    assert (tmp_path / "out" / "report.json").is_file()


def test_the_sweep_covers_both_axes_and_holds_the_assumed_split(tmp_path: Path) -> None:
    report = build(
        FLAT,
        ROOM,
        tmp_path / "out",
        usd_per_hour_per_card=RATE,
        low_crossovers_hz=(800.0, 1000.0),
        high_crossovers_hz=(2500.0, 4000.0),
    )
    assert len(report["sweep"]) == 4
    assumed = report["assumed_split"]
    assert assumed["low_crossover_hz"] == 1000.0
    assert assumed["high_crossover_hz"] == 4000.0
    assert assumed["total_usd"] == pytest.approx(sum(_price().values()), rel=1e-6)


def test_a_shorter_high_band_window_moves_only_the_high_band(tmp_path: Path) -> None:
    report = build(
        FLAT,
        ROOM,
        tmp_path / "out",
        usd_per_hour_per_card=RATE,
        low_crossovers_hz=(1000.0,),
        high_crossovers_hz=(4000.0,),
        windows_s={"high": 0.030},
    )
    bands = {band["band"]: band for band in report["assumed_split"]["bands"]}
    assert bands["high"]["usd"] == pytest.approx(0.5 * _price()["high"], rel=0.02)
    assert bands["mid"]["usd"] == pytest.approx(_price()["mid"], rel=1e-6)
