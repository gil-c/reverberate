"""The comparison a campaign is judged by, on signals whose difference is known."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.accel.verify import compare_encoded, compare_signals


def a_field(points: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((points, 4, 2000)) * np.exp(-np.arange(2000) / 500.0)).astype(
        np.float32
    )


class TestCompareSignals:
    def test_identical_signals_report_zero_and_every_point_identical(self) -> None:
        x = a_field()
        report = compare_signals(x, x, 48000.0)
        assert report["max_over_peak"]["max"] == 0.0
        assert report["float32_identical"] == 3 and report["within_tolerance"] == 3
        assert max(report["octave_level_db"]["max_abs"]) == 0.0

    def test_a_scaled_point_is_found_with_its_level(self) -> None:
        x = a_field()
        y = x.copy()
        y[1] *= 1.01
        report = compare_signals(y, x, 48000.0, tolerance=1e-6)
        assert report["max_over_peak"]["worst_point"] == 1
        assert 0.009 < report["max_over_peak"]["max"] < 0.011
        assert report["float32_identical"] == 2 and report["within_tolerance"] == 2
        assert abs(max(report["octave_level_db"]["max_abs"]) - 20 * np.log10(1.01)) < 1e-6


class TestCompareEncoded:
    def test_files_are_joined_on_point_index(self, tmp_path: Path) -> None:
        x = a_field(points=4)
        for name, points, index in (
            ("a.h5", x[[0, 1, 2]], [0, 1, 2]),
            ("b.h5", x[[2, 1, 3]], [2, 1, 3]),
        ):
            with h5py.File(tmp_path / name, "w") as handle:
                handle.create_dataset("signals", data=points)
                handle.create_dataset("point_index", data=np.asarray(index))
                handle.attrs["sample_rate_hz"] = 48000.0
        report = compare_encoded(tmp_path / "a.h5", tmp_path / "b.h5")
        assert report["points"] == 2
        assert report["float32_identical"] == 2
        assert report["points_only_in_a"] == 1 and report["points_only_in_b"] == 1


class TestCommandLine:
    def test_two_fields_are_compared_and_the_report_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from reverberate.accel.verify import main

        rng = np.random.default_rng(0)
        for name, scale in (("a", 1.0), ("b", 1.0 + 1e-8)):
            with h5py.File(tmp_path / f"{name}.h5", "w") as handle:
                handle.create_dataset(
                    "ir", data=(rng.standard_normal((2, 4, 512)) * scale).astype(np.float32)
                )
                handle.create_dataset("point_index", data=np.array([0, 1]))
                handle.create_dataset("has_high", data=np.array([True, True]))
                handle.attrs["sample_rate_hz"] = 48000.0
                handle.attrs["mid_on_high_gain"] = 0.5
            rng = np.random.default_rng(0)
        assert (
            main(
                [
                    "field",
                    str(tmp_path / "a.h5"),
                    str(tmp_path / "b.h5"),
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )
            == 0
        )
        report = json.loads((tmp_path / "r.json").read_text())
        assert report["points"] == 2 and report["within_tolerance"] == 2
        assert report["flags_differing"] == {"has_high": 0}
        assert "per_point_max_over_peak" not in capsys.readouterr().out
