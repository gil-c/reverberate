"""The encode job, the merge of shards, and the levelling of a borrowed low band."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from reverberate.experiments.w40_volume_field.assemble import level_borrowed_low
from reverberate.experiments.w40_volume_field.encode import encode_job, merge_encoded

PLAN = {
    "sound_speed_m_s": 343.2,
    "lowcut_hz": 40.0,
    "lowcut_order": 8,
    "encoder": {"order": 7, "fit_order": 10},
    "bands": {
        "mid": {
            "rows": [[0, 5], [5, 12], None, [12, 20]],
            "centres": [[0, 1, 0], [1, 1, 0], None, [2, 1, 0]],
            "grid_step_m": 0.00817,
            "fmax_hz": 4000.0,
            "sample_rate_hz": 72819.0,
        }
    },
}


class TestEncodeJob:
    def test_a_shard_job_names_its_rows_offset_pressure_and_workers(self) -> None:
        job = encode_job(
            PLAN, "mid", {"name": "S1"}, "/root/run/S1/mid", shard=1, shards=2, workers=9
        )
        assert job["rows"] == [None, [0, 7], None, [7, 15]]
        assert job["row_offset"] == 5
        assert job["pressure"].endswith("S1__mid.shard1.h5")
        assert job["workers"] == 9 and job["order"] == 7 and job["fit_order"] == 10
        assert job["output"] == "/root/run/S1/mid/encoded.h5"


def _write_encoded(path: Path, points: list[int], samples: int = 16) -> None:
    with h5py.File(path, "w") as handle:
        signals = np.asarray(
            [np.full((4, samples), float(p), dtype=np.float32) for p in points], dtype=np.float32
        ).reshape(len(points), 4, samples)
        handle.create_dataset("signals", data=signals)
        handle.create_dataset("point_index", data=np.asarray(points))
        handle.create_dataset(
            "centres", data=np.asarray([[p, 0.0, 0.0] for p in points], dtype=float).reshape(-1, 3)
        )
        handle.attrs["sample_rate_hz"] = 48000.0
        handle.attrs["order"] = 1
        handle.attrs["band"] = "mid"
        handle.attrs["source"] = "S1"


class TestMergeEncoded:
    def test_shards_merge_in_plan_order_whatever_their_arrival(self, tmp_path: Path) -> None:
        _write_encoded(tmp_path / "s0.h5", [4, 7])
        _write_encoded(tmp_path / "s1.h5", [0, 2])
        _write_encoded(tmp_path / "empty.h5", [])
        merge_encoded(
            [tmp_path / "s0.h5", tmp_path / "empty.h5", tmp_path / "s1.h5"], tmp_path / "m.h5"
        )
        with h5py.File(tmp_path / "m.h5") as handle:
            assert list(handle["point_index"][...]) == [0, 2, 4, 7]
            assert list(handle["signals"][:, 0, 0]) == [0.0, 2.0, 4.0, 7.0]
            assert list(handle["centres"][:, 0]) == [0.0, 2.0, 4.0, 7.0]
            assert handle.attrs["order"] == 1 and handle.attrs["band"] == "mid"


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
