"""The batched encoder against the point-by-point one it replaces.

On numpy the accelerated pipeline calls the same filters as the encode child
and fits with the same matrices, so its float32 output is expected to be the
child's exactly; the test says so on a small array. The card's numbers are
compared on the card, where they can differ in the last bits of a transform
or a solve, and that difference is what the campaign reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate import audio
from reverberate.accel.encode import BandPipeline, encode_band, geometry_key
from reverberate.spatial.array import ArrayDesign, design_array
from reverberate.spatial.encode import EncoderSettings, encode
from reverberate.wave.comms import Grid

C = 343.2
RATE = 72818.9528707636
FMAX = 4000.0
SAMPLES = 1500


def a_grid(nodes: int = 120, h: float = 0.00817) -> Grid:
    axis = np.arange(nodes) * h
    return Grid(h=h, Ts=h / C / np.sqrt(3.0), l2=1 / 3, fcc_flag=0, xv=axis, yv=axis, zv=axis)


def small_array(grid: Grid, centre_index: int = 60) -> ArrayDesign:
    centre = np.array([grid.xv[centre_index]] * 3)
    return design_array(centre, grid, radii_m=(0.02, 0.05), counts=(16, 40), fit_order=4)


def a_plan(fit_order: int = 4, order: int = 3) -> dict[str, Any]:
    return {
        "sound_speed_m_s": C,
        "lowcut_hz": 40.0,
        "lowcut_order": 8,
        "encoder": {"order": order, "fit_order": fit_order},
        "bands": {
            "mid": {
                "sample_rate_hz": RATE,
                "samples": SAMPLES,
                "fmax_hz": FMAX,
                "grid_step_m": 0.00817,
            }
        },
    }


def child_path(u: np.ndarray, design: ArrayDesign, order: int, fit_order: int) -> np.ndarray:
    """What the CPU encode child did for one point, step by step."""
    s = audio.reduce_nodes(u, np.ones((design.count, 1)))
    s = audio.integrate_and_lowcut(s, 1.0 / RATE, differentiated=True, fcut=40.0, order=8)
    s = audio.lowpass(s, RATE, FMAX)
    s = audio.resample_to(s, RATE, 48000.0)
    s = audio.apply_air_absorption(s, 48000.0, sound_speed_m_s=C, atmosphere=audio.Atmosphere())
    offsets = design.positions - design.centre
    d = ArrayDesign(
        centre=design.centre,
        positions=design.positions,
        offsets=offsets,
        radii=np.linalg.norm(offsets, axis=1),
        shell=np.zeros(design.count, dtype=int),
        nominal_radii=(float(np.linalg.norm(offsets, axis=1).max()),),
        grid_step_m=design.grid_step_m,
        requested_centre=design.centre,
    )
    settings = EncoderSettings(order=order, fit_order=fit_order, max_frequency_hz=FMAX)
    return encode(s, 48000.0, d, sound_speed_m_s=C, settings=settings).signals.astype(np.float32)


def pressure(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((count, SAMPLES)) * np.exp(-np.arange(SAMPLES) / 400.0)[None, :]
    return u.astype(np.float32).astype(np.float64)


class TestOnePoint:
    def test_the_numpy_pipeline_is_the_child_s_float32_for_float32(self) -> None:
        grid = a_grid()
        design = small_array(grid)
        u = pressure(design.count)
        expected = child_path(u, design, order=3, fit_order=4)
        pipeline = BandPipeline.from_plan(a_plan(), "mid", "S1", differentiated=True)
        got = pipeline.point(u, np.ones((design.count, 1)), design.positions, design.centre, np)
        assert got.shape == expected.shape == (16, int(SAMPLES * 48000.0 / RATE))
        assert np.array_equal(got, expected)

    def test_two_points_of_one_geometry_share_a_preparation(self) -> None:
        grid = a_grid()
        first, second = small_array(grid, 50), small_array(grid, 70)
        assert geometry_key(first.positions - first.centre) == geometry_key(
            second.positions - second.centre
        )
        pipeline = BandPipeline.from_plan(a_plan(), "mid", "S1", differentiated=True)
        for design in (first, second):
            pipeline.point(
                pressure(design.count),
                np.ones((design.count, 1)),
                design.positions,
                design.centre,
                np,
            )
        assert len(pipeline.encoders) == 1


class TestABand:
    def test_encode_band_writes_the_child_s_file(self, tmp_path: Path) -> None:
        grid = a_grid()
        designs = [small_array(grid, 50), None, small_array(grid, 70)]
        rows: list[list[int] | None] = []
        cursor = 0
        positions = []
        for design in designs:
            if design is None:
                rows.append(None)
                continue
            rows.append([cursor, cursor + design.count])
            positions.append(design.positions)
            cursor += design.count
        stacked = np.vstack(positions)
        plan = a_plan()
        plan["bands"]["mid"]["rows"] = rows
        plan["bands"]["mid"]["centres"] = [
            None if d is None else [float(v) for v in d.centre] for d in designs
        ]
        u = pressure(cursor, seed=7)
        with h5py.File(tmp_path / "sim_outs.h5", "w") as handle:
            handle.create_dataset("u_out", data=u)
        with h5py.File(tmp_path / "comms_out.h5", "w") as handle:
            handle.create_dataset("out_alpha", data=np.ones((cursor, 1)))
            handle.create_dataset("diff", data=np.int8(1))
        report = encode_band(
            plan,
            "mid",
            {"name": "S1"},
            pressure=tmp_path / "sim_outs.h5",
            comms=tmp_path / "comms_out.h5",
            positions=stacked,
            output=tmp_path / "encoded.h5",
            xp=np,
        )
        assert report["points"] == 2 and report["geometries"] == 1
        with h5py.File(tmp_path / "encoded.h5") as handle:
            assert list(handle["point_index"][...]) == [0, 2]
            assert handle["signals"].shape == (2, 16, int(SAMPLES * 48000.0 / RATE))
            assert handle["signals"].dtype == np.float32
            assert handle.attrs["band"] == "mid" and handle.attrs["source"] == "S1"
            assert handle.attrs["sample_rate_hz"] == 48000.0
            second = np.asarray(handle["signals"][1])
        design = designs[2]
        assert design is not None
        start, stop = rows[2]  # type: ignore[misc]
        assert np.array_equal(second, child_path(u[start:stop], design, order=3, fit_order=4))


class TestRecordLength:
    def test_the_record_keeps_every_sample_the_engine_wrote(self) -> None:
        """One step more than the plan says, as the engine runs: the child keeps it, so do we.

        Three samples fewer in the low band of hssd_0076 moved every synthesised
        tail of the field, because the assembly reads the band's length as the
        response's duration.
        """
        grid = a_grid()
        design = small_array(grid)
        rng = np.random.default_rng(3)
        longer = SAMPLES + 1
        u = rng.standard_normal((design.count, longer)) * np.exp(-np.arange(longer) / 400.0)
        u = u.astype(np.float32).astype(np.float64)
        expected = child_path(u, design, order=3, fit_order=4)
        pipeline = BandPipeline.from_plan(a_plan(), "mid", "S1", differentiated=True)
        got = pipeline.point(u, np.ones((design.count, 1)), design.positions, design.centre, np)
        assert got.shape[1] == expected.shape[1] == int(longer * 48000.0 / RATE)
        assert np.array_equal(got, expected)


def _write_encoded(path: Path, points: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "signals", data=np.asarray(points, dtype=np.float32).reshape(-1, 1, 1)
        )
        handle.create_dataset("point_index", data=np.asarray(points, dtype=np.int64))
        handle.create_dataset(
            "centres", data=np.asarray([[p, 1.0, 0.0] for p in points]).reshape(-1, 3)
        )
        handle.attrs["sample_rate_hz"] = 48000.0
        handle.attrs["order"] = 1
        handle.attrs["band"] = "mid"
        handle.attrs["source"] = "S1"


class TestMergeEncoded:
    def test_parts_merge_in_plan_order_whatever_their_arrival(self, tmp_path: Path) -> None:
        from reverberate.accel.encode import merge_encoded

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
