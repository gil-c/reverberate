"""The campaign of low band pairs against a fake engine: what is solved, cached and skipped.

The grid is a small empty box written as a cache entry's files, the engine is
a function that writes the pressure a real one would leave (the same pulse at
every node, whose level names the source position), and everything between,
the arrays, the comms, the filters, the fit and the cache, is the real code
on ``numpy``. No card and no PFFDTD.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.accel import solve as solve_module
from reverberate.accel.cli import main
from reverberate.accel.pairs import (
    KIND,
    PairCache,
    PairsCampaign,
    cost_record,
    encoder_record,
    estimate,
    install_pairs,
    prepare_pairs_bundle,
    source_positions,
)
from reverberate.spatial.lowband import pair_key

C = 343.2
H = 0.0327
NODES = 60
KEY = "lowgrid"
FMAX = 1768.0
DURATION = 0.05


def write_entry(data: Path) -> Path:
    """An empty box of air as the four files of a cache entry, under ``data``'s cache."""
    entry = data / "cache" / "vox" / KEY
    entry.mkdir(parents=True)
    axis = np.arange(NODES) * H
    with h5py.File(entry / "sim_consts.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(H))
        handle.create_dataset("Ts", data=np.float64(H / C / np.sqrt(3.0)))
        handle.create_dataset("l2", data=np.float64(1 / 3))
        handle.create_dataset("fcc_flag", data=np.int8(0))
    with h5py.File(entry / "cart_grid.h5", "w") as handle:
        for name in ("xv", "yv", "zv"):
            handle.create_dataset(name, data=axis)
    with h5py.File(entry / "vox_out.h5", "w") as handle:
        handle.create_dataset("bn_ixyz", data=np.array([0], dtype=np.int64))
    (entry / "sim_mats.h5").write_bytes(b"materials")
    (entry / "manifest.json").write_text(json.dumps({"fmax": FMAX}))
    return entry


SOURCES = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.6], [0.5, 0.6, 0.5]])
CELLS = np.array([[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]])
HEARD_AT = [[0, 1], [1], [0]]


def write_bundle(bundle: Path, heard_at: list[list[int]] | None = None) -> None:
    bundle.mkdir(parents=True)
    heard = HEARD_AT if heard_at is None else heard_at
    spec = {
        "kind": KIND,
        "scene_id": "104862621_172226772",
        "dwelling": "hssd_0076",
        "models": "models/storey",
        "model_json": "models/storey/apartment_full.json",
        "storey_scene": "apartment_full",
        "materials": "models/materials",
        "bands": {"low": {"fmax_hz": FMAX, "duration_s": DURATION, "cache_key": KEY, "nh": 4}},
        "ppw": 10.5,
        "tc": 20.0,
        "rh": 50.0,
        "ram_gb": 64.0,
        "order": 2,
        "fit_order": 3,
        "points": 2,
        "pairs": sum(len(c) for c in heard),
        "solver": "fake/1",
        "encoder": encoder_record(2, 3, FMAX),
    }
    (bundle / "campaign.json").write_text(json.dumps(spec))
    np.save(bundle / "sources.npy", SOURCES)
    np.save(bundle / "cells.npy", CELLS)
    (bundle / "heard_at.json").write_text(json.dumps(heard))


class Driven(PairsCampaign):
    """The real campaign with the grid already voxelised."""

    def __post_init__(self) -> None:
        self.gpu = False
        super().__post_init__()

    def voxelise(self) -> dict[str, Any]:
        return {"low": {"cached": True}}


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A data root holding the grid, a bundle, and an engine that counts its solves."""
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    write_entry(tmp_path / "data")
    write_bundle(tmp_path / "bundle")
    solved: list[float] = []

    def fake_engine(job_dir: Path, **kwargs: object) -> solve_module.EngineResult:
        with h5py.File(job_dir / "comms_out.h5", "r") as handle:
            receivers, steps = int(handle["Nr"][()]), int(handle["Nt"][()])
            level = float(np.abs(np.asarray(handle["in_sigs"])).sum())
        pulse = np.zeros(steps, dtype=np.float32)
        pulse[100:140] = np.hanning(40)
        with h5py.File(job_dir / "sim_outs.h5", "w") as handle:
            handle.create_dataset("u_out", data=np.tile(pulse * (1 + len(solved)), (receivers, 1)))
        (job_dir / "engine.log").write_text("Running [100.0%]\n")
        solved.append(level)
        return solve_module.EngineResult(job_dir / "sim_outs.h5", 0.5, job_dir / "engine.log", 1)

    monkeypatch.setattr(solve_module, "run_engine", fake_engine)
    return {"bundle": tmp_path / "bundle", "out": tmp_path / "out", "solved": solved}


def a_campaign(machine: dict[str, Any], **options: Any) -> Driven:
    return Driven(
        bundle=machine["bundle"], out=machine["out"], pffdtd_dir=machine["out"], **options
    )


class TestTheCampaign:
    def test_each_source_position_is_solved_once_and_each_pair_cached(
        self, machine: dict[str, Any]
    ) -> None:
        campaign = a_campaign(machine)
        report = campaign.run()
        assert len(machine["solved"]) == 3, "one solve a source position, whatever its cells"
        assert report["pairs_solved"] == 4 and report["pairs_in_cache"] == 4
        assert (machine["out"] / "campaign.done").is_file()
        assert not (machine["out"] / "jobs").exists() or not any(
            (machine["out"] / "jobs").iterdir()
        ), "the pressure is deleted once it is responses"
        for source, cells in enumerate(HEARD_AT):
            for cell in cells:
                response = campaign.cache.read(campaign.key_of(source, cell))
                assert response.shape == (9, 200) and response.dtype == np.float32
        assert [r["cells"] for r in report["solves"]] == [2, 1, 1]

    def test_a_pair_is_a_response_at_4_khz_on_the_field_s_scale(
        self, machine: dict[str, Any]
    ) -> None:
        """One pressure at every node is, nearly, an omnidirectional field: channel 0 above all."""
        campaign = a_campaign(machine)
        campaign.run()
        first = campaign.cache.read(campaign.key_of(0, 0))
        again = campaign.cache.read(campaign.key_of(0, 1))
        energy = (first**2).sum(axis=1)
        assert energy[0] > 0 and energy[1:].max() < 0.05 * energy[0]
        assert np.abs(first - again).max() < 1e-6 * np.abs(first).max(), "the same solve"
        # The pulse sits 100 steps of the grid's clock in: 5.5 ms, sample 22 at 4 kHz.
        assert abs(int(np.argmax(np.abs(first[0]))) - 24) <= 6
        record = campaign.cache.records()[campaign.key_of(0, 0)]
        assert record["scale"] == pytest.approx(FMAX / 8000.0)
        assert record["cell_m"] == [1.0, 1.0, 1.0]
        # The array stands on a node of the grid, not on the cell asked for.
        assert np.abs(np.array(record["centre_m"]) - 1.0).max() <= H / 2 + 1e-9
        assert record["centre_m"] != record["cell_m"]

    def test_a_second_run_solves_nothing_and_a_new_pair_solves_its_source_alone(
        self, machine: dict[str, Any]
    ) -> None:
        a_campaign(machine).run()
        report = a_campaign(machine).run()
        assert len(machine["solved"]) == 3 and report["pairs_solved"] == 0
        (machine["bundle"] / "heard_at.json").write_text(json.dumps([[0, 1], [0, 1], [0]]))
        campaign = a_campaign(machine)
        report = campaign.run()
        assert len(machine["solved"]) == 4, "source position 1 alone, for its new cell"
        assert [r["source"] for r in report["solves"]] == [1]
        assert report["solves"][0]["cells"] == 1 and report["pairs_in_cache"] == 5

    def test_several_source_positions_at_once_give_the_same_cache(
        self, machine: dict[str, Any]
    ) -> None:
        campaign = a_campaign(machine, solvers=3)
        report = campaign.run()
        assert report["pairs_solved"] == 4 and len(machine["solved"]) == 3
        assert sorted(campaign.cache.records()) == sorted(
            campaign.key_of(s, c) for s, cells in enumerate(HEARD_AT) for c in cells
        )

    def test_a_cell_with_no_free_ball_is_refused_and_its_pairs_left_out(
        self, machine: dict[str, Any]
    ) -> None:
        cells = CELLS.copy()
        cells[1] = [0.1, 1.0, 1.0]  # its ball leaves the grid, whichever way it is shifted
        np.save(machine["bundle"] / "cells.npy", cells)
        campaign = a_campaign(machine)
        report = campaign.run()
        assert report["placed"]["refused"] == [1]
        assert report["pairs_solved"] == 2 and len(machine["solved"]) == 2

    def test_a_failing_solve_marks_the_run_failed(
        self, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(job_dir: Path, **kwargs: object) -> solve_module.EngineResult:
            raise RuntimeError("the engine exited with 1")

        monkeypatch.setattr(solve_module, "run_engine", broken)
        with pytest.raises(RuntimeError):
            a_campaign(machine).run()
        assert "engine exited" in (machine["out"] / "campaign.failed").read_text()

    def test_the_command_line_runs_a_bundle_of_pairs_as_a_campaign(
        self, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pairs_module = importlib.import_module("reverberate.accel.pairs")
        monkeypatch.setattr(pairs_module.PairsCampaign, "voxelise", Driven.voxelise)
        code = main(
            ["campaign", "--bundle", str(machine["bundle"]), "--out", str(machine["out"]), "--cpu"]
        )
        assert code == 0 and len(machine["solved"]) == 3
        assert json.loads((machine["out"] / "report.json").read_text())["kind"] == KIND

    def test_a_field_s_bundle_is_not_a_bundle_of_pairs(self, machine: dict[str, Any]) -> None:
        spec = json.loads((machine["bundle"] / "campaign.json").read_text())
        (machine["bundle"] / "campaign.json").write_text(json.dumps({**spec, "kind": "field"}))
        with pytest.raises(ValueError, match="low band pairs"):
            a_campaign(machine)


class TestTheCache:
    def test_pairs_brought_home_are_installed_once_and_published(
        self, machine: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reverberate.store import MemoryStore

        campaign = a_campaign(machine)
        campaign.run()
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "laptop"))
        found = install_pairs(machine["out"] / "pairs")
        assert len(found["installed"]) == 4
        cache = PairCache.local(KEY)
        key = campaign.key_of(0, 0)
        assert np.array_equal(cache.read(key), campaign.cache.read(key))
        assert install_pairs(machine["out"] / "pairs")["installed"] == []
        store = MemoryStore()
        assert sorted(cache.publish(store)) == sorted(cache.records())
        assert cache.publish(store) == [], "what the store holds is not sent again"
        elsewhere = PairCache(tmp_path / "elsewhere", KEY)
        assert elsewhere.fetch(store, key) and not elsewhere.fetch(store, "0" * 64)
        assert np.array_equal(elsewhere.read(key), cache.read(key))

    def test_a_missing_pair_is_a_key_error(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            PairCache(tmp_path, KEY).read("ab" * 32)


RECIPE: dict[str, Any] = {
    "dwelling": {"floor_y_m": 0.0},
    "heights": {"standing_m": 1.7, "seated_m": 1.2},
    "stations": [
        {"id": "armchair", "kind": "seat", "position": [0.0, 1.2, 0.0]},
        {"id": "counter", "kind": "stand", "position": [1.0, 1.7, 0.0]},
        {"id": "tv", "kind": "stand", "position": [3.0, 1.7, 3.0]},
    ],
    "rails": [
        {"id": "r", "a": "armchair", "b": "counter", "points": [[0, 0], [1, 0]], "pitch_m": 0.08}
    ],
    "sources": [
        {
            "id": "v1",
            "segments": [
                {"type": "dwell", "station": "armchair", "height": "seated"},
                {"type": "rise", "station": "armchair", "to": "standing"},
                {"type": "travel", "rail": "r", "from": "armchair", "to": "counter"},
                {"type": "dwell", "station": "counter", "height": "standing"},
            ],
        },
        {"id": "n1", "segments": [{"type": "dwell", "station": "tv", "height": "standing"}]},
    ],
}


class TestTheBundle:
    def test_a_recipe_s_positions_are_its_stations_and_its_rails_every_pitch(self) -> None:
        found = source_positions(RECIPE)
        positions = found["positions"]
        # The seat, six more up its vertical rail (0.08 to 0.48, then the top),
        # thirteen along the rail past its start (0.08 to 0.96, then the end), the television.
        assert positions.shape == (1 + 7 + 13 + 1, 3)
        assert positions[0].tolist() == [0.0, 1.2, 0.0]
        assert [1.0, 1.7, 0.0] in positions.tolist() and [0.96, 1.7, 0.0] in positions.tolist()
        assert len(found["by_source"]["v1"]) == 21 and len(found["by_source"]["n1"]) == 1
        assert len({tuple(p) for p in positions.tolist()}) == positions.shape[0]

    def test_the_bundle_is_a_campaign_the_rental_can_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import reverberate.experiments.run as run_module

        class Spec:
            key, ppw, nh = "key1500", 10.5, 4

        seen: list[float] = []

        def fake_spec(models: Path, scene: str, fmax: float) -> tuple[Spec, None, int]:
            seen.append(fmax)
            return Spec(), None, 0

        monkeypatch.setattr(run_module, "scene_spec", fake_spec)
        earlier = tmp_path / "earlier" / "storey"
        earlier.mkdir(parents=True)
        (earlier / "manifest.json").write_text("{}")
        (earlier / "apartment_full.json").write_text("{}")
        (earlier.parent / "materials").mkdir()
        spec = prepare_pairs_bundle(
            tmp_path / "bundle",
            scene_id="104862621_172226772",
            sources=SOURCES,
            cells=CELLS,
            heard_at=HEARD_AT,
            models_from=earlier,
        )
        assert seen == [pytest.approx(1500.0)], "the grid a 1 kHz crossover needs"
        assert spec["kind"] == KIND and list(spec["bands"]) == ["low"]
        assert spec["bands"]["low"]["cache_key"] == "key1500"
        assert (spec["source_positions"], spec["cells"], spec["pairs"], spec["points"]) == (
            3,
            2,
            4,
            2,
        )
        assert spec["estimate"]["usd"] > 0
        assert json.loads((tmp_path / "bundle" / "heard_at.json").read_text()) == HEARD_AT
        assert (tmp_path / "bundle" / "models" / "storey" / "apartment_full.json").is_file()
        with pytest.raises(ValueError, match="does not hold"):
            prepare_pairs_bundle(
                tmp_path / "other",
                scene_id="104862621_172226772",
                sources=SOURCES,
                cells=CELLS,
                heard_at=[[0], [2], [0]],
                models_from=earlier,
            )

    def test_a_pair_s_key_is_the_format_s_and_the_same_on_every_machine(
        self, machine: dict[str, Any]
    ) -> None:
        campaign = a_campaign(machine)
        assert campaign.key_of(1, 1) == pair_key(
            KEY,
            SOURCES[1],
            CELLS[1],
            encoder=encoder_record(2, 3, FMAX),
            solver="fake/1",
            window_s=DURATION,
        )


class TestTheCost:
    def test_the_estimate_is_a_price_per_source_position_and_one_per_pair(self) -> None:
        one = estimate(1, 0, fmax_hz=1000.0)
        # The low band of the line campaign: 4.26e11 node updates.
        assert one["stencil_s_per_source"] == pytest.approx(3.15, abs=0.01)
        many = estimate(400, 7000, fmax_hz=1500.0)
        assert many["stencil_s_per_source"] == pytest.approx(3.15 * 1.5**4, rel=0.01)
        expected = 104.4 * 1.5 + 400 * many["stencil_s_per_source"] + 7000 * many["cell_s"]
        assert many["seconds"] == pytest.approx(expected, rel=1e-3)
        assert many["usd"] == pytest.approx(many["seconds"] / 3600 * 1.74, abs=0.01)
        assert many["cache_gb"] == pytest.approx(7000 * 1.2288e-3, rel=1e-3)

    def test_a_cost_record_carries_its_rate(self) -> None:
        report = {"total_s": 7200.0, "device": {"gpu": "NVIDIA A100 80GB PCIe", "devices": 2}}
        record = cost_record(report, billed_rate_usd_per_hour=1.74, instance=54075335)
        assert record == {
            "stage": "low",
            "seconds": 7200.0,
            "card": "NVIDIA A100 80GB PCIe",
            "cards": 2,
            "billed_rate_usd_per_hour": 1.74,
            "usd": 3.48,
            "instance": 54075335,
        }
