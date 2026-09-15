"""What the laptop prepares for the machine, against fakes of the export and of HSSD."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.accel.bundle import prepare_bundle


class Room:
    def __init__(self, name: str) -> None:
        self.name = name
        self.label = name
        self.regions = (name,)
        self.outdoor = False
        from shapely.geometry import box

        self.polygon = box(0, 0, 4, 3)


class Spec:
    def __init__(self, fmax: float) -> None:
        self.key = f"key{int(fmax)}"
        self.ppw = 10.5
        self.nh = 8


def fake_hssd(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import reverberate.experiments.run as run_module
    import reverberate.experiments.w40_volume_field.plan as plan_module

    campaign_module = importlib.import_module("reverberate.experiments.w40_volume_field.storey")
    exported: list[str] = []

    def fake_export(hssd_root: Path, scene_id: str, models: Path) -> Path:
        exported.append(scene_id)
        models.mkdir(parents=True, exist_ok=True)
        (models / "manifest.json").write_text(
            json.dumps({"dwelling": "hssd_test", "room": "living room"})
        )
        (models / "apartment_full.json").write_text("{}")
        (models / "room_only.json").write_text("{}")
        (models.parent / "materials").mkdir(exist_ok=True)
        (models.parent / "materials" / "wall.h5").write_bytes(b"m")
        return models

    monkeypatch.setattr(campaign_module, "export_scene", fake_export)
    monkeypatch.setattr(run_module, "scene_spec", lambda models, scene, fmax: (Spec(fmax), None, 0))
    monkeypatch.setattr(
        plan_module,
        "plan_points",
        lambda root, scene_id, *, height_m, pitch_m: (np.zeros((3, 3)), ["living room"] * 3, 12.0),
    )
    monkeypatch.setattr(
        plan_module, "free_floor", lambda root, scene_id: (None, None, [Room("living room")])
    )
    return exported


class TestPrepareBundle:
    def test_the_bundle_holds_the_storey_the_grid_the_rooms_and_the_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exported = fake_hssd(monkeypatch)
        sources: list[dict[str, Any]] = [
            {"name": "S1", "room": "living room", "position": [0, 1.7, 0]}
        ]
        spec = prepare_bundle(
            tmp_path / "bundle", hssd_root=tmp_path, scene_id="102344022", sources=sources
        )
        assert exported == ["102344022"]
        assert (
            spec["dwelling"] == "hssd_0002"
            and spec["points"] == 3
            and spec["free_floor_m2"] == 12.0
        )
        assert spec["bands"]["low"]["cache_key"] == "key1000" and spec["bands"]["high"]["nh"] == 8
        assert spec["model_json"] == "models/storey/apartment_full.json"
        bundle = tmp_path / "bundle"
        assert not (bundle / "models" / "storey" / "room_only.json").exists(), (
            "only the storey is uploaded"
        )
        assert np.load(bundle / "points.npy").shape == (3, 3)
        rooms = json.loads((bundle / "rooms.json").read_text())
        assert rooms[0]["name"] == "living room" and rooms[0]["polygon"].startswith("POLYGON")
        assert json.loads((bundle / "campaign.json").read_text())["sources"] == sources

    def test_an_earlier_export_is_reused_rather_than_exported_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exported = fake_hssd(monkeypatch)
        earlier = tmp_path / "earlier" / "living_room"
        earlier.mkdir(parents=True)
        (earlier / "manifest.json").write_text(
            json.dumps({"dwelling": "hssd_0002", "room": "living room"})
        )
        (earlier / "apartment_full.json").write_text('{"earlier": true}')
        (earlier.parent / "materials").mkdir()
        (earlier.parent / "materials" / "wall.h5").write_bytes(b"m")
        prepare_bundle(
            tmp_path / "bundle",
            hssd_root=tmp_path,
            scene_id="102344022",
            sources=[],
            models_from=earlier,
        )
        assert exported == []
        assert (
            tmp_path / "bundle" / "models" / "storey" / "apartment_full.json"
        ).read_text() == '{"earlier": true}'
        assert (tmp_path / "bundle" / "models" / "materials" / "wall.h5").is_file()

    def test_an_export_with_another_ppw_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_hssd(monkeypatch)
        with pytest.raises(ValueError, match="points per wavelength"):
            prepare_bundle(
                tmp_path / "bundle", hssd_root=tmp_path, scene_id="102344022", sources=[], ppw=8.0
            )
