"""The app's audit views against what each solver read.

The mirror's layers are linked into the site only when they are the derived
scene the engine read, triangle for triangle and label for label; one moved
corner or one relabelled triangle refuses them and says so. The materials the
page shows for a mirror field are the ones the engine's own functions give
from that field's parameters. The wave view's absorption is the export's
table beside the model the voxel cache names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.mirror.audit import write_geometry_layers
from reverberate.mirror.calibrate import Parameters
from reverberate.mirror.geometry import derive, write_derived
from reverberate.viz.app_payload import WalkRun, build_run, write_synthetic_run
from reverberate.viz.audit_payload import PFFDTD_BANDS_HZ, mirror_audit, wave_materials
from test_accel_scene import write_scene


def _mirror_run(tmp_path: Path) -> tuple[WalkRun, Any]:
    """A synthetic run carrying a derived box scene, its layers and a B field's report."""
    run = write_synthetic_run(tmp_path / "run", field_step_m=1.1, field_box_m=2.2)
    derived = derive(write_scene(tmp_path / "box.json", extra_material=True))
    write_derived(derived, run.path / "mirror" / "scene")
    write_geometry_layers(derived, run.path / "mirror" / "audit")
    parameters = Parameters(absorption_scale=tuple(2.0 for _ in range(7)), scattering_scale=0.5)
    report = {"parameters": parameters.record(), "scene": {"key": derived.key}}
    (run.path / "mirror" / "report_S1.json").write_text(json.dumps(report))
    (run.path / "mirror" / "card_S1.json").write_text(json.dumps(report))
    manifest = json.loads((run.path / "walk.json").read_text())
    manifest["sources"][0]["field_mirror"] = "fields/S1.h5"
    manifest["mirror"] = {
        "scene": "mirror/scene.json",
        "key": derived.key,
        "audit": "mirror/audit",
        "paths": {},
    }
    (run.path / "walk.json").write_text(json.dumps(manifest))
    return WalkRun.read(run.path), derived


def test_layers_that_are_the_scene_are_linked_with_the_materials_each_field_read(
    tmp_path: Path,
) -> None:
    run, derived = _mirror_run(tmp_path)
    site = tmp_path / "site"
    record = build_run(run, site)["mirror"]
    check = record["check"]

    assert check["problems"] == [] and check["material_problems"] == []
    assert record["audit"] == "mirror/audit"
    assert (site / "mirror" / "audit" / "layers.json").is_file()
    assert check["labels"] == ["carpet", "wall"]
    assert sum(f["triangles"] for f in check["facets"]) == derived.reflector_vertices.shape[0]
    np.testing.assert_allclose(
        check["materials"]["absorption"], derived.materials.absorption, atol=1e-6
    )
    variant = check["variants"]["B:S1"]
    assert variant["variant"] == "B"
    expected = np.clip(derived.materials.absorption * 2.0, 0.0, 0.999)
    np.testing.assert_allclose(variant["rays"]["absorption"], expected, atol=1e-6)
    np.testing.assert_allclose(
        variant["rays"]["scattering"], derived.materials.scattering * 0.5, atol=1e-6
    )
    # Without an image scale the images read the rays' absorption.
    assert variant["images"]["absorption"] == variant["rays"]["absorption"]


def test_one_moved_corner_refuses_the_layers_and_names_them(tmp_path: Path) -> None:
    run, _ = _mirror_run(tmp_path)
    corners = run.path / "mirror" / "audit" / "occluders.f32"
    data = np.fromfile(corners, dtype=np.float32)
    data[0] += 0.01
    data.tofile(corners)

    record = build_run(run, tmp_path / "site")["mirror"]

    assert "audit" not in record
    assert any("occluders: 1 triangles differ" in p for p in record["check"]["problems"])
    assert not (tmp_path / "site" / "mirror" / "audit").exists()


def test_one_relabelled_triangle_refuses_the_layers(tmp_path: Path) -> None:
    run, _ = _mirror_run(tmp_path)
    labels = run.path / "mirror" / "audit" / "reflectors_label.i16"
    data = np.fromfile(labels, dtype=np.int16)
    data[:4] = 0 if data[0] else 1
    data.tofile(labels)

    check = mirror_audit(run.path, run.mirror, run.sources)

    assert any("reflectors: 1 triangles carry another label" in p for p in check["problems"])


def test_layers_of_another_scene_are_refused(tmp_path: Path) -> None:
    run, _ = _mirror_run(tmp_path)
    layers = run.path / "mirror" / "audit" / "layers.json"
    record = json.loads(layers.read_text())
    record["key"] = "0" * 32
    layers.write_text(json.dumps(record))

    check = mirror_audit(run.path, run.mirror, run.sources)

    assert any("layers.json is of scene" in p for p in check["problems"])


def test_parameters_that_differ_between_card_and_report_are_said(tmp_path: Path) -> None:
    run, _ = _mirror_run(tmp_path)
    card = run.path / "mirror" / "card_S1.json"
    record = json.loads(card.read_text())
    record["parameters"]["scattering_scale"] = 3.0
    card.write_text(json.dumps(record))

    check = mirror_audit(run.path, run.mirror, run.sources)

    assert check["problems"] == []
    assert any("name other parameters" in p for p in check["material_problems"])


def test_the_wave_absorption_is_the_export_table_beside_the_voxelised_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    model = tmp_path / "models" / "apartment_full.json"
    model.parent.mkdir(parents=True)
    model.write_text("{}")
    table = {"shell": [0.1] * 11, "table": [0.2] * 11}
    (model.parent / "manifest.json").write_text(json.dumps({"materials": table}))
    cache = tmp_path / "data" / "cache" / "vox" / "k1"
    cache.mkdir(parents=True)
    manifest = {
        "materials": {"shell": "shell.h5", "table": "table.h5"},
        "model_json": str(model),
        "h_m": 0.004,
        "fmax": 8000.0,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))

    found = wave_materials("k1", ["shell", "table"])
    assert found["problems"] == []
    assert found["bands_hz"] == PFFDTD_BANDS_HZ and len(PFFDTD_BANDS_HZ) == 11
    assert found["absorption"] == [[0.1] * 11, [0.2] * 11]

    manifest["model_sha256"] = "0" * 64
    (cache / "manifest.json").write_text(json.dumps(manifest))
    assert any("changed since" in p for p in wave_materials("k1", ["shell", "table"])["problems"])
    assert wave_materials("k2", ["shell"])["problems"] == ["voxel cache k2 is not on this machine"]
