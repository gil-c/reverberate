"""The mirror solver's audit is written from the scene and the paths it read and wrote."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reverberate.mirror.files import write_paths
from reverberate.mirror.geometry import write_derived
from reverberate.mirror.ism import grow_tree, paths_for
from reverberate.viz.mirror_audit import mirror_record
from test_mirror_ism import RECEIVER, SOURCE, box_scene


def test_the_audit_draws_the_scene_s_triangles_and_each_point_s_paths(tmp_path: Path) -> None:
    scene = box_scene(alpha=0.3)
    write_derived(scene, tmp_path / "mirror" / "scene")
    paths = [paths_for(scene, grow_tree(scene, SOURCE), RECEIVER)]
    write_paths(paths, tmp_path / "mirror" / "paths_S1.npz")
    record = mirror_record(
        tmp_path / "mirror" / "scene",
        {"S1": tmp_path / "mirror" / "paths_S1.npz"},
        tmp_path / "site",
    )
    assert record is not None and record["key"] == scene.key
    layers = json.loads((tmp_path / "site" / record["audit"] / "layers.json").read_text())
    reflectors = layers["layers"]["reflectors"]
    assert reflectors["triangles"] == scene.reflector_vertices.shape[0]
    corners = np.fromfile(tmp_path / "site" / record["audit"] / "reflectors.f32", dtype="<f4")
    assert corners.size == reflectors["quads"] * 4 * 3
    assert sum(f["triangles"] for f in layers["facets"]) == reflectors["triangles"]
    drawn = json.loads((tmp_path / "site" / record["paths"]["S1"]).read_text())["points"][0]
    assert len(drawn) == paths[0].count and drawn[0][0] == 0
    assert mirror_record(tmp_path / "none", {}, tmp_path / "site") is None
