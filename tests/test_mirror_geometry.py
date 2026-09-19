"""The derived geometry against a box whose facets are known.

A closed unit box of twelve triangles, air inside, must become six facets of
one square metre each, every one a reflector, with normals pointing into the
air; a carpet patch on its floor under the area threshold must reach the
occluders and not the reflectors; and the census must account for every
square metre of the model. Written and read back, the scene must carry the
same key.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reverberate.mirror.geometry import (
    GeometryRules,
    derive,
    load_derived,
    material_table,
    shell_surface_of,
    write_derived,
)
from test_accel_scene import write_scene


def test_a_closed_box_becomes_six_reflecting_facets_facing_the_air(tmp_path: Path) -> None:
    derived = derive(write_scene(tmp_path / "box.json"))
    assert len(derived.facets) == 6
    assert all(f.area == pytest.approx(1.0) for f in derived.facets)
    assert derived.reflector_vertices.shape == (12, 3, 3)
    for facet in derived.facets:
        # The centroid of the box, (0.5, 0.5, 0.5), lies on the normal's side.
        assert float(facet.normal @ np.full(3, 0.5)) > facet.offset
    assert derived.census["totals"]["model_area_m2"] == pytest.approx(6.0)
    assert derived.census["totals"]["reflector_area_m2"] == pytest.approx(6.0)
    assert derived.census["totals"]["diffuse_area_m2"] == pytest.approx(0.0)


def test_a_small_patch_reaches_the_occluders_and_not_the_reflectors(tmp_path: Path) -> None:
    derived = derive(write_scene(tmp_path / "box.json", extra_material=True))
    assert derived.labels == ("carpet", "wall")
    carpet = derived.census["labels"]["carpet"]
    assert carpet["facets"] == 0 and carpet["area_m2"] == pytest.approx(0.36)
    assert carpet["diffuse_area_m2"] == pytest.approx(0.36)
    assert int(np.count_nonzero(derived.occluder_label == derived.labels.index("carpet"))) == 2
    assert len(derived.facets) == 6


def test_the_rules_travel_and_a_larger_threshold_drops_the_box_walls(tmp_path: Path) -> None:
    rules = GeometryRules(reflector_area_m2=2.0)
    derived = derive(write_scene(tmp_path / "box.json"), rules=rules)
    assert len(derived.facets) == 0
    assert derived.census["rules"]["reflector_area_m2"] == 2.0
    assert derived.occluder_vertices.shape[0] == 12


def test_written_and_read_back_the_scene_keeps_its_key(tmp_path: Path) -> None:
    derived = derive(write_scene(tmp_path / "box.json", extra_material=True))
    path = write_derived(derived, tmp_path / "mirror" / "scene")
    record = json.loads(path.read_text())
    again = load_derived(tmp_path / "mirror" / "scene")
    assert again.key == derived.key == record["key"]
    assert again.summary() == derived.summary()
    assert [f.kind for f in again.facets] == [f.kind for f in derived.facets]
    np.testing.assert_array_equal(again.occluder_vertices, derived.occluder_vertices)


def test_the_derivation_is_deterministic_across_seeds_of_nothing(tmp_path: Path) -> None:
    a = derive(write_scene(tmp_path / "a.json"))
    b = derive(write_scene(tmp_path / "b.json"))
    assert a.key == b.key


def test_materials_come_from_the_solver_when_the_manifest_says_so() -> None:
    manifest = {"materials": {"wall": [0.5] * 11}}
    table = material_table(("wall", "carpet"), manifest)
    assert np.allclose(table.absorption[0], 0.5)
    assert table.source[0].startswith("solver manifest")
    assert table.scattering[1] == pytest.approx(0.30)
    assert table.absorption[1][3] == pytest.approx(0.69)  # carpet_thick at 1 kHz


def test_the_shell_is_told_apart_by_its_air_facing_normal() -> None:
    kinds = shell_surface_of(np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0]], dtype=float))
    assert list(kinds) == ["floor", "ceiling", "wall"]
