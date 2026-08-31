"""That the scene says which side of each surface the air is on.

The values under test are PFFDTD's, read from ``room_geo.py`` and
``vox_scene.py`` rather than assumed: ``2`` means *front side only*, not "two
sided". B0 and B1 wrote ``2`` everywhere meaning the latter, which made every
surface whose normal pointed the wrong way rigid in the solver while keeping
its area in every report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from reverberate.geometry.apartment import Storey, build_storey
from reverberate.geometry.hssd_room import RoomRegion
from reverberate.geometry.orientation import BOTH, FRONT, orient_for_air
from reverberate.geometry.sim_geometry import shell_assignments, simulation_geometry


def square_storey(size: float = 4.0) -> Storey:
    loop = np.array([[0, 0, 0], [size, 0, 0], [size, 0, size], [0, 0, size]], dtype=float)
    region = RoomRegion(
        name="room", label="bedroom", poly_loop=loop, floor_height=0.0, extrusion_height=2.8
    )
    tiny = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    return build_storey([region], tiny)


def normals_point_away_from(mesh: trimesh.Trimesh, point: np.ndarray) -> bool:
    outward = mesh.triangles_center - point
    return bool(np.all(np.einsum("ij,ij->i", outward, mesh.face_normals) > 0))


def test_a_solid_in_air_has_its_normals_pointing_out() -> None:
    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    oriented = orient_for_air(box, "outside")
    assert oriented.authoritative
    assert np.all(oriented.sides == FRONT)
    assert normals_point_away_from(oriented.mesh, np.zeros(3))


def test_a_room_holding_air_has_its_normals_pointing_in() -> None:
    """The air is inside a shell, so the material must face inwards.

    Written as ``FRONT`` on an inverted winding rather than as ``BACK`` on the
    original one, so that a scene file only ever carries a single convention.
    """
    box = trimesh.creation.box(extents=(4.0, 3.0, 2.5))
    oriented = orient_for_air(box, "inside")
    assert oriented.authoritative
    assert np.all(oriented.sides == FRONT)
    assert not normals_point_away_from(oriented.mesh, np.zeros(3))


def test_an_inverted_solid_is_turned_back_around() -> None:
    """An asset shipped inside out must not be taken at its word."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.invert()
    oriented = orient_for_air(box, "outside")
    assert oriented.authoritative
    assert normals_point_away_from(oriented.mesh, np.zeros(3))


def test_an_open_sheet_is_declared_unknown_rather_than_guessed() -> None:
    """A curtain has no inside, so nothing entitles the export to pick a side.

    ``BOTH`` is both the honest answer and the safe one: a surface active on
    both sides can be wasteful, but it can never be silently rigid.
    """
    sheet = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
    )
    oriented = orient_for_air(sheet, "outside")
    assert not oriented.authoritative
    assert np.all(oriented.sides == BOTH)


def test_the_source_mesh_is_never_mutated() -> None:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    before = box.face_normals.copy()
    orient_for_air(box, "inside")
    assert np.array_equal(box.face_normals, before)


def test_every_face_of_a_scene_carries_a_sidedness() -> None:
    assignments = shell_assignments(square_storey())
    for assignment in assignments:
        assert assignment.sides is not None
        assert len(assignment.sides) == len(assignment.mesh.faces)


def test_the_shell_parts_face_the_room_they_enclose() -> None:
    """Per part, because the split is where an orientation is easiest to lose.

    Orientation is derived on the whole enclosure and then sliced, since a
    submesh of a box is an open sheet that could not be oriented on its own.
    """
    storey = square_storey(size=4.0)
    inside = np.array([2.0, 1.4, 2.0])
    for assignment in shell_assignments(storey):
        assert assignment.sides is not None
        assert np.all(assignment.sides == FRONT)
        towards = inside - assignment.mesh.triangles_center
        dots = np.einsum("ij,ij->i", towards, assignment.mesh.face_normals)
        assert np.all(dots > 0), assignment.name


def test_the_summary_counts_what_could_not_be_oriented() -> None:
    storey = square_storey()
    _, summary = simulation_geometry(hssd_root=Path("/nonexistent"), storey=storey, instances=[])
    assert summary.unoriented_faces == 0
    assert "faces oriented" in summary.summary()
