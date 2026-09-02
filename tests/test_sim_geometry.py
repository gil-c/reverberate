"""Tests for the geometry the solver receives.

The property under test throughout is that the simulator and the acoustic view
read from the same place, so that a picture of one is a picture of the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point

from reverberate.geometry.apartment import Storey, build_storey
from reverberate.geometry.hssd_room import FurnitureInstance, RoomRegion
from reverberate.geometry.sim_geometry import (
    obstacle_assignments,
    obstacle_collider,
    sample_points,
    sample_source_receiver,
    shell_assignments,
    simulation_geometry,
)


def square_storey(size: float = 4.0) -> Storey:
    loop = np.array(
        [[0, 0, 0], [size, 0, 0], [size, 0, size], [0, 0, size]],
        dtype=float,
    )
    region = RoomRegion(
        name="room", label="bedroom", poly_loop=loop, floor_height=0.0, extrusion_height=2.8
    )
    tiny = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    return build_storey([region], tiny)


def test_shell_is_split_into_floor_wall_and_ceiling_materials() -> None:
    assignments = shell_assignments(square_storey())
    assert sorted(a.name for a in assignments) == [
        "shell_ceiling",
        "shell_floor",
        "shell_wall",
    ]


def test_shell_parts_together_cover_every_face_exactly_once() -> None:
    storey = square_storey()
    assignments = shell_assignments(storey)
    from reverberate.geometry.apartment import extrude_storey

    assert sum(len(a.mesh.faces) for a in assignments) == len(extrude_storey(storey).faces)


def test_floor_and_ceiling_get_different_absorption() -> None:
    """Carpet and plasterboard are the two ends of the range; averaging them
    away would flatten the signal the model is meant to learn."""
    by_name = {a.name: a for a in shell_assignments(square_storey())}
    floor = np.mean(by_name["shell_floor"].material.energy_absorption["coeffs"])
    ceiling = np.mean(by_name["shell_ceiling"].material.energy_absorption["coeffs"])
    assert floor != ceiling


def build_object_tree(root: Path) -> int:
    """A minimal objects tree holding one dense collider."""
    directory = root / "objects" / "a"
    directory.mkdir(parents=True)
    dense = trimesh.creation.icosphere(subdivisions=4)
    exported = dense.export(file_type="glb")
    assert isinstance(exported, bytes)
    (directory / "abc.glb").write_bytes(exported)
    (directory / "abc.collider.glb").write_bytes(exported)
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "hssd_obj_semantics_condensed.csv").write_text(
        "hash,art,pick,condensed,primary,,\nabc,No,No,sofa,sofa,,\n"
    )
    return len(dense.faces)


def instance(template: str, x: float = 0.0) -> FurnitureInstance:
    return FurnitureInstance(
        template_name=template,
        translation=np.array([x, 0.0, 0.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.ones(3),
    )


def test_the_collider_reaches_the_solver_with_every_triangle(tmp_path: Path) -> None:
    """Nothing is decimated any more, so the count must match exactly.

    ``<=`` would pass while a reduction quietly came back; this is the
    assertion that makes its return a failing test.
    """
    dense_faces = build_object_tree(tmp_path)
    assignments, unresolved = obstacle_assignments(tmp_path, [instance("abc")])
    assert unresolved == []
    assert len(assignments[0].mesh.faces) == dense_faces


def test_the_viewer_and_the_simulator_read_the_same_collider(tmp_path: Path) -> None:
    """The whole point: the acoustic view must not be a flattering picture.

    ``obstacle_collider`` is the single place that mesh is chosen, and both the
    exported mesh and the simulated one come from it.
    """
    build_object_tree(tmp_path)
    exported = obstacle_collider(tmp_path, "abc")
    simulated = obstacle_assignments(tmp_path, [instance("abc")])[0][0].mesh
    assert exported is not None
    assert len(exported.faces) == len(simulated.faces)


def test_an_unresolvable_template_is_reported_not_dropped(tmp_path: Path) -> None:
    build_object_tree(tmp_path)
    assignments, unresolved = obstacle_assignments(tmp_path, [instance("missing")])
    assert assignments == []
    assert unresolved == ["missing"]


def test_summary_counts_the_walls_the_exporter_will_build(tmp_path: Path) -> None:
    build_object_tree(tmp_path)
    storey = square_storey()
    assignments, summary = simulation_geometry(tmp_path, storey, [instance("abc")])
    assert summary.shell_watertight
    assert summary.obstacle_count == 1
    assert summary.total_walls == sum(len(a.mesh.faces) for a in assignments)


def test_instances_keep_their_placement_in_the_simulated_geometry(tmp_path: Path) -> None:
    build_object_tree(tmp_path)
    assignments, _ = obstacle_assignments(tmp_path, [instance("abc", x=3.0)])
    assert assignments[0].mesh.centroid[0] == pytest.approx(3.0, abs=0.1)


def test_sampled_points_stay_clear_of_the_walls() -> None:
    """Close to a surface the image source model is dominated by one early
    reflection, which is not what a listener in the room hears."""
    storey = square_storey(size=8.0)
    rng = np.random.default_rng(0)
    points = sample_points(storey, 20, rng, min_wall_distance=0.5)
    for point in points:
        assert storey.walkable.exterior.distance(Point(point[0], point[2])) >= 0.5 - 1e-9


def test_sampled_points_sit_at_the_requested_height_above_the_floor() -> None:
    storey = square_storey()
    points = sample_points(storey, 5, np.random.default_rng(0), height=1.2)
    assert all(point[1] == pytest.approx(storey.floor_height + 1.2) for point in points)


def test_sampling_is_reproducible_from_its_seed() -> None:
    storey = square_storey()
    first = sample_points(storey, 4, np.random.default_rng(7))
    second = sample_points(storey, 4, np.random.default_rng(7))
    assert np.allclose(first, second)


def test_a_room_too_narrow_for_the_clearance_is_reported_not_fudged() -> None:
    storey = square_storey(size=0.6)
    with pytest.raises(ValueError):
        sample_points(storey, 1, np.random.default_rng(0), min_wall_distance=0.5)


def test_a_pair_in_one_room_is_labelled_as_such() -> None:
    storey = square_storey(size=8.0)
    pair = sample_source_receiver(storey, np.random.default_rng(1), same_room=True)
    assert pair.same_room
    assert pair.source_room == pair.receiver_room


def test_instances_from_another_storey_are_not_simulated(tmp_path: Path) -> None:
    """The caller should not have to remember to filter; passing a whole
    scene's furniture must not simulate the floor above."""
    build_object_tree(tmp_path)
    storey = square_storey()
    upstairs = FurnitureInstance(
        template_name="abc",
        translation=np.array([1.0, 9.0, 1.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.ones(3),
    )
    _, summary = simulation_geometry(tmp_path, storey, [instance("abc", x=1.0), upstairs])
    assert summary.obstacle_count == 1
