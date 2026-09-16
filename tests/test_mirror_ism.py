"""The image source twin against pyroomacoustics on a shoebox, and against geometry.

pyroomacoustics grows a shoebox's images by a closed formula; the twin grows
them on six facets and must land on the same 25 positions up to order two,
with the same reflection factors, and every one visible from a receiver in
an empty box. An obstacle between the source and the receiver must cut the
direct path and not the reflections that go round it; a facet that reflects
on one side only must make no image of a source behind it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import pytest

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.geometry import (
    DerivedScene,
    Facet,
    GeometryRules,
    MaterialTable,
    derive,
)
from reverberate.mirror.ism import IsmSettings, grow_tree, paths_for, ray_triangles
from test_accel_scene import write_scene

SIZE = np.array([4.0, 3.0, 2.5])


def box_scene(
    alpha: float = 0.2,
    scattering: float = 0.0,
    obstacle: tuple[np.ndarray, np.ndarray] | None = None,
    size: np.ndarray = SIZE,
) -> DerivedScene:
    """Six facets of a box with air inside, and an optional slab as an occluder only."""
    corners = np.array(
        [[x, y, z] for x in (0.0, size[0]) for y in (0.0, size[1]) for z in (0.0, size[2])]
    )
    # Faces as (normal into the air, offset, four corner indices in a loop).
    faces = {
        "x0": (np.array([1.0, 0, 0]), [0, 1, 3, 2]),
        "x1": (np.array([-1.0, 0, 0]), [4, 6, 7, 5]),
        "y0": (np.array([0, 1.0, 0]), [0, 4, 5, 1]),
        "y1": (np.array([0, -1.0, 0]), [2, 3, 7, 6]),
        "z0": (np.array([0, 0, 1.0]), [0, 2, 6, 4]),
        "z1": (np.array([0, 0, -1.0]), [1, 5, 7, 3]),
    }
    triangles: list[np.ndarray] = []
    facets = []
    for normal, loop in faces.values():
        quad = corners[loop]
        start = len(triangles)
        triangles.append(quad[[0, 1, 2]])
        triangles.append(quad[[0, 2, 3]])
        facets.append(
            Facet(
                label=0,
                normal=normal,
                offset=float(normal @ quad[0]),
                area=float(np.linalg.norm(np.cross(quad[1] - quad[0], quad[3] - quad[0]))),
                triangles=np.arange(start, start + 2, dtype=np.int32),
                kind="shell_wall",
            )
        )
    reflectors = np.asarray(triangles, dtype=float)
    occluders = reflectors.copy()
    labels = np.zeros(len(occluders), dtype=np.int16)
    if obstacle is not None:
        lo, hi = obstacle
        # A vertical slab in the plane x = lo[0], spanning lo to hi in y and z.
        a = np.array([lo[0], lo[1], lo[2]])
        b = np.array([lo[0], hi[1], lo[2]])
        c = np.array([lo[0], hi[1], hi[2]])
        d = np.array([lo[0], lo[1], hi[2]])
        slab = np.array([[a, b, c], [a, c, d]])
        occluders = np.concatenate([occluders, slab])
        labels = np.concatenate([labels, np.zeros(2, dtype=np.int16)])
    materials = MaterialTable(
        ("wall",),
        np.full((1, len(OCTAVE_BANDS)), alpha),
        np.array([scattering]),
    )
    return DerivedScene(
        labels=("wall",),
        materials=materials,
        facets=tuple(facets),
        reflector_vertices=reflectors,
        reflector_facet=np.repeat(np.arange(6, dtype=np.int32), 2),
        occluder_vertices=occluders,
        occluder_label=labels,
        occluder_sides=np.full(len(occluders), 2, dtype=np.int8),
        rules=GeometryRules(),
        bmin=np.zeros(3),
        bmax=size,
    )


SOURCE = np.array([1.0, 1.2, 1.1])
RECEIVER = np.array([2.5, 1.8, 1.3])


def oracle(alpha: float, max_order: int) -> pra.Room:
    room = pra.ShoeBox(list(SIZE), fs=16000, materials=pra.Material(alpha), max_order=max_order)
    room.add_source(list(SOURCE))
    room.add_microphone(list(RECEIVER))
    room.image_source_model()
    return room


def test_the_tree_lands_on_the_shoebox_images_of_pyroomacoustics() -> None:
    tree = grow_tree(box_scene(), SOURCE, IsmSettings(max_order=2))
    ours = np.unique(np.round(tree.positions, 9), axis=0)
    theirs = np.unique(np.round(oracle(0.2, 2).sources[0].images.T, 9), axis=0)
    assert ours.shape == theirs.shape == (25, 3)
    # pyroomacoustics keeps its images in single precision.
    np.testing.assert_allclose(ours, theirs, atol=1e-5)
    assert list(np.bincount(tree.order)) == [1, 6, 30]


def test_every_shoebox_image_is_heard_once_with_the_oracle_s_reflection_factor() -> None:
    alpha = 0.2
    scene = box_scene(alpha)
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=2))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=2))
    # Of the 37 sequences, exactly the 25 distinct images validate: one order
    # of the two commuting mirrors is geometrically consistent, never both.
    positions = np.unique(np.round(tree.positions[paths.image], 9), axis=0)
    assert paths.count == 25 and positions.shape[0] == 25
    room = oracle(alpha, 2)
    theirs = room.sources[0]
    by_position = {
        tuple(np.round(p.astype(float), 4).tolist()): float(d)
        for p, d in zip(theirs.images.T, theirs.damping[0], strict=True)
    }
    for k in range(paths.count):
        image = tuple(np.round(tree.positions[paths.image[k]], 4).tolist())
        factor = paths.gain[k, 0] * paths.length_m[k]
        assert factor == pytest.approx(by_position[image], abs=1e-6)
        assert paths.length_m[k] == pytest.approx(
            float(np.linalg.norm(tree.positions[paths.image[k]] - RECEIVER))
        )
    direct = int(np.flatnonzero(paths.order == 0)[0])
    np.testing.assert_allclose(
        paths.direction[direct], (SOURCE - RECEIVER) / np.linalg.norm(SOURCE - RECEIVER)
    )


def test_the_hit_points_lie_on_their_facets_and_the_legs_add_up() -> None:
    scene = box_scene()
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=2))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=2))
    for k in range(paths.count):
        order = int(paths.order[k])
        points = paths.points[k, : order + 2]
        legs = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        assert legs == pytest.approx(paths.length_m[k], abs=1e-9)
        for bounce in range(order):
            facet = scene.facets[int(paths.sequence[k, bounce])]
            hit = points[bounce + 1]
            assert float(facet.normal @ hit) == pytest.approx(facet.offset, abs=1e-9)
            assert np.all(hit >= -1e-9) and np.all(hit <= SIZE + 1e-9)


def test_a_slab_between_the_two_cuts_the_direct_path_and_not_the_reflections() -> None:
    lo = np.array([1.75, 0.5, 0.5])
    hi = np.array([1.75, 2.5, 2.0])
    scene = box_scene(obstacle=(lo, hi))
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=1))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=1))
    assert 0 not in paths.order
    # The floor and ceiling paths cross the slab's plane below and above it,
    # the side wall paths cross it beyond its edges; both end wall paths
    # cross it through the middle and are cut.
    kinds = {int(paths.sequence[k, 0]) for k in range(paths.count)}
    assert kinds == {2, 3, 4, 5}


def test_a_one_sided_facet_makes_no_image_of_a_source_behind_it() -> None:
    scene = box_scene()
    outside = np.array([-1.0, 1.0, 1.0])  # behind the x = 0 wall
    tree = grow_tree(scene, outside, IsmSettings(max_order=1))
    assert 0 not in set(tree.sequence[tree.order == 1, 0].tolist())
    assert tree.count == 1 + 5


def test_the_furniture_budget_bounds_the_sequences() -> None:
    scene = box_scene()
    furnished = DerivedScene(
        **{
            **scene.__dict__,
            "facets": tuple(
                Facet(f.label, f.normal, f.offset, f.area, f.triangles, "furniture")
                for f in scene.facets
            ),
        }
    )
    tree = grow_tree(furnished, SOURCE, IsmSettings(max_order=3, furniture_bounces=1))
    assert tree.order.max() == 1


def test_segments_against_triangles_by_moller_trumbore() -> None:
    triangle = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
    origins = np.array([[0.2, 0.2, 0.0], [0.9, 0.9, 0.0], [0.2, 0.2, 0.0]])
    ends = np.array([[0.2, 0.2, 2.0], [0.9, 0.9, 2.0], [0.2, 0.2, 0.5]])
    assert ray_triangles(origins, ends, triangle).tolist() == [True, False, False]
    # A segment starting on the plane is spared by the strict margin.
    on_plane = np.array([[0.2, 0.2, 1.0]])
    assert (
        ray_triangles(on_plane, np.array([[0.2, 0.2, 2.0]]), triangle, strict=0.01)[0] is np.False_
    )


def test_the_derived_box_of_the_solver_model_gives_the_same_tree(tmp_path: Path) -> None:
    derived = derive(write_scene(tmp_path / "box.json"))
    tree = grow_tree(derived, np.array([0.3, 0.4, 0.5]), IsmSettings(max_order=1))
    assert tree.count == 7
    paths = paths_for(derived, tree, np.array([0.6, 0.5, 0.5]), IsmSettings(max_order=1))
    assert paths.count == 7


def test_images_beyond_the_window_are_pruned_with_their_descendants() -> None:
    scene = box_scene()
    settings = IsmSettings(max_order=2, window_s=0.010)  # 3.4 m of path
    region = (RECEIVER - 0.1, RECEIVER + 0.1)
    pruned = grow_tree(scene, SOURCE, settings, region=region)
    whole = grow_tree(scene, SOURCE, settings)
    assert 1 < pruned.count < whole.count
    reach = settings.sound_speed_m_s * settings.window_s
    assert np.all(np.linalg.norm(pruned.positions[1:] - RECEIVER, axis=1) <= reach + 0.2)
