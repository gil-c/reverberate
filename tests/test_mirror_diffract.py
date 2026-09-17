"""The diffracted onset: round a wall through its doorway, later and duller than the line.

A box cut in two by a wall with a doorway: the receiver behind the wall has
no line of sight to the source. The onset must be found, pass through the
doorway (longer than the straight line, no shorter than the path by the
door's edge), lose more at high frequencies than at low, and arrive from the
doorway's side.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from reverberate.mirror.diffract import DiffractionSettings, diffracted_paths, maekawa_db
from test_mirror_ism import box_scene

# The wall at x = 2 spans the box but for a doorway at 2.0 <= z <= 2.5 (the
# box is 4 x 3 x 2.5), from the floor to y = 2.1.
WALL_X = 2.0
DOOR = (2.0, 2.5, 2.1)


def _rect(x: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    a = [x, y0, z0]
    b = [x, y1, z0]
    c = [x, y1, z1]
    d = [x, y0, z1]
    return np.asarray([[a, b, c], [a, c, d]], dtype=float)


def walled_box() -> Any:
    scene = box_scene(alpha=0.2)
    z0, z1, top = DOOR
    wall = np.concatenate(
        [
            _rect(WALL_X, 0.0, 3.0, 0.0, z0),  # beside the doorway
            _rect(WALL_X, top, 3.0, z0, z1),  # above it
        ]
    )
    occluders = np.concatenate([scene.occluder_vertices, wall])
    labels = np.concatenate([scene.occluder_label, np.zeros(len(wall), dtype=np.int16)])
    return replace(scene, occluder_vertices=occluders, occluder_label=labels)


def test_maekawa_is_the_shadow_boundary_at_no_detour_and_grows_with_frequency() -> None:
    bands = np.array([125.0, 1000.0, 8000.0])
    at_edge = maekawa_db(0.0, bands, 343.0, 24.0)
    np.testing.assert_allclose(at_edge, 10 * np.log10(3.0))
    deep = maekawa_db(0.5, bands, 343.0, 24.0)
    assert deep[0] < deep[1] < deep[2] <= 24.0


@pytest.mark.parametrize("cell_m", [0.1])
def test_the_onset_goes_through_the_doorway(cell_m: float) -> None:
    scene = walled_box()
    source = np.array([1.0, 1.2, 0.8])
    receivers = np.array([[3.0, 1.2, 0.8], [1.5, 1.2, 1.0]])
    onsets, record = diffracted_paths(
        scene,
        source,
        receivers,
        [0],
        sound_speed_m_s=343.2,
        settings=DiffractionSettings(cell_m=cell_m),
    )
    assert record["found"] == 1 and 0 in onsets and 1 not in onsets
    path = onsets[0]
    straight = float(np.linalg.norm(receivers[0] - source))
    # Round the door's near jamb at z = 2.0: the shortest path in the plane y = 1.2.
    jamb = np.array([WALL_X, 1.2, DOOR[0]])
    by_jamb = float(np.linalg.norm(jamb - source) + np.linalg.norm(receivers[0] - jamb))
    assert path.length_m[0] > straight + 0.5
    assert by_jamb - 0.05 <= path.length_m[0] <= by_jamb + 0.4
    loss = -20.0 * np.log10(path.gain[0] * path.length_m[0])
    assert np.all(np.diff(loss) >= -1e-9) and loss[0] > 4.0 and loss[-1] > loss[0] + 6.0
    # It arrives from the doorway's side: towards +z and back towards the wall.
    assert path.direction[0, 2] > 0.5 and path.direction[0, 0] < 0.0


def test_the_edge_reflects_in_the_receivers_own_room() -> None:
    """The corner the sound bends round is a source, and the room round it answers."""
    scene = walled_box()
    source = np.asarray([1.0, 1.2, 1.2])
    receivers = np.asarray([[3.0, 1.2, 1.2]])
    plain, plain_record = diffracted_paths(
        scene,
        source,
        receivers,
        [0],
        sound_speed_m_s=343.2,
        settings=DiffractionSettings(reflections=0),
    )
    with_edge, record = diffracted_paths(
        scene,
        source,
        receivers,
        [0],
        sound_speed_m_s=343.2,
        settings=DiffractionSettings(reflections=1),
    )
    assert plain_record["edge_trees"] == 0
    if 0 not in plain:
        return  # the geodesic found no way round in this fixture
    assert 0 in with_edge
    # The onset itself is untouched: same shortest path, same time, same gain.
    assert with_edge[0].length_m.min() == pytest.approx(plain[0].length_m[0])
    np.testing.assert_allclose(
        with_edge[0].gain[np.argmin(with_edge[0].length_m)], plain[0].gain[0], rtol=1e-9
    )
    # ... and what is added arrives later and quieter, as a reflection must.
    assert with_edge[0].length_m.size >= plain[0].length_m.size
    if with_edge[0].length_m.size > 1:
        order = np.argsort(with_edge[0].length_m)
        first, second = order[0], order[1]
        assert with_edge[0].length_m[second] > with_edge[0].length_m[first]
        assert record["edge_trees"] >= 1


def test_the_edges_are_chosen_by_length_and_material() -> None:
    """A rim under the length floor, or on a face that absorbs, is not an edge."""
    from reverberate.mirror.diffract import diffracting_edges

    scene = walled_box()
    edges = diffracting_edges(scene, DiffractionSettings(min_edge_m=0.3))
    assert edges.count > 0
    assert float(edges.length_m.min()) >= 0.3
    # Nothing survives a material the sound does not reach.
    deaf = diffracting_edges(scene, DiffractionSettings(max_edge_absorption=0.0))
    assert deaf.count < edges.count
    # ... nor an impossible length floor.
    assert diffracting_edges(scene, DiffractionSettings(min_edge_m=1e6)).count == 0


def test_a_shadowed_point_hears_several_edges() -> None:
    """The geodesic gives one way round; the edges give the ones that exist."""
    scene = walled_box()
    source = np.array([1.0, 1.2, 0.8])
    receivers = np.array([[3.0, 1.2, 0.8]])
    one, _ = diffracted_paths(
        scene,
        source,
        receivers,
        [0],
        sound_speed_m_s=343.2,
        settings=DiffractionSettings(edges=False, reflections=0),
    )
    many, record = diffracted_paths(
        scene,
        source,
        receivers,
        [0],
        sound_speed_m_s=343.2,
        settings=DiffractionSettings(edges=True, reflections=0, max_edge_detour_m=4.0),
    )
    assert record["edges_kept"] > 0
    assert many[0].length_m.size > one[0].length_m.size
    # Every one of them is a real way round: longer than the straight line.
    straight = float(np.linalg.norm(receivers[0] - source))
    assert float(many[0].length_m.min()) > straight
    # The energy is the one barrier's, shared, not one barrier's for each edge.
    shared = float(np.sum(many[0].gain[:, 4] ** 2))
    alone = float(one[0].gain[0, 4] ** 2)
    assert shared == pytest.approx(alone, rel=0.05)


def test_a_corner_snaps_onto_the_edge_it_stands_near() -> None:
    """The grid puts a corner at a cell centre; the edge puts it where it belongs."""
    from reverberate.mirror.diffract import diffracting_edges, snap_to_edges

    scene = walled_box()
    edges = diffracting_edges(scene, DiffractionSettings())
    settings = DiffractionSettings(snap_m=0.5)
    # A corner 12 cm off the first edge, between two points either side of it.
    edge = 0
    middle = 0.5 * (edges.a[edge] + edges.b[edge])
    off = np.array([0.12, 0.0, 0.0])
    corners = [middle + np.array([1.0, 0.0, 1.0]), middle + off, middle - np.array([1.0, 0.0, 1.0])]
    moved = snap_to_edges([c.copy() for c in corners], edges, settings)
    assert len(moved) == 3
    np.testing.assert_allclose(moved[0], corners[0])
    np.testing.assert_allclose(moved[2], corners[2])
    # The middle point now lies on an edge, and the way round is no longer.
    on_edge = min(
        float(np.linalg.norm(np.cross(edges.b[k] - edges.a[k], moved[1] - edges.a[k])))
        / max(float(np.linalg.norm(edges.b[k] - edges.a[k])), 1e-9)
        for k in range(edges.count)
    )
    assert on_edge < 1e-6

    def walk(points: list[np.ndarray]) -> float:
        return float(
            sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:], strict=True))
        )

    assert walk(moved) <= walk(corners) + 1e-9

    # With no edge within reach the corners are left exactly where they were.
    far = snap_to_edges([c.copy() for c in corners], edges, DiffractionSettings(snap_m=1e-6))
    for a, b in zip(far, corners, strict=True):
        np.testing.assert_allclose(a, b)
