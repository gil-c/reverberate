"""The rays through a tree of boxes: the same hits as the grid, and the same tail.

The tree must return the triangle a test of every triangle returns, the
lower index on a tie, on scenes of random triangles with coincident ones
among them; its build must be the same twice and hold every item once. The
twin through the tree must count what the grid's twin counts on a furnished
room, to the integer; the C text must count what the twin counts and visit
the nodes it visits; and the text in single precision must let no ray out of
a closed room, aimed at its edges and its corners or cast at random, and
give a tail that differs from the double precision's by less than two seeds
of the double precision differ.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.compute import Devices
from reverberate.mirror import bvh, tracer
from reverberate.mirror.engine import histogram_on_devices
from reverberate.mirror.geometry import DerivedScene, Facet, GeometryRules, MaterialTable
from reverberate.mirror.rays import RaySettings, _nearest_hit, trace
from reverberate.mirror.shared import Store

native = pytest.mark.skipif(not tracer.available(), reason="needs a C compiler on the host")

#: A room that is on no round number, as a dwelling's coordinates are not.
CORNER = np.array([-13.37, 0.013, -7.91])
SIZE = np.array([4.1, 2.7, 3.3])
CABINET = (np.array([1.1, 0.0, 0.4]), np.array([0.9, 1.3, 0.55]))
SOURCE = CORNER + np.array([2.9, 1.2, 2.1])
CELLS = CORNER + np.array([[0.9, 1.5, 2.4], [3.2, 1.1, 0.9], [2.1, 2.0, 1.6]])


def _sheet(origin: np.ndarray, u: np.ndarray, v: np.ndarray, cuts: int) -> np.ndarray:
    """A parallelogram in ``cuts`` by ``cuts`` quads of two triangles, normal along u x v."""
    out = []
    for i in range(cuts):
        for j in range(cuts):
            a = origin + u * (i / cuts) + v * (j / cuts)
            b = origin + u * ((i + 1) / cuts) + v * (j / cuts)
            c = origin + u * ((i + 1) / cuts) + v * ((j + 1) / cuts)
            d = origin + u * (i / cuts) + v * ((j + 1) / cuts)
            out += [[a, b, c], [a, c, d]]
    return np.asarray(out, dtype=float)


def _box(corner: np.ndarray, size: np.ndarray, cuts: int, inwards: bool) -> list[np.ndarray]:
    """The six faces of a box, each a sheet, normals into the box or out of it."""
    x, y, z = np.diag(size)
    far = corner + size
    faces = [
        (corner, y, z),
        (corner, z, x),
        (corner, x, y),
        (far, -z, -y),
        (far, -x, -z),
        (far, -y, -x),
    ]
    return [_sheet(o, u, v, cuts) if inwards else _sheet(o, v, u, cuts) for o, u, v in faces]


def furnished_room(cuts: int = 3, alpha: float = 0.3, scattering: float = 0.4) -> DerivedScene:
    """A closed room of ``cuts`` by ``cuts`` quads a wall, and a closed cabinet standing in it."""
    walls = _box(CORNER, SIZE, cuts, inwards=True)
    cabinet = _box(CORNER + CABINET[0], CABINET[1], 2, inwards=False)
    sheets = walls + cabinet
    triangles = np.concatenate(sheets)
    facets, facet_of, first = [], [], 0
    for k, sheet in enumerate(sheets):
        normal = np.cross(sheet[0, 1] - sheet[0, 0], sheet[0, 2] - sheet[0, 0])
        normal /= np.linalg.norm(normal)
        facets.append(
            Facet(
                label=0 if k < 6 else 1,
                normal=normal,
                offset=float(normal @ sheet[0, 0]),
                area=float(
                    0.5
                    * np.linalg.norm(
                        np.cross(sheet[:, 1] - sheet[:, 0], sheet[:, 2] - sheet[:, 0]), axis=1
                    ).sum()
                ),
                triangles=np.arange(first, first + len(sheet), dtype=np.int32),
                kind="shell_wall" if k < 6 else "furniture",
            )
        )
        facet_of += [k] * len(sheet)
        first += len(sheet)
    labels = np.asarray([0 if f < 6 else 1 for f in facet_of], dtype=np.int16)
    bands = len(OCTAVE_BANDS)
    materials = MaterialTable(
        ("wall", "cabinet"),
        np.stack([np.full(bands, alpha), np.linspace(0.5, 0.2, bands)]),
        np.array([scattering, 0.8]),
    )
    return DerivedScene(
        labels=("wall", "cabinet"),
        materials=materials,
        facets=tuple(facets),
        reflector_vertices=triangles,
        reflector_facet=np.asarray(facet_of, dtype=np.int32),
        occluder_vertices=triangles.copy(),
        occluder_label=labels,
        occluder_sides=np.full(len(triangles), 2, dtype=np.int8),
        rules=GeometryRules(),
        bmin=CORNER.copy(),
        bmax=CORNER + SIZE,
    )


def _random_triangles(count: int, seed: int) -> np.ndarray:
    """Small triangles in a cube of five metres, a twentieth of them twice."""
    rng = np.random.default_rng(seed)
    triangles = rng.uniform(0.0, 5.0, (count, 1, 3)) + rng.normal(0.0, 0.5, (count, 3, 3))
    return np.concatenate([triangles, triangles[::20]])


def _counts(histogram: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.rint(histogram.energy * 2.0**40).astype(np.int64)  # type: ignore[attr-defined]
    moments = np.rint(histogram.moments * 2.0**40).astype(np.int64)  # type: ignore[attr-defined]
    return energy, moments, histogram.hits  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# the tree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_the_tree_returns_the_hit_a_test_of_every_triangle_returns(seed: int) -> None:
    triangles = _random_triangles(400, seed)
    tree = bvh.build(*bvh.triangle_boxes(triangles))
    rng = np.random.default_rng(seed + 10)
    origins = rng.uniform(0.0, 5.0, (300, 3))
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    counts: dict[str, int] = {}
    distance, triangle = bvh.nearest(
        tree, triangles[tree.order], origins, directions, counts=counts
    )
    everything = np.arange(len(triangles))
    hits = 0
    for k in range(len(origins)):
        want_t, want = _nearest_hit(origins[k], directions[k], triangles, everything, -1)
        assert triangle[k] == want
        hits += want >= 0
        if want >= 0:
            assert distance[k] == pytest.approx(want_t, rel=1e-12)
    # Enough rays hit, some of them a triangle that is there twice: the lower index is kept.
    assert hits > 150
    assert np.any((triangle >= 0) & (triangle % 20 == 0) & (triangle < 400))
    assert not np.any(triangle >= 400)
    # And the tree spares the tests: far fewer than every triangle a ray.
    assert counts["tests"] < 0.1 * len(origins) * len(triangles)
    # The ray leaves a triangle: that one is not hit again.
    again, other = bvh.nearest(tree, triangles[tree.order], origins, directions, triangle)
    assert np.all((other != triangle) | (triangle < 0))
    assert np.all(again >= distance)


def test_the_build_is_the_same_twice_and_holds_every_item_in_one_leaf() -> None:
    triangles = _random_triangles(700, 3)
    lo, hi = bvh.triangle_boxes(triangles)
    first, second = bvh.build(lo, hi), bvh.build(lo, hi)
    for name, array in first.arrays().items():
        np.testing.assert_array_equal(array, second.arrays()[name])
    start, held = first.leaves()
    assert sorted(first.order.tolist()) == list(range(len(triangles)))
    assert held.max() <= bvh.LEAF and int(held.sum()) == len(triangles)
    covered = np.zeros(len(triangles), dtype=int)
    for a, n in zip(start, held, strict=True):
        covered[a : a + n] += 1
    assert np.all(covered == 1)
    # A child's box holds its triangles, grown by the pad and never shrunk.
    packed = first.packed()
    assert packed.shape == (first.nodes, 16) and packed.dtype == np.float32
    np.testing.assert_array_equal(packed.view(np.int32)[:, 12:14], first.links)
    assert first.boxes[0, :, :3].min() <= lo.min() - 0.5 * first.pad_m
    assert first.boxes[0, :, 3:].max() >= hi.max() + 0.5 * first.pad_m
    again = bvh.Bvh.from_arrays(first.arrays())
    assert again.depth == first.depth and again.pad_m == first.pad_m


@pytest.mark.parametrize("count", [0, 1, 2, 60])
def test_a_tree_of_nothing_of_one_and_of_items_on_one_point_is_a_tree(count: int) -> None:
    triangle = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    triangles = np.repeat(triangle[None, :, :], count, axis=0)
    tree = bvh.build(*bvh.triangle_boxes(triangles))
    assert sorted(tree.order.tolist()) == list(range(count))
    distance, found = bvh.nearest(
        tree, triangles[tree.order], np.array([[0.2, 0.2, 0.0]]), np.array([[0.0, 0.0, 1.0]])
    )
    # Sixty triangles in one place: the first of them is the one hit.
    assert found[0] == (0 if count else -1)
    assert distance[0] == (1.0 if count else np.inf)


# --------------------------------------------------------------------------
# the twin through the tree, and the C text
# --------------------------------------------------------------------------


@pytest.mark.parametrize("skip_order,window_s", [(0, 0.0), (3, 0.01)])
def test_the_twin_through_the_tree_counts_what_the_grid_s_twin_counts(
    skip_order: int, window_s: float
) -> None:
    scene = furnished_room()
    settings = RaySettings(
        rays=40,
        duration_s=0.07,
        bin_s=0.002,
        receiver_radius_m=0.5,
        seed=6,
        skip_specular_order=skip_order,
        skip_window_s=window_s,
    )
    grid = _counts(trace(scene, SOURCE, CELLS, settings))
    stats: dict[str, int] = {}
    tree = _counts(tracer.trace_tree(scene, SOURCE, CELLS, settings, stats=stats))
    assert grid[2].sum() > 30
    np.testing.assert_array_equal(tree[2], grid[2])
    np.testing.assert_array_equal(tree[0], grid[0])
    # A direction may differ in its last bit, and a moment by a count of 2^40 with it.
    assert np.abs(tree[1] - grid[1]).max() <= 2
    assert stats["escapes"] == 0 and stats["segments"] > 40 * 10
    assert stats["deposits"] == int(grid[2].sum())


def test_shares_of_the_rays_through_the_tree_sum_to_the_whole() -> None:
    scene = furnished_room()
    settings = RaySettings(rays=300, duration_s=0.08, receiver_radius_m=0.5, seed=7)
    held = tracer.prepare(scene)
    whole = _counts(tracer.trace_tree(scene, SOURCE, CELLS, settings, held=held))
    parts = [
        _counts(
            tracer.trace_tree(scene, SOURCE, CELLS, settings, held=held, ray_start=a, ray_count=n)
        )
        for a, n in ((0, 100), (100, 150), (250, 50))
    ]
    for k in range(3):
        np.testing.assert_array_equal(sum(p[k] for p in parts), whole[k])
    # A cell's histogram is its own: traced alone, it is its rows of the three's.
    alone = _counts(tracer.trace_tree(scene, SOURCE, CELLS[1:2], settings, held=held))
    for k in range(3):
        np.testing.assert_array_equal(alone[k][0], whole[k][1])


def test_the_tree_is_kept_in_a_store_and_does_not_depend_on_materials(tmp_path: Path) -> None:
    scene = furnished_room()
    store = Store(tmp_path / "store")
    made = tracer.prepare(scene, store)
    assert store.made == {"rays_tree": 1}
    louder = replace(scene.materials, absorption=scene.materials.absorption * 0.5)
    again = Store(tmp_path / "store")
    found = tracer.prepare(replace(scene, materials=louder), again)
    assert again.found == {"rays_tree": 1} and again.made == {}
    np.testing.assert_array_equal(found.tree.links, made.tree.links)
    np.testing.assert_array_equal(found.meta, made.meta)
    np.testing.assert_array_equal(found.triangles, made.triangles)
    # Furniture and what lies on no reflector are written beside each triangle's label.
    of_scene = made.of_scene()
    np.testing.assert_array_equal(of_scene["label"], scene.occluder_label)
    assert of_scene["furniture"].sum() == 6 * 2 * 2 * 2 and not of_scene["off"].any()


@native
@pytest.mark.parametrize("skip_order", [0, 3])
def test_the_c_text_counts_what_the_twin_counts_and_visits_its_nodes(skip_order: int) -> None:
    scene = furnished_room()
    settings = RaySettings(
        rays=1500,
        duration_s=0.12,
        bin_s=0.002,
        receiver_radius_m=0.4,
        seed=9,
        skip_specular_order=skip_order,
        skip_window_s=0.012 if skip_order else 0.0,
    )
    held = tracer.prepare(scene)
    twin_stats: dict[str, int] = {}
    twin = _counts(tracer.trace_tree(scene, SOURCE, CELLS, settings, held=held, stats=twin_stats))
    text_stats: dict[str, int] = {}
    text = tracer.counts_on_host(scene, held, SOURCE, CELLS, settings, stats=text_stats)
    assert twin[2].sum() > 500
    for k in range(3):
        np.testing.assert_array_equal(text[k], twin[k])
    for name in ("segments", "nodes", "tests", "escapes", "deposits", "ended_by_floor"):
        assert text_stats[name] == twin_stats[name], name
    # The same rays twice, and in shares, are the same counts.
    again = tracer.counts_on_host(scene, held, SOURCE, CELLS, settings)
    half = [
        tracer.counts_on_host(scene, held, SOURCE, CELLS, settings, ray_start=a, ray_count=750)
        for a in (0, 750)
    ]
    for k in range(3):
        np.testing.assert_array_equal(again[k], text[k])
        np.testing.assert_array_equal(half[0][k] + half[1][k], text[k])


@native
def test_the_c_text_s_first_hit_is_the_twin_s_on_random_triangles() -> None:
    triangles = _random_triangles(600, 5)
    tree = bvh.build(*bvh.triangle_boxes(triangles))
    order = tree.order.astype(np.int64)
    held = tracer.TracerScene(
        tree=tree,
        triangles=np.ascontiguousarray(triangles[order].reshape(-1, 9)),
        meta=np.stack([order, np.zeros_like(order)], axis=1).astype(np.int32),
        centre=np.full(3, 2.5),
    )
    rng = np.random.default_rng(8)
    origins = rng.uniform(0.0, 5.0, (2000, 3))
    directions = rng.normal(size=(2000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    want_t, want = bvh.nearest(tree, triangles[order], origins, directions)
    got_t, got = tracer.first_hits(held, origins, directions)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got_t, want_t)
    # Single precision finds the same triangle but where a ray grazes an edge.
    single_t, single = tracer.first_hits(held, origins, directions, precision="single")
    assert np.mean(single == want) > 0.995
    same = (single == want) & (want >= 0)
    np.testing.assert_allclose(single_t[same], want_t[same], rtol=2e-5, atol=2e-6)


@native
def test_no_ray_aimed_at_an_edge_or_a_corner_leaves_a_closed_room() -> None:
    """Single precision's test is watertight, which Moller and Trumbore's is not.

    A ray aimed at a point of an edge is, to the last bit, outside one
    triangle or the other, and Moller and Trumbore's test may find it
    outside both: in double precision three in a hundred of the rays aimed
    here pass between two triangles (171 of 5292 from the first origin, on
    the machine this was written on). A ray cast at random does not come
    that near an edge once in a storey's rays, which is why the tails never
    showed it. Single precision's rounding is a billion times coarser: its
    test is the one whose two triangles take the same products on their
    edge, and none of the 15 876 rays aimed here leaves the room.
    """
    scene = furnished_room(cuts=7)
    held = tracer.prepare(scene)
    walls = scene.occluder_vertices[: 6 * 7 * 7 * 2]
    corners = walls.reshape(-1, 3)
    middles = 0.5 * (walls + np.roll(walls, 1, axis=1)).reshape(-1, 3)
    thirds = (walls * (2 / 3) + np.roll(walls, 1, axis=1) / 3).reshape(-1, 3)
    targets = np.concatenate([corners, middles, thirds])
    for origin in (SOURCE, CORNER + np.array([0.3, 2.5, 3.0]), CORNER + np.array([3.9, 0.2, 0.1])):
        directions = targets - origin[None, :]
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        origins = np.repeat(origin[None, :], len(targets), axis=0)
        _, single = tracer.first_hits(held, origins, directions, precision="single")
        assert len(targets) == 5292 and np.all(single >= 0)


@native
def test_single_precision_keeps_the_rays_in_and_gives_the_double_s_tail() -> None:
    """The tail in single precision against the double's, by the measure of two seeds.

    Per band, in windows of 50 ms: the level of single precision against
    double precision at one seed must be nearer than the farthest two of
    three seeds of double precision are from one another, in the worst
    band and window. The windows hold thousands of crossings each.
    """
    scene = furnished_room(cuts=5, alpha=0.12, scattering=0.5)
    held = tracer.prepare(scene)
    settings = RaySettings(rays=8000, duration_s=0.3, bin_s=0.002, receiver_radius_m=0.5)

    def levels(seed: int, precision: str) -> np.ndarray:
        stats: dict[str, int] = {}
        counted = tracer.counts_on_host(
            scene,
            held,
            SOURCE,
            CELLS,
            replace(settings, seed=seed, precision=precision),
            stats=stats,
        )
        # A closed room: no ray meets nothing, in either precision, in 400 000 segments.
        assert stats["escapes"] == 0 and stats["segments"] > 400_000
        assert counted[2].sum() > 20_000
        windows = counted[0].reshape(len(CELLS), 6, 25, -1).sum(axis=2)
        return np.asarray(10.0 * np.log10(windows[:, 1:].astype(float)))

    double = [levels(seed, "double") for seed in (0, 1, 2)]
    single = levels(0, "single")
    between_seeds = max(
        float(np.abs(double[a] - double[b]).max()) for a, b in ((0, 1), (0, 2), (1, 2))
    )
    assert float(np.abs(single - double[0]).max()) <= between_seeds
    assert between_seeds < 1.0


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cores", [1, 3])
def test_through_the_tree_the_engine_gives_the_grid_s_histogram(cores: int) -> None:
    scene = furnished_room()
    settings = RaySettings(rays=45, duration_s=0.06, receiver_radius_m=0.5, seed=2)
    grid = trace(scene, SOURCE, CELLS, settings)
    stats: dict[str, int] = {}
    held: dict[object, object] = {}
    tree = histogram_on_devices(
        scene,
        SOURCE,
        CELLS,
        settings,
        devices=Devices.host(cores),
        structure="tree",
        held=held,
        stats=stats,
    )
    np.testing.assert_array_equal(tree.hits, grid.hits)
    np.testing.assert_array_equal(tree.energy, grid.energy)
    assert np.abs(tree.moments - grid.moments).max() * 2.0**40 <= 2
    assert stats["segments"] > 45 * 5 and "tree" in held
    with pytest.raises(ValueError, match="single precision is the tree's"):
        histogram_on_devices(
            scene, SOURCE, CELLS, replace(settings, precision="single"), structure="grid"
        )
    with pytest.raises(ValueError, match="'grid' or 'tree'"):
        tracer.structure("octree")


def test_the_tree_is_what_the_rays_go_through_and_the_grid_can_be_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(tracer.STRUCTURE_VARIABLE, raising=False)
    assert tracer.structure() == "tree" and tracer.structure("grid") == "grid"
    monkeypatch.setenv(tracer.STRUCTURE_VARIABLE, "grid")
    assert tracer.structure() == "grid" and tracer.structure("tree") == "tree"


def test_a_precision_is_recorded_where_it_is_not_the_reference_s() -> None:
    assert "precision" not in RaySettings().record()
    assert RaySettings(precision="single").record()["precision"] == "single"
    assert RaySettings(**RaySettings(precision="single").record()).precision == "single"
