"""What a pack was computed on, derived for the audit: grids, their difference, rays, paths.

Two rooms parted by a sheet with a doorway are voxelised by the project's
own voxeliser at two steps (:mod:`reverberate.viz.computed_demo`), so the
grids here are what a dwelling's are: engine axes that are not the scene's,
boundary nodes, cut links. On them:

- a node's place and the air a source reaches are the solver's own
  (``wave/lowband/problem.py``, ``wave/comms.py``), node for node;
- the walls drawn are the cut links, each once, on the model's planes, and
  merging changes their count and not their area;
- the comparison of two steps finds the doorway's change of width, a gap
  that seals with the room behind it, and walls moved by half a step at most;
- the archive a trace writes holds all of it in a few kilobytes;
- the rays traced again for the page are the tracer's, hit for hit.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reverberate.mirror import rays as rays_module
from reverberate.mirror.rays import RaySettings, trace
from reverberate.render.pack import synthetic_free_field, write_pack
from reverberate.spatial.array import design_array
from reverberate.trace.computed import (
    NAME,
    grid_record,
    read_as_computed,
    save_as_computed,
    snap_sources,
    write_as_computed,
)
from reverberate.viz import computed_grid as cg
from reverberate.viz.audit_api import Binary
from reverberate.viz.computed_api import ComputedService, packed
from reverberate.viz.computed_demo import ROOMS_M, WALL_X_M, rooms_grid
from reverberate.viz.computed_rays import path_points, ray_paths
from reverberate.viz.scene_api import SceneError
from reverberate.wave.comms import engine_indices, load_grid, nearest_node
from reverberate.wave.lowband.problem import load_problem
from test_mirror_ism import RECEIVER, SOURCE, box_scene

FINE_M, COARSE_M = 0.05, 0.08
SOURCE_M = np.array([[1.0, 1.2, 1.5]])


@pytest.fixture(scope="module")
def rooms(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The two rooms at two steps with a doorway, and at the coarse step with a 6 cm gap."""
    tmp = tmp_path_factory.mktemp("rooms")
    fine, entry, _ = rooms_grid(tmp / "fine", FINE_M)
    coarse, _, _ = rooms_grid(tmp / "coarse", COARSE_M)
    sealed, _, _ = rooms_grid(tmp / "sealed", COARSE_M, door_m=0.06)
    return {"tmp": tmp, "fine": fine, "coarse": coarse, "sealed": sealed, "entry": entry}


def test_a_node_s_place_and_the_air_reached_are_the_solver_s(rooms: dict[str, Any]) -> None:
    grid, entry = rooms["fine"], rooms["entry"]
    lattice = load_grid(entry)
    # The engine's axes are not the scene's here: the test would pass by accident otherwise.
    assert grid.order != (0, 1, 2)
    rng = np.random.default_rng(3)
    for point in rng.uniform([0.2, 0.2, 0.2], np.asarray(ROOMS_M) - 0.2, size=(40, 3)):
        position, index = nearest_node(point, lattice)
        assert np.allclose(grid.positions(np.array([index]))[0], position, atol=1e-12)
        assert int(grid.flat_of(position)[0]) == index
        assert int(grid.engine_index(np.array([index]))[0]) == int(
            engine_indices(np.array([index]), lattice)[0]
        )
    nodes, weights = snap_sources(SOURCE_M, lattice)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights[0] @ grid.positions(nodes[0]), SOURCE_M[0])
    problem = load_problem(entry, engine_indices(nodes[0], lattice))
    assert int(grid.reached.sum()) == problem.reached
    air = np.flatnonzero(grid.reached.reshape(-1))
    air = air[~np.isin(air, grid.boundary)]
    compact = problem.compact(grid.engine_index(air))
    assert (compact >= 0).all()
    assert (problem.kind.reshape(-1)[compact] > 0).all()
    assert air.size == int(np.count_nonzero(problem.kind))
    assert np.array_equal(
        np.sort(problem.compact(grid.engine_index(grid.boundary))), np.sort(problem.bn_index)
    )


def test_the_walls_are_the_cut_links_on_the_model_s_planes(rooms: dict[str, Any]) -> None:
    grid = rooms["fine"]
    held = cg.walls(grid)
    cut = sum(int(np.count_nonzero((grid.adjacency >> d) & 1 == 0)) for d in range(6))
    assert held["faces"] == cut == cg.facts(grid)["cut_links"]
    assert held["quads"] < held["faces"] // 100  # flat walls merge
    corners = held["corners"].astype(float)
    area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]), axis=1
    ).sum()
    assert area == pytest.approx(cut * grid.step_m**2, rel=1e-5)
    # A wall square is half a step from its node, so within a step of the plane it stands for.
    planes = [(0, 0.0), (0, WALL_X_M), (0, ROOMS_M[0]), (1, 0.0), (1, ROOMS_M[1])]
    planes += [(2, 0.0), (2, ROOMS_M[2])]
    flat = np.ptp(corners, axis=1).argmin(axis=1)
    for quad, axis in enumerate(flat):
        level = corners[quad, 0, axis]
        assert min(abs(level - at) for a, at in planes if a == axis) <= grid.step_m
    # The partition's squares carry the partition's material, but for the nodes in its
    # corners with the shell: a node has one material, its nearest triangle's.
    partition = grid.labels.index("partition")
    on_it = (flat == 0) & (np.abs(corners[:, 0, 0] - WALL_X_M) <= grid.step_m)
    areas = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]), axis=1
    )
    assert areas[on_it & (held["material"] == partition)].sum() > 0.9 * areas[on_it].sum()
    # Two sides of a sheet, less the doorway: twice its area, to a step round its rim.
    sheet = ROOMS_M[1] * ROOMS_M[2] - 0.9 * 2.0
    assert areas[on_it].sum() == pytest.approx(2.0 * sheet, rel=0.06)
    near = cg.walls(grid, np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert 0 < near["faces"] < held["faces"]


def test_a_slice_is_a_layer_of_nodes_and_nothing_else(rooms: dict[str, Any]) -> None:
    grid = rooms["fine"]
    classes, layer = cg.slice_classes(grid, 1.2)
    nx, _, nz = grid.shape
    assert classes.shape == (nz, nx)
    assert abs(grid.origin_m[1] + grid.step_m * layer - 1.2) <= 0.5 * grid.step_m + 1e-9
    assert np.array_equal(classes.T != cg.NOT_REACHED, grid.reached[:, layer, :])
    on_layer = grid.subs(grid.boundary)[:, 1] == layer
    assert int(np.count_nonzero(classes >= cg.RIGID)) == int(on_layer.sum())
    # Beside the partition the boundary nodes are the partition's.
    column = int(np.floor((WALL_X_M - grid.origin_m[0]) / grid.step_m))
    row = int(np.rint((0.5 - grid.origin_m[2]) / grid.step_m))
    assert classes[row, column] == cg.FIRST_MATERIAL + grid.labels.index("partition")
    low, high = cg.absorbing_layer(grid)
    assert np.allclose(low, grid.origin_m + grid.step_m)
    assert np.allclose(high, grid.origin_m + grid.step_m * (np.asarray(grid.shape) - 2))


def test_two_steps_differ_by_the_doorway_s_width_and_half_a_step(rooms: dict[str, Any]) -> None:
    fine, coarse = rooms["fine"], rooms["coarse"]
    held = cg.compare(fine, coarse)
    numbers = held.numbers
    assert numbers["resolution_m"] == FINE_M
    assert numbers["air_a_m3"] == pytest.approx(np.prod(ROOMS_M), rel=0.01)
    # Both rooms are reached on both grids: no part opens or seals.
    assert numbers["regions_air_only_in_a"] == [] and numbers["regions_air_only_in_b"] == []
    assert numbers["differs_m3"] < 0.03 * numbers["air_a_m3"]
    # A staircase moves a wall by half of the coarser step at most.
    assert 0.0 < numbers["largest_wall_displacement_m"] <= 0.5 * COARSE_M + 1e-9
    changes = cg.opening_changes(fine, coarse, 1.2)
    assert changes["openings_a"] == changes["openings_b"] == 1
    (door,) = changes["changed"]
    assert door["across"] == "x" and door["centre_m"][0] == pytest.approx(WALL_X_M, abs=FINE_M)
    assert (door["a"]["width_nodes"], door["b"]["width_nodes"]) == (18, 11)
    assert door["a"]["width_m"] == pytest.approx(0.9) and door["b"]["width_m"] == pytest.approx(
        0.88
    )
    # The layer of each grid nearest the height asked for: not one height.
    assert changes["layer_a_m"] != changes["layer_b_m"]
    classes, _ = cg.diff_slice(held, 1.2)
    assert set(np.unique(classes)) <= {0, 1, 2, 3, 4}
    assert int(np.count_nonzero(classes == 2)) == int(held.only_a[:, cg.layer_of(fine, 1.2)].sum())


def test_a_gap_that_seals_takes_the_room_behind_it(rooms: dict[str, Any]) -> None:
    fine, sealed = rooms["fine"], rooms["sealed"]
    numbers = cg.compare(fine, sealed).numbers
    (region,) = numbers["regions_air_only_in_a"]
    second_room = (ROOMS_M[0] - WALL_X_M) * ROOMS_M[1] * ROOMS_M[2]
    assert region["volume_m3"] == pytest.approx(second_room, rel=0.02)
    assert region["centre_m"] == pytest.approx([3.0, 1.2, 1.5], abs=0.05)
    assert numbers["regions_air_only_in_b"] == []
    assert numbers["air_b_m3"] == pytest.approx(numbers["air_a_m3"] - second_room, rel=0.05)
    changes = cg.opening_changes(fine, sealed, 1.2)
    (door,) = changes["changed"]
    assert door["a"]["width_nodes"] == 18 and door["b"] is None
    skin = cg.diff_surfaces(cg.compare(fine, sealed))
    assert skin["corners"].shape[0] > 0 and set(np.unique(skin["value"])) == {0}


def test_merging_squares_covers_them_exactly() -> None:
    rng = np.random.default_rng(1)
    mask = rng.random((3, 9, 11)) < 0.6
    value = rng.integers(0, 2, mask.shape)
    w, u, v = np.nonzero(mask)
    rw, u0, u1, v0, v1, rv = cg.merge_faces(w, u, v, value[mask])
    covered = np.zeros(mask.shape, dtype=int)
    for k in range(rw.size):
        covered[rw[k], u0[k] : u1[k] + 1, v0[k] : v1[k] + 1] += 1
        assert (value[rw[k], u0[k] : u1[k] + 1, v0[k] : v1[k] + 1] == rv[k]).all()
    assert np.array_equal(covered, mask.astype(int))
    assert rw.size < int(mask.sum())


def test_the_archive_holds_the_grid_the_arrays_and_the_sources(rooms: dict[str, Any]) -> None:
    grid, tmp = rooms["fine"], rooms["tmp"]
    lattice = load_grid(rooms["entry"])
    nodes, weights = snap_sources(SOURCE_M, lattice)
    array = grid.flat_of(grid.positions(np.array([nodes[0, 0]])) + grid.step_m * np.eye(3))
    path = save_as_computed(
        tmp / NAME,
        record={"voxel_low_key": grid.key, "solver": "test", "cells_in_pack": 1},
        grid=grid,
        cell_asked=np.array([[1.01, 1.2, 1.5], [9.0, 9.0, 9.0]]),
        cell_centre=np.array([[1.0, 1.2, 1.5], [np.nan] * 3]),
        cell_offsets=np.array([0, 3, 3]),
        cell_nodes=array,
        source_position=SOURCE_M,
        source_nodes=nodes,
        source_weights=weights,
    )
    # 335 104 nodes and 26 228 boundary nodes, in a few kilobytes.
    assert path.stat().st_size < 12_000
    held = read_as_computed(path)
    assert held.grid is not None
    assert np.array_equal(held.grid.reached, grid.reached)
    assert np.array_equal(held.grid.boundary, grid.boundary)
    assert np.array_equal(held.grid.adjacency, grid.adjacency)
    assert np.array_equal(held.grid.material, grid.material)
    assert held.grid.order == grid.order and held.grid.labels == grid.labels
    assert np.array_equal(held.nodes_of(0), array) and held.nodes_of(1).size == 0
    assert np.array_equal(held.source_nodes, nodes)
    (tmp / "other.npz").write_bytes(b"")
    np.savez(tmp / "other.npz", record=np.frombuffer(b'{"schema": "x"}', dtype=np.uint8))
    with pytest.raises(ValueError, match="not an as-computed archive"):
        read_as_computed(tmp / "other.npz")
    again = grid_record(rooms["entry"], SOURCE_M, key=grid.key, labels=grid.labels)
    assert np.array_equal(again.reached, grid.reached)


# --------------------------------------------------------------------------
# rays and paths
# --------------------------------------------------------------------------


def test_the_rays_traced_again_are_the_tracer_s_hit_for_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = box_scene()
    settings = RaySettings(rays=400, duration_s=0.08, skip_specular_order=1, skip_window_s=0.02)
    count = 12
    seen: list[int] = []
    real = rays_module._nearest_hit

    def watched(*args: Any) -> tuple[float, int]:
        distance, triangle = real(*args)
        seen.append(triangle)
        return distance, triangle

    monkeypatch.setattr(rays_module, "_nearest_hit", watched)
    histogram = trace(scene, SOURCE, RECEIVER[None, :], settings, ray_count=count)
    monkeypatch.undo()
    held = ray_paths(scene, SOURCE, settings, count=count, receivers=RECEIVER[None, :])
    # Every hit the tracer looked up, in its order: the same triangles, and the same ends.
    starts = held["offsets"]
    mine = [
        int(t) for ray in range(count) for t in held["triangle"][starts[ray] + 1 : starts[ray + 1]]
    ]
    assert mine == seen
    assert held["offsets"].size == count + 1 and held["truncated"] == 0
    assert np.allclose(held["points"][held["offsets"][:-1]], SOURCE)
    # The energy only falls, and the crossings counted are the histogram's hits.
    for ray in range(count):
        energy = held["energy"][held["offsets"][ray] : held["offsets"][ray + 1]]
        assert (np.diff(energy, axis=0) <= 1e-15).all() and (energy[0] == 1.0).all()
    assert len(held["crossings"]) == int(histogram.hits.sum())
    short = ray_paths(scene, SOURCE, settings, count=count, max_bounces=2)
    assert short["truncated"] > 0 and int(np.diff(short["offsets"]).max()) <= 3


def test_an_image_path_s_corners_lie_on_its_facets_and_reflect() -> None:
    normals = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    offsets = np.array([0.0, 3.0])
    source, listener = np.array([1.0, 1.0, 0.5]), np.array([2.0, 2.0, 0.0])
    corners = path_points(source, listener, np.array([0, 1, -1]), normals, offsets)
    assert corners.shape == (4, 3)
    assert np.allclose(corners[0], source) and np.allclose(corners[-1], listener)
    assert corners[1, 1] == pytest.approx(0.0) and corners[2, 0] == pytest.approx(3.0)
    # Its length is the distance from the listener to the source's image.
    image = source * [1, -1, 1]
    image = np.array([6.0 - image[0], image[1], image[2]])
    length = np.linalg.norm(np.diff(corners, axis=0), axis=1).sum()
    assert length == pytest.approx(np.linalg.norm(image - listener))
    direct = path_points(source, listener, np.array([-1, -1]), normals, offsets)
    assert np.allclose(direct, [source, listener])


# --------------------------------------------------------------------------
# the endpoints
# --------------------------------------------------------------------------


class Packs:
    """What the service asks of the audit: the packs it knows."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def packs(self) -> list[dict[str, Any]]:
        return [{"id": f"pack{k}", "path": str(p)} for k, p in enumerate(self.paths)]


def read(answer: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    assert isinstance(answer, Binary)
    spec = json.loads(answer.headers["X-Computed"])
    arrays = {
        a["name"]: np.frombuffer(
            answer.payload, dtype=a["dtype"], count=int(np.prod(a["shape"])), offset=a["offset"]
        ).reshape(a["shape"])
        for a in spec["arrays"]
    }
    return spec, arrays


def test_arrays_go_end_to_end_each_on_its_own_alignment() -> None:
    answer = packed(
        {"a": np.arange(3, dtype=np.uint8), "b": np.array([1.5, 2.5], dtype=np.float32)}, said=1
    )
    spec, arrays = read(answer)
    assert spec["said"] == 1 and spec["arrays"][1]["offset"] == 4
    assert arrays["a"].tolist() == [0, 1, 2] and arrays["b"].tolist() == [1.5, 2.5]


def test_the_page_is_told_what_a_pack_was_computed_on_or_why_not(
    rooms: dict[str, Any], tmp_path: Path
) -> None:
    fine, coarse = rooms["fine"], rooms["coarse"]
    homes = []
    for name in ("a", "b", "bare"):
        home = tmp_path / name / "pulled"
        home.mkdir(parents=True)
        write_pack(home / "pack.h5", synthetic_free_field(listener_start=(1.0, 1.2, 1.5)))
        homes.append(home / "pack.h5")
    for path, grid in ((homes[0], fine), (homes[1], coarse)):
        save_as_computed(
            path.parent / NAME,
            record={"voxel_low_key": grid.key, "solver": "test", "cells_in_pack": 1},
            grid=grid,
            cell_asked=np.array([[1.01, 1.2, 1.5]]),
            cell_centre=np.array([[1.0, 1.2, 1.5]]),
        )
    service = ComputedService(Packs(homes), tmp_path / "cache")  # type: ignore[arg-type]

    def ask(path: str, **query: Any) -> Any:
        return service.handle("GET", path.split("/"), {k: str(v) for k, v in query.items()})

    about = ask("about", pack="pack0")
    own = about["grid"]["own"]
    assert own == f"{fine.key}@pack0" and about["grid"]["why_none"] == ""
    assert about["snapped"]["from"] == f"{NAME} beside the pack"
    assert (
        about["mirror"]["directory"] is None
        and "not beside the pack" in about["mirror"]["why_none"]
    )
    bare = ask("about", pack="pack2")
    assert bare["grid"]["own"] is None and bare["grid"]["offered"] == []
    assert "names no low grid" in bare["grid"]["why_none"]
    assert bare["snapped"]["from"] is None

    facts = ask("grid/facts", pack="pack0", grid=own)
    assert facts["step_m"] == FINE_M and facts["nodes"] == fine.nodes
    spec, arrays = read(ask("grid/slice", pack="pack0", grid=own, y=1.2))
    assert spec["exact"] and arrays["classes"].shape == (fine.shape[2], fine.shape[0])
    assert np.array_equal(arrays["classes"], cg.slice_classes(fine, 1.2)[0])
    spec, arrays = read(ask("grid/walls", pack="pack0", grid=own, x=2, y=1.2, z=1.5, r=1))
    assert 0 < spec["quads"] == arrays["corners"].shape[0] and spec["faces"] < facts["cut_links"]
    # The second pack's own grid is asked for by its own name, from the first pack's page.
    other = f"{coarse.key}@pack1"
    numbers = ask("diff", pack="pack0", a=own, b=other)
    assert numbers["a"]["key"] == fine.key and numbers["b"]["key"] == coarse.key
    assert numbers["largest_wall_displacement_m"] <= 0.5 * COARSE_M + 1e-9
    changed = ask("diff/openings", pack="pack0", a=own, b=other, y=1.2)["changed"]
    assert [(row["a"]["width_nodes"], row["b"]["width_nodes"]) for row in changed] == [(18, 11)]
    _, arrays = read(ask("diff/slice", pack="pack0", a=own, b=other, y=1.2))
    assert arrays["classes"].shape == (fine.shape[2], fine.shape[0])
    refused: tuple[tuple[str, dict[str, Any]], ...] = (
        ("grid/slice", {"pack": "pack0", "grid": "nothing", "y": 1}),
        ("mirror/facets", {"pack": "pack0"}),
        ("about", {"pack": "none"}),
        ("nowhere", {}),
    )
    for asked, query in refused:
        with pytest.raises(SceneError, match="no |not on this machine"):
            ask(asked, **query)


# --------------------------------------------------------------------------
# what the trace brings home
# --------------------------------------------------------------------------


class Said:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, message: str) -> None:
        self.lines.append(message)


def a_trace(out: Path, engine: Any) -> Any:
    return SimpleNamespace(
        engine=engine,
        out=out,
        journal=Said(),
        asked=np.array([[1.01, 1.2, 1.5], [1.0, 1.2, 0.02]]),
        patch_cells=np.zeros((0, 3)),
        placed=np.array([True, False]),
        cells=np.array([[1.0, 1.2, 1.5], [1.0, 1.2, 0.02]]),
    )


def test_the_trace_leaves_what_it_computed_on_beside_the_pack(
    rooms: dict[str, Any], tmp_path: Path
) -> None:
    grid, lattice = rooms["fine"], load_grid(rooms["entry"])
    design = design_array(np.array([1.01, 1.2, 1.5]), lattice, fit_order=4, outer_radius_m=0.6)
    campaign = SimpleNamespace(
        entry_path=rooms["entry"],
        designs=[design, None],
        sources=SOURCE_M,
        cells=np.array([[1.01, 1.2, 1.5], [1.0, 1.2, 0.02]]),
    )
    engine = SimpleNamespace(campaign=campaign, voxel_low_key=grid.key, solver="a solver")
    told = write_as_computed(a_trace(tmp_path, engine))
    assert told["written"] and told["grid"] and told["bytes"] < 20_000
    held = read_as_computed(tmp_path / NAME)
    assert held.grid is not None and np.array_equal(held.grid.reached, grid.reached)
    assert held.record["voxel_low_key"] == grid.key and held.record["cells_in_pack"] == 2
    # The array's own nodes, in its order, and its centre a node of the grid.
    assert np.allclose(held.grid.positions(held.nodes_of(0)), design.positions)
    assert np.allclose(held.cell_centre[0], design.centre) and np.isnan(held.cell_centre[1]).all()
    assert held.nodes_of(1).size == 0
    spread = held.source_weights[0] @ held.grid.positions(held.source_nodes[0])
    assert np.allclose(spread, SOURCE_M[0])
    # Without a grid (the free field of the tests) the centres alone; and never a failure.
    bare = a_trace(tmp_path / "bare", SimpleNamespace(voxel_low_key="free-field", solver="none"))
    bare.out.mkdir()
    assert write_as_computed(bare)["grid"] is False
    held = read_as_computed(bare.out / NAME)
    assert held.grid is None and "not solved on a grid" in held.record["why_no_grid"]
    assert np.allclose(held.cell_centre[0], [1.0, 1.2, 1.5]) and np.isnan(held.cell_centre[1]).all()
    broken = a_trace(tmp_path / "nowhere", engine)
    assert write_as_computed(broken)["written"] is False
    assert broken.journal.lines and "not written" in broken.journal.lines[0]
