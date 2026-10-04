"""The batched trace of a moving scene against the present pipeline, one position at a time.

At rest the table must be the pipeline's: the same paths in the same order,
so its own renderer gives the same samples. Moving, every step must hold
what the pipeline finds for that step's source and listener alone, whatever
anchors, short lists and blocks the batch went through; a path must keep its
identity from step to step; and the departure direction must point at the
path's first reflection.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from reverberate.mirror.diffract import diffracted_paths
from reverberate.mirror.geometry import DerivedScene, Facet
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, occluder_grid, paths_for
from reverberate.mirror.moving import (
    KIND_DIFFRACTED,
    KIND_DIFFRACTED_REFLECTED,
    KIND_DIRECT,
    KIND_SPECULAR,
    MovingScene,
    MovingSettings,
    grow_candidates,
    path_id,
    prepare,
    render_early,
    trace_early,
)
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.parameters import Parameters, regain
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.render import RenderSettings, early_signals
from test_mirror_diffract import walled_box
from test_mirror_ism import box_scene

C = 343.2
SETTINGS = MirrorSettings(
    ism=IsmSettings(max_order=3, flutter_order=6),
    parameters=Parameters(image_absorption_scale=(1.5, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3)),
)
RENDER = RenderSettings(order=3, duration_s=0.2)
SOURCE = np.array([1.0, 1.2, 0.8])


def present(
    ms: MovingScene,
    source: np.ndarray,
    receiver: np.ndarray,
    region: tuple[np.ndarray, np.ndarray] | None,
) -> Paths:
    """What ``pipeline.trace`` holds for one point: the twin's paths, with the images' gains."""
    tree = grow_tree(ms.scene, source, ms.ism, region=region)
    found = paths_for(ms.scene, tree, receiver, ms.ism, grid=occluder_grid(ms.scene))
    return regain(found, ms.images)


def tilted(scene: DerivedScene) -> tuple[DerivedScene, np.ndarray]:
    """The scene turned off the axes, and the rotation."""
    a, b = 0.3, 0.2
    turn = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ np.array(
        [[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]
    )
    facets = tuple(replace(f, normal=turn @ f.normal) for f in scene.facets)
    occluders = scene.occluder_vertices @ turn.T
    return (
        replace(
            scene,
            facets=facets,
            reflector_vertices=scene.reflector_vertices @ turn.T,
            occluder_vertices=occluders,
            bmin=occluders.reshape(-1, 3).min(axis=0),
            bmax=occluders.reshape(-1, 3).max(axis=0),
        ),
        turn,
    )


def test_at_rest_the_table_renders_the_pipeline_s_early_part_sample_for_sample() -> None:
    """Source on a station, listener on lattice points, some in view and some behind the wall."""
    ms = prepare(walled_box(), SETTINGS)
    lattice = np.array(
        [[x, 1.2, z] for x in (0.6, 1.4, 2.6, 3.4) for z in (0.6, 1.4, 2.2)], dtype=float
    )
    region = (lattice.min(axis=0) - 0.5, lattice.max(axis=0) + 0.5)
    table = trace_early(
        ms, np.repeat(SOURCE[None, :], len(lattice), axis=0), lattice, region=region
    )
    shadowed = 0
    for step, receiver in enumerate(lattice):
        want = present(ms, SOURCE, receiver, region)
        mine, onset = table.paths(step)
        assert onset is None
        np.testing.assert_array_equal(mine.sequence, want.sequence)
        np.testing.assert_array_equal(mine.length_m, want.length_m)
        np.testing.assert_array_equal(mine.direction, want.direction)
        np.testing.assert_array_equal(mine.gain, want.gain)
        np.testing.assert_array_equal(
            render_early(table, step, RENDER), early_signals(want, RENDER, C)
        )
        shadowed += int(not np.any(want.order == 0))
    assert 0 < shadowed < len(lattice)
    assert table.record["without_direct"] == shadowed


def test_at_rest_a_shadowed_point_has_the_pipeline_s_diffracted_onset_too() -> None:
    catalogue = walled_box()
    ms = prepare(catalogue, SETTINGS)
    for receiver in (np.array([3.0, 1.2, 0.8]), np.array([3.4, 1.9, 1.1])):
        region = (receiver - 0.5, receiver + 0.5)
        want = present(ms, SOURCE, receiver, region)
        assert not np.any(want.order == 0)
        onsets, _ = diffracted_paths(catalogue, SOURCE, receiver[None, :], [0], sound_speed_m_s=C)
        held = onset_field(catalogue, np.stack([SOURCE, receiver]), sound_speed_m_s=C)
        table = trace_early(ms, SOURCE[None, :], receiver[None, :], region=region, onsets=held)
        _, onset = table.paths(0)
        assert onset is not None and onset.count == onsets[0].count > 1
        np.testing.assert_array_equal(onset.length_m, onsets[0].length_m)
        np.testing.assert_array_equal(onset.gain, onsets[0].gain)
        np.testing.assert_array_equal(onset.order, onsets[0].order)
        both = early_signals(want, RENDER, C) + early_signals(onsets[0], RENDER, C)
        np.testing.assert_array_equal(render_early(table, 0, RENDER), both)
        kinds = set(table.kind.tolist())
        assert kinds == {KIND_SPECULAR, KIND_DIFFRACTED, KIND_DIFFRACTED_REFLECTED}
        # A diffracted path leaves the source towards the doorway, not through the wall.
        diffracted = table.kind == KIND_DIFFRACTED
        assert np.all(table.departure[diffracted][:, 2] > 0.3)
        np.testing.assert_allclose(np.linalg.norm(table.departure, axis=1), 1.0, atol=1e-12)


def test_off_the_axes_the_render_is_the_pipeline_s_sample_for_sample() -> None:
    """Planes that are not axis aligned: no product is exact, and the two still agree to the bit.

    The tilted box keeps the box's degenerate legs, which graze a facet's
    edge to a unit in the last place. Both codes take every scalar product
    through :func:`reverberate.mirror.ism.dot3`, so such a leg is a path for
    both or for neither, whatever the machine's linear algebra library.
    """
    catalogue, turn = tilted(walled_box())
    ms = prepare(catalogue, SETTINGS)
    source = turn @ SOURCE
    for point in ([1.5, 1.0, 1.8], [3.0, 1.2, 0.8]):
        receiver = turn @ np.asarray(point)
        region = (receiver - 0.5, receiver + 0.5)
        want = present(ms, source, receiver, region)
        signals = early_signals(want, RENDER, C)
        held = onset_field(catalogue, np.stack([source, receiver]), sound_speed_m_s=C)
        if not np.any(want.order == 0):
            onsets, _ = diffracted_paths(
                catalogue, source, receiver[None, :], [0], sound_speed_m_s=C
            )
            signals = signals + early_signals(onsets[0], RENDER, C)
        table = trace_early(ms, source[None, :], receiver[None, :], region=region, onsets=held)
        mine, _ = table.paths(0)
        np.testing.assert_array_equal(mine.sequence, want.sequence)
        np.testing.assert_array_equal(mine.length_m, want.length_m)
        np.testing.assert_array_equal(render_early(table, 0, RENDER), signals)


def a_walk(steps: int) -> tuple[np.ndarray, np.ndarray]:
    """A source crossing its room and a listener going through the doorway and back."""
    u = np.linspace(0.0, 1.0, steps)[:, None]
    source = np.array([0.4, 1.2, 0.5]) + u * np.array([1.2, 0.3, 1.6])
    listener = np.array([1.5, 1.4, 2.25]) + np.sin(np.pi * u) * np.array([1.9, 0.2, -0.9])
    return source, listener


def test_every_step_of_a_walk_holds_what_the_pipeline_finds_there() -> None:
    catalogue = walled_box()
    # Small anchors and blocks, so the walk crosses many cells and many blocks.
    moving = MovingSettings(
        source_pitch_m=0.3, listener_pitch_m=0.3, pairs_per_block=600, pairs_per_validation=200
    )
    ms = prepare(catalogue, SETTINGS, moving)
    source, listener = a_walk(30)
    both = np.concatenate([source, listener])
    region = (both.min(axis=0) - 0.5, both.max(axis=0) + 0.5)
    table = trace_early(ms, source, listener, region=region)
    assert table.record["anchors"] > 3 and table.record["anchor_pairs"] > 10
    # The sieve leaves the validation a small part of what the short lists hold.
    assert table.record["pairs"] < table.record["sieved"]
    for step in range(table.steps):
        want = present(ms, source[step], listener[step], region)
        mine, _ = table.paths(step)
        np.testing.assert_array_equal(mine.sequence, want.sequence)
        np.testing.assert_allclose(mine.length_m, want.length_m, rtol=0, atol=1e-12)
        np.testing.assert_allclose(mine.gain, want.gain, rtol=1e-12)
        rows = table.rows(step)
        ranked = np.argsort(table.rank[rows], kind="stable")
        first = (
            np.where((want.order > 0)[:, None], want.points[:, 1], listener[step][None, :])
            - source[step]
        )
        np.testing.assert_allclose(
            table.departure[rows][ranked],
            first / np.linalg.norm(first, axis=1, keepdims=True),
            atol=1e-12,
        )


def test_a_path_keeps_its_identity_and_the_table_is_the_pack_s() -> None:
    ms = prepare(walled_box(), SETTINGS)
    source, listener = a_walk(40)
    audible = np.ones(40, dtype=bool)
    audible[10:15] = False
    table = trace_early(ms, source, listener, audible=audible)
    assert table.offsets.shape == (41,) and table.offsets[0] == 0
    assert np.all(np.diff(table.offsets) >= 0) and table.offsets[-1] == table.path_id.size
    assert np.all(np.diff(table.offsets)[10:15] == 0)
    for step in range(table.steps):
        ids = table.path_id[table.rows(step)]
        assert np.all(ids[1:] > ids[:-1])
    # An identity is its kind and its facets: the same sequence, the same number, at any step.
    names: dict[int, tuple[int, ...]] = {}
    for identity, order, sequence in zip(table.path_id, table.order, table.sequence, strict=True):
        facets = tuple(int(f) for f in sequence[:order])
        assert names.setdefault(int(identity), facets) == facets
    assert len(set(names.values())) == len(names)
    direct = int.from_bytes(hashlib.sha256(bytes([0])).digest()[:8], "little")
    assert path_id(KIND_DIRECT) == direct
    assert np.all(table.path_id[table.kind == KIND_DIRECT] == direct)
    np.testing.assert_allclose(table.delay_s, table.length_m / C)
    packed = table.pack()
    assert packed["path_id"].dtype == np.uint64 and packed["delay_s"].dtype == np.float64
    assert packed["arrival"].dtype == packed["departure"].dtype == np.float32
    assert packed["gain"].dtype == np.float32 and packed["gain"].shape[1] == 7
    assert packed["order"].dtype == packed["kind"].dtype == np.uint8
    np.testing.assert_allclose(np.linalg.norm(packed["arrival"], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(packed["departure"], axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(packed["gain"])) and np.all(packed["gain"] >= 0.0)


def test_positions_met_before_are_not_traced_again_and_blocks_do_not_change_the_answer() -> None:
    catalogue = walled_box()
    source, listener = a_walk(12)
    # Out and back: every position twice.
    source = np.concatenate([source, source[::-1]])
    listener = np.concatenate([listener, listener[::-1]])
    whole = trace_early(prepare(catalogue, SETTINGS), source, listener)
    assert whole.record["jobs"] == 12
    small = MovingSettings(pairs_per_block=300, pairs_per_validation=50, source_pitch_m=0.2)
    again = trace_early(prepare(catalogue, SETTINGS, small), source, listener)
    np.testing.assert_array_equal(whole.offsets, again.offsets)
    np.testing.assert_array_equal(whole.path_id, again.path_id)
    np.testing.assert_array_equal(whole.length_m, again.length_m)
    np.testing.assert_array_equal(whole.departure, again.departure)
    for step in range(12):
        np.testing.assert_array_equal(
            whole.length_m[whole.rows(step)], whole.length_m[whole.rows(23 - step)]
        )


def test_the_candidates_hold_the_tree_of_every_source_of_their_cell() -> None:
    scene = box_scene(obstacle=(np.array([2.0, 0.5, 0.5]), np.array([2.0, 2.5, 2.0])))
    # A one sided reflector hung in the room, so the air side test has something to refuse.
    hung = Facet(
        label=0,
        normal=np.array([0.0, 0.0, 1.0]),
        offset=1.25,
        area=1.0,
        triangles=np.array([12, 13], dtype=np.int32),
        kind="furniture",
    )
    panel = np.array(
        [
            [[1.0, 1.0, 1.25], [2.0, 1.0, 1.25], [2.0, 2.0, 1.25]],
            [[1.0, 1.0, 1.25], [2.0, 2.0, 1.25], [1.0, 2.0, 1.25]],
        ]
    )
    scene = replace(
        scene,
        facets=(*scene.facets, hung),
        reflector_vertices=np.concatenate([scene.reflector_vertices, panel]),
    )
    ms = prepare(scene, SETTINGS)
    anchor = np.array([1.5, 1.5, 1.25])
    slack = 0.3
    candidates = grow_candidates(ms, anchor, slack)
    held = {tuple(row) for row in candidates.sequence.tolist()}
    rng = np.random.default_rng(3)
    fewest = 10**9
    for _ in range(6):
        offset = rng.normal(size=3)
        source = anchor + slack * rng.random() * offset / np.linalg.norm(offset)
        tree = grow_tree(ms.scene, source, ms.ism)
        mine = {tuple(row) for row in tree.sequence.tolist()}
        assert mine <= held
        fewest = min(fewest, len(mine))
        # An image is an affine function of its source.
        index = {tuple(row): k for k, row in enumerate(candidates.sequence.tolist())}
        rows = np.array([index[tuple(row)] for row in tree.sequence.tolist()])
        moved = np.einsum("nij,j->ni", candidates.rotation[rows], source)
        np.testing.assert_allclose(moved + candidates.translation[rows], tree.positions, atol=1e-9)
    # Under the panel its air side is not the source's: that tree is smaller.
    assert fewest < candidates.count
    with pytest.raises(ValueError, match="must be"):
        trace_early(ms, np.zeros((2, 3)), np.zeros((3, 3)))
