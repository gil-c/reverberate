"""Build a pyroomacoustics Room from reconstructed HSSD geometry and simulate it.

Follows the official ``examples/room_from_stl.py`` pattern: one
``pra.wall_factory`` call per triangle, for both the room shell and every
furniture obstacle, each with its own per-triangle absorption and scattering
coefficients. Uses the hybrid image source plus ray tracing mode required by
the roadmap (section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra
import trimesh

from reverberate.metrics import BandMetrics, measure


@dataclass
class MeshMaterialAssignment:
    """One mesh (room shell or one furniture obstacle) with its material."""

    mesh: trimesh.Trimesh
    material: pra.Material
    name: str = ""
    #: Set when the mesh was decimated and its absorption rescaled to keep the
    #: obstacle's absorbing power. Carried here so the viewer and the audit
    #: panel quote the simulator's own figures rather than recomputing them.
    compensation: object | None = None
    #: PFFDTD's per-face sidedness, derived once by
    #: :func:`reverberate.geometry.orientation.orient_for_air` and carried with
    #: the mesh so that an exporter never has to invent it. ``None`` only for
    #: assignments built by hand in a test or by an older caller.
    sides: np.ndarray | None = None


def walls_from_mesh(assignment: MeshMaterialAssignment) -> list[pra.wall.Wall]:
    """One pra wall per triangle, as in the official STL example."""
    absorption = np.asarray(assignment.material.energy_absorption["coeffs"])
    scattering = np.asarray(assignment.material.scattering["coeffs"])
    triangles = assignment.mesh.vertices[assignment.mesh.faces]  # (n_faces, 3, 3)
    return [
        pra.wall_factory(
            triangle.T,
            absorption,
            scattering,
            name=f"{assignment.name}_{i}" if assignment.name else "",
        )
        for i, triangle in enumerate(triangles)
    ]


HIT_TARGET = 600
"""Mean ray hits per histogram bin, which is what actually sets the noise floor.

pyroomacoustics sizes its own ray count from Vorlaender (2008, eq. 11.12),
``n = target * V / (pi r^2 c dt)``, and defaults ``target`` to 20. Twenty is
enough to draw a decay curve; it is not enough to *compare* two of them. On the
living room of scene 102344049, stripped to its 24-face shell so that seeds were
cheap, four seeds per setting gave, as spread of T30 across bands (the
acceptance threshold is 5%):

    target   20  ->    661 rays ->  6-14%   -- pyroomacoustics' default
    target   60  ->  2 000 rays ->    12%
    target  180  ->  6 000 rays ->   7.5%
    target  600  -> 20 000 rays ->   3.4%   -- this value
    target 1800  -> 60 000 rays ->   5.8%

The last line is not a regression, it is the floor of a four-seed estimate of a
standard deviation. 600 is the first setting that clears the threshold.
"""

RECEIVER_RADIUS = 1.0
"""Radius of the sphere that counts a ray as heard, in metres.

This is a sampling knob and not a physical one, which had to be measured rather
than assumed: the required ray count falls as ``1/r^2``, so a small receiver is
expensive, but a large one averages over space and could plausibly bias the
answer. It does not. Driven to 200 000 rays, radii of 0.15, 0.5 and 1.0 m agree
on T30 to within 3-6%, well inside the 5% threshold, so the radius is free to be
chosen for cost alone. Widening it from the 0.15 m used previously buys a factor
of 44 in rays for the same noise.

Below convergence the error is not symmetric. Starving the histogram truncates
the tail, so the decay looks steeper and T30 comes out *short*: the old setting
of 200 rays at 0.15 m reported 0.66 s where the converged answer is 1.3 s, an
underestimate of a factor of two that no amount of averaging would have revealed.
"""


def build_room(
    assignments: list[MeshMaterialAssignment],
    fs: int = 16000,
    max_order: int = 3,
    n_rays: int | None = None,
    receiver_radius: float = RECEIVER_RADIUS,
    hit_target: int = HIT_TARGET,
) -> pra.Room:
    """A room whose ray count is derived from its volume, not fixed by hand.

    ``n_rays`` is left ``None`` in normal use. A fixed count cannot be correct
    for every room, because the number needed scales with volume, and passing
    one silently makes the answer depend on room size. It is kept overridable
    only so experiments can sweep it.
    """
    walls: list[pra.wall.Wall] = []
    for assignment in assignments:
        walls.extend(walls_from_mesh(assignment))
    room = pra.Room(
        walls,
        fs=fs,
        max_order=max_order,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(n_rays=n_rays, receiver_radius=receiver_radius)
    if n_rays is None:
        # set_ray_tracing has just sized itself for 20 hits per bin, off this
        # room's own volume. Rescaling that is safer than recomputing the
        # formula here, which would mean duplicating pyroomacoustics' volume
        # estimate and its assumptions about what counts as enclosed.
        sized = int(room.rt_args["n_rays"] * hit_target / 20)
        room.set_ray_tracing(n_rays=sized, receiver_radius=receiver_radius)
    return room


@dataclass(frozen=True)
class PairResponse:
    """One source/receiver pair's impulse response and its per-band metrics."""

    source: np.ndarray
    receiver: np.ndarray
    rir: np.ndarray
    bands: BandMetrics


def simulate_pairs(
    assignments: list[MeshMaterialAssignment],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    fs: int = 16000,
    max_order: int = 2,
    n_rays: int | None = None,
    room: pra.Room | None = None,
    seed: int | None = None,
) -> list[PairResponse]:
    """Several source/receiver pairs against one room, built once.

    Construction is the dominant cost and it is paid per triangle: one
    ``pra.wall_factory`` call per face from a Python loop, measured at ~65 s of
    a ~114 s run on an apartment. That cost is independent of where the source
    and the receiver stand, so paying it once per *geometry* rather than once
    per *pair* is close to a free order of magnitude, and it is what makes a
    positional sweep affordable at all.

    Pairs are kept paired: pyroomacoustics computes a response for every
    source/microphone combination, so ``rir[i][j]`` is microphone ``i`` hearing
    source ``j`` and only the diagonal is asked for. The off-diagonal responses
    are computed and discarded, which is still far cheaper than rebuilding the
    walls, but it does mean the ray tracing cost grows with the number of
    pairs even though the construction cost does not.

    **The ray tracer is stochastic and its randomness is global.** Diffuse
    reflection draws from one process-wide generator inside ``libroom``, so two
    runs of the identical scene return different responses, and adding a second
    source shifts the draws the first one gets. ``seed`` pins it. That matters
    beyond reproducibility: because adding a pair perturbs the other pairs'
    draws, one sweep is comparable to another only at the same seed and with
    the same pairs in the same order.

    Seeding bounds the scatter but does not remove it, so a comparison between
    two geometries is meaningful only once the residual spread is below the
    threshold being tested. That is what ``build_room``'s ray sizing is for,
    and it is why ``n_rays`` defaults to ``None`` here rather than to a
    constant.

    Pass ``room`` to reuse a room that has already been built *and has no
    source or microphone attached yet*; otherwise one is built from
    ``assignments``.
    """
    if not pairs:
        return []
    if seed is not None:
        pra.random.seed(seed)
    if room is None:
        room = build_room(assignments, fs=fs, max_order=max_order, n_rays=n_rays)
    for source, _ in pairs:
        room.add_source(np.asarray(source, dtype=float))
    room.add_microphone_array(np.array([np.asarray(mic, dtype=float) for _, mic in pairs]).T)
    room.image_source_model()
    room.ray_tracing()
    room.compute_rir()

    responses = []
    for index, (source, receiver) in enumerate(pairs):
        response = np.asarray(room.rir[index][index], dtype=float)
        responses.append(
            PairResponse(
                source=np.asarray(source, dtype=float),
                receiver=np.asarray(receiver, dtype=float),
                rir=response,
                bands=measure(response, fs),
            )
        )
    return responses


@dataclass
class SimulationResult:
    rt60_broadband: float
    room_volume: float
    room_surface_area: float
    mean_absorption: float
    sabine_rt60: float
    eyring_rt60: float
    #: Every per-octave-band measure of the response, from 125 Hz to 8 kHz.
    #: The broadband figure above is kept only for continuity with the
    #: Sabine/Eyring guard, which is itself a broadband estimate; it is not the
    #: quantity this project predicts, and it can average away a large error in
    #: one band against an opposite one in another.
    bands: BandMetrics | None = None
    #: The impulse response itself, so a caller can compare two simulations
    #: without re-running either.
    rir: np.ndarray | None = None


def sabine_rt60(volume: float, surface_area: float, mean_absorption: float) -> float:
    """Sabine formula, seconds. Guard against zero/near-total absorption."""
    if mean_absorption <= 0:
        return float("inf")
    return 0.161 * volume / (surface_area * mean_absorption)


def eyring_rt60(volume: float, surface_area: float, mean_absorption: float) -> float:
    """Eyring formula, seconds. More accurate than Sabine at high absorption."""
    if mean_absorption >= 1.0:
        return 0.0
    denom = -surface_area * np.log(1.0 - mean_absorption)
    if denom <= 0:
        return float("inf")
    return 0.161 * volume / float(denom)


def simulate_and_validate(
    assignments: list[MeshMaterialAssignment],
    source: np.ndarray,
    mic: np.ndarray,
    fs: int = 16000,
    max_order: int = 3,
    n_rays: int = 10000,
    room_volume: float | None = None,
) -> SimulationResult:
    """Run the hybrid ISM + ray tracing simulation and compute the Sabine/Eyring guard.

    ``room_volume`` should be passed explicitly whenever the shell is split
    across more than one ``MeshMaterialAssignment`` (for example floor, wall
    and ceiling assigned separately so each can carry its own material): each
    part's own ``.mesh.volume`` is not the enclosed air volume in that case,
    only the volume of a thin, degenerate slab. When omitted, the volume of
    ``assignments[0].mesh`` is used, which is only correct when the first
    assignment is the single, complete, watertight shell.
    """
    room = build_room(assignments, fs=fs, max_order=max_order, n_rays=n_rays)
    room.add_source(source)
    room.add_microphone_array(np.c_[mic])
    room.image_source_model()
    room.ray_tracing()
    room.compute_rir()

    rt60 = float(room.measure_rt60()[0, 0])
    response = np.asarray(room.rir[0][0], dtype=float)
    band_metrics = measure(response, fs)

    total_area = 0.0
    weighted_absorption = 0.0
    for assignment in assignments:
        area = float(assignment.mesh.area)
        mean_coeff = float(np.mean(assignment.material.energy_absorption["coeffs"]))
        total_area += area
        weighted_absorption += area * mean_coeff
    mean_absorption = weighted_absorption / total_area if total_area else 0.0

    if room_volume is not None:
        volume = room_volume
    else:
        # Fallback for the common single-mesh-shell case (roadmap 5.1's own
        # example): the first assignment, by convention, carries the whole
        # enclosed shell. Callers with a split shell must pass room_volume.
        volume = float(assignments[0].mesh.volume) if assignments else 0.0

    return SimulationResult(
        rt60_broadband=rt60,
        room_volume=volume,
        room_surface_area=total_area,
        mean_absorption=mean_absorption,
        sabine_rt60=sabine_rt60(volume, total_area, mean_absorption),
        eyring_rt60=eyring_rt60(volume, total_area, mean_absorption),
        bands=band_metrics,
        rir=response,
    )
