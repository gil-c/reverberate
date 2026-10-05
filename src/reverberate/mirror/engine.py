"""The paths of every receiver and the rays of the tail, on whatever :class:`Devices` holds.

On cards, the kernels of :mod:`reverberate.mirror.kernels`; on the host, the
numpy code of :mod:`reverberate.mirror.ism` and :mod:`reverberate.mirror.rays`,
which is their specification. Receivers are split over the devices for the
paths, rays for the histogram, and results are merged in share order: a
receiver's paths do not depend on which device validated them and the
histogram is a sum of integers. The paths and the histogram are the same on
the host and on any number of cards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.compute import Devices, raw_kernel
from reverberate.mirror import tracer
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import (
    ImageTree,
    IsmSettings,
    Paths,
    _facet_arrays,
    _gains,
    occluder_grid,
    paths_for,
)
from reverberate.mirror.kernels import PATHS_KERNEL, RAYS_KERNEL
from reverberate.mirror.rays import (
    HISTOGRAM_SCALE,
    Histogram,
    RaySettings,
    UniformGrid,
    grid_like,
    reflector_of_triangles,
    trace,
)
from reverberate.mirror.shared import Store
from reverberate.mirror.tracer import STATS, TracerScene
from reverberate.spatial.sh import channel_count

__all__ = ["DeviceScene", "histogram_on_devices", "paths_on_devices", "upload"]


@dataclass
class DeviceScene:
    """The scene, its grid and the tree as one device holds them."""

    device: int
    arrays: dict[str, Any]
    grid: UniformGrid
    facet_start: np.ndarray
    facet_count: np.ndarray


def _facet_ranges(scene: DerivedScene) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(
        [int(f.triangles[0]) if f.triangles.size else 0 for f in scene.facets], dtype=np.int32
    )
    count = np.asarray([int(f.triangles.size) for f in scene.facets], dtype=np.int32)
    return start, count


#: The facets' own buckets aim at this many triangles a cell, at most this many cells a side.
_BUCKET_LOAD = 4.0
_BUCKET_SIDE = 512
_BUCKET_PAD_M = 1e-6


def facet_buckets(
    scene: DerivedScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per facet, a uniform grid in its plane of the triangles whose padded boxes meet each cell.

    Returns ``frame [facet, 9]`` (origin u, origin v, cell, axis u, axis v),
    ``shape [facet, 2]``, ``base [facet]`` (the facet's first bucket),
    ``offsets [bucket + 1]`` and ``members`` (global reflector triangle
    indices). The probe of the paths kernel runs along the facet's normal,
    so a triangle it hits has the crossing's plane coordinates inside its
    box: the bucket of the crossing holds every such triangle.
    """
    count = len(scene.facets)
    frame = np.zeros((count, 9))
    shape = np.ones((count, 2), dtype=np.int32)
    base = np.zeros(count, dtype=np.int64)
    bucket_lists: list[np.ndarray] = []
    bucket_counts: list[np.ndarray] = []
    total = 0
    normals, _, _, _ = _facet_arrays(scene)
    for f, facet in enumerate(scene.facets):
        n = np.asarray(normals[f], dtype=float)
        helper = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        indices = np.asarray(facet.triangles, dtype=np.int64)
        tris = scene.reflector_vertices[indices] if indices.size else np.zeros((0, 3, 3))
        pu = tris @ u
        pv = tris @ v
        if indices.size:
            u_lo, u_hi = pu.min(axis=1) - _BUCKET_PAD_M, pu.max(axis=1) + _BUCKET_PAD_M
            v_lo, v_hi = pv.min(axis=1) - _BUCKET_PAD_M, pv.max(axis=1) + _BUCKET_PAD_M
            origin_u, origin_v = float(u_lo.min()), float(v_lo.min())
            extent_u = float(u_hi.max()) - origin_u
            extent_v = float(v_hi.max()) - origin_v
            cells = max(indices.size / _BUCKET_LOAD, 1.0)
            cell = max(np.sqrt(max(extent_u * extent_v, 1e-12) / cells), 1e-3)
            cell = max(cell, extent_u / _BUCKET_SIDE, extent_v / _BUCKET_SIDE)
            su = int(min(max(np.ceil(extent_u / cell), 1), _BUCKET_SIDE))
            sv = int(min(max(np.ceil(extent_v / cell), 1), _BUCKET_SIDE))
        else:
            origin_u = origin_v = 0.0
            cell = 1.0
            su = sv = 1
        frame[f] = [origin_u, origin_v, cell, *u, *v]
        shape[f] = (su, sv)
        base[f] = total
        grid_count = np.zeros(su * sv, dtype=np.int64)
        members = np.zeros(0, dtype=np.int64)
        if indices.size:
            iu0 = np.clip(np.floor((u_lo - origin_u) / cell), 0, su - 1).astype(np.int64)
            iu1 = np.clip(np.floor((u_hi - origin_u) / cell), 0, su - 1).astype(np.int64)
            iv0 = np.clip(np.floor((v_lo - origin_v) / cell), 0, sv - 1).astype(np.int64)
            iv1 = np.clip(np.floor((v_hi - origin_v) / cell), 0, sv - 1).astype(np.int64)
            wide = iu1 - iu0 + 1
            tall = iv1 - iv0 + 1
            spans = wide * tall
            owner = np.repeat(np.arange(indices.size), spans)
            rank = np.arange(owner.size) - np.repeat(np.cumsum(spans) - spans, spans)
            cu = iu0[owner] + rank // tall[owner]
            cv = iv0[owner] + rank % tall[owner]
            flat = cu * sv + cv
            order = np.argsort(flat, kind="stable")
            members = indices[owner[order]]
            grid_count = np.bincount(flat, minlength=su * sv)
        bucket_lists.append(members)
        bucket_counts.append(grid_count)
        total += su * sv
    counts = np.concatenate(bucket_counts) if bucket_counts else np.zeros(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    members = (
        np.concatenate(bucket_lists).astype(np.int32) if bucket_lists else np.zeros(0, np.int32)
    )
    return frame, shape, base, offsets, members


def upload(
    scene: DerivedScene,
    tree: ImageTree,
    *,
    device: int = 0,
    grid: UniformGrid | None = None,
) -> DeviceScene:
    """Everything the kernels read, on ``device``."""
    import cupy

    grid = grid or occluder_grid(scene)
    normals, offsets, _, _ = _facet_arrays(scene)
    facet_start, facet_count = _facet_ranges(scene)
    frame, shape, base, bucket_offsets, bucket_members = facet_buckets(scene)
    tri_reflector, tri_furniture = reflector_of_triangles(scene)
    with cupy.cuda.Device(device):
        arrays = {
            "tree_pos": cupy.asarray(np.ascontiguousarray(tree.positions, dtype=np.float64)),
            "tree_order": cupy.asarray(np.ascontiguousarray(tree.order, dtype=np.int32)),
            "tree_parent": cupy.asarray(np.ascontiguousarray(tree.parent, dtype=np.int32)),
            "tree_seq": cupy.asarray(np.ascontiguousarray(tree.sequence, dtype=np.int32)),
            "facet_normal": cupy.asarray(np.ascontiguousarray(normals, dtype=np.float64)),
            "facet_offset": cupy.asarray(np.ascontiguousarray(offsets, dtype=np.float64)),
            "facet_frame": cupy.asarray(np.ascontiguousarray(frame, dtype=np.float64)),
            "facet_shape": cupy.asarray(np.ascontiguousarray(shape, dtype=np.int32)),
            "facet_base": cupy.asarray(np.ascontiguousarray(base, dtype=np.int64)),
            "bucket_offsets": cupy.asarray(bucket_offsets),
            "bucket_members": cupy.asarray(bucket_members),
            "reflector_tris": cupy.asarray(
                np.ascontiguousarray(scene.reflector_vertices.reshape(-1, 9), dtype=np.float64)
            ),
            "occ_tris": cupy.asarray(
                np.ascontiguousarray(scene.occluder_vertices.reshape(-1, 9), dtype=np.float64)
            ),
            "tri_label": cupy.asarray(np.ascontiguousarray(scene.occluder_label, dtype=np.int16)),
            "tri_reflector": cupy.asarray(np.ascontiguousarray(tri_reflector, dtype=np.int32)),
            "tri_furniture": cupy.asarray(np.ascontiguousarray(tri_furniture, dtype=np.int32)),
            "grid_origin": cupy.asarray(np.asarray(grid.origin, dtype=np.float64)),
            "grid_shape": cupy.asarray(np.asarray(grid.shape, dtype=np.int32)),
            "cell_offsets": cupy.asarray(np.asarray(grid.offsets, dtype=np.int64)),
            "cell_members": cupy.asarray(np.asarray(grid.members, dtype=np.int32)),
            "absorption": cupy.asarray(
                np.ascontiguousarray(scene.materials.absorption, dtype=np.float64)
            ),
            "scattering": cupy.asarray(
                np.ascontiguousarray(scene.materials.scattering, dtype=np.float64)
            ),
            "source": cupy.asarray(np.asarray(tree.source, dtype=np.float64)),
        }
    return DeviceScene(device, arrays, grid, facet_start, facet_count)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def _paths_kernel_args(held: DeviceScene, tree: ImageTree) -> tuple[Any, ...]:
    a = held.arrays
    return (
        a["tree_pos"],
        a["tree_order"],
        a["tree_parent"],
        a["tree_seq"],
        np.int32(tree.sequence.shape[1]),
        a["facet_normal"],
        a["facet_offset"],
        a["facet_frame"],
        a["facet_shape"],
        a["facet_base"],
        a["bucket_offsets"],
        a["bucket_members"],
        a["reflector_tris"],
        a["occ_tris"],
        a["grid_origin"],
        np.float64(held.grid.cell_m),
        a["grid_shape"],
        a["cell_offsets"],
        a["cell_members"],
    )


def paths_on_device(
    scene: DerivedScene,
    tree: ImageTree,
    receivers: np.ndarray,
    settings: IsmSettings,
    held: DeviceScene,
    *,
    threads: int = 128,
    pairs_per_launch: int = 8_000_000,
    say: Any = None,
) -> list[Paths]:
    """The paths of every receiver, validated on one device; one :class:`Paths` per receiver."""
    import cupy

    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    n_images = tree.count
    width = tree.sequence.shape[1]
    kernel = raw_kernel(PATHS_KERNEL, "paths")
    out: list[Paths] = []
    with cupy.cuda.Device(held.device):
        recv = cupy.asarray(np.ascontiguousarray(receivers, dtype=np.float64))
        static = _paths_kernel_args(held, tree)
        per_launch = max(1, pairs_per_launch // n_images)
        for first in range(0, receivers.shape[0], per_launch):
            chunk = min(per_launch, receivers.shape[0] - first)
            npairs = chunk * n_images
            pair_image = cupy.tile(cupy.arange(n_images, dtype=cupy.int32), chunk)
            pair_receiver = cupy.repeat(
                cupy.arange(first, first + chunk, dtype=cupy.int32), n_images
            )
            valid = cupy.zeros(npairs, dtype=cupy.uint8)
            blocks = (npairs + threads - 1) // threads
            t0 = time.time()
            kernel(
                (blocks,),
                (threads,),
                (
                    *static,
                    recv,
                    held.arrays["source"],
                    pair_image,
                    pair_receiver,
                    np.int32(npairs),
                    np.float64(settings.epsilon_m),
                    valid,
                    valid,  # no points on this pass
                    np.int32(0),
                ),
            )
            cupy.cuda.Stream.null.synchronize()
            kept = cupy.flatnonzero(valid)
            kept_image = pair_image[kept]
            kept_receiver = pair_receiver[kept]
            points = cupy.zeros((int(kept.size), width + 2, 3), dtype=cupy.float64)
            if kept.size:
                blocks = (int(kept.size) + threads - 1) // threads
                kernel(
                    (blocks,),
                    (threads,),
                    (
                        *static,
                        recv,
                        held.arrays["source"],
                        kept_image,
                        kept_receiver,
                        np.int32(int(kept.size)),
                        np.float64(settings.epsilon_m),
                        cupy.zeros(int(kept.size), dtype=cupy.uint8),
                        points,
                        np.int32(1),
                    ),
                )
                cupy.cuda.Stream.null.synchronize()
            if say is not None:
                say(
                    f"paths: receivers {first}..{first + chunk} on device {held.device},"
                    f" {int(kept.size)} of {npairs} pairs valid, {time.time() - t0:.1f} s"
                )
            host_image = cupy.asnumpy(kept_image)
            host_receiver = cupy.asnumpy(kept_receiver)
            host_points = cupy.asnumpy(points)
            for r in range(first, first + chunk):
                mine = np.flatnonzero(host_receiver == r)
                out.append(
                    _paths_of(scene, tree, receivers[r], host_image[mine], host_points[mine])
                )
    return out


def _paths_of(
    scene: DerivedScene,
    tree: ImageTree,
    receiver: np.ndarray,
    images: np.ndarray,
    points: np.ndarray,
) -> Paths:
    """A :class:`Paths` from the images the card validated, in image order as the twin."""
    order = np.argsort(images, kind="stable")
    images = images[order].astype(np.int32)
    points = points[order]
    lengths = np.linalg.norm(tree.positions[images] - receiver[None, :], axis=1)
    direction = (tree.positions[images] - receiver[None, :]) / np.maximum(lengths, 1e-12)[:, None]
    return Paths(
        receiver=np.asarray(receiver, dtype=float),
        image=images,
        order=tree.order[images],
        length_m=lengths,
        direction=direction,
        gain=_gains(scene, tree.sequence[images], lengths),
        points=points,
        sequence=tree.sequence[images],
    )


def paths_on_devices(
    scene: DerivedScene,
    tree: ImageTree,
    receivers: np.ndarray,
    settings: IsmSettings | None = None,
    *,
    devices: Devices | None = None,
    grid: UniformGrid | None = None,
    say: Any = None,
) -> list[Paths]:
    """The paths of every receiver, receivers split over the devices."""
    settings = settings or IsmSettings()
    devices = devices or Devices.detect()
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    grid = grid or occluder_grid(scene)

    def work(card: int, share: np.ndarray) -> list[Paths]:
        if card < 0:
            return [paths_for(scene, tree, r, settings, grid=grid) for r in receivers[share]]
        held = upload(scene, tree, device=card, grid=grid)
        return paths_on_device(scene, tree, receivers[share], settings, held, say=say)

    return [p for part in devices.map(work, devices.split(receivers.shape[0])) for p in part]


# --------------------------------------------------------------------------
# rays
# --------------------------------------------------------------------------


def histogram_on_device(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings,
    held: DeviceScene,
    *,
    ray_start: int,
    ray_count: int,
    threads: int = 128,
    rays_per_launch: int = 200_000,
    say: Any = None,
    stats: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The integer histogram of rays ``ray_start`` to ``ray_start + ray_count`` on one device."""
    import cupy

    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    bands = scene.materials.absorption.shape[1]
    channels = channel_count(settings.order)
    bins = int(np.ceil(settings.duration_s / settings.bin_s))
    reach = settings.sound_speed_m_s * settings.duration_s
    radius = settings.receiver_radius_m
    rec_grid = grid_like(held.grid, receivers - radius, receivers + radius)
    if rec_grid.shape != held.grid.shape:
        raise RuntimeError("the receiver grid must share the occluder grid's cells")
    # The kernel holds a ray's bands and harmonics in fixed arrays.
    if bands > 8 or channels > 16:
        raise ValueError(
            f"the ray kernel holds 8 bands and order 3, asked {bands} and {settings.order}"
        )
    kernel = raw_kernel(RAYS_KERNEL, "rays")
    with cupy.cuda.Device(held.device):
        energy = cupy.zeros((receivers.shape[0], bins, bands), dtype=cupy.int64)
        moments = cupy.zeros((receivers.shape[0], bins, bands, channels), dtype=cupy.int64)
        hits = cupy.zeros((receivers.shape[0], bins), dtype=cupy.int64)
        counted = cupy.zeros(len(STATS), dtype=cupy.int64)
        recv = cupy.asarray(np.ascontiguousarray(receivers, dtype=np.float64))
        rec_offsets = cupy.asarray(np.asarray(rec_grid.offsets, dtype=np.int64))
        rec_members = cupy.asarray(np.asarray(rec_grid.members, dtype=np.int32))
        a = held.arrays
        for first in range(ray_start, ray_start + ray_count, rays_per_launch):
            count = min(rays_per_launch, ray_start + ray_count - first)
            blocks = (count + threads - 1) // threads
            t0 = time.time()
            kernel(
                (blocks,),
                (threads,),
                (
                    np.uint64(settings.seed),
                    np.int32(first),
                    np.int32(count),
                    np.int32(settings.rays),
                    cupy.asarray(np.asarray(source, dtype=np.float64)),
                    np.float64(reach),
                    np.int32(bins),
                    np.float64(settings.sound_speed_m_s * settings.bin_s),
                    np.float64(radius),
                    np.int32(receivers.shape[0]),
                    recv,
                    rec_offsets,
                    rec_members,
                    a["occ_tris"],
                    a["tri_label"],
                    a["grid_origin"],
                    np.float64(held.grid.cell_m),
                    a["grid_shape"],
                    a["cell_offsets"],
                    a["cell_members"],
                    a["absorption"],
                    a["scattering"],
                    np.int32(bands),
                    np.int32(channels),
                    np.float64(settings.energy_floor / settings.rays),
                    a["tri_reflector"],
                    a["tri_furniture"],
                    np.int32(settings.skip_specular_order),
                    np.int32(settings.skip_furniture_bounces),
                    np.float64(
                        settings.sound_speed_m_s * settings.skip_window_s
                        if settings.skip_window_s > 0
                        else 1e300
                    ),
                    energy,
                    moments,
                    hits,
                    counted,
                ),
            )
            cupy.cuda.Stream.null.synchronize()
            if say is not None:
                say(
                    f"rays: {first}..{first + count} on device {held.device},"
                    f" {time.time() - t0:.1f} s"
                )
        _counted(stats, cupy.asnumpy(counted))
        return cupy.asnumpy(energy), cupy.asnumpy(moments), cupy.asnumpy(hits)


def _counted(stats: dict[str, int] | None, counted: np.ndarray) -> None:
    """A launch's counts (:data:`reverberate.mirror.tracer.STATS`) added to ``stats``."""
    if stats is not None:
        for name, value in zip(STATS, counted, strict=True):
            stats[name] = stats.get(name, 0) + int(value)


def tree_upload(
    scene: DerivedScene, held: TracerScene, *, device: int, single: bool
) -> dict[str, Any]:
    """What the tree's text reads of a scene, on ``device``: the same for every source."""
    import cupy

    with cupy.cuda.Device(device):
        return {
            "nodes": cupy.asarray(held.nodes_single if single else held.nodes),
            "tris": cupy.asarray(held.single if single else held.triangles),
            "meta": cupy.asarray(held.meta),
            "absorption": cupy.asarray(
                np.ascontiguousarray(scene.materials.absorption, dtype=np.float64)
            ),
            "scattering": cupy.asarray(
                np.ascontiguousarray(scene.materials.scattering, dtype=np.float64)
            ),
        }


def histogram_tree_on_device(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings,
    held: TracerScene,
    on_card: dict[str, Any],
    *,
    device: int,
    ray_start: int,
    ray_count: int,
    threads: int = 128,
    rays_per_launch: int = 1_000_000,
    say: Any = None,
    stats: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The integer histogram of a share of the rays through the tree, on one device.

    A thread a ray, the ray's whole life in it: the rays of a site end
    within a few bounces of one another (nearly every one by the energy
    floor on a furnished storey), so a warp has little to wait for and
    nothing is compacted between bounces.
    """
    import cupy

    single = settings.precision == "single"
    arrays = tracer.shot_arrays(scene, held, source, receivers, settings)
    scalars = tracer.shot_scalars(scene, settings)
    kernel = raw_kernel(tracer.device_source(single), "rays_tree")
    cells = int(arrays["receivers"].shape[0])
    # The card's text takes a number for "never": the grid kernel's.
    skip_reach = scalars["skip_reach"] if np.isfinite(scalars["skip_reach"]) else 1e300
    with cupy.cuda.Device(device):
        energy = cupy.zeros((cells, scalars["bins"], scalars["bands"]), dtype=cupy.int64)
        moments = cupy.zeros(
            (cells, scalars["bins"], scalars["bands"], scalars["channels"]), dtype=cupy.int64
        )
        hits = cupy.zeros((cells, scalars["bins"]), dtype=cupy.int64)
        counted = cupy.zeros(len(STATS), dtype=cupy.int64)
        mine = {
            name: cupy.asarray(arrays[name])
            for name in ("rec_nodes", "rec_order", "receivers", "source")
        }
        for first in range(ray_start, ray_start + ray_count, rays_per_launch):
            count = min(rays_per_launch, ray_start + ray_count - first)
            t0 = time.time()
            kernel(
                ((count + threads - 1) // threads,),
                (threads,),
                (
                    on_card["nodes"],
                    on_card["tris"],
                    on_card["meta"],
                    mine["rec_nodes"],
                    mine["rec_order"],
                    mine["receivers"],
                    on_card["absorption"],
                    on_card["scattering"],
                    mine["source"],
                    np.uint64(scalars["seed"]),
                    np.float64(scalars["reach"]),
                    np.float64(scalars["bin_length"]),
                    np.float64(scalars["radius"]),
                    np.float64(scalars["floor_energy"]),
                    np.float64(skip_reach),
                    np.int32(scalars["total_rays"]),
                    np.int32(scalars["bins"]),
                    np.int32(scalars["bands"]),
                    np.int32(scalars["channels"]),
                    np.int32(scalars["skip_order"]),
                    np.int32(scalars["skip_furniture"]),
                    np.int32(first),
                    np.int32(count),
                    energy,
                    moments,
                    hits,
                    counted,
                ),
            )
            cupy.cuda.Stream.null.synchronize()
            if say is not None:
                say(
                    f"rays: {first}..{first + count} through the tree on device {device},"
                    f" {time.time() - t0:.2f} s"
                )
        _counted(stats, cupy.asnumpy(counted))
        return cupy.asnumpy(energy), cupy.asnumpy(moments), cupy.asnumpy(hits)


def histogram_on_devices(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings | None = None,
    *,
    devices: Devices | None = None,
    grid: UniformGrid | None = None,
    say: Any = None,
    held: dict[Any, Any] | None = None,
    structure: str | None = None,
    store: Store | None = None,
    stats: dict[str, int] | None = None,
) -> Histogram:
    """The histogram of every receiver, rays split over the devices.

    ``held`` keeps each card's upload from one source to the next: the rays
    read the scene and its grid or its tree, which do not depend on the
    source, and a caller with many sources hands the same dictionary every
    time. ``structure`` is ``grid`` or ``tree``
    (:func:`reverberate.mirror.tracer.structure`): what the triangles are
    looked up in, which changes the time and not the histogram. ``store``
    keeps the tree from one process to the next. ``stats`` gains the counts
    of :data:`reverberate.mirror.tracer.STATS`.
    """
    settings = settings or RaySettings()
    devices = devices or Devices.detect()
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    source = np.asarray(source, dtype=float).reshape(3)
    if tracer.structure(structure) == "tree":
        return _histogram_through_tree(
            scene, source, receivers, settings, devices, held, store, say, stats
        )
    if settings.precision != "double":
        raise ValueError("single precision is the tree's: the grid is double precision")
    grid = grid or occluder_grid(scene, settings.cell_m)
    tree = ImageTree(
        source=source,
        positions=source[None, :],
        order=np.zeros(1, dtype=np.int32),
        parent=np.full(1, -1, dtype=np.int32),
        sequence=np.full((1, 1), -1, dtype=np.int32),
    )

    def work(card: int, share: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first, count = int(share[0]), int(share.size)
        if card < 0:
            h = trace(
                scene, source, receivers, settings, grid=grid, ray_start=first, ray_count=count
            )
            return (
                np.rint(h.energy * HISTOGRAM_SCALE).astype(np.int64),
                np.rint(h.moments * HISTOGRAM_SCALE).astype(np.int64),
                h.hits,
            )
        on_card = None if held is None else held.get(card)
        if on_card is None:
            on_card = upload(scene, tree, device=card, grid=grid)
            if held is not None:
                held[card] = on_card
        return histogram_on_device(
            scene,
            source,
            receivers,
            settings,
            on_card,
            ray_start=first,
            ray_count=count,
            say=say,
            stats=stats,
        )

    parts = devices.map(work, devices.split(settings.rays))
    return _summed(parts, scene, settings)


def _summed(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    scene: DerivedScene,
    settings: RaySettings,
) -> Histogram:
    """The shares' integer histograms as one."""
    return Histogram.from_counts(
        np.sum([p[0] for p in parts], axis=0),
        np.sum([p[1] for p in parts], axis=0),
        np.sum([p[2] for p in parts], axis=0),
        bin_s=settings.bin_s,
        bands_hz=scene.materials.bands_hz,
        order=settings.order,
        rays=settings.rays,
    )


def _histogram_through_tree(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings,
    devices: Devices,
    held: dict[Any, Any] | None,
    store: Store | None,
    say: Any,
    stats: dict[str, int] | None,
) -> Histogram:
    """:func:`histogram_on_devices` with the triangles under their tree.

    On a card, the C text a thread a ray. On the host, the same text a
    share of the rays a core where the machine has a compiler, and the
    numpy twin where it has none (double precision only): the three give
    one histogram.
    """
    kept = held if held is not None else {}
    if "tree" not in kept:
        kept["tree"] = tracer.prepare(scene, store)
    under: TracerScene = kept["tree"]
    single = settings.precision == "single"
    Share = tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]

    def work(card: int, share: np.ndarray) -> Share:
        first, count = int(share[0]), int(share.size)
        # A share's own counts, added once the shares are back: a forked core's are its own.
        mine: dict[str, int] = {}
        if card >= 0:
            name = ("tree", card, single)
            if name not in kept:
                kept[name] = tree_upload(scene, under, device=card, single=single)
            found = histogram_tree_on_device(
                scene,
                source,
                receivers,
                settings,
                under,
                kept[name],
                device=card,
                ray_start=first,
                ray_count=count,
                say=say,
                stats=mine,
            )
        elif tracer.available():
            found = tracer.counts_on_host(
                scene,
                under,
                source,
                receivers,
                settings,
                ray_start=first,
                ray_count=count,
                stats=mine,
            )
        else:
            h = tracer.trace_tree(
                scene,
                source,
                receivers,
                settings,
                held=under,
                ray_start=first,
                ray_count=count,
                stats=mine,
            )
            found = (
                np.rint(h.energy * HISTOGRAM_SCALE).astype(np.int64),
                np.rint(h.moments * HISTOGRAM_SCALE).astype(np.int64),
                h.hits,
            )
        return (*found, mine)

    parts = devices.map(work, devices.split(settings.rays))
    if stats is not None:
        for part in parts:
            for name, value in part[3].items():
                stats[name] = stats.get(name, 0) + value
    return _summed([(p[0], p[1], p[2]) for p in parts], scene, settings)
