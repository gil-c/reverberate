"""The mirror on the card: the paths of every receiver, the rays of the tail, on every device.

:mod:`reverberate.mirror.ism` and :mod:`reverberate.mirror.rays` are the
twins; this module runs the kernels of :mod:`reverberate.mirror.kernels` on
what a scene, a tree and a set of receivers upload to, and hands back the
same :class:`Paths` and :class:`Histogram` the twins would. Without a
card (``REVERBERATE_NO_GPU=1``, or no cupy) every function falls back to
the twin, so a campaign on the host's cores still runs, slowly.

**Every device of the host takes a share.** Receivers are split over the
devices for the paths, rays over the devices for the histogram, and the
results are merged in a fixed order: the paths of a receiver do not depend
on which card validated them, and the histogram is a sum of integers, so
the field is the same with one card or eight.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.accel.backend import cuda_available, raw_kernel
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
    Histogram,
    RaySettings,
    UniformGrid,
    grid_like,
    reflector_of_triangles,
    trace,
)
from reverberate.spatial.sh import channel_count

__all__ = ["DeviceScene", "device_count", "histogram_on_devices", "paths_on_devices", "upload"]


def device_count() -> int:
    """How many CUDA devices this process may use; zero without a card."""
    if not cuda_available():
        return 0
    import cupy

    return int(cupy.cuda.runtime.getDeviceCount())


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
    tri_reflector, tri_furniture = reflector_of_triangles(scene)
    with cupy.cuda.Device(device):
        arrays = {
            "tree_pos": cupy.asarray(np.ascontiguousarray(tree.positions, dtype=np.float64)),
            "tree_order": cupy.asarray(np.ascontiguousarray(tree.order, dtype=np.int32)),
            "tree_parent": cupy.asarray(np.ascontiguousarray(tree.parent, dtype=np.int32)),
            "tree_seq": cupy.asarray(np.ascontiguousarray(tree.sequence, dtype=np.int32)),
            "facet_normal": cupy.asarray(np.ascontiguousarray(normals, dtype=np.float64)),
            "facet_offset": cupy.asarray(np.ascontiguousarray(offsets, dtype=np.float64)),
            "facet_start": cupy.asarray(facet_start),
            "facet_count": cupy.asarray(facet_count),
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
        a["facet_start"],
        a["facet_count"],
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
    devices: list[int] | None = None,
    grid: UniformGrid | None = None,
    say: Any = None,
) -> list[Paths]:
    """The paths of every receiver, receivers split over the devices; the twin without a card."""
    settings = settings or IsmSettings()
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    grid = grid or occluder_grid(scene)
    if device_count() == 0:
        return [paths_for(scene, tree, r, settings, grid=grid) for r in receivers]
    devices = devices if devices is not None else list(range(device_count()))
    shares = np.array_split(np.arange(receivers.shape[0]), len(devices))
    results: list[Paths | None] = [None] * receivers.shape[0]

    def on_device(device: int, share: np.ndarray) -> list[Paths]:
        held = upload(scene, tree, device=device, grid=grid)
        return paths_on_device(scene, tree, receivers[share], settings, held, say=say)

    for share, found in zip(shares, _on_each_device(devices, shares, on_device), strict=True):
        for index, paths in zip(share, found, strict=True):
            results[int(index)] = paths
    return [p for p in results if p is not None]


def _on_each_device(devices: list[int], shares: list[np.ndarray], work: Any) -> list[Any]:
    """``work(device, share)`` on every device at once, one thread each; results in share order.

    The cards run concurrently and the merge that follows is in the order of
    the shares, never of completion, so two cards give the one card's numbers.
    An empty share gives ``None``.
    """
    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as pool:
        futures = [
            pool.submit(work, device, share) if share.size else None
            for device, share in zip(devices, shares, strict=True)
        ]
        return [None if f is None else f.result() for f in futures]


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
    kernel = raw_kernel(RAYS_KERNEL, "rays")
    with cupy.cuda.Device(held.device):
        energy = cupy.zeros((receivers.shape[0], bins, bands), dtype=cupy.int64)
        moments = cupy.zeros((receivers.shape[0], bins, bands, channels), dtype=cupy.int64)
        hits = cupy.zeros((receivers.shape[0], bins), dtype=cupy.int64)
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
                    a["source"]
                    if source is None
                    else cupy.asarray(np.asarray(source, dtype=np.float64)),
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
                    energy,
                    moments,
                    hits,
                ),
            )
            cupy.cuda.Stream.null.synchronize()
            if say is not None:
                say(
                    f"rays: {first}..{first + count} on device {held.device},"
                    f" {time.time() - t0:.1f} s"
                )
        return cupy.asnumpy(energy), cupy.asnumpy(moments), cupy.asnumpy(hits)


def histogram_on_devices(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings | None = None,
    *,
    devices: list[int] | None = None,
    grid: UniformGrid | None = None,
    say: Any = None,
) -> Histogram:
    """The histogram of every receiver, rays split over the devices; the twin without a card."""
    settings = settings or RaySettings()
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    source = np.asarray(source, dtype=float).reshape(3)
    grid = grid or occluder_grid(scene, settings.cell_m)
    if device_count() == 0:
        return trace(scene, source, receivers, settings, grid=grid)
    devices = devices if devices is not None else list(range(device_count()))
    shares = np.array_split(np.arange(settings.rays), len(devices))
    bands = scene.materials.absorption.shape[1]
    channels = channel_count(settings.order)
    bins = int(np.ceil(settings.duration_s / settings.bin_s))
    energy = np.zeros((receivers.shape[0], bins, bands), dtype=np.int64)
    moments = np.zeros((receivers.shape[0], bins, bands, channels), dtype=np.int64)
    hits = np.zeros((receivers.shape[0], bins), dtype=np.int64)
    tree = ImageTree(
        source=source,
        positions=source[None, :],
        order=np.zeros(1, dtype=np.int32),
        parent=np.full(1, -1, dtype=np.int32),
        sequence=np.full((1, 1), -1, dtype=np.int32),
    )

    def on_device(device: int, share: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        held = upload(scene, tree, device=device, grid=grid)
        return histogram_on_device(
            scene,
            source,
            receivers,
            settings,
            held,
            ray_start=int(share[0]),
            ray_count=int(share.size),
            say=say,
        )

    for part in _on_each_device(devices, shares, on_device):
        if part is None:
            continue
        e, m, h = part
        energy += e
        moments += m
        hits += h
    return Histogram.from_counts(
        energy,
        moments,
        hits,
        bin_s=settings.bin_s,
        bands_hz=scene.materials.bands_hz,
        order=settings.order,
        rays=settings.rays,
    )
