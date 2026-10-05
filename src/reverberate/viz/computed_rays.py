"""A few of the tail's rays, bounce by bounce, and an image path through its facets.

The tail of a pack is a histogram of a hundred thousand rays; nothing of the
rays themselves is kept. To show what was traced, a sample is traced again
here, on the host, by the law the kernel follows.

:func:`ray_paths` is :func:`reverberate.mirror.rays.trace` with the
histogram taken out and the vertices kept: the same directions
(:func:`reverberate.mirror.rays.ray_directions`, so ray ``k`` here is ray
``k`` of the hundred thousand), the same nearest hit on the same occluder
triangles through the same grid, the same hashed draw between a specular
and a scattered bounce, the same absorption per band, the same floor and
reach. The tests hold it to :func:`trace` hit for hit. It is a **sample**:
what the page draws is ``count`` rays of the many the histogram summed, and
the page says so.

:func:`path_points` gives an image path's corners from its facet sequence,
by the image method: the source mirrored in each facet's plane in turn, and
the path walked back from the listener through the images.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.rays import (
    RaySettings,
    UniformGrid,
    _lambert,
    _nearest_hit,
    _sphere_crossings,
    hash_uniform,
    ray_directions,
    reflector_of_triangles,
    triangle_grid,
)

__all__ = ["path_points", "ray_paths"]


def ray_paths(
    scene: DerivedScene,
    source: np.ndarray,
    settings: RaySettings,
    *,
    count: int,
    start: int = 0,
    receivers: np.ndarray | None = None,
    grid: UniformGrid | None = None,
    max_bounces: int | None = None,
) -> dict[str, Any]:
    """Rays ``start`` to ``start + count`` of the tracer's, each as its vertices.

    ``scene`` is the scene the tracer reads, with its calibrated materials.
    Returns ``offsets`` ``[ray + 1]`` into ``points`` ``[vertex, 3]``, the
    energy carried on leaving each vertex ``[vertex, band]`` as a share of
    the ray's own at the source, the triangle hit at each vertex (``-1`` at
    the source and at an end in the air) and whether the bounce there was
    scattered, and ``crossings``: per receiver the vertices after which a
    ray entered its sphere **and was counted**, with the time. ``max_bounces``
    stops a ray early; ``truncated`` then counts the rays it stopped.
    """
    source = np.asarray(source, dtype=float).reshape(3)
    receivers = (
        np.zeros((0, 3)) if receivers is None else np.atleast_2d(np.asarray(receivers, dtype=float))
    )
    triangles = scene.occluder_vertices
    labels = scene.occluder_label
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    absorption = scene.materials.absorption
    scattering = scene.materials.scattering
    bands = absorption.shape[1]
    if grid is None:
        grid = triangle_grid(triangles, settings.cell_m, scene.bmin, scene.bmax)
    reach = settings.sound_speed_m_s * settings.duration_s
    directions = ray_directions(count, settings.seed, start=start)
    floor = settings.energy_floor
    one = np.array([0])
    tri_reflector, tri_furniture = reflector_of_triangles(scene)
    skip_reach = (
        settings.sound_speed_m_s * settings.skip_window_s if settings.skip_window_s > 0 else np.inf
    )
    offsets = [0]
    points: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    hit: list[int] = []
    scattered: list[bool] = []
    crossings: list[tuple[int, int, float]] = []
    truncated = 0
    for ray in range(start, start + count):
        position = source.copy()
        direction = directions[ray - start]
        carried = np.ones(bands)
        travelled = 0.0
        last_triangle = -1
        specular_only = True
        furniture_bounces = 0
        points.append(position.copy())
        energy.append(carried.copy())
        hit.append(-1)
        scattered.append(False)
        for bounce in range(100_000):
            candidates = grid.candidates(position, position + direction * (reach - travelled))
            distance, triangle = _nearest_hit(
                position, direction, triangles, candidates, last_triangle
            )
            segment = min(distance, reach - travelled)
            covered = (
                specular_only
                and 1 <= bounce <= settings.skip_specular_order
                and furniture_bounces <= settings.skip_furniture_bounces
            )
            if receivers.shape[0]:
                which, entries = _sphere_crossings(
                    position, direction, segment, receivers, settings.receiver_radius_m
                )
                if covered:
                    # What the image tree renders is not the rays', unless it comes late.
                    keep = (
                        travelled + entries > skip_reach
                        if skip_reach < np.inf
                        else np.zeros(which.size, dtype=bool)
                    )
                    which, entries = which[keep], entries[keep]
                for r, entry in zip(which, entries, strict=True):
                    at = (travelled + float(entry)) / settings.sound_speed_m_s
                    if at < settings.duration_s:
                        crossings.append((int(r), len(points) - 1, at))
            if triangle < 0 or travelled + distance >= reach:
                points.append(position + direction * segment)
                energy.append(carried.copy())
                hit.append(-1)
                scattered.append(False)
                break
            travelled += distance
            position = position + direction * distance
            label = int(labels[triangle])
            carried = carried * (1.0 - absorption[label])
            points.append(position.copy())
            energy.append(carried.copy())
            hit.append(int(triangle))
            if np.all(carried < floor):
                scattered.append(False)
                break
            normal = normals[triangle]
            if float(normal @ direction) > 0.0:
                normal = -normal
            kind = float(
                hash_uniform(settings.seed, np.array([ray]), np.array([bounce + 1]), one)[0]
            )
            if tri_furniture[triangle]:
                furniture_bounces += 1
            if kind < scattering[label]:
                direction = _lambert(normal, settings.seed, ray, bounce + 1)
                specular_only = False
                scattered.append(True)
            else:
                direction = direction - 2.0 * float(direction @ normal) * normal
                if tri_reflector[triangle] < 0:
                    specular_only = False
                scattered.append(False)
            last_triangle = triangle
            if max_bounces is not None and bounce + 1 >= max_bounces:
                truncated += 1
                break
        offsets.append(len(points))
    return {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "points": np.asarray(points, dtype=float).reshape(-1, 3),
        "energy": np.asarray(energy, dtype=float).reshape(-1, bands),
        "triangle": np.asarray(hit, dtype=np.int64),
        "scattered": np.asarray(scattered, dtype=bool),
        "crossings": crossings,
        "truncated": truncated,
        "first": start,
        "count": count,
    }


def path_points(
    source: np.ndarray,
    listener: np.ndarray,
    sequence: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """An image path's corners: the source, a point on each facet in bounce order, the listener.

    ``sequence`` names the facets, ``-1`` padded; ``normals`` and ``offsets``
    are every facet's plane, ``normal . x = offset``.
    """
    facets = [int(f) for f in np.asarray(sequence).reshape(-1) if int(f) >= 0]
    source = np.asarray(source, dtype=float)
    listener = np.asarray(listener, dtype=float)
    images = [source]
    for facet in facets:
        n = np.asarray(normals[facet], dtype=float)
        images.append(images[-1] - 2.0 * (float(n @ images[-1]) - float(offsets[facet])) * n)
    corners = [listener]
    here = listener
    for k in range(len(facets), 0, -1):
        n = np.asarray(normals[facets[k - 1]], dtype=float)
        leg = images[k] - here
        along = float(n @ leg)
        share = (float(offsets[facets[k - 1]]) - float(n @ here)) / along if along != 0.0 else 0.0
        here = here + share * leg
        corners.append(here)
    corners.append(source)
    return np.asarray(corners[::-1], dtype=float)
