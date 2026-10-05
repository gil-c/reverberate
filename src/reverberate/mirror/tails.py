"""The ray tail of a scene that moves: histograms at a few places, weights at every step.

The late part does not need the source's position to 8 cm nor the
listener's to a step. Rays are traced from every station and from positions
every 0.80 m along the rails (:func:`tail_sites`); the receiver spheres stand
on the listening cells the pack names. A step then reads up to four
histograms, two source positions by two cells, and sums them with weights
linear in arc length on the source's side and in distance on the listener's
(:func:`source_weights`, :func:`cell_weights`): an interpolation in energy,
never of waveforms (``docs/formats/scene-pack.md``, group ``tail``).

A histogram at a cell depends on the source position, the cell, the scene
with its calibrated materials and the ray settings, and on nothing else, not
on which other cells were traced with it: it is kept under :func:`tail_key`,
one entry a (site, cell), and traced once for every recipe of the dwelling
that reads it. A launch still casts a site's rays once for all the cells it
is asked.

:func:`interpolation_error_db` is the measure of what the interpolation
costs: the level, band by band, of the weighted sum against a histogram
traced where the source really is.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.compute import Devices, to_numpy
from reverberate.mirror import tracer
from reverberate.mirror.engine import histogram_on_devices
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import occluder_grid
from reverberate.mirror.moving import MovingScene, _blocked
from reverberate.mirror.parameters import apply_parameters
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import Histogram, RaySettings
from reverberate.mirror.render import _band_map, band_pulse_energy
from reverberate.mirror.shared import Store

__all__ = [
    "SPACING_M",
    "TailCache",
    "TailSites",
    "TailTable",
    "cell_weights",
    "histograms",
    "interpolation_error_db",
    "shared_scene",
    "site_key",
    "sites_read",
    "source_weights",
    "tail_key",
    "tail_scale",
    "tail_sites",
    "tail_table",
]

#: Rays are traced from rail positions this far apart at most, m.
SPACING_M = 0.80
#: A source or a head nearer than this to a traced position is on it, m.
ON_M = 1e-3


@dataclass(frozen=True)
class TailSites:
    """The source positions rays are traced from: stations, and samples along rails."""

    positions: np.ndarray
    #: ``[site]``: the rail the site samples, ``-1`` for a station.
    rail: np.ndarray
    #: ``[site]``: its arc length along that rail, m; zero for a station.
    arc_m: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


def tail_sites(
    stations: np.ndarray, rails: list[np.ndarray] | None = None, spacing_m: float = SPACING_M
) -> TailSites:
    """Every station, then each rail sampled evenly, both ends included.

    A rail is a polyline ``[vertex, 3]``. Its samples are equally spaced in
    arc length, as few as keep them at most ``spacing_m`` apart.
    """
    stations = np.asarray(stations, dtype=float).reshape(-1, 3)
    positions = [stations]
    rail_of = [np.full(stations.shape[0], -1, dtype=np.int32)]
    arcs = [np.zeros(stations.shape[0])]
    for index, rail in enumerate(rails or []):
        rail = np.asarray(rail, dtype=float).reshape(-1, 3)
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rail, axis=0), axis=1))])
        pieces = max(1, int(np.ceil(along[-1] / spacing_m - 1e-9)))
        arc = np.linspace(0.0, along[-1], pieces + 1)
        positions.append(np.stack([np.interp(arc, along, rail[:, k]) for k in range(3)], axis=1))
        rail_of.append(np.full(arc.size, index, dtype=np.int32))
        arcs.append(arc)
    return TailSites(np.concatenate(positions), np.concatenate(rail_of), np.concatenate(arcs))


def source_weights(position: np.ndarray, sites: TailSites) -> tuple[np.ndarray, np.ndarray]:
    """Per step: the two sites either side of the source, and the weight of the second.

    ``slots [step, 2]`` are rows of ``sites``, ``-1`` where there is no second;
    ``weight [step]`` is slot 1's share of the energy, slot 0 having the rest.
    A source on a site (a station, or a rail's own sample) reads that site
    alone. Elsewhere it reads the two consecutive samples of the rail piece
    it is nearest to, with a weight linear in arc length between them.
    """
    position = np.atleast_2d(np.asarray(position, dtype=float))
    steps = position.shape[0]
    slots = np.full((steps, 2), -1, dtype=np.int32)
    weight = np.zeros(steps, dtype=float)
    gaps = np.linalg.norm(position[:, None, :] - sites.positions[None, :, :], axis=2)
    nearest = np.argmin(gaps, axis=1)
    on = gaps[np.arange(steps), nearest] <= ON_M
    slots[on, 0] = nearest[on]
    # The rail pieces: consecutive samples of one rail.
    first = np.flatnonzero((sites.rail[:-1] >= 0) & (sites.rail[:-1] == sites.rail[1:]))
    off = np.flatnonzero(~on)
    if off.size and first.size == 0:
        raise ValueError(f"step {int(off[0])}: the source is on no station and there is no rail")
    if off.size:
        a = sites.positions[first]
        b = sites.positions[first + 1]
        span = b - a
        along = np.einsum("spk,pk->sp", position[off][:, None, :] - a[None, :, :], span)
        t = np.clip(along / np.maximum(np.einsum("pk,pk->p", span, span), 1e-18)[None, :], 0, 1)
        foot = a[None, :, :] + t[:, :, None] * span[None, :, :]
        piece = np.argmin(np.linalg.norm(position[off][:, None, :] - foot, axis=2), axis=1)
        share = t[np.arange(off.size), piece]
        slots[off, 0] = first[piece]
        slots[off, 1] = first[piece] + 1
        weight[off] = share
    return slots, weight


def cell_weights(
    listener: np.ndarray,
    cells: np.ndarray,
    ms: MovingScene | None = None,
    *,
    nearest: int = 6,
    xp: Any = np,
) -> tuple[np.ndarray, np.ndarray]:
    """Per step: the two cells the head reads its tail from, and the weight of the second.

    ``slots [step, 2]`` are rows of ``cells``, slot 0 the nearer, ``-1`` where
    there is no second; ``weight [step]`` is slot 1's share, ``d0 / (d0 + d1)``,
    linear in distance between the two. A head on a cell reads it alone.

    With ``ms``, a cell the head does not see is passed over among the
    ``nearest`` cells: the cell behind a wall is near and is another room's
    tail. Where none is seen the nearest serves.
    """
    listener = np.atleast_2d(np.asarray(listener, dtype=float))
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    steps = listener.shape[0]
    take = min(nearest, cells.shape[0])
    gaps = np.linalg.norm(listener[:, None, :] - cells[None, :, :], axis=2)
    ranked = np.argsort(gaps, axis=1, kind="stable")[:, :take]
    distance = np.take_along_axis(gaps, ranked, axis=1)
    seen = np.ones((steps, take), dtype=bool)
    if ms is not None and steps:
        cut = _blocked(
            xp,
            ms.on(xp),
            xp.asarray(np.repeat(listener, take, axis=0)),
            xp.asarray(cells[ranked.reshape(-1)]),
            ms.ism.epsilon_m,
            ms.moving.tests_per_block,
        )
        seen = ~to_numpy(cut).reshape(steps, take)
        seen[~seen.any(axis=1), 0] = True
    slots = np.full((steps, 2), -1, dtype=np.int32)
    weight = np.zeros(steps, dtype=float)
    for step in range(steps):
        usable = np.flatnonzero(seen[step])
        slots[step, 0] = ranked[step, usable[0]]
        d0 = float(distance[step, usable[0]])
        if d0 <= ON_M or usable.size < 2:
            continue
        slots[step, 1] = ranked[step, usable[1]]
        weight[step] = d0 / (d0 + float(distance[step, usable[1]]))
    return slots, weight


# --------------------------------------------------------------------------
# the histograms and their cache
# --------------------------------------------------------------------------


def tail_key(
    scene: DerivedScene | str, position: np.ndarray, cell: np.ndarray, rays: RaySettings
) -> str:
    """The identity of one source position's histogram at one cell.

    The scene's own key, which holds its geometry and the materials the rays
    read; the position and the cell in whole millimetres; the ray settings.
    Sixty-four hexadecimal characters. Nothing of the other cells a launch
    held: a second recipe of the dwelling, with another set of cells, finds
    the cells it shares. ``scene`` may be that key itself: it is a digest of
    every triangle, and a caller with many positions reads it once.
    """
    record = {
        "scene": scene if isinstance(scene, str) else scene.key,
        "position_mm": [int(v) for v in np.rint(np.asarray(position, dtype=float) * 1000.0)],
        "cell_mm": [int(v) for v in np.rint(np.asarray(cell, dtype=float).reshape(3) * 1000.0)],
        "rays": rays.record(),
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def site_key(scene: DerivedScene | str, position: np.ndarray, rays: RaySettings) -> str:
    """The name of one source position's rays, whatever cells they are counted at."""
    record = {
        "scene": scene if isinstance(scene, str) else scene.key,
        "position_mm": [int(v) for v in np.rint(np.asarray(position, dtype=float) * 1000.0)],
        "rays": rays.record(),
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


@dataclass
class TailCache:
    """Histograms by :func:`tail_key`, a cell an entry: ``<key>.npz`` files, or memory alone.

    An entry is one cell's rows of a site's histogram. :meth:`site` gives a
    site's histogram over a list of cells from their entries, and holds the
    last few it gave.
    """

    directory: Path | None = None
    #: The sites' histograms held in memory at most, the last read kept; ``None`` is every
    #: one. A process among several that each read the whole scene's sites holds a few.
    keep: int | None = None

    def __post_init__(self) -> None:
        self._held: dict[str, Histogram] = {}
        self._cells: dict[str, Histogram] = {}
        self._shared: dict[str, Any] = {}
        #: Sites found whole, and sites asked for of which a cell at least was not there.
        self.hits = 0
        self.misses = 0
        #: The same a (site, cell) at a time: entries found, and entries not there.
        self.cells_found = 0
        self.cells_absent = 0

    def _hold(self, name: str, histogram: Histogram) -> None:
        self._held.pop(name, None)
        self._held[name] = histogram
        if self.keep is not None and self.directory is not None:
            while len(self._held) > max(1, self.keep):
                self._held.pop(next(iter(self._held)))

    def path(self, key: str) -> Path | None:
        """Where an entry is kept: two characters of its key, then the key."""
        return None if self.directory is None else Path(self.directory) / key[:2] / f"{key}.npz"

    def has(self, key: str) -> bool:
        path = self.path(key)
        return key in self._cells if path is None else path.is_file()

    def absent(self, keys: list[str]) -> list[int]:
        """The places in ``keys`` of the entries that are not kept."""
        return [k for k, key in enumerate(keys) if not self.has(key)]

    def _read(self, key: str) -> Histogram | None:
        path = self.path(key)
        if path is None:
            return self._cells.get(key)
        if not path.is_file():
            return None
        with np.load(path) as arrays:
            return Histogram(
                energy=arrays["energy"],
                moments=arrays["moments"],
                hits=arrays["hits"],
                bin_s=float(arrays["bin_s"]),
                bands_hz=tuple(int(v) for v in arrays["bands_hz"]),
                order=int(arrays["order"]),
                rays=int(arrays["rays"]),
            )

    def site(self, keys: list[str], *, counted: bool = True) -> Histogram | None:
        """A site's histogram over the cells of ``keys``, or ``None`` when an entry is absent.

        ``counted`` is whether the call is one of those :attr:`hits` and
        :attr:`misses` count: not the reading back of what was just traced.
        """
        name = hashlib.sha256("".join(keys).encode()).hexdigest()
        found = self._held.get(name)
        if found is not None:
            self._hold(name, found)
            if counted:
                self.hits += 1
                self.cells_found += len(keys)
            return found
        cells = [self._read(key) for key in keys]
        there = [c for c in cells if c is not None]
        whole = len(there) == len(cells) and bool(cells)
        if counted:
            self.hits += int(whole)
            self.misses += int(not whole)
            self.cells_found += len(there)
            self.cells_absent += len(cells) - len(there)
        if not whole:
            return None
        found = Histogram(
            energy=np.stack([c.energy for c in there]),
            moments=np.stack([c.moments for c in there]),
            hits=np.stack([c.hits for c in there]),
            bin_s=there[0].bin_s,
            bands_hz=there[0].bands_hz,
            order=there[0].order,
            rays=there[0].rays,
        )
        self._hold(name, found)
        return found

    def hold(self, keys: list[str], histogram: Histogram) -> None:
        """A site's histogram over the cells of ``keys``, held as :meth:`site` would give it.

        What was just traced over every cell of a site is the site's
        histogram: reading its entries back from the disk took as long as a
        fifth of the rays (0.09 s of 53 entries).
        """
        self._hold(hashlib.sha256("".join(keys).encode()).hexdigest(), histogram)

    def put(self, keys: list[str], histogram: Histogram) -> None:
        """A histogram over cells, kept a cell an entry under ``keys``, in the cells' order."""
        if len(keys) != histogram.energy.shape[0]:
            raise ValueError(f"{len(keys)} keys for {histogram.energy.shape[0]} cells")
        for row, key in enumerate(keys):
            path = self.path(key)
            if path is None:
                self._cells[key] = Histogram(
                    energy=histogram.energy[row],
                    moments=histogram.moments[row],
                    hits=histogram.hits[row],
                    bin_s=histogram.bin_s,
                    bands_hz=histogram.bands_hz,
                    order=histogram.order,
                    rays=histogram.rays,
                )
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            # Whole or not at all: another process may be reading the directory.
            partial = path.with_suffix(f".{os.getpid()}.partial.npz")
            np.savez(
                partial,
                energy=histogram.energy[row],
                moments=histogram.moments[row],
                hits=histogram.hits[row],
                bin_s=histogram.bin_s,
                bands_hz=np.asarray(histogram.bands_hz),
                order=histogram.order,
                rays=histogram.rays,
            )
            partial.replace(path)


def shared_scene(
    cache: TailCache, catalogue: DerivedScene, settings: MirrorSettings
) -> dict[str, Any]:
    """What every call on one scene shares, kept in ``cache``: read once, whoever asks.

    The scene with the calibration's materials, its key (a digest of every
    triangle), the occluders' grid once a histogram is traced, and each
    card's upload.
    """
    rays = settings.traced_rays()
    shared = cache._shared
    if (
        shared.get("catalogue") is not catalogue
        or shared.get("parameters") != settings.parameters.key
        or shared.get("cell_m") != rays.cell_m
    ):
        scene = apply_parameters(catalogue, settings.parameters)
        shared = {
            "catalogue": catalogue,
            "parameters": settings.parameters.key,
            "cell_m": rays.cell_m,
            "scene": scene,
            "key": scene.key,
            "grid": None,
            "uploads": {},
        }
        cache._shared = shared
    return shared


def sites_read(
    source: np.ndarray, sites: TailSites, audible: np.ndarray | None = None
) -> np.ndarray:
    """The rows of ``sites`` a source's audible steps read: those :func:`tail_table` traces."""
    source = np.atleast_2d(np.asarray(source, dtype=float))
    heard = np.ones(source.shape[0], dtype=bool) if audible is None else np.asarray(audible, bool)
    if not bool(heard.any()):
        return np.zeros(0, dtype=np.int64)
    slots, weight = source_weights(source[heard], sites)
    slots[weight <= 0.0, 1] = -1
    return np.asarray(np.unique(slots[slots >= 0]), dtype=np.int64)


def histograms(
    catalogue: DerivedScene,
    settings: MirrorSettings,
    positions: np.ndarray,
    cells: np.ndarray,
    *,
    devices: Devices | None = None,
    cache: TailCache | None = None,
    say: Any = None,
    store: Store | None = None,
    stats: dict[str, int] | None = None,
) -> list[Histogram]:
    """One :class:`Histogram` over ``cells`` per source position, traced or read from the cache.

    The rays are the present pipeline's: the scene with the calibration's
    materials, :meth:`MirrorSettings.traced_rays`, so a histogram here is the
    one :func:`reverberate.mirror.pipeline.trace` computes for that source
    over those receivers. A position whose cells are not all kept is traced
    over those that are not, in one launch, and each is kept on its own.
    ``store`` keeps the rays' tree (:func:`reverberate.mirror.tracer.prepare`).
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    cache = cache if cache is not None else TailCache()
    rays = settings.traced_rays()
    shared = shared_scene(cache, catalogue, settings)
    scene = shared["scene"]
    out = []
    through_grid = tracer.structure() == "grid"
    for position in positions:
        keys = [tail_key(shared["key"], position, cell, rays) for cell in cells]
        found = cache.site(keys)
        if found is None:
            if through_grid and shared["grid"] is None:
                shared["grid"] = occluder_grid(scene, rays.cell_m)
            absent = cache.absent(keys)
            traced = histogram_on_devices(
                scene,
                position,
                cells[absent],
                rays,
                devices=devices,
                grid=shared["grid"],
                say=say,
                held=shared["uploads"],
                store=store,
                stats=stats,
            )
            cache.put([keys[k] for k in absent], traced)
            # Traced over every cell (or over none: the empty histogram), what was traced
            # is the answer; over some, the site is put together from its entries.
            if len(absent) == len(keys):
                found = traced
                cache.hold(keys, traced)
            else:
                found = cache.site(keys, counted=False)
            if found is None:
                raise RuntimeError("a histogram just traced is not in its cache")
        out.append(found)
    return out


def tail_scale(
    ms: MovingScene,
    position: np.ndarray,
    cells: np.ndarray,
    histogram: Histogram,
    *,
    rate: float,
    receiver_radius_m: float,
    xp: Any = np,
) -> np.ndarray:
    """``[cell, bank band]``: what turns each cell's histogram energy into the response's.

    ``scale_per_band`` of :func:`reverberate.mirror.render.render_point`. A
    cell that sees the source and was reached by rays is scaled on its own
    direct sound: the energy a sphere of the receivers' radius catches at
    that distance against the direct pulse's. The others take the median of
    those, as :func:`reverberate.mirror.pipeline.render` gives the points
    without a direct path. Where no cell sees the source, a voice in another
    room than every cell, they take what a cell beyond the sphere would
    have: the spreading and the sphere's share both go as the distance
    squared, so that scale is the pulse's energy times four over the
    radius squared, wherever the cell stands.
    """
    position = np.asarray(position, dtype=float).reshape(3)
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    _, picks = _band_map(rate)
    distance = np.linalg.norm(cells - position[None, :], axis=1)
    cut = to_numpy(
        _blocked(
            xp,
            ms.on(xp),
            xp.asarray(cells),
            xp.asarray(np.repeat(position[None, :], cells.shape[0], axis=0)),
            ms.ism.epsilon_m,
            ms.moving.tests_per_block,
        )
    )
    heard = histogram.hits.sum(axis=1) > 0
    own = ~cut & heard
    expected = receiver_radius_m**2 / (4.0 * np.maximum(distance, 1.05 * receiver_radius_m) ** 2)
    # The direct path's gain is its spreading alone, the same in every band.
    amplitude = 1.0 / np.maximum(distance, 1e-12)
    scale = (amplitude**2 / expected)[:, None] * band_pulse_energy(rate)[None, : len(picks)]
    beyond = 4.0 / receiver_radius_m**2 * band_pulse_energy(rate)[: len(picks)]
    fallback = np.median(scale[own], axis=0) if bool(own.any()) else beyond
    scale[~own] = fallback
    return np.asarray(scale, dtype=float)


# --------------------------------------------------------------------------
# the pack's group
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TailTable:
    """One source's ``tail`` group: the histograms it reads, and what each step reads of them."""

    #: ``[hist, bin, band]`` and ``[hist, bin, band, channel]``.
    energy: np.ndarray
    moments: np.ndarray
    #: ``[hist, bank band]``.
    scale: np.ndarray
    hist_position: np.ndarray
    #: ``[hist]``: row of the cells the table was built on.
    hist_cell: np.ndarray
    #: ``[step, 2, 2]``: rows of ``energy``, source slot by cell slot; ``-1`` where none.
    hist: np.ndarray
    position_weight: np.ndarray
    cell_weight: np.ndarray

    def pack(self) -> dict[str, np.ndarray]:
        """The datasets of ``/sources/<id>/tail``, in the pack's types."""
        return {
            "energy": self.energy.astype(np.float32),
            "moments": self.moments.astype(np.float32),
            "scale": self.scale.astype(np.float64),
            "hist_position": self.hist_position.astype(np.float64),
            "hist_cell": self.hist_cell.astype(np.int32),
            "hist": self.hist.astype(np.int32),
            "position_weight": self.position_weight.astype(np.float32),
            "cell_weight": self.cell_weight.astype(np.float32),
        }

    def weights(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """The step's histograms and their weights: up to four rows, weights summing to one."""
        p, c = float(self.position_weight[step]), float(self.cell_weight[step])
        rows = self.hist[step].reshape(-1)
        share = np.asarray([(1 - p) * (1 - c), (1 - p) * c, p * (1 - c), p * c])
        used = (rows >= 0) & (share > 0.0)
        return rows[used], share[used]

    def at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """The step's own histogram: energy ``[bin, band]`` and moments, summed in energy."""
        rows, share = self.weights(step)
        if rows.size == 0:
            return np.zeros(self.energy.shape[1:]), np.zeros(self.moments.shape[1:])
        return (
            np.tensordot(share, self.energy[rows], axes=1),
            np.tensordot(share, self.moments[rows], axes=1),
        )


def tail_table(
    ms: MovingScene,
    settings: MirrorSettings,
    source: np.ndarray,
    listener: np.ndarray,
    sites: TailSites,
    cells: np.ndarray,
    *,
    audible: np.ndarray | None = None,
    rate: float = 48000.0,
    devices: Devices | None = None,
    cache: TailCache | None = None,
    xp: Any = np,
    say: Any = None,
) -> TailTable:
    """One source's tail along its trajectory and the listener's.

    Rays are traced, or read from ``cache``, from the sites an audible step
    reads and from no other; a histogram is stored once however many steps
    read it.
    """
    source = np.atleast_2d(np.asarray(source, dtype=float))
    listener = np.atleast_2d(np.asarray(listener, dtype=float))
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    steps = source.shape[0]
    heard = np.ones(steps, dtype=bool) if audible is None else np.asarray(audible, dtype=bool)
    site_slot = np.full((steps, 2), -1, dtype=np.int32)
    cell_slot = np.full((steps, 2), -1, dtype=np.int32)
    position_weight = np.zeros(steps)
    cell_weight = np.zeros(steps)
    if bool(heard.any()):
        site_slot[heard], position_weight[heard] = source_weights(source[heard], sites)
        cell_slot[heard], cell_weight[heard] = cell_weights(listener[heard], cells, ms, xp=xp)
    # A slot nothing is read from is no slot.
    site_slot[position_weight <= 0.0, 1] = -1
    cell_slot[cell_weight <= 0.0, 1] = -1
    pair = np.stack(
        [site_slot[:, [0, 0, 1, 1]].reshape(-1), cell_slot[:, [0, 1, 0, 1]].reshape(-1)], axis=1
    )
    valid = (pair >= 0).all(axis=1)
    used, inverse = np.unique(pair[valid], axis=0, return_inverse=True)
    hist = np.full(steps * 4, -1, dtype=np.int32)
    hist[valid] = np.asarray(inverse).reshape(-1)
    used_sites = np.unique(used[:, 0]) if used.size else np.zeros(0, dtype=np.int64)
    traced = histograms(
        ms.catalogue,
        settings,
        sites.positions[used_sites],
        cells,
        devices=devices,
        cache=cache,
        say=say,
    )
    by_site = dict(zip((int(s) for s in used_sites), traced, strict=True))
    scales = {
        site: tail_scale(
            ms,
            sites.positions[site],
            cells,
            histogram,
            rate=rate,
            receiver_radius_m=settings.rays.receiver_radius_m,
            xp=xp,
        )
        for site, histogram in by_site.items()
    }
    bins = int(np.ceil(settings.rays.duration_s / settings.rays.bin_s))
    bands = int(ms.scene.materials.absorption.shape[1])
    channels = (settings.rays.order + 1) ** 2
    energy = np.zeros((used.shape[0], bins, bands))
    moments = np.zeros((used.shape[0], bins, bands, channels))
    scale = np.zeros((used.shape[0], len(_band_map(rate)[1])))
    for row, (site, cell) in enumerate(used):
        energy[row] = by_site[int(site)].energy[cell]
        moments[row] = by_site[int(site)].moments[cell]
        scale[row] = scales[int(site)][cell]
    return TailTable(
        energy=energy,
        moments=moments,
        scale=scale,
        hist_position=sites.positions[used[:, 0]] if used.size else np.zeros((0, 3)),
        hist_cell=used[:, 1].astype(np.int32) if used.size else np.zeros(0, dtype=np.int32),
        hist=hist.reshape(steps, 2, 2),
        position_weight=position_weight,
        cell_weight=cell_weight,
    )


def interpolation_error_db(
    low: np.ndarray, high: np.ndarray, truth: np.ndarray, weight: float, *, from_bin: int = 0
) -> np.ndarray:
    """Per band, the level of an interpolated tail against the traced one, dB.

    ``low``, ``high`` and ``truth`` are energies ``[bin, band]``: the two
    histograms either side, and the one traced where the source or the head
    is; ``weight`` is ``high``'s. The level is the energy summed from
    ``from_bin`` on. Read the worst band, not the median.
    """
    made = (1.0 - weight) * low[from_bin:].sum(axis=0) + weight * high[from_bin:].sum(axis=0)
    want = truth[from_bin:].sum(axis=0)
    return np.asarray(10.0 * np.log10(np.maximum(made, 1e-300) / np.maximum(want, 1e-300)))
