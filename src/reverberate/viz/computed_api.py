"""What the page asks in order to show a pack's geometry as the computation used it.

The scene view draws the furnished model. None of the engines computed on
it: the wave solver read a staircase of nodes, the image sources and the
rays a decimated scene of facets and triangles, each listening array stood
on nodes and not on the cell asked for. This module serves those, for any
pack, each with the key of the asset it is and each saying whether it is
the thing itself, a view derived from it, or a sample traced again.

Under ``/api/computed/``:

- ``GET about?pack=``: the pack's asset keys; its grid, or why none is
  here; the grids offered beside it; every listening cell as asked and as
  its array stood; every solved source position and its nodes; the tail's
  sites and cells; the mirror's scene, or why none is here.
- ``GET grid/slice?grid=&y=``: one layer of nodes, a byte a node (exact).
- ``GET grid/walls?grid=&x=&y=&z=&r=``: the solver's walls about a place,
  merged squares (exact where drawn; the box bounds what is sent).
- ``GET grid/array?pack=&cell=``: the nodes a cell's array sampled.
- ``GET diff?a=&b=``: two grids' numbers; ``diff/slice``, ``diff/walls``
  and ``diff/openings`` as the grid's own.
- ``GET mirror/facets?pack=``: the facet catalogue; ``mirror/triangles``
  the triangles of the reflectors or the occluders; ``mirror/edges`` the
  diffracting edges.
- ``GET rays?pack=&source=&row=&count=``: a sample of the tail's rays from a
  site, traced again by the tracer's law.
- ``GET tail?pack=&source=&row=``: one histogram of the pack, and its shape.
- ``GET paths?pack=&source=&step=``: the step's image paths as polylines
  through the facets, recomputed by the trace's own code and matched to the
  pack's rows by path id.
- ``GET low?pack=&source=&step=``: which cells and pairs serve the head.

**Where things are found.** A pack's grid is ``as_computed.npz`` beside it
(:mod:`reverberate.trace.computed`); failing that the voxel cache entry of
its key, cut here as the solver cuts it and kept under the cache folder;
failing that nothing, and the page says so. A pack traced before the
archive existed gets its arrays from ``pairs_plan.json`` and a lattice
rebuilt from its bundle's export. The mirror's scene is the bundle's
(``<home>/bundle/trace/mirror`` for ``<home>/pulled/pack.h5``) or one given
at start, and is shown against the key the pack names.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.pack import ScenePack, read_pack
from reverberate.trace.computed import (
    NAME,
    AsComputed,
    GridRecord,
    grid_record,
    read_as_computed,
    save_as_computed,
)
from reverberate.viz import computed_grid as cg
from reverberate.viz.audit_api import AuditService, Binary
from reverberate.viz.scene_api import SceneError

__all__ = ["PREFIX", "ComputedService", "low_at", "packed"]

#: The path every endpoint lives under.
PREFIX = "api/computed"

#: The walls are sent about a place, this far at most, m.
MAX_RADIUS_M = 8.0
#: The most rays one request traces again, and the bounces a ray is followed for.
MAX_RAYS = 600
RAY_BOUNCES = 60
#: The directions a histogram's shape is given at: elevations by azimuths.
SHAPE_GRID = (13, 24)


def packed(arrays: Mapping[str, np.ndarray], **said: Any) -> Binary:
    """Arrays end to end as bytes, with what they are in the ``X-Computed`` header.

    Each is padded to its item size, so a page reads it as a typed array in place.
    """
    parts: list[bytes] = []
    spec = []
    at = 0
    for name, array in arrays.items():
        array = np.ascontiguousarray(array)
        pad = (-at) % max(array.dtype.itemsize, 1)
        parts.append(b"\0" * pad)
        at += pad
        spec.append(
            {
                "name": name,
                "dtype": array.dtype.newbyteorder("<").str,
                "shape": list(array.shape),
                "offset": at,
            }
        )
        raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        parts.append(raw)
        at += len(raw)
    return Binary(b"".join(parts), {"X-Computed": json.dumps({"arrays": spec, **said})})


def _text(value: Any) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _list(values: Any, decimals: int = 4) -> Any:
    return np.round(np.asarray(values, dtype=float), decimals).tolist()


@dataclass
class _Mirror:
    """A mirror's scene, loaded once: the assets, the moving scene, the rays' grid."""

    directory: Path
    assets: Any
    ms: Any
    sheets: np.ndarray
    grid: Any = None
    edges: Any = None


def low_at(pack: ScenePack, source: Any, step: int) -> dict[str, Any]:
    """What the band under the crossover reads at a step: its cells and their source positions."""
    held = source.low
    mode = int(held.mode[step])
    head = np.asarray(pack.listener.position[step], dtype=float)
    here = np.asarray(source.position[step], dtype=float)
    weight = float(held.position_weight[step])
    knot = 0
    if held.slot_knots_hz is not None:
        knot = int(np.argmin(np.abs(np.asarray(held.slot_knots_hz) - pack.crossover.cutoff_hz)))
    cells = []
    for slot in range(2):
        cell = int(held.cell[step, slot])
        if cell < 0:
            continue
        centre = np.asarray(pack.cells.position[cell], dtype=float)
        pairs = []
        sides: list[tuple[int, float]] = [
            (int(held.pair[step, 0, slot]), 1.0 - weight),
            (int(held.pair[step, 1, slot]), weight),
        ]
        if held.slot_pair is not None and held.slot_weight is not None:
            # More than two positions: each with its weight at the crossover.
            sides = [
                (int(row), float(held.slot_weight[step, side, knot]))
                for side, row in enumerate(held.slot_pair[step, :, slot])
            ]
        for row, share in sides:
            if row < 0 or (share <= 0.0 and held.slot_pair is None):
                continue
            position = np.asarray(held.pair_position[row], dtype=float)
            pairs.append(
                {
                    "row": row,
                    "weight": round(share, 4),
                    "position_m": _list(position, 4),
                    "from_source_mm": round(float(np.linalg.norm(position - here)) * 1000.0, 1),
                    "key": _text(held.pair_key[row]),
                }
            )
        cells.append(
            {
                "cell": cell,
                "centre_m": _list(centre, 4),
                "translation_mm": round(float(np.linalg.norm(head - centre)) * 1000.0, 1),
                "pairs": pairs,
            }
        )
    return {
        "exact": "the pack's own tables at this step"
        if held.slot_knots_hz is None
        else "the pack's own tables at this step; a position's weight changes with "
        f"frequency and is given at {float(held.slot_knots_hz[knot]):.0f} Hz",
        "step": step,
        "mode": mode,
        "mode_name": {0: "inaudible", 1: "exact", 2: "translated", 3: "fused"}.get(mode, "?"),
        "listener_m": _list(head, 4),
        "source_m": _list(here, 4),
        "cells": cells,
    }


class ComputedService:
    """The endpoints over the audit's packs, some grids given at start and one cache folder."""

    def __init__(
        self,
        audit: AuditService,
        cache_root: Path,
        grids: Iterable[Path] = (),
        mirrors: Iterable[Path] = (),
        vox_cache: Path | None = None,
    ) -> None:
        self.audit = audit
        self.cache_root = Path(cache_root)
        self.offered = [Path(p) for p in grids]
        self.mirror_dirs = [Path(p) for p in mirrors]
        self.vox_cache = vox_cache
        self._lock = threading.RLock()
        self._packs: dict[str, tuple[Path, ScenePack]] = {}
        self._archives: dict[str, AsComputed | None] = {}
        self._grids: dict[str, GridRecord] = {}
        self._sources: dict[str, dict[str, Any]] = {}
        self._diffs: dict[tuple[str, str], cg.Comparison] = {}
        self._mirrors: dict[str, _Mirror] = {}
        self._rays: dict[tuple[Any, ...], dict[str, Any]] = {}

    # -- packs ----------------------------------------------------------------

    def _pack(self, query: Mapping[str, Any]) -> tuple[str, Path, ScenePack]:
        wanted = str(query.get("pack", ""))
        with self._lock:
            if wanted not in self._packs:
                found = next((p for p in self.audit.packs() if p["id"] == wanted), None)
                if found is None:
                    raise SceneError(404, f"no pack {wanted!r}")
                path = Path(found["path"])
                try:
                    self._packs[wanted] = (path, read_pack(path, check=False))
                except (OSError, ValueError) as error:
                    raise SceneError(422, f"{path.name}: {error}") from error
            path, pack = self._packs[wanted]
        return wanted, path, pack

    @staticmethod
    def _whole(query: Mapping[str, Any], name: str, low: int, high: int) -> int:
        try:
            value = int(query[name])
        except (KeyError, TypeError, ValueError) as error:
            raise SceneError(400, f"{name} is not a whole number") from error
        if not low <= value < high:
            raise SceneError(400, f"{name} = {value} is outside {low} to {high - 1}")
        return value

    @staticmethod
    def _number(query: Mapping[str, Any], name: str, default: float | None = None) -> float:
        try:
            return float(query[name])
        except (KeyError, TypeError, ValueError) as error:
            if default is not None and name not in query:
                return default
            raise SceneError(400, f"{name} is not a number") from error

    def _source(self, pack: ScenePack, query: Mapping[str, Any]) -> Any:
        name = str(query.get("source", ""))
        if name not in pack.sources:
            raise SceneError(400, f"the pack holds no source {name!r}")
        return pack.sources[name]

    def _archive(self, pack_id: str, path: Path) -> AsComputed | None:
        with self._lock:
            if pack_id not in self._archives:
                beside = path.parent / NAME
                try:
                    self._archives[pack_id] = read_as_computed(beside) if beside.is_file() else None
                except (OSError, ValueError, KeyError):
                    self._archives[pack_id] = None
            return self._archives[pack_id]

    # -- grids ----------------------------------------------------------------

    def _vox_entry(self, key: str, home: Path) -> Path | None:
        roots = [home / "cache"]
        if self.vox_cache is not None:
            roots.append(self.vox_cache)
        for root in roots:
            if (root / key / "vox_out.h5").is_file():
                return root / key
        return None

    def _solved_positions(self, pack: ScenePack) -> np.ndarray:
        """Every source position the pack's low band was solved from."""
        rows = [
            np.asarray(source.low.pair_position, dtype=float).reshape(-1, 3)
            for source in pack.sources.values()
            if source.low is not None
        ]
        held = np.concatenate(rows) if rows else np.zeros((0, 3))
        return np.unique(held, axis=0) if held.shape[0] else held

    def _cut(self, entry: Path, key: str, sources: np.ndarray) -> GridRecord:
        """A cache entry cut to its sources' reach, kept under the cache folder by key."""
        digest = hashlib.sha256(np.ascontiguousarray(sources).tobytes()).hexdigest()[:10]
        kept = self.cache_root / f"{key}-{digest}.npz"
        if kept.is_file():
            try:
                found = read_as_computed(kept).grid
                if found is not None:
                    return found
            except (OSError, ValueError, KeyError):
                pass
        record = grid_record(entry, sources, key=key)
        try:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            save_as_computed(
                kept,
                record={"voxel_low_key": key, "solver": "", "cells_in_pack": 0},
                grid=record,
                cell_asked=np.zeros((0, 3)),
                cell_centre=np.zeros((0, 3)),
                source_position=sources,
            )
        except OSError:
            pass
        return record

    def _grid_sources(self, pack_id: str, path: Path, pack: ScenePack) -> dict[str, Any]:
        """Where each grid the page may ask for comes from; nothing heavy is read."""
        with self._lock:
            if pack_id in self._sources:
                return self._sources[pack_id]
            key = str(dict(pack.header.provenance).get("assets", {}).get("voxel_low_key", ""))
            found: dict[str, Any] = {}
            archive = self._archive(pack_id, path)
            own = None
            why = ""
            if archive is not None and archive.grid is not None:
                own = f"{archive.grid.key}@{pack_id}"
                found[own] = {
                    "key": archive.grid.key,
                    "from": f"{NAME} beside the pack",
                    "load": lambda a=archive: a.grid,
                }
            elif key:
                entry = self._vox_entry(key, path.parent.parent)
                if entry is not None:
                    own = f"{key}@{pack_id}"
                    positions = self._solved_positions(pack)
                    found[own] = {
                        "key": key,
                        "from": f"the voxel cache entry {entry}, cut here to the pack's sources",
                        "load": lambda e=entry, k=key, p=positions: self._cut(e, k, p),
                    }
                else:
                    why = (
                        (archive.record.get("why_no_grid") if archive is not None else "")
                        or f"neither {NAME} beside the pack nor the voxel cache entry {key}"
                        " is on this machine"
                    )
            else:
                why = "the pack names no low grid"
            for given in self.offered:
                try:
                    if given.is_dir():
                        name = given.name
                        positions = self._solved_positions(pack)
                        found[name] = {
                            "key": name,
                            "from": f"given at start: the voxel cache entry {given}",
                            "load": lambda e=given, k=name, p=positions: self._cut(e, k, p),
                        }
                    else:
                        held = read_as_computed(given).grid
                        if held is not None:
                            found[held.key] = {
                                "key": held.key,
                                "from": f"given at start: {given}",
                                "load": lambda g=held: g,
                            }
                except (OSError, ValueError, KeyError):
                    continue
            record = {"own": own, "why": why, "key": key, "grids": found}
            self._sources[pack_id] = record
            return record

    def _grid(self, query: Mapping[str, Any], name: str = "grid") -> GridRecord:
        """The grid a request names: ``<key>@<pack>`` is that pack's own, any other one offered."""
        wanted = str(query.get(name, ""))
        owner = wanted.rsplit("@", 1)[1] if "@" in wanted else query.get("pack", "")
        pack_id, path, pack = self._pack({"pack": owner})
        sources = self._grid_sources(pack_id, path, pack)["grids"]
        if wanted not in sources:
            raise SceneError(404, f"no grid {wanted!r} for this pack")
        with self._lock:
            if wanted not in self._grids:
                try:
                    self._grids[wanted] = sources[wanted]["load"]()
                except (OSError, ValueError, NotImplementedError) as error:
                    raise SceneError(422, f"grid {wanted}: {error}") from error
                # Two grids are held, the two of a comparison; a third takes the oldest's place.
                while len(self._grids) > 3:
                    oldest = next(iter(self._grids))
                    del self._grids[oldest]
                    self._diffs = {k: v for k, v in self._diffs.items() if oldest not in k}
            return self._grids[wanted]

    def _box(self, query: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        centre = np.array([self._number(query, axis) for axis in ("x", "y", "z")])
        radius = min(MAX_RADIUS_M, max(0.2, self._number(query, "r", 3.0)))
        return centre - radius, centre + radius

    def slice(self, query: Mapping[str, Any]) -> Binary:
        grid = self._grid(query)
        classes, layer = cg.slice_classes(grid, self._number(query, "y"))
        return packed(
            {"classes": classes},
            exact="one layer of nodes, a byte a node, none resampled",
            layer=layer,
            y_m=float(grid.origin_m[1] + grid.step_m * layer),
            origin_m=[float(grid.origin_m[0]), float(grid.origin_m[2])],
            step_m=grid.step_m,
            labels=list(grid.labels),
            key=grid.key,
        )

    def walls(self, query: Mapping[str, Any]) -> Binary:
        grid = self._grid(query)
        low, high = self._box(query)
        held = cg.walls(grid, low, high)
        return packed(
            {"corners": held["corners"], "material": held["material"]},
            exact="every link a reached boundary node may not read, as a square half a step "
            "from the node; coplanar squares of one material merged",
            faces=held["faces"],
            quads=held["quads"],
            box_m=held["box_m"],
            labels=list(grid.labels),
            key=grid.key,
            step_m=grid.step_m,
        )

    def _comparison(self, query: Mapping[str, Any]) -> cg.Comparison:
        a, b = self._grid(query, "a"), self._grid(query, "b")
        name = (str(query["a"]), str(query["b"]))
        with self._lock:
            if name not in self._diffs:
                self._diffs = {name: cg.compare(a, b)}
            return self._diffs[name]

    def diff(self, query: Mapping[str, Any]) -> dict[str, Any]:
        return dict(self._comparison(query).numbers)

    def diff_slice(self, query: Mapping[str, Any]) -> Binary:
        held = self._comparison(query)
        classes, layer = cg.diff_slice(held, self._number(query, "y"))
        return packed(
            {"classes": classes},
            derived=held.numbers["method"],
            layer=layer,
            y_m=float(held.a.origin_m[1] + held.a.step_m * layer),
            origin_m=[float(held.a.origin_m[0]), float(held.a.origin_m[2])],
            step_m=held.a.step_m,
        )

    def diff_walls(self, query: Mapping[str, Any]) -> Binary:
        held = self._comparison(query)
        low, high = self._box(query)
        skin = cg.diff_surfaces(held, low, high)
        return packed(
            {"corners": skin["corners"], "value": skin["value"]},
            derived="the skin of the air only A has (0) and only B has (1), on A's nodes",
            step_m=held.a.step_m,
        )

    def diff_openings(self, query: Mapping[str, Any]) -> dict[str, Any]:
        held = self._comparison(query)
        return cg.opening_changes(held.a, held.b, self._number(query, "y"))

    # -- arrays and sources ---------------------------------------------------

    def _legacy(self, path: Path) -> dict[str, Any] | None:
        """A pack older than the archive: its arrays from the plan and a lattice rebuilt.

        The lattice is the voxeliser's own arithmetic on the bundle's export
        (:func:`reverberate.accel.lattice.cart_grid`), which needs the model's
        bounds and nothing of the voxelisation.
        """
        home = path.parent.parent
        plan, bundle = path.parent / "pairs_plan.json", home / "bundle" / "pairs"
        if not plan.is_file() or not (bundle / "campaign.json").is_file():
            return None
        from reverberate.accel.lattice import cart_grid, sim_constants
        from reverberate.accel.pairs import OUTER_RADIUS_M
        from reverberate.experiments.w38_ambisonic_bands import outer_radius_for
        from reverberate.spatial.array import design_array
        from reverberate.trace.computed import snap_arrays, snap_sources
        from reverberate.wave.comms import Grid

        spec = json.loads((bundle / "campaign.json").read_text())
        told = json.loads(plan.read_text())
        model = bundle / str(spec["model_json"])
        if not model.is_file():
            return None
        bounds = (
            self.cache_root / f"bounds-{hashlib.sha256(str(model).encode()).hexdigest()[:12]}.json"
        )
        stamp = [model.stat().st_size, model.stat().st_mtime_ns]
        held = json.loads(bounds.read_text()) if bounds.is_file() else {}
        if held.get("stamp") != stamp:
            groups = json.loads(model.read_text())["mats_hash"].values()
            points = np.concatenate([np.asarray(g["pts"], dtype=float) for g in groups])
            held = {
                "stamp": stamp,
                "bmin": points.min(axis=0).tolist(),
                "bmax": points.max(axis=0).tolist(),
            }
            try:
                self.cache_root.mkdir(parents=True, exist_ok=True)
                bounds.write_text(json.dumps(held))
            except OSError:
                pass
        band = spec["bands"]["low"]
        constants = sim_constants(
            float(spec["tc"]), float(spec["rh"]), float(band["fmax_hz"]), float(spec["ppw"])
        )
        lattice = cart_grid(constants.h, 3.5, np.asarray(held["bmin"]), np.asarray(held["bmax"]))
        grid = Grid(
            h=lattice.h,
            Ts=constants.ts,
            l2=constants.l2,
            fcc_flag=0,
            xv=lattice.xv,
            yv=lattice.yv,
            zv=lattice.zv,
        )
        if abs(float(told.get("grid_step_m", grid.h)) - grid.h) > 1e-9 * grid.h:
            return None
        asked = np.load(bundle / "cells.npy").reshape(-1, 3)
        radius = outer_radius_for(grid.h, OUTER_RADIUS_M)
        designs: list[Any] = []
        on_lattice = True
        for centre in told["centres"]:
            if centre is None:
                designs.append(None)
                continue
            design = design_array(
                np.asarray(centre, dtype=float),
                grid,
                fit_order=int(spec["fit_order"]),
                outer_radius_m=radius,
            )
            on_lattice &= bool(np.allclose(design.centre, centre, rtol=0.0, atol=1e-6 * grid.h))
            designs.append(design)
        centres, offsets, nodes = snap_arrays(designs, grid)
        sources = np.load(bundle / "sources.npy").reshape(-1, 3)
        source_nodes, source_weights = snap_sources(sources, grid)
        return {
            "from": (
                "pairs_plan.json and a lattice rebuilt from the bundle's export: the pack is "
                f"older than {NAME}"
                + ("" if on_lattice else "; A RECORDED CENTRE IS NOT A NODE OF THE REBUILT LATTICE")
            ),
            "origin_m": np.array([grid.xv[0], grid.yv[0], grid.zv[0]]),
            "step_m": grid.h,
            "shape": grid.shape,
            "asked": asked,
            "centre": centres,
            "offsets": offsets,
            "nodes": nodes,
            "source_position": sources,
            "source_nodes": source_nodes,
            "source_weights": source_weights,
        }

    def _snapped(self, pack_id: str, path: Path) -> dict[str, Any] | None:
        archive = self._archive(pack_id, path)
        if archive is not None:
            lattice = archive.grid
            return {
                "from": f"{NAME} beside the pack",
                "origin_m": None if lattice is None else lattice.origin_m,
                "step_m": None if lattice is None else lattice.step_m,
                "shape": None if lattice is None else lattice.shape,
                "asked": archive.cell_asked,
                "centre": archive.cell_centre,
                "offsets": archive.cell_offsets,
                "nodes": archive.cell_nodes,
                "source_position": archive.source_position,
                "source_nodes": archive.source_nodes,
                "source_weights": archive.source_weights,
            }
        with self._lock:
            name = f"legacy:{pack_id}"
            if name not in self._sources:
                try:
                    self._sources[name] = {"held": self._legacy(path)}
                except (OSError, ValueError, KeyError) as error:
                    self._sources[name] = {"held": None, "why": repr(error)[:200]}
            held: dict[str, Any] | None = self._sources[name]["held"]
            return held

    @staticmethod
    def _node_positions(snapped: Mapping[str, Any], flat: np.ndarray) -> np.ndarray:
        subs = np.stack(
            np.unravel_index(np.asarray(flat, dtype=np.int64), snapped["shape"]), axis=-1
        )
        return np.asarray(snapped["origin_m"] + snapped["step_m"] * subs)

    def array(self, query: Mapping[str, Any]) -> Binary:
        pack_id, path, pack = self._pack(query)
        snapped = self._snapped(pack_id, path)
        if snapped is None or snapped["shape"] is None:
            raise SceneError(404, "the nodes of this pack's arrays are not on this machine")
        cell = self._whole(query, "cell", 0, int(snapped["asked"].shape[0]))
        start, stop = int(snapped["offsets"][cell]), int(snapped["offsets"][cell + 1])
        nodes = self._node_positions(snapped, snapped["nodes"][start:stop])
        return packed(
            {"nodes": nodes.astype(np.float32)},
            exact="the nodes the array sampled",
            cell=cell,
            centre_m=_list(snapped["centre"][cell], 5),
            asked_m=_list(snapped["asked"][cell], 5),
        )

    # -- about ------------------------------------------------------------------

    def about(self, query: Mapping[str, Any]) -> dict[str, Any]:
        pack_id, path, pack = self._pack(query)
        provenance = dict(pack.header.provenance)
        assets = dict(provenance.get("assets", {}))
        grids = self._grid_sources(pack_id, path, pack)
        snapped = self._snapped(pack_id, path)
        cells_in_pack = int(pack.cells.position.shape[0])
        cells = []
        for cell in range(cells_in_pack):
            centre = np.asarray(pack.cells.position[cell], dtype=float)
            row: dict[str, Any] = {
                "cell": cell,
                "centre_m": _list(centre, 5),
                "kind": int(pack.cells.kind[cell]),
                "clearance_m": round(float(pack.cells.clearance_m[cell]), 3),
                "asked_m": None,
                "moved_mm": None,
                "nodes": None,
            }
            if snapped is not None and cell < snapped["asked"].shape[0]:
                asked = np.asarray(snapped["asked"][cell], dtype=float)
                row["asked_m"] = _list(asked, 5)
                row["moved_mm"] = round(float(np.linalg.norm(centre - asked)) * 1000.0, 2)
                if snapped["shape"] is not None:
                    row["nodes"] = int(snapped["offsets"][cell + 1] - snapped["offsets"][cell])
            cells.append(row)
        solved = []
        if snapped is not None and snapped["shape"] is not None:
            for row_index in range(int(snapped["source_position"].shape[0])):
                position = np.asarray(snapped["source_position"][row_index], dtype=float)
                nodes = self._node_positions(snapped, snapped["source_nodes"][row_index])
                weights = np.asarray(snapped["source_weights"][row_index], dtype=float)
                heaviest = nodes[int(np.argmax(weights))]
                solved.append(
                    {
                        "position_m": _list(position, 5),
                        "nodes_m": _list(nodes, 5),
                        "weights": _list(weights, 5),
                        "nearest_mm": round(float(np.linalg.norm(heaviest - position)) * 1000.0, 2),
                    }
                )
        tails = {}
        for name, source in pack.sources.items():
            if source.tail is None:
                continue
            tails[name] = [
                {
                    "row": row,
                    "site_m": _list(source.tail.hist_position[row], 4),
                    "cell": int(source.tail.hist_cell[row]),
                }
                for row in range(int(np.asarray(source.tail.hist_cell).shape[0]))
            ]
        mirror_dir = self._mirror_dir(path)
        return {
            "pack": pack_id,
            "name": path.stem if path.stem != "pack" else path.parent.parent.name,
            "recipe_sha256": pack.header.recipe_sha256,
            "assets": assets,
            "solver": str(provenance.get("solver", "")),
            "code_version": str(provenance.get("code_version", "")),
            "steps": pack.header.steps,
            "step_s": pack.header.step_s,
            "bands_hz": [int(v) for v in pack.header.bands_hz],
            "grid": {
                "own": grids["own"],
                "why_none": grids["why"],
                "key": grids["key"],
                "offered": [
                    {
                        "id": name,
                        "key": told["key"],
                        "from": told["from"],
                        "own": name == grids["own"],
                    }
                    for name, told in grids["grids"].items()
                ],
            },
            "snapped": {
                "from": None if snapped is None else snapped["from"],
                "why_none": ""
                if snapped is not None
                else f"neither {NAME} nor pairs_plan.json with the bundle is beside the pack",
                "step_m": None if snapped is None else snapped["step_m"],
            },
            "cells": cells,
            "solved_sources": solved,
            "tails": tails,
            "receiver_radius_m": float(pack.mirror.receiver_radius_m),
            "mirror": {
                "key": str(assets.get("mirror_scene_key", "")),
                "directory": None if mirror_dir is None else str(mirror_dir),
                "why_none": ""
                if mirror_dir is not None
                else "the bundle's trace/mirror is not beside the pack and none was given at start",
            },
        }

    def grid_facts(self, query: Mapping[str, Any]) -> dict[str, Any]:
        return cg.facts(self._grid(query))

    # -- the mirror -------------------------------------------------------------

    def _mirror_dir(self, path: Path) -> Path | None:
        candidates = [path.parent.parent / "bundle" / "trace" / "mirror", path.parent / "mirror"]
        for directory in [*candidates, *self.mirror_dirs]:
            if (directory / "scene.npz").is_file() and (directory / "alignment.json").is_file():
                return directory
        return None

    def _mirror(self, path: Path) -> _Mirror:
        directory = self._mirror_dir(path)
        if directory is None:
            raise SceneError(404, "the mirror's scene of this pack is not on this machine")
        with self._lock:
            name = str(directory)
            if name not in self._mirrors:
                from reverberate.mirror.ism import sheet_layers
                from reverberate.mirror.moving import MovingSettings, prepare
                from reverberate.trace.assets import MirrorAssets

                assets = MirrorAssets.load(directory)
                ms = prepare(assets.catalogue, assets.settings, MovingSettings())
                self._mirrors = {
                    name: _Mirror(directory, assets, ms, np.asarray(sheet_layers(ms.scene)))
                }
            return self._mirrors[name]

    def facets(self, query: Mapping[str, Any]) -> dict[str, Any]:
        _, path, pack = self._pack(query)
        mirror = self._mirror(path)
        scene = mirror.ms.scene
        named = str(dict(pack.header.provenance).get("assets", {}).get("mirror_scene_key", ""))
        found = mirror.assets.catalogue.key
        materials = scene.materials
        return {
            "key": found,
            "pack_names": named,
            "is_the_packs": bool(named) and named == found,
            "exact": "the facets and triangles the image sources and the rays read, with the "
            "calibration's absorption",
            "directory": str(mirror.directory),
            "summary": scene.summary(),
            "labels": list(scene.labels),
            "bands_hz": [int(v) for v in materials.bands_hz],
            "absorption": _list(materials.absorption, 4),
            "scattering": _list(materials.scattering, 4),
            "sheets": int(mirror.sheets.sum()),
            "coincident_facets": str(mirror.ms.ism.coincident_facets),
            "facets": [
                {
                    "label": int(f.label),
                    "kind": f.kind,
                    "normal": _list(f.normal, 5),
                    "offset": round(float(f.offset), 5),
                    "area_m2": round(float(f.area), 3),
                    "triangles": int(f.triangles.size),
                    "two_sided": int(f.sides) == 3,
                    "sheet_layer": bool(mirror.sheets[k]),
                }
                for k, f in enumerate(scene.facets)
            ],
            "occluder_triangles": int(scene.occluder_vertices.shape[0]),
            "reflector_triangles": int(scene.reflector_vertices.shape[0]),
            "rays": mirror.assets.settings.traced_rays().record(),
            "ism": mirror.assets.settings.ism.record(),
        }

    def triangles(self, query: Mapping[str, Any]) -> Binary:
        _, path, _ = self._pack(query)
        scene = self._mirror(path).ms.scene
        layer = str(query.get("layer", "reflectors"))
        if layer == "reflectors":
            return packed(
                {
                    "vertices": np.asarray(scene.reflector_vertices, dtype=np.float32),
                    "facet": np.asarray(scene.reflector_facet, dtype=np.int32),
                },
                exact="the reflector triangles, each with its facet",
            )
        if layer == "occluders":
            return packed(
                {
                    "vertices": np.asarray(scene.occluder_vertices, dtype=np.float32),
                    "label": np.asarray(scene.occluder_label, dtype=np.int16),
                },
                exact="the occluder triangles, which the rays bounce on and the paths are "
                "blocked by, each with its material",
            )
        raise SceneError(400, "layer is reflectors or occluders")

    def edges(self, query: Mapping[str, Any]) -> Binary:
        _, path, _ = self._pack(query)
        mirror = self._mirror(path)
        if mirror.edges is None:
            from reverberate.mirror.edges import diffracting_edges

            mirror.edges = diffracting_edges(mirror.assets.catalogue)
        held = mirror.edges
        return packed(
            {
                "a": np.asarray(held.a, dtype=np.float32),
                "b": np.asarray(held.b, dtype=np.float32),
                "facet": np.asarray(held.facet, dtype=np.int32),
            },
            exact="the edges sound is bent round for a step that does not see its source",
            **held.record(),
        )

    def paths(self, query: Mapping[str, Any]) -> dict[str, Any]:
        """A step's rows of ``early``, each with its corners where this code finds it again."""
        from reverberate.mirror.moving import trace_early
        from reverberate.viz.computed_rays import path_points

        _, path, pack = self._pack(query)
        source = self._source(pack, query)
        step = self._whole(query, "step", 0, pack.header.steps)
        mirror = self._mirror(path)
        rows = source.early.rows(step)
        ids = [int(v) for v in np.asarray(source.early.path_id[rows])]
        kinds = [int(v) for v in np.asarray(source.early.kind[rows])]
        orders = [int(v) for v in np.asarray(source.early.order[rows])]
        delays = np.asarray(source.early.delay_s[rows], dtype=float)
        here = np.asarray(source.position[step], dtype=float)
        head = np.asarray(pack.listener.position[step], dtype=float)
        listed: list[dict[str, Any]] = [
            {"kind": k, "order": o, "delay_s": float(d), "points": None, "facets": None}
            for k, o, d in zip(kinds, orders, delays, strict=True)
        ]
        extra = 0
        worst = 0.0
        if ids:
            heads = np.concatenate([pack.listener.position, pack.cells.position])
            region = (heads.min(axis=0) - 0.5, heads.max(axis=0) + 0.5)
            with self._lock:
                table = trace_early(mirror.ms, here[None, :], head[None, :], region=region)
            where = {name: row for row, name in enumerate(ids)}
            for row in range(int(table.path_id.shape[0])):
                at = where.get(int(table.path_id[row]))
                if at is None:
                    extra += 1
                    continue
                sequence = [int(f) for f in table.sequence[row] if int(f) >= 0]
                corners = path_points(
                    here, head, table.sequence[row], mirror.ms.normals, mirror.ms.offsets
                )
                listed[at]["points"] = _list(corners, 4)
                listed[at]["facets"] = sequence
                worst = max(worst, abs(float(table.delay_s[row]) - float(delays[at])))
        found = sum(1 for row in listed if row["points"] is not None)
        return {
            "step": step,
            "source_m": _list(here, 4),
            "listener_m": _list(head, 4),
            "rows": listed,
            "found": found,
            "not_found": len(listed) - found,
            "not_in_the_pack": extra,
            "worst_delay_difference_s": worst,
            "recomputed": "by this machine's reverberate.mirror.moving.trace_early for this step, "
            "matched to the pack's rows by path id; a diffracted onset is not recomputed, and a "
            "row this code does not find again is drawn as its direction alone",
            "pack_code_version": str(dict(pack.header.provenance).get("code_version", "")),
        }

    def rays(self, query: Mapping[str, Any]) -> Binary:
        from reverberate.mirror.ism import occluder_grid
        from reverberate.viz.computed_rays import ray_paths

        pack_id, path, pack = self._pack(query)
        source = self._source(pack, query)
        if source.tail is None:
            raise SceneError(404, "the source has no tail")
        hists = int(np.asarray(source.tail.hist_cell).shape[0])
        row = self._whole(query, "row", 0, max(hists, 1))
        count = min(MAX_RAYS, max(1, int(self._number(query, "count", 200))))
        mirror = self._mirror(path)
        settings = mirror.assets.settings.traced_rays()
        site = np.asarray(source.tail.hist_position[row], dtype=float)
        cell = int(source.tail.hist_cell[row])
        name = (pack_id, tuple(np.round(site, 6)), cell, count)
        with self._lock:
            if name not in self._rays:
                if mirror.grid is None:
                    mirror.grid = occluder_grid(mirror.ms.scene, settings.cell_m)
                self._rays = {
                    name: ray_paths(
                        mirror.ms.scene,
                        site,
                        settings,
                        count=count,
                        receivers=np.asarray(pack.cells.position[cell], dtype=float)[None, :],
                        grid=mirror.grid,
                        max_bounces=RAY_BOUNCES,
                    )
                }
            held = self._rays[name]
        crossings = np.asarray(held["crossings"], dtype=float).reshape(-1, 3)
        return packed(
            {
                "offsets": held["offsets"].astype(np.int32),
                "points": held["points"].astype(np.float32),
                "energy": held["energy"].astype(np.float32),
                "scattered": held["scattered"].astype(np.uint8),
                "crossing_vertex": crossings[:, 1].astype(np.int32),
                "crossing_s": crossings[:, 2].astype(np.float32),
            },
            sample=f"rays {held['first']} to {held['first'] + count - 1} of the "
            f"{settings.rays} the histogram summed, traced again on this machine by the "
            f"tracer's law (same directions, same seed {settings.seed}, same scene); each "
            f"followed for {RAY_BOUNCES} bounces at most",
            count=count,
            of=settings.rays,
            truncated=held["truncated"],
            site_m=_list(site, 4),
            cell=cell,
            cell_m=_list(pack.cells.position[cell], 4),
            receiver_radius_m=settings.receiver_radius_m,
            bands_hz=[int(v) for v in mirror.ms.scene.materials.bands_hz],
        )

    def tail(self, query: Mapping[str, Any]) -> dict[str, Any]:
        from reverberate.spatial.sh import real_sh, scene_to_ambisonic

        _, _, pack = self._pack(query)
        source = self._source(pack, query)
        if source.tail is None:
            raise SceneError(404, "the source has no tail")
        hists = int(np.asarray(source.tail.hist_cell).shape[0])
        row = self._whole(query, "row", 0, max(hists, 1))
        energy = np.asarray(source.tail.energy[row], dtype=float)
        moments = np.asarray(source.tail.moments[row], dtype=float)
        rows, columns = SHAPE_GRID
        el = np.linspace(-0.5 * np.pi, 0.5 * np.pi, rows)[:, None]
        az = (2.0 * np.pi * np.arange(columns) / columns)[None, :]
        grid = np.stack(
            [np.cos(el) * np.cos(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.sin(az)],
            axis=-1,
        )
        basis = real_sh(int(pack.mirror.histogram_order), scene_to_ambisonic(grid.reshape(-1, 3)))
        shape = np.maximum(basis[:, : moments.shape[-1]] @ moments.sum(axis=0).T, 0.0)
        peak = float(shape.max())
        return {
            "exact": "the pack's own histogram: energy against time per band, and the shape its "
            "directional moments give, summed over time",
            "row": row,
            "site_m": _list(source.tail.hist_position[row], 4),
            "cell": int(source.tail.hist_cell[row]),
            "bin_s": float(pack.mirror.histogram_bin_s),
            "bands_hz": [int(v) for v in pack.header.bands_hz],
            "energy": [[float(f"{v:.4g}") for v in band] for band in energy.T],
            "shape": {
                "elevations": rows,
                "azimuths": columns,
                "directions": _list(grid.reshape(-1), 4),
                # ``[band][direction]``, the largest value of all one.
                "energy": _list(shape.T / peak, 3) if peak > 0.0 else None,
            },
        }

    def low(self, query: Mapping[str, Any]) -> dict[str, Any]:
        _, _, pack = self._pack(query)
        source = self._source(pack, query)
        step = self._whole(query, "step", 0, pack.header.steps)
        if source.low is None:
            raise SceneError(404, "the source has no low band")
        return low_at(pack, source, step)

    # -- routing ----------------------------------------------------------------

    def handle(self, method: str, parts: list[str], query: Mapping[str, Any]) -> Any:
        """Answer ``parts``, the path below :data:`PREFIX`; raises :class:`SceneError`."""
        if method == "GET":
            match parts:
                case ["about"]:
                    return self.about(query)
                case ["grid", "facts"]:
                    return self.grid_facts(query)
                case ["grid", "slice"]:
                    return self.slice(query)
                case ["grid", "walls"]:
                    return self.walls(query)
                case ["grid", "array"]:
                    return self.array(query)
                case ["diff"]:
                    return self.diff(query)
                case ["diff", "slice"]:
                    return self.diff_slice(query)
                case ["diff", "walls"]:
                    return self.diff_walls(query)
                case ["diff", "openings"]:
                    return self.diff_openings(query)
                case ["mirror", "facets"]:
                    return self.facets(query)
                case ["mirror", "triangles"]:
                    return self.triangles(query)
                case ["mirror", "edges"]:
                    return self.edges(query)
                case ["paths"]:
                    return self.paths(query)
                case ["rays"]:
                    return self.rays(query)
                case ["tail"]:
                    return self.tail(query)
                case ["low"]:
                    return self.low(query)
        raise SceneError(404, f"no such endpoint: {method} /{PREFIX}/{'/'.join(parts)}")
