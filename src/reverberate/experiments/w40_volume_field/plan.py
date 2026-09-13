"""The listening grid, the arrays of every point, and the plan of one campaign.

A plan is one JSON file, ``plan.json``: the points, their rooms, the sources,
and per band the cache key of the grid it is solved on, the rows of every
point's array in the engine's output, and the cost. Everything downstream
reads the plan and nothing else.

**Every listening point is a whole array, not a receiver.** Order 7 fitted at
10 needs about a thousand grid nodes per point per band, so the engine's
output is ``points x 1000`` rows; the number of points one solve can carry is
set by the host's RAM, and :mod:`.solve` slices a band that does not fit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from reverberate.experiments.engine import sim_consts, write_record
from reverberate.experiments.run import entry_from_key, sound_speed
from reverberate.experiments.w10_ambisonic import COMMS_NAME, cost_record
from reverberate.experiments.w38_ambisonic_bands import outer_radius_for
from reverberate.spatial.array import ArrayDesign, design_array
from reverberate.spatial.encode import EncoderSettings
from reverberate.wave.comms import Grid, engine_indices, load_grid, write_comms
from reverberate.wave.voxelise import CacheEntry

__all__ = [
    "COMMS_NAME",
    "DELIVERY_RATE_HZ",
    "bands_of_source",
    "free_floor",
    "grid_points",
    "pitch_for",
    "place_arrays",
    "plan_field",
    "points_per_run",
    "prepare_field",
    "slug",
]

DELIVERY_RATE_HZ = 48000.0

#: The low cut the owner kept for W39: 40 Hz, order 8.
LOWCUT_HZ = 40.0
LOWCUT_ORDER = 8

#: Bytes the engine holds in host memory per receiver row and per step.
BYTES_PER_SAMPLE = 8

#: Free floor: the walkable outline pulled in from the walls, and the
#: furniture pushed out, by this much. The mid band ball is 16.5 cm plus a
#: 2 cm margin, so a point closer than that to anything fails its clearance.
WALL_SETBACK_M = 0.25
FURNITURE_CLEARANCE_M = 0.22

#: Horizontal shifts tried, in order, for a low band ball that touches a
#: boundary; the low band ball is 41 cm and a listener stands closer than that
#: to a wall all the time. Below 800 Hz a quarter of a metre is under a
#: quarter wavelength, and the shift is recorded per point rather than hidden.
LOW_SHIFTS_M = (0.10, 0.20, 0.30, 0.40)

TRICK = (
    "one solve per source and band carries every listening point of the grid "
    "as its own order 7 array; the pressure is encoded where it was computed "
    "and only the spherical harmonic signals come home"
)


# --------------------------------------------------------------------------
# the listening grid
# --------------------------------------------------------------------------


def free_floor(hssd_root: Path, scene_id: str) -> tuple[Any, Polygon | MultiPolygon, list[Any]]:
    """The storey, the floor a listener may stand on, and the everyday rooms."""
    from reverberate.geometry.apartment import build_apartment, instances_on_storey
    from reverberate.geometry.hssd_room import load_object_instances
    from reverberate.geometry.placement import furniture_footprints, sampling_area

    storeys = build_apartment(hssd_root, scene_id)
    storey = storeys[0]
    instances = instances_on_storey(
        load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json"),
        storey,
        storeys,
    )
    footprints = furniture_footprints(hssd_root, instances, storey.floor_height)
    area = sampling_area(
        storey, footprints, min_wall_distance=WALL_SETBACK_M, clearance=FURNITURE_CLEARANCE_M
    )
    rooms = [room for room in storey.everyday if not room.outdoor]
    return storey, area, rooms


def grid_points(
    area: Polygon | MultiPolygon, rooms: list[Any], *, pitch_m: float, height_m: float
) -> tuple[np.ndarray, list[str]]:
    """A square lattice over the free floor, each point labelled with its room."""
    min_x, min_z, max_x, max_z = area.bounds
    xs = np.arange(min_x, max_x + 1e-9, pitch_m)
    zs = np.arange(min_z, max_z + 1e-9, pitch_m)
    points, labels = [], []
    for x in xs:
        for z in zs:
            here = Point(float(x), float(z))
            if not area.contains(here):
                continue
            room = next((r.name for r in rooms if r.polygon.contains(here)), None)
            if room is None:
                continue
            points.append([float(x), height_m, float(z)])
            labels.append(room)
    return np.asarray(points, dtype=float).reshape(-1, 3), labels


def pitch_for(area_m2: float, max_points: int, *, step_m: float = 0.05) -> float:
    """The finest pitch, on a 5 cm ladder, that keeps the grid under ``max_points``."""
    pitch = math.sqrt(area_m2 / max_points)
    return math.ceil(pitch / step_m) * step_m


def points_per_run(ram_gb: float, nodes_per_point: int, steps: int, *, safety: float = 1.5) -> int:
    """How many listening points one solve can carry on a machine with this RAM."""
    per_point = nodes_per_point * steps * BYTES_PER_SAMPLE * safety
    return int(ram_gb * 1e9 / per_point)


# --------------------------------------------------------------------------
# clearance of many balls at once
# --------------------------------------------------------------------------


class Boundary:
    """The boundary nodes of one grid, sorted once, for many ball checks."""

    def __init__(self, entry_path: Path, grid: Grid) -> None:
        with h5py.File(entry_path / "vox_out.h5", "r") as handle:
            nodes = np.asarray(handle["bn_ixyz"][...], dtype=np.int64)
        self.nodes = np.sort(nodes)
        self.grid = grid

    def ball(self, centre: np.ndarray, radius: float) -> np.ndarray | None:
        """Engine indices of every node in the ball, or ``None`` at the grid's edge."""
        grid = self.grid
        axes = (grid.xv, grid.yv, grid.zv)
        ranges = []
        for j in range(3):
            low = int(np.searchsorted(axes[j], centre[j] - radius, side="left"))
            high = int(np.searchsorted(axes[j], centre[j] + radius, side="right"))
            if low <= 0 or high >= len(axes[j]):
                return None
            ranges.append(np.arange(low, high, dtype=np.int64))
        nx, ny, nz = grid.shape
        ix, iy, iz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
        inside = (
            (axes[0][ix] - centre[0]) ** 2
            + (axes[1][iy] - centre[1]) ** 2
            + (axes[2][iz] - centre[2]) ** 2
        ) <= radius**2
        flat = (ix * (ny * nz) + iy * nz + iz)[inside]
        return engine_indices(flat, grid)

    def hits(self, centre: np.ndarray, radius: float) -> int | None:
        ball = self.ball(centre, radius)
        if ball is None:
            return None
        position = np.searchsorted(self.nodes, ball)
        position = np.clip(position, 0, self.nodes.size - 1)
        return int(np.count_nonzero(self.nodes[position] == ball))


def place_arrays(
    points: np.ndarray,
    entry_path: Path,
    *,
    settings: EncoderSettings,
    outer_radius_m: float,
    margin_m: float = 0.02,
    shifts_m: tuple[float, ...] = (),
) -> tuple[list[ArrayDesign | None], list[dict[str, Any]]]:
    """One array per point on this grid, or ``None`` where no clear ball exists.

    A point whose ball touches a boundary is tried again at each of
    ``shifts_m`` in eight horizontal directions before it is given up; the
    shift that worked is recorded on the point.
    """
    grid = load_grid(entry_path)
    boundary = Boundary(entry_path, grid)
    directions = [
        np.array([math.cos(a), 0.0, math.sin(a)])
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
    ]
    designs: list[ArrayDesign | None] = []
    records: list[dict[str, Any]] = []
    for point in points:
        chosen: ArrayDesign | None = None
        record: dict[str, Any] = {"shift_m": 0.0, "hits": None}
        candidates = [point] + [point + s * d for s in shifts_m for d in directions]
        for candidate in candidates:
            try:
                design = design_array(
                    candidate, grid, fit_order=settings.fit_order, outer_radius_m=outer_radius_m
                )
            except ValueError:
                continue
            hits = boundary.hits(design.centre, float(design.radii.max()) + margin_m)
            if hits == 0:
                chosen = design
                record = {
                    "shift_m": round(float(np.linalg.norm(candidate - point)), 3),
                    "hits": 0,
                }
                break
            if record["hits"] is None:
                record["hits"] = hits
        designs.append(chosen)
        records.append(record)
    return designs, records


# --------------------------------------------------------------------------
# plan and prepare
# --------------------------------------------------------------------------


def _settings(order: int, fit_order: int, fmax_hz: float) -> EncoderSettings:
    return EncoderSettings(order=order, fit_order=fit_order, max_frequency_hz=fmax_hz)


def plan_field(
    out: Path,
    *,
    hssd_root: Path,
    scene_id: str,
    sources: list[dict[str, Any]],
    storey_keys: dict[str, str],
    room_keys: dict[str, str],
    durations_s: dict[str, float],
    height_m: float,
    ram_gb: float,
    pitch_m: float | None = None,
    order: int = 7,
    fit_order: int = 10,
    outer_radius_m: float = 0.16,
    high_key: str | None = None,
) -> dict[str, Any]:
    """Choose the grid, place every array on every band's grid, write the plan.

    ``sources`` are ``{"name", "room", "position"}``; ``room_keys`` maps a room
    name to its high band cache key; ``storey_keys`` holds ``low`` and ``mid``.
    """
    out.mkdir(parents=True, exist_ok=True)
    storey, area, rooms = free_floor(hssd_root, scene_id)
    nodes_estimate = 6 * int(math.ceil(1.4 * (fit_order + 1) ** 2)) + 1
    entries = {name: entry_from_key(key) for name, key in storey_keys.items()}
    steps = {
        name: int(round(durations_s[name] * sim_consts(entries[name].path).sample_rate))
        for name in ("low", "mid")
    }
    cap = min(points_per_run(ram_gb, nodes_estimate, steps[name]) for name in ("low", "mid"))
    if pitch_m is None:
        pitch_m = pitch_for(float(area.area), cap)
    points, labels = grid_points(area, rooms, pitch_m=pitch_m, height_m=height_m)
    if points.shape[0] > cap:
        raise ValueError(
            f"{points.shape[0]} points at {pitch_m:.2f} m exceed the {cap} the RAM budget "
            f"of {ram_gb:.0f} GB allows; coarsen the pitch"
        )

    bands: dict[str, dict[str, Any]] = {}
    per_point: dict[str, list[dict[str, Any]]] = {}
    arrays: dict[str, list[ArrayDesign | None]] = {}
    for name in ("low", "mid"):
        entry = entries[name]
        fmax = float(str(entry.manifest["fmax"]))
        step = float(str(entry.manifest["h_m"]))
        designs, records = place_arrays(
            points,
            entry.path,
            settings=_settings(order, fit_order, fmax),
            outer_radius_m=outer_radius_for(step, outer_radius_m),
            shifts_m=LOW_SHIFTS_M if name == "low" else (0.05, 0.10),
        )
        arrays[name] = designs
        per_point[name] = records
        bands[name] = _band_record(name, entry, fmax, step, durations_s[name], designs)
    # The high band is per room: one entry per source room, and only the
    # points standing in that room get an array on it.
    for room, key in room_keys.items():
        entry = entry_from_key(key)
        fmax = float(str(entry.manifest["fmax"]))
        step = float(str(entry.manifest["h_m"]))
        inside = np.array([label == room for label in labels])
        designs_room, records_room = place_arrays(
            points[inside],
            entry.path,
            settings=_settings(order, fit_order, fmax),
            outer_radius_m=outer_radius_for(step, outer_radius_m),
            shifts_m=(0.05, 0.10),
        )
        room_designs: list[ArrayDesign | None] = [None] * points.shape[0]
        room_records: list[dict[str, Any]] = [{"shift_m": None, "hits": None}] * points.shape[0]
        for index, design, record in zip(
            np.flatnonzero(inside), designs_room, records_room, strict=True
        ):
            room_designs[int(index)] = design
            room_records[int(index)] = record
        band_name = f"high:{room}"
        arrays[band_name] = room_designs
        per_point[band_name] = room_records
        bands[band_name] = _band_record(
            band_name, entry, fmax, step, durations_s["high"], room_designs
        )

    # A storey solved at 8 kHz carries every point in one high band: no sealed
    # doorway, no point without its top octave, no level seam between rooms.
    if high_key:
        entry = entry_from_key(high_key)
        fmax = float(str(entry.manifest["fmax"]))
        step = float(str(entry.manifest["h_m"]))
        designs_all, records_all = place_arrays(
            points,
            entry.path,
            settings=_settings(order, fit_order, fmax),
            outer_radius_m=outer_radius_for(step, outer_radius_m),
            shifts_m=(0.05, 0.10),
        )
        arrays["high"] = designs_all
        per_point["high"] = records_all
        bands["high"] = _band_record("high", entry, fmax, step, durations_s["high"], designs_all)

    # Row layout per band: point after point, its nodes contiguous, so the
    # encoder reads one slice per point.
    for name, designs in arrays.items():
        rows: list[list[int] | None] = []
        positions = []
        cursor = 0
        for _index, design in enumerate(designs):
            if design is None:
                rows.append(None)
                continue
            rows.append([cursor, cursor + design.count])
            positions.append(design.positions)
            cursor += design.count
        bands[name]["rows"] = rows
        bands[name]["centres"] = [
            None if d is None else [float(v) for v in d.centre] for d in designs
        ]
        bands[name]["receivers"] = cursor
        stacked = np.vstack(positions) if positions else np.zeros((0, 3))
        np.save(out / f"array_positions_{slug(name)}.npy", stacked)

    plan = {
        "run": out.name,
        "scene_id": scene_id,
        "dwelling": _dwelling(scene_id),
        "kind": "ambisonic field over the walkable volume",
        "trick": TRICK,
        "height_m": height_m,
        "pitch_m": pitch_m,
        "ram_budget_gb": ram_gb,
        "points_cap": cap,
        "free_floor_m2": round(float(area.area), 2),
        "points": points.tolist(),
        "rooms": labels,
        "sources": sources,
        "sound_speed_m_s": sound_speed(),
        "encoder": _settings(order, fit_order, 0.0).record(),
        "lowcut_hz": LOWCUT_HZ,
        "lowcut_order": LOWCUT_ORDER,
        "bands": bands,
        "per_point": per_point,
    }
    write_record(out, "plan.json", plan)
    return plan


def _band_record(
    name: str,
    entry: CacheEntry,
    fmax: float,
    step: float,
    duration_s: float,
    designs: list[ArrayDesign | None],
) -> dict[str, Any]:
    constants = sim_consts(entry.path)
    samples = int(round(duration_s * constants.sample_rate))
    placed = [d for d in designs if d is not None]
    receivers = sum(d.count for d in placed)
    return {
        "band": name,
        "cache_key": entry.key,
        "cache_root": str(entry.path.parent),
        "fmax_hz": fmax,
        "grid_step_m": step,
        "sample_rate_hz": constants.sample_rate,
        "duration_s": duration_s,
        "samples": samples,
        "points_placed": len(placed),
        "points_missing": len(designs) - len(placed),
        "nodes_per_point": round(receivers / len(placed), 1) if placed else None,
        "cost": cost_record(int(str(entry.manifest["grid_points"])), samples, receivers),
    }


def _dwelling(scene_id: str) -> str:
    try:
        from reverberate.geometry.scene_ids import local_name

        return local_name(scene_id)
    except (KeyError, ImportError):
        return scene_id


def slug(band: str) -> str:
    return band.replace(":", "__").replace(" ", "_").replace(".", "_")


def bands_of_source(plan: dict[str, Any], source: dict[str, Any]) -> list[str]:
    if "high" in plan["bands"]:
        return ["low", "mid", "high"]
    high = f"high:{source['room']}"
    return ["low", "mid"] + ([high] if high in plan["bands"] else [])


def prepare_field(out: Path) -> list[Path]:
    """One comms file per source and band, from the plan."""
    plan = json.loads((out / "plan.json").read_text())
    written = []
    for source in plan["sources"]:
        for band in bands_of_source(plan, source):
            record = plan["bands"][band]
            entry = entry_from_key(record["cache_key"])
            positions = np.load(out / f"array_positions_{slug(band)}.npy")
            comms_dir = out / source["name"] / slug(band) / "comms"
            comms_dir.mkdir(parents=True, exist_ok=True)
            path = write_comms(
                entry.path,
                np.asarray(source["position"], dtype=float),
                positions,
                record["samples"] / record["sample_rate_hz"],
                diff_source=True,
                out_path=comms_dir / COMMS_NAME,
                interpolation="nearest",
            )
            written.append(path)
            print(f"{source['name']} {band}: {positions.shape[0]} receivers -> {path}")
    return written
