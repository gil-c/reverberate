"""The impulse response field: what the app reads, and a mock of it.

A field is what the producer of a run records for one source: an ambisonic
response at every listening cell of a regular lattice over the flat's air, so
that a listener can be decoded to two ears anywhere without another solve. It
travels as one HDF5 per source, in the shape agreed with the producer session
and written down in ``docs/formats/response-field.md``.

The browser reads the file itself, by HTTP range requests: :func:`build_site`
writes an ``index.json`` naming the byte range of every cell's response, which
works because the responses are stored uncompressed, one contiguous block per
position. :func:`mock_field` writes a field of the same shape from nothing the
solver produced, for building and testing the app without one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic

__all__ = ["FieldHeader", "build_site", "check", "mock_field", "read_header", "write_field"]

ORDERING = "ACN"
NORMALISATION = "N3D"
SOUND_SPEED_M_S = 343.2
REQUIRED_DATASETS = ("ir", "positions", "cell_index", "direct_path_m")
PER_POSITION_DATASETS = ("rooms", "has_high", "solved_to_hz")
REQUIRED_ATTRIBUTES = (
    "order",
    "ordering",
    "normalisation",
    "sample_rate_hz",
    "source_id",
    "source_position",
    "directivity",
    "gain",
    "grid_origin_m",
    "grid_step_m",
    "grid_shape",
    "scene_id",
    "dwelling",
)


@dataclass(frozen=True)
class FieldHeader:
    """What a field says about itself, before any response is read."""

    source_id: str
    source_position: tuple[float, ...]
    directivity: str
    order: int
    channels: int
    sample_rate_hz: float
    samples: int
    positions: int
    grid_origin_m: tuple[float, ...]
    grid_step_m: tuple[float, ...]
    grid_shape: tuple[int, ...]
    gain: float
    scene_id: str
    dwelling: str


def _attr(handle: h5py.File, name: str) -> Any:
    value = handle.attrs[name]
    return value.decode() if isinstance(value, bytes) else value


def _ranges(dataset: h5py.Dataset) -> list[tuple[int, int]]:
    """Byte offset and size of each leading-index slab of a dataset in its file.

    Contiguous storage puts every slab at a stride; chunked storage with one
    chunk per leading index puts each where the file says. Anything else, a
    compression or a chunk spanning positions, cannot be range-read and is
    refused by name, because the page would otherwise read garbage as sound.
    """
    name = dataset.name.strip("/")
    if dataset.compression is not None or dataset.dtype.str != "<f4":
        raise ValueError(
            f"/{name} must be uncompressed little-endian float32 to be range read; "
            f"it is {dataset.dtype.str} with compression {dataset.compression!r}"
        )
    count = int(dataset.shape[0])
    slab = int(np.prod(dataset.shape[1:])) * 4
    if dataset.chunks is None:
        base = dataset.id.get_offset()
        if base is None:
            raise ValueError(f"/{name} has no storage offset")
        return [(int(base) + index * slab, slab) for index in range(count)]
    if tuple(dataset.chunks) != (1, *dataset.shape[1:]):
        raise ValueError(
            f"/{name} is chunked as {dataset.chunks}; a range read needs one chunk "
            f"per position, {(1, *dataset.shape[1:])}"
        )
    ranges: list[tuple[int, int] | None] = [None] * count
    for index in range(dataset.id.get_num_chunks()):
        info = dataset.id.get_chunk_info(index)
        ranges[int(info.chunk_offset[0])] = (int(info.byte_offset), int(info.size))
    if any(r is None for r in ranges):
        raise ValueError(f"/{name} has positions with no chunk written")
    return [r for r in ranges if r is not None]


def _problems(handle: h5py.File) -> list[str]:
    """Everything wrong with a field, in one pass, so the producer fixes it once."""
    found = [f"missing dataset /{name}" for name in REQUIRED_DATASETS if name not in handle]
    found += [
        f"missing root attribute {name!r}"
        for name in REQUIRED_ATTRIBUTES
        if name not in handle.attrs
    ]
    if found:
        return found
    for name, wanted in (("ordering", ORDERING), ("normalisation", NORMALISATION)):
        if _attr(handle, name) != wanted:
            found.append(f"{name} is {_attr(handle, name)!r}, the app reads {wanted} only")
    ir = handle["ir"]
    if ir.ndim != 3:
        return found + [f"/ir has {ir.ndim} axes, expected [position, channel, sample]"]
    order = int(_attr(handle, "order"))
    if ir.shape[1] != channel_count(order):
        found.append(f"/ir has {ir.shape[1]} channels, order {order} needs {channel_count(order)}")
    try:
        _ranges(ir)
    except ValueError as error:
        found.append(str(error))
    n = ir.shape[0]
    shapes = {"positions": (n, 3), "cell_index": (n, 3), "direct_path_m": (n,)}
    shapes.update({name: (n,) for name in PER_POSITION_DATASETS if name in handle})
    for name, shape in shapes.items():
        if tuple(handle[name].shape) != shape:
            found.append(f"/{name} is {tuple(handle[name].shape)}, expected {shape}")
    grid = np.asarray(_attr(handle, "grid_shape"))
    cells = np.asarray(handle["cell_index"][:])
    if cells.size and ((cells < 0).any() or (cells >= grid[None, :]).any()):
        found.append("/cell_index has an index outside grid_shape")
    if cells.size and len(np.unique(cells, axis=0)) != n:
        found.append("/cell_index names a cell twice")
    return found


def check(path: Path) -> list[str]:
    """The problems a field has, empty when the app can read it."""
    with h5py.File(path, "r") as handle:
        return _problems(handle)


def read_header(path: Path) -> FieldHeader:
    """Read and validate a field's header; raise on anything the app cannot take."""
    with h5py.File(path, "r") as handle:
        problems = _problems(handle)
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
        ir = handle["ir"]
        return FieldHeader(
            source_id=str(_attr(handle, "source_id")),
            source_position=tuple(float(v) for v in _attr(handle, "source_position")),
            directivity=str(_attr(handle, "directivity")),
            order=int(_attr(handle, "order")),
            channels=int(ir.shape[1]),
            sample_rate_hz=float(_attr(handle, "sample_rate_hz")),
            samples=int(ir.shape[2]),
            positions=int(ir.shape[0]),
            grid_origin_m=tuple(float(v) for v in _attr(handle, "grid_origin_m")),
            grid_step_m=tuple(float(v) for v in _attr(handle, "grid_step_m")),
            grid_shape=tuple(int(v) for v in _attr(handle, "grid_shape")),
            gain=float(_attr(handle, "gain")),
            scene_id=str(_attr(handle, "scene_id")),
            dwelling=str(_attr(handle, "dwelling")),
        )


def write_field(
    path: Path,
    *,
    ir: np.ndarray,
    positions: np.ndarray,
    cell_index: np.ndarray,
    direct_path_m: np.ndarray,
    grid_origin_m: Sequence[float],
    grid_step_m: Sequence[float],
    grid_shape: Sequence[int],
    order: int,
    sample_rate_hz: float,
    source_id: str,
    source_position: Sequence[float],
    directivity: str,
    gain: float,
    scene_id: str,
    dwelling: str,
    provenance: dict[str, Any],
    rooms: Sequence[str] | None = None,
    has_high: np.ndarray | None = None,
    solved_to_hz: np.ndarray | None = None,
) -> Path:
    """Write a field in the agreed shape, one chunk per position so it range reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "ir", data=np.asarray(ir, dtype=np.float32), chunks=(1, *ir.shape[1:])
        )
        handle["positions"] = np.asarray(positions, dtype=np.float64)
        handle["cell_index"] = np.asarray(cell_index, dtype=np.int32)
        handle["direct_path_m"] = np.asarray(direct_path_m, dtype=np.float64)
        if rooms is not None:
            handle["rooms"] = np.asarray(list(rooms), dtype=h5py.string_dtype())
        if has_high is not None:
            handle["has_high"] = np.asarray(has_high, dtype=bool)
        if solved_to_hz is not None:
            handle["solved_to_hz"] = np.asarray(solved_to_hz, dtype=np.float64)
        handle.attrs.update(
            {
                "order": int(order),
                "ordering": ORDERING,
                "normalisation": NORMALISATION,
                "sample_rate_hz": float(sample_rate_hz),
                "source_id": source_id,
                "source_position": np.asarray(source_position, dtype=np.float64),
                "directivity": directivity,
                "gain": float(gain),
                "grid_origin_m": np.asarray(grid_origin_m, dtype=np.float64),
                "grid_step_m": np.asarray(grid_step_m, dtype=np.float64),
                "grid_shape": np.asarray(grid_shape, dtype=np.int64),
                "scene_id": scene_id,
                "dwelling": dwelling,
                "provenance_json": json.dumps(provenance, sort_keys=True),
            }
        )
    return path


def build_site(field: Path, target: Path) -> dict[str, Any]:
    """Describe a field to the page: ``index.json`` beside a link to the file.

    The index carries the header, every cell's position and lattice index,
    the optional per-position datasets, and the byte range of every cell, so
    the page reads a cell straight out of the HDF5. Rewritten only when the
    file changes, by size and modification time.
    """
    field, target = Path(field), Path(target)
    stamp = {"size": field.stat().st_size, "mtime": field.stat().st_mtime}
    target.mkdir(parents=True, exist_ok=True)
    link = target / "field.h5"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(field.resolve())
    index_file = target / "index.json"
    if index_file.is_file():
        previous: dict[str, Any] = json.loads(index_file.read_text())
        if previous.get("field") == stamp:
            return previous
    header = read_header(field)
    with h5py.File(field, "r") as handle:
        ranges = _ranges(handle["ir"])
        record: dict[str, Any] = {
            "field": stamp,
            **asdict(header),
            "ordering": ORDERING,
            "normalisation": NORMALISATION,
            "cell_index": np.asarray(handle["cell_index"][:]).tolist(),
            "positions": np.round(np.asarray(handle["positions"][:]), 4).tolist(),
            "direct_path_m": np.round(np.asarray(handle["direct_path_m"][:]), 4).tolist(),
            "file": "field.h5",
            "offsets": [offset for offset, _ in ranges],
            "cell_bytes": ranges[0][1],
            "provenance": json.loads(str(_attr(handle, "provenance_json"))),
        }
        for name in PER_POSITION_DATASETS:
            if name not in handle:
                record[name] = None
            elif name == "rooms":
                record[name] = [
                    v.decode() if isinstance(v, bytes) else str(v) for v in handle[name][:]
                ]
            else:
                record[name] = np.asarray(handle[name][:]).tolist()
    index_file.write_text(json.dumps(record))
    # What a later call reads back, so the two answers compare equal.
    written: dict[str, Any] = json.loads(index_file.read_text())
    return written


def _images(source: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    """The six first-order mirror images of ``source`` in the box ``lo``..``hi``."""
    images = []
    for axis in range(3):
        for wall in (lo[axis], hi[axis]):
            image = source.copy()
            image[axis] = 2.0 * wall - source[axis]
            images.append(image)
    return images


def mock_field(
    path: Path,
    *,
    source_id: str,
    source_position: Sequence[float],
    box_lo: Sequence[float],
    box_hi: Sequence[float],
    step_m: float = 0.4,
    heights_m: Sequence[float] = (1.7,),
    margin_m: float = 0.3,
    order: int = 7,
    sample_rate_hz: float = 48_000.0,
    total_s: float = 1.2,
    t60_s: float = 0.5,
    wall_gain: float = 0.6,
    scene_id: str = "102344022",
    dwelling: str = "hssd_0002",
    room_name: str = "box",
    seed: int = 0,
) -> Path:
    """A field over one box room, from geometry alone.

    Direct arrival and six wall images as delayed impulses carrying the
    harmonics of their direction, then a diffuse tail of channel-independent
    decaying noise. It has the right shape, the right frame and a level that
    falls with distance, and it measures nothing.
    """
    rng = np.random.default_rng(seed)
    source = np.asarray(source_position, dtype=float)
    lo, hi = np.asarray(box_lo, dtype=float), np.asarray(box_hi, dtype=float)
    heights = np.asarray(heights_m, dtype=float) + lo[1]
    xs = np.arange(lo[0] + margin_m, hi[0] - margin_m + 1e-9, step_m)
    zs = np.arange(lo[2] + margin_m, hi[2] - margin_m + 1e-9, step_m)
    # One layer has no step in y; zero says so rather than inventing one.
    step = [step_m, float(heights[1] - heights[0]) if len(heights) > 1 else 0.0, step_m]
    channels = channel_count(order)
    total = int(round(total_s * sample_rate_hz))
    tail_start = int(round(0.150 * sample_rate_hz))
    envelope = 10.0 ** (-60.0 * np.arange(total - tail_start) / sample_rate_hz / (20.0 * t60_s))
    arrivals = [(source, 1.0)] + [(image, wall_gain) for image in _images(source, lo, hi)]

    positions, cells, direct, responses = [], [], [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(heights):
            for k, z in enumerate(zs):
                listener = np.array([x, y, z])
                ir = np.zeros((channels, total))
                for point, gain in arrivals:
                    offset = point - listener
                    distance = float(np.linalg.norm(offset))
                    sample = int(round(distance / SOUND_SPEED_M_S * sample_rate_hz))
                    if sample >= tail_start or distance < 1e-6:
                        continue
                    direction = scene_to_ambisonic(offset[None, :])[0] / distance
                    ir[:, sample] += gain / distance * real_sh(order, direction[None, :])[0]
                level = wall_gain * 0.3 / max(float(np.linalg.norm(source - listener)), 0.5)
                ir[:, tail_start:] = (
                    rng.standard_normal((channels, total - tail_start)) * envelope * level
                )
                positions.append(listener)
                cells.append([i, j, k])
                direct.append(float(np.linalg.norm(source - listener)))
                responses.append(ir)
    ir_all = np.asarray(responses, dtype=np.float32)
    gain = 0.891 / max(float(np.abs(ir_all).max()), 1e-9)
    count = len(positions)
    return write_field(
        path,
        ir=ir_all * gain,
        positions=np.asarray(positions),
        cell_index=np.asarray(cells),
        direct_path_m=np.asarray(direct),
        grid_origin_m=[xs[0], heights[0], zs[0]],
        grid_step_m=step,
        grid_shape=[len(xs), len(heights), len(zs)],
        order=order,
        sample_rate_hz=sample_rate_hz,
        source_id=source_id,
        source_position=source.tolist(),
        directivity="omni",
        gain=gain,
        scene_id=scene_id,
        dwelling=dwelling,
        provenance={"kind": "mock", "note": "geometry only, not a solve", "seed": seed},
        rooms=[room_name] * count,
        has_high=np.ones(count, dtype=bool),
        solved_to_hz=np.full(count, 8000.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check an impulse response field for the app")
    parser.add_argument("field", type=Path)
    arguments = parser.parse_args(argv)
    problems = check(arguments.field)
    for problem in problems:
        print(f"  {problem}")
    if problems:
        return 1
    header = read_header(arguments.field)
    print(f"{arguments.field}: {header.positions} cells, order {header.order}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
