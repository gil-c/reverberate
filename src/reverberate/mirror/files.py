"""What the mirror reads and writes, and the reference field's clock and scale.

A mirror field is one HDF5 per source in the format of
``docs/formats/ambisonic-field.md``: the reference's lattice and per point
datasets copied, ``/ir`` and the provenance the mirror's. It is rendered on
the reference's clock and scale: one lead and one gain per source, medians
over the direct sound of the points that have one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.mirror.direct import DIRECT_WINDOW_S, direct_arrival
from reverberate.mirror.ism import Paths
from reverberate.spatial.encode import Ambisonic

__all__ = [
    "Alignment",
    "align_to_reference",
    "lattice_of",
    "load_paths",
    "read_omni",
    "write_field",
    "write_paths",
]

_PATH_COLUMNS = (
    "image",
    "order",
    "length_m",
    "direction",
    "gain",
    "points",
    "sequence",
)


def lattice_of(run: Path, source: str) -> tuple[np.ndarray, float, int]:
    """Positions, sample rate and ambisonic order of a source's lattice.

    From ``field/<source>.h5`` when the reference is there, else from
    ``mirror/lattice_<source>.npz`` (``positions``, ``sample_rate_hz``, ``order``).
    """
    reference = Path(run) / "field" / f"{source}.h5"
    if reference.is_file():
        with h5py.File(reference, "r") as handle:
            return (
                np.asarray(handle["positions"][...], dtype=float),
                float(handle.attrs["sample_rate_hz"]),
                int(handle.attrs["order"]),
            )
    lattice = Path(run) / "mirror" / f"lattice_{source}.npz"
    if not lattice.is_file():
        raise FileNotFoundError(f"neither {reference} nor {lattice}")
    with np.load(lattice) as arrays:
        return (
            np.asarray(arrays["positions"], dtype=float),
            float(arrays["sample_rate_hz"]),
            int(arrays["order"]),
        )


def read_omni(reference: Path, indices: list[int]) -> tuple[list[np.ndarray], float]:
    """The omni channel of the reference at ``indices``, and its sample rate."""
    with h5py.File(reference, "r") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        return [np.asarray(handle["ir"][i][0], dtype=float) for i in indices], rate


def write_paths(every: list[Paths], target: Path) -> Path:
    """The paths of every point in one ``.npz``: concatenated, with offsets."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([p.count for p in every])]).astype(np.int64)

    def stack(name: str, empty_shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        rows = [getattr(p, name) for p in every if p.count]
        if not rows:
            return np.zeros((0, *empty_shape), dtype=dtype)
        return np.concatenate([np.asarray(r) for r in rows], axis=0)

    bands = int(every[0].gain.shape[1]) if every and every[0].gain.ndim == 2 else 7
    width = max((int(p.sequence.shape[1]) for p in every if p.count), default=1)
    rows = max((int(p.points.shape[1]) for p in every if p.count), default=2)
    np.savez_compressed(
        target,
        offsets=offsets,
        receivers=np.asarray([p.receiver for p in every], dtype=float).reshape(-1, 3),
        image=stack("image", (), np.int64),
        order=stack("order", (), np.int64),
        length_m=stack("length_m", (), float),
        direction=stack("direction", (3,), float),
        gain=stack("gain", (bands,), float),
        points=stack("points", (rows, 3), float),
        sequence=stack("sequence", (width,), np.int64),
    )
    return target.with_suffix(".npz")


def load_paths(target: Path) -> list[Paths]:
    """The inverse of :func:`write_paths`.

    Files written before ``receivers`` existed give each point a receiver of NaN.
    """
    with np.load(Path(target)) as arrays:
        offsets = arrays["offsets"]
        columns = {name: arrays[name] for name in _PATH_COLUMNS}
        count = len(offsets) - 1
        receivers = arrays["receivers"] if "receivers" in arrays else np.full((count, 3), np.nan)
    return [
        Paths(
            receiver=np.asarray(receivers[i], dtype=float),
            **{name: columns[name][offsets[i] : offsets[i + 1]] for name in _PATH_COLUMNS},
        )
        for i in range(count)
    ]


@dataclass(frozen=True)
class Alignment:
    """The reference's clock and scale for the mirror."""

    lead_s: float
    gain: float
    points_used: int

    def record(self) -> dict[str, Any]:
        return {"lead_s": round(self.lead_s, 7), "gain": self.gain, "points_used": self.points_used}


def align_to_reference(
    reference: Path, mirror_direct_energy: dict[int, float], *, sound_speed_m_s: float
) -> Alignment:
    """The lead and gain that put the mirror on the reference's clock and scale.

    ``mirror_direct_energy`` maps a point to the mirror's direct energy at
    gain one and no lead (:func:`reverberate.mirror.direct.direct_energy`);
    the reference's is read the same way on the same points.
    """
    leads = []
    ratios = []
    with h5py.File(reference, "r") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        direct_path = np.asarray(handle["direct_path_m"][...], dtype=float)
        half = int(round(DIRECT_WINDOW_S * rate)) // 2
        for point, energy in sorted(mirror_direct_energy.items()):
            if energy <= 0.0:
                continue
            signals = np.asarray(handle["ir"][point], dtype=float)
            try:
                start = direct_arrival(signals, rate)
            except ValueError:
                continue
            leads.append(start / rate - direct_path[point] / sound_speed_m_s)
            omni = signals[0, max(start - half, 0) : start + half + 1]
            ratios.append(float(np.sum(omni**2)) / energy)
    if not leads:
        raise ValueError("no point with a direct sound on both sides to align on")
    return Alignment(
        lead_s=float(np.median(leads)),
        gain=float(np.sqrt(np.median(ratios))),
        points_used=len(leads),
    )


def write_field(
    target: Path,
    reference: Path,
    responses: Mapping[int, Ambisonic],
    *,
    provenance: dict[str, Any],
    gain: float = 1.0,
) -> Path:
    """The reference's lattice with the mirror's responses, each times ``gain``.

    ``responses`` may be lazy; a point with none is written silent and
    listed in ``/mirror_silent``.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(reference, "r") as source:
        datasets = {
            name: np.asarray(source[name][...])
            for name in source
            if name != "ir" and isinstance(source[name], h5py.Dataset)
        }
        attributes = {name: source.attrs[name] for name in source.attrs}
        points, channels, samples = (int(v) for v in source["ir"].shape)
    silent = []
    with h5py.File(target, "w") as handle:
        ir = handle.create_dataset(
            "ir", shape=(points, channels, samples), dtype=np.float32, chunks=(1, channels, samples)
        )
        for point in range(points):
            response = responses.get(point)
            if response is None:
                ir[point] = np.zeros((channels, samples), dtype=np.float32)
                silent.append(point)
                continue
            if response.signals.shape[0] != channels:
                raise ValueError(
                    f"point {point}: {response.signals.shape[0]} channels, the field has {channels}"
                )
            block = np.zeros((channels, samples), dtype=np.float64)
            take = min(samples, response.signals.shape[1])
            block[:, :take] = response.signals[:, :take] * gain
            ir[point] = block.astype(np.float32)
        for name, data in datasets.items():
            handle.create_dataset(name, data=data)
        handle.create_dataset("mirror_silent", data=np.asarray(silent, dtype=np.int32))
        for name, value in attributes.items():
            if name != "provenance_json":
                handle.attrs[name] = value
        handle.attrs["gain"] = 1.0
        handle.attrs["mirror_silent_count"] = len(silent)
        handle.attrs["provenance_json"] = json.dumps(
            {**provenance, "reference": str(reference), "applied_gain": gain}, sort_keys=True
        )
    return target
