"""The mirror's field on disk, in the reference's own format, aligned to the reference.

A mirror field is one HDF5 per source, ``field_mirror/<source>.h5``, in the
shape of ``docs/formats/ambisonic-field.md``: the same lattice, the same
points in the same order, the same attributes, so the walk-through app reads
it as it reads the reference and can switch between the two at a cell. The
lattice and the per point datasets are copied from the reference field; only
``/ir`` and the provenance are the mirror's.

**Alignment.** The reference's chain places the direct sound later than the
path length says (the crossover filters' group delay and the encoder's own
lead), at an arbitrary pressure scale. The mirror is rendered on the
reference's clock and scale: one lead and one gain per source, read on the
direct sound of the points that have one, as medians, and written to the
provenance. The criteria never needed either; the ear switching between the
two does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.mirror.criteria import CriteriaSettings, direct_arrival
from reverberate.spatial.encode import Ambisonic

__all__ = ["Alignment", "align_to_reference", "read_lattice", "write_mirror_field"]


@dataclass(frozen=True)
class Alignment:
    """The reference's clock and scale for the mirror."""

    lead_s: float
    gain: float
    points_used: int
    lead_spread_s: float
    gain_spread_db: float

    def record(self) -> dict[str, Any]:
        return {
            "lead_s": round(self.lead_s, 7),
            "gain": self.gain,
            "points_used": self.points_used,
            "lead_spread_s": round(self.lead_spread_s, 7),
            "gain_spread_db": round(self.gain_spread_db, 3),
        }


def align_to_reference(
    reference: Path,
    mirror_direct_energy: dict[int, float],
    *,
    sound_speed_m_s: float,
    settings: CriteriaSettings | None = None,
) -> Alignment:
    """The lead and the gain that put the mirror on the reference's clock and scale.

    ``mirror_direct_energy`` maps a point index to the mirror's direct
    pulse energy over half a millisecond, at gain one and no lead; the
    reference's is read the same way on the same points.
    """
    settings = settings or CriteriaSettings()
    leads = []
    ratios = []
    with h5py.File(reference, "r") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        direct_path = np.asarray(handle["direct_path_m"][...], dtype=float)
        window = int(round(settings.window_s * rate))
        for point, energy in sorted(mirror_direct_energy.items()):
            if energy <= 0.0:
                continue
            signals = np.asarray(handle["ir"][point], dtype=float)
            try:
                start = direct_arrival(signals, rate, settings)
            except ValueError:
                continue
            leads.append(start / rate - direct_path[point] / sound_speed_m_s)
            omni = signals[0, max(start - window // 2, 0) : start + window // 2 + 1]
            ratios.append(float(np.sum(omni**2)) / energy)
    if not leads:
        raise ValueError("no point with a direct sound on both sides to align on")
    lead_array = np.asarray(leads)
    ratio_array = np.asarray(ratios)
    return Alignment(
        lead_s=float(np.median(lead_array)),
        gain=float(np.sqrt(np.median(ratio_array))),
        points_used=len(leads),
        lead_spread_s=float(np.percentile(lead_array, 90) - np.percentile(lead_array, 10)),
        gain_spread_db=float(
            10.0 * np.log10(np.percentile(ratio_array, 90) / np.percentile(ratio_array, 10))
        ),
    )


def read_lattice(reference: Path) -> dict[str, Any]:
    """Everything of a field that is not the responses: the datasets and attributes to copy."""
    with h5py.File(reference, "r") as handle:
        datasets = {
            name: np.asarray(handle[name][...])
            for name in handle
            if name != "ir" and isinstance(handle[name], h5py.Dataset)
        }
        attributes = {name: handle.attrs[name] for name in handle.attrs}
        shape = tuple(int(v) for v in handle["ir"].shape)
    return {"datasets": datasets, "attributes": attributes, "ir_shape": shape}


def write_mirror_field(
    target: Path,
    reference: Path,
    responses: Mapping[int, Ambisonic],
    *,
    provenance: dict[str, Any],
    gain: float = 1.0,
) -> Path:
    """The mirror field: the reference's lattice with the mirror's responses.

    ``responses`` maps a point index to its response at the reference's rate
    and order, and may be lazy (a mapping that reads from disk on access);
    a point with none is written silent and listed in ``/mirror_silent``.
    ``gain`` scales every response.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lattice = read_lattice(reference)
    points, channels, samples = lattice["ir_shape"]
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
        for name, data in lattice["datasets"].items():
            handle.create_dataset(name, data=data)
        handle.create_dataset("mirror_silent", data=np.asarray(silent, dtype=np.int32))
        for name, value in lattice["attributes"].items():
            if name == "provenance_json":
                continue
            handle.attrs[name] = value
        handle.attrs["gain"] = 1.0
        handle.attrs["mirror"] = "geometric mirror of the reference field; see provenance_json"
        handle.attrs["mirror_silent_count"] = len(silent)
        handle.attrs["provenance_json"] = json.dumps(
            {**provenance, "reference": str(reference), "applied_gain": gain}, sort_keys=True
        )
    return target
