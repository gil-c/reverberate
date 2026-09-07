"""Measurements that say whether an encoded response is of the room it claims.

Every function here answers a question a level cannot: which way did the sound
come from, how much of it is in the orders that were kept, and does the tail
behave the way a diffuse field has to. They exist because the failures this
package can have are silent. A mirrored frame, a swapped ear or an inverted
odd order all leave the spectrum and the decay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import acn, degrees_of, scene_to_ambisonic

__all__ = [
    "DirectionOfArrival",
    "direct_arrival_sample",
    "direction_of_arrival",
    "energy_per_order",
]


@dataclass(frozen=True)
class DirectionOfArrival:
    """Where a windowed part of a response came from, and how sure that is."""

    unit_vector: np.ndarray
    azimuth_deg: float
    elevation_deg: float
    #: Length of the mean intensity vector over the window, in ``[0, 1]``. One
    #: is a single plane wave; near zero is a diffuse or ambiguous arrival, and
    #: a direction read off it means nothing.
    concentration: float

    def angle_to_deg(self, other: np.ndarray) -> float:
        """Angle to another direction, in degrees. The number a test asserts on."""
        reference = np.asarray(other, dtype=float)
        reference = reference / np.linalg.norm(reference)
        return float(np.degrees(np.arccos(np.clip(self.unit_vector @ reference, -1.0, 1.0))))

    def record(self) -> dict[str, Any]:
        return {
            "azimuth_deg": round(self.azimuth_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "concentration": round(self.concentration, 4),
        }


def direct_arrival_sample(ambisonic: Ambisonic) -> int:
    """The sample the direct sound arrives at, taken from the omnidirectional channel."""
    return int(np.argmax(np.abs(ambisonic.signals[0])))


def direction_of_arrival(
    ambisonic: Ambisonic,
    *,
    start: int | None = None,
    length: int | None = None,
) -> DirectionOfArrival:
    """The intensity vector's direction over a window, pointing **at** the source.

    For a plane wave arriving from ``s`` the first order channels are
    ``sqrt(3) s`` times the omnidirectional one, so the product of the two,
    averaged over the window, points back along the arrival path. That is the
    quantity a wrong frame or an inverted odd order shows up in immediately,
    and nothing else in this package does.
    """
    signals = ambisonic.signals
    if ambisonic.order < 1:
        raise ValueError("a direction needs at least first order")
    if start is None:
        start = max(direct_arrival_sample(ambisonic) - 8, 0)
    if length is None:
        length = min(64, signals.shape[1] - start)
    window = signals[:, start : start + length]
    pressure = window[0]
    velocity = np.stack([window[acn(1, 1)], window[acn(1, -1)], window[acn(1, 0)]]) / np.sqrt(3.0)
    intensity = np.mean(pressure[None, :] * velocity, axis=1)
    magnitude = float(np.linalg.norm(intensity))
    scalar = float(np.mean(pressure**2 + np.sum(velocity**2, axis=0)) / 2.0)
    if magnitude == 0.0:
        raise ValueError("the window carries no energy, so it has no direction")
    unit = intensity / magnitude
    return DirectionOfArrival(
        unit_vector=unit,
        azimuth_deg=float(np.degrees(np.arctan2(unit[1], unit[0]))),
        elevation_deg=float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0)))),
        concentration=magnitude / scalar if scalar > 0 else 0.0,
    )


def direction_to_scene_point(ambisonic: Ambisonic, point: np.ndarray) -> np.ndarray:
    """The unit vector from the array's centre to a point in scene coordinates.

    The reference a measured direction of arrival is compared against, in the
    one frame conversion this project has.
    """
    offset = np.asarray(point, dtype=float) - np.asarray(ambisonic.centre, dtype=float)
    rotated = scene_to_ambisonic(offset[None, :])[0]
    norm = float(np.linalg.norm(rotated))
    if norm == 0.0:
        raise ValueError("the point is the array's own centre")
    return np.asarray(rotated / norm)


def energy_per_order(
    ambisonic: Ambisonic, *, frame_s: float = 0.005, hop_s: float = 0.0025
) -> tuple[np.ndarray, np.ndarray]:
    """Energy in each degree against time, as ``(times, energy [frame, order])``.

    What it is for: a response whose energy climbs into the highest kept order
    as the tail develops is behaving like a diffuse field, where every order
    carries the same energy. One whose high orders stay empty has been
    truncated somewhere, and one whose high orders dominate the direct sound is
    fitting noise.
    """
    frame = max(int(round(frame_s * ambisonic.sample_rate_hz)), 8)
    hop = max(int(round(hop_s * ambisonic.sample_rate_hz)), 1)
    degrees = degrees_of(ambisonic.order)
    times, rows = [], []
    for start in range(0, ambisonic.signals.shape[1] - frame + 1, hop):
        block = ambisonic.signals[:, start : start + frame]
        energy = np.sum(block**2, axis=1)
        rows.append([float(energy[degrees == n].sum()) for n in range(ambisonic.order + 1)])
        times.append((start + frame / 2.0) / ambisonic.sample_rate_hz)
    return np.asarray(times), np.asarray(rows)
