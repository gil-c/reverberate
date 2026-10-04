"""How a source radiates with the angle from where it faces: the pack's ``/directivity``.

A model is a table ``gain_db [band, angle]`` on the mirror's seven octave
bands and on the angles 0 to 180 degrees by 5, a figure of revolution about
the facing direction ``(cos yaw, 0, -sin yaw)``, **normalised to unit mean
power over the sphere in every band** (``docs/formats/scene-pack.md``): a
directional source radiates what the omnidirectional one does, so the tail
and the band under the crossover, which stay omnidirectional, keep their
level. :func:`directivity_gain` is what the signal engine multiplies a
path's band gains by.

``voice_v1``
------------
**A parametric fit, not a measured table; to be confirmed by the owner.**
The pattern of each band is

    ``gain_db(angle) = -back_db * (1 - cos(angle)) / 2``

before normalisation: level on the axis, ``back_db`` lower straight behind,
half of it at the side. ``back_db`` per octave band is :data:`VOICE_V1_BACK_DB`,
round figures for the front to back difference of conversational speech, of
the order reported by

- W. T. Chu and A. C. C. Warnock, *Detailed directivity of sound fields
  around human talkers*, National Research Council Canada, Institute for
  Research in Construction, report IRC-RR-104, 2002 (forty talkers, 92
  positions on a sphere, third octave bands); and, for the 8 kHz band,
- B. B. Monson, E. J. Hunter and B. H. Story, *Horizontal directivity of low-
  and high-frequency energy in speech and singing*, J. Acoust. Soc. Am.
  132 (1), 2012, pp. 433 to 441.

The reports' own tables were not at hand when this was written: the figures
are a reading of their trend (a voice is nearly omnidirectional at 125 Hz
and loses of the order of 15 to 20 dB behind the head at 8 kHz), the shape
is the simplest that holds it, and the measured dip under the chin and lobe
above the head are not in it, a figure of revolution having no up or down.
Replacing the figures by the reports' tables changes the model's digest and
nothing else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS

__all__ = [
    "ANGLES_DEG",
    "VOICE_V1_BACK_DB",
    "Directivity",
    "directivity_gain",
    "facing",
    "mean_power",
    "model",
    "omni",
    "voice_v1",
]

#: The angles a model is tabulated on, degrees from the facing direction.
ANGLES_DEG = np.arange(0.0, 180.0 + 1e-9, 5.0)

#: ``voice_v1``: how far under the axis the level is straight behind, dB, per octave band.
VOICE_V1_BACK_DB = (2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 18.0)

#: Angles the mean power is summed on, per degree of the table's step.
_FINE = 50


@dataclass(frozen=True)
class Directivity:
    """One model: the pack's group ``/directivity/<name>``."""

    name: str
    #: ``[band, angle]``, dB, on :data:`OCTAVE_BANDS` and :attr:`angles_deg`.
    gain_db: np.ndarray
    angles_deg: np.ndarray
    bands_hz: tuple[int, ...] = tuple(OCTAVE_BANDS)

    @property
    def digest(self) -> str:
        """The SHA-256 of ``gain_db`` as the pack stores it: the recipe's asset key."""
        return hashlib.sha256(np.ascontiguousarray(self.gain_db, dtype="<f4").tobytes()).hexdigest()

    def pack(self) -> dict[str, Any]:
        """The group's dataset and attribute, in the pack's types."""
        return {
            "gain_db": np.ascontiguousarray(self.gain_db, dtype=np.float32),
            "angles_deg": np.asarray(self.angles_deg, dtype=np.float64),
        }


def _at(gain_db: np.ndarray, angles_deg: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """``[..., band]``: the table read at ``angle_deg``, linear in decibels between its angles."""
    step = float(angles_deg[1] - angles_deg[0])
    place = np.clip((np.asarray(angle_deg, dtype=float) - angles_deg[0]) / step, 0.0, None)
    low = np.minimum(np.floor(place).astype(int), angles_deg.size - 2)
    weight = np.clip(place - low, 0.0, 1.0)
    table = np.asarray(gain_db, dtype=float).T  # [angle, band]
    return np.asarray(
        table[low] * (1.0 - weight)[..., None] + table[low + 1] * weight[..., None], dtype=float
    )


def mean_power(gain_db: np.ndarray, angles_deg: np.ndarray = ANGLES_DEG) -> np.ndarray:
    """Per band, the pattern's power averaged over the sphere, as the engine reads the table.

    ``1/2`` the integral of ``10^(gain_db / 10) sin(angle)`` over the angle,
    the table interpolated as :func:`directivity_gain` interpolates it.
    """
    count = (angles_deg.size - 1) * _FINE
    edges = np.linspace(float(angles_deg[0]), float(angles_deg[-1]), count + 1)
    middle = 0.5 * (edges[:-1] + edges[1:])
    power = 10.0 ** (_at(gain_db, angles_deg, middle) / 10.0)  # [angle, band]
    # Each slice's own share of the sphere, exactly: the sum is one for an omni.
    share = 0.5 * (np.cos(np.radians(edges[:-1])) - np.cos(np.radians(edges[1:])))
    return np.asarray(share @ power, dtype=float)


def model(name: str, gain_db: np.ndarray, angles_deg: np.ndarray = ANGLES_DEG) -> Directivity:
    """A model from a pattern in dB of any level: normalised to unit mean power per band."""
    gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.shape != (len(OCTAVE_BANDS), angles_deg.size):
        raise ValueError(
            f"a pattern is [{len(OCTAVE_BANDS)}, {angles_deg.size}], not {gain_db.shape}"
        )
    level = 10.0 * np.log10(mean_power(gain_db, angles_deg))
    normalised = gain_db - level[:, None]
    normalised.setflags(write=False)
    return Directivity(name=name, gain_db=normalised, angles_deg=np.asarray(angles_deg))


def omni() -> Directivity:
    """The source that radiates alike in every direction: zeros."""
    zeros = np.zeros((len(OCTAVE_BANDS), ANGLES_DEG.size))
    zeros.setflags(write=False)
    return Directivity(name="omni", gain_db=zeros, angles_deg=ANGLES_DEG)


def voice_v1() -> Directivity:
    """A talker's mouth, by the parametric fit of the module's docstring."""
    back = np.asarray(VOICE_V1_BACK_DB, dtype=float)[:, None]
    shape = 0.5 * (1.0 - np.cos(np.radians(ANGLES_DEG)))[None, :]
    return model("voice_v1", -back * shape)


def facing(yaw_deg: np.ndarray | float) -> np.ndarray:
    """The direction a source faces, scene frame: ``(cos yaw, 0, -sin yaw)``."""
    yaw = np.radians(np.asarray(yaw_deg, dtype=float))
    return np.asarray(np.stack([np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)], axis=-1))


def directivity_gain(
    table: Directivity, departure: np.ndarray, facing_yaw: np.ndarray | float
) -> np.ndarray:
    """``[path, band]`` pressure gains of ``table`` for paths leaving along ``departure``.

    ``departure`` is ``[path, 3]``, the pack's unit vectors from the source
    along each path's first leg; ``facing_yaw`` the source's yaw in degrees,
    one for all the paths or one per path. Linear, to multiply a path's
    ``gain`` by.
    """
    departure = np.atleast_2d(np.asarray(departure, dtype=float))
    ahead = np.broadcast_to(facing(facing_yaw), departure.shape)
    cosine = np.einsum("ij,ij->i", departure, ahead) / np.maximum(
        np.linalg.norm(departure, axis=1), 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.asarray(10.0 ** (_at(table.gain_db, table.angles_deg, angle) / 20.0))
