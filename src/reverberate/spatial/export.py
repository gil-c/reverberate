"""Writing an ambisonic or binaural response out in formats other tools read.

``docs/formats/impulse-response.md`` already fixes the point receiver case:
SOFA ``SingleRoomSRIR`` beside a raw HDF5, one receiver per measurement,
explicitly because loose points in a room are not a rigid array. An ambisonic
response is the opposite case and needs the opposite answer: it **is** one
array, its channels are not positions at all, and SOFA has a receiver type for
exactly that.

Three files, and each is for a different reader:

- ``.wav`` in ambiX, which is ACN ordering and SN3D normalisation, because that
  is what every ambisonic tool and every game engine reads. The conversion from
  the N3D this package computes in is one diagonal matrix and it happens here
  and nowhere else.
- ``.sofa`` with ``ReceiverPosition_Type`` set to spherical harmonics, for a
  reader who wants the provenance with the samples.
- a binaural ``.sofa`` with two real receivers, one measurement per head
  orientation, which is what a listening test or a training pipeline consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reverberate.response import Provenance, to_sofa_coordinates
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count, degrees_of, orders_of, sn3d_scale

__all__ = [
    "AMBIX_NOTE",
    "to_ambix",
    "write_ambisonic_sofa",
    "write_ambix_wav",
    "write_brir_sofa",
]

AMBIX_NOTE = (
    "ambiX: ACN channel ordering, SN3D normalisation. The channels are "
    "spherical harmonic coefficients about one point, not microphone signals."
)


def to_ambix(ambisonic: Ambisonic) -> np.ndarray:
    """The signals in SN3D, which is what ambiX means by normalisation."""
    return np.asarray(ambisonic.signals * sn3d_scale(ambisonic.order)[:, None], dtype=float)


def write_ambix_wav(
    ambisonic: Ambisonic,
    path: Path,
    *,
    headroom_db: float = 1.0,
    gain: float | None = None,
) -> tuple[Path, float]:
    """Write an ambiX WAV and return the path and the single gain applied.

    One gain for every channel: scaling channels separately would destroy the
    directional information, which here is carried entirely by the ratios
    between them.

    ``gain`` is the gain a caller computed over a whole set of files that have
    to stay comparable, and passing it is the only way to keep this file's level
    against theirs. **Without it this function normalises**, and asking for zero
    headroom does not disable that: it asks for a peak of exactly one, which is
    a normalisation with no room left. That is how this file came out at full
    scale beside binaural files at a fifth of it, while the report claimed one
    shared gain for all of them.
    """
    import soundfile

    signals = to_ambix(ambisonic)
    if gain is None:
        peak = float(np.max(np.abs(signals)))
        gain = 1.0 if peak == 0.0 else 10.0 ** (-headroom_db / 20.0) / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        str(path),
        (signals * gain).T,
        int(round(ambisonic.sample_rate_hz)),
        subtype="FLOAT",
        format="WAV",
    )
    return path, gain


def _sofa_common(sofa: Any, *, title: str, licence: str, provenance: Provenance) -> None:
    sofa.GLOBAL_Title = title
    sofa.GLOBAL_License = licence
    sofa.GLOBAL_ApplicationName = "reverberate"
    sofa.GLOBAL_RoomType = "reverberant"
    sofa.GLOBAL_Comment = provenance.to_json()


def write_ambisonic_sofa(
    ambisonic: Ambisonic,
    path: Path,
    *,
    source_position: np.ndarray,
    provenance: Provenance,
    title: str,
    licence: str,
) -> Path:
    """One measurement whose receivers are spherical harmonic coefficients.

    ``ReceiverPosition_Type`` is ``spherical harmonics``, which is AES69's own
    way of saying that a receiver index is a channel number and not a place.
    The ACN index and the degree and order of each channel go into
    ``ReceiverDescriptions``, so a reader who has only this file can still tell
    which channel is which without knowing this project's conventions.
    """
    import sofar

    sofa = sofar.Sofa("SingleRoomSRIR")
    channels = channel_count(ambisonic.order)
    sofa.Data_IR = ambisonic.signals[np.newaxis, :, :]
    sofa.Data_SamplingRate = float(ambisonic.sample_rate_hz)
    sofa.Data_Delay = np.zeros((1, channels))
    sofa.ListenerPosition = to_sofa_coordinates(ambisonic.centre)
    sofa.ListenerView = np.array([[1.0, 0.0, 0.0]])
    sofa.ListenerUp = np.array([[0.0, 0.0, 1.0]])
    degrees, orders = degrees_of(ambisonic.order), orders_of(ambisonic.order)
    sofa.ReceiverPosition = np.stack(
        [degrees.astype(float), orders.astype(float), np.zeros(channels)], axis=1
    )[:, :, np.newaxis]
    sofa.ReceiverPosition_Type = "spherical harmonics"
    sofa.ReceiverPosition_Units = "degree, degree, metre"
    # A harmonic channel has no orientation, but the convention wants one row
    # per receiver and refuses a single shared one.
    sofa.ReceiverView = np.tile(np.array([1.0, 0.0, 0.0]), (channels, 1))[:, :, np.newaxis]
    sofa.ReceiverUp = np.tile(np.array([0.0, 0.0, 1.0]), (channels, 1))[:, :, np.newaxis]
    sofa.ReceiverDescriptions = np.array(
        [
            f"ACN {index}, n={n}, m={m}"
            for index, (n, m) in enumerate(zip(degrees, orders, strict=True))
        ]
    )
    sofa.SourcePosition = to_sofa_coordinates(np.asarray(source_position, dtype=float))
    sofa.EmitterPosition = np.zeros((1, 3, 1))
    sofa.MeasurementDate = np.zeros(1)
    _sofa_common(sofa, title=title, licence=licence, provenance=provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    sofar.write_sofa(str(path), sofa)
    return path


def write_brir_sofa(
    brirs: np.ndarray,
    yaw_deg: np.ndarray,
    path: Path,
    *,
    sample_rate_hz: float,
    listener_position: np.ndarray,
    source_position: np.ndarray,
    ear_positions: np.ndarray,
    provenance: Provenance,
    title: str,
    licence: str,
) -> Path:
    """Binaural room responses, one measurement per head orientation.

    ``brirs`` is ``[orientation, ear, sample]``. The head orientation travels
    as ``ListenerView`` rather than being baked into the file name, because a
    reader that does not know which way the head was facing cannot use a
    binaural response for anything.
    """
    import sofar

    responses = np.asarray(brirs, dtype=float)
    if responses.ndim != 3 or responses.shape[1] != 2:
        raise ValueError(f"expected [orientation, 2, sample], got {responses.shape}")
    count = responses.shape[0]
    yaw = np.radians(np.asarray(yaw_deg, dtype=float))
    if yaw.size != count:
        raise ValueError(f"{yaw.size} orientations for {count} responses")

    sofa = sofar.Sofa("SingleRoomSRIR")
    sofa.Data_IR = responses
    sofa.Data_SamplingRate = float(sample_rate_hz)
    sofa.Data_Delay = np.zeros((1, 2))
    sofa.ListenerPosition = np.tile(to_sofa_coordinates(listener_position), (count, 1))
    sofa.ListenerView = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(count)], axis=1)
    sofa.ListenerUp = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    ears = np.asarray(ear_positions, dtype=float)
    if ears.shape != (2, 3):
        raise ValueError(f"expected two ear positions of three coordinates, got {ears.shape}")
    sofa.ReceiverPosition = ears[:, :, np.newaxis]
    sofa.ReceiverView = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))[:, :, np.newaxis]
    sofa.ReceiverUp = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1))[:, :, np.newaxis]
    # Named rather than left to the reader's guess: which row is the left ear
    # is the one thing a binaural file cannot afford to be ambiguous about.
    sofa.ReceiverDescriptions = np.array(["left ear", "right ear"])
    sofa.SourcePosition = np.tile(to_sofa_coordinates(source_position), (count, 1))
    sofa.EmitterPosition = np.zeros((1, 3, 1))
    sofa.MeasurementDate = np.zeros(count)
    _sofa_common(sofa, title=title, licence=licence, provenance=provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    sofar.write_sofa(str(path), sofa)
    return path
