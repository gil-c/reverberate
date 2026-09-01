"""What an impulse response is as an artefact, in two files that agree.

The roadmap says the deliverable is "the full impulse response, not a decay
envelope", spatially resolved and re-decodable. Until now nothing said what
that response *is* on disk: the solver writes ``sim_outs.h5`` at the grid rate
in its own index space, and there the trail stopped. This module is the
declaration.

**Two files, and the split is not redundancy.**

``response.sofa`` is the canonical, portable artefact, AES69-2022 under the
``SingleRoomSRIR`` convention. Anyone outside this project reads only this. It
is what makes the dataset usable by the hearing aid work that consumes it,
without importing a line of this repository.

``response.h5`` is the raw internal record: the solver's own sample rate, no
resampling, float64 as the engine writes it, and the provenance block. SOFA
could hold that provenance as free text in ``GLOBAL_Comment``, and burying it
in a string is exactly what would make ``reverberate audit`` impossible. So the
provenance is typed here and mirrored into the SOFA comment for a reader who
has only that file.

**Units, stated once.** Positions in metres, sample rates in hertz, durations in
seconds, sound speed in metres per second, grid step in metres. Impulse
responses are pressure in arbitrary units: the solver's excitation is not
calibrated to a source power, so an absolute level would be a fiction. What is
comparable between two responses of the same run is their relative level, and
that is preserved exactly because no per response normalisation is applied.

**Coordinates.** The scene is Y up, as HSSD and glTF are, and every position in
this project is ``(x, height, z)``. SOFA is Z up. The conversion is a single
right handed rotation about X, :func:`to_sofa_coordinates`, applied on write and
inverted on read, so a SOFA reader gets the convention it expects and the
internal file keeps the convention the geometry uses. Doing this silently in
two places is how a dataset acquires a mirrored axis, so it is one function
with one test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "SOFA_CONVENTION",
    "Provenance",
    "ResponseSet",
    "from_sofa_coordinates",
    "read_raw",
    "to_sofa_coordinates",
    "write_raw",
    "write_sofa",
]

#: The AES69-2022 convention used for a room response from one source to a set
#: of listener positions. ``SingleRoomMIMOSRIR`` is the right answer once the
#: receiver batching of the roadmap's section 8 puts several sources in one run;
#: it is deliberately not used yet, because nothing writes several sources into
#: one file and a convention nobody exercises is a convention nobody checks.
SOFA_CONVENTION = "SingleRoomSRIR"

#: Bumped when the raw layout changes in a way a reader must notice.
RAW_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    """Everything needed to answer "where did this response come from".

    Every field is required except the ones that genuinely may not exist: a run
    on a laptop has no billed rate, and a run whose scene was not published has
    no scene digest yet.
    """

    #: SHA-256 of the serialised surface list handed to the solver. This is the
    #: constraint 9 anchor: the viewer renders the object with this digest, so
    #: a picture and a response can be proved to be of the same thing.
    scene_sha256: str
    #: The voxelisation cache key, which the roadmap fixes as a hash of the
    #: model file alone, so it is shared by every source and receiver pair in
    #: the same scene at the same grid step.
    mats_hash: str
    #: ``cuda`` or ``cpu``.
    engine: str
    #: Which of the roadmap's three bands this response belongs to.
    band: str
    #: Upper frequency the grid was built for, in hertz.
    fmax_hz: float
    #: Grid step in metres, and the points per wavelength it realises.
    grid_step_m: float
    points_per_wavelength: float
    #: Sound speed in metres per second, as handed to the solver.
    sound_speed_m_s: float
    #: The seed that produced the placement, so the geometry of the run is
    #: reproducible and not merely recorded.
    seed: int
    #: Run directory name, which is also the key under ``runs/`` in the store.
    run_id: str
    solver_commit: str = ""
    #: Present only for a rented run. Roadmap constraint 10: a cost figure
    #: without its hourly rate is not a measurement.
    billed_rate_usd_per_hour: float | None = None
    wall_clock_s: float | None = None
    #: Free text for what a reader must be told and no field can carry, such as
    #: "two bare omnidirectional points, this is not a binaural response".
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Provenance:
        return cls(**json.loads(payload))


@dataclass(frozen=True)
class ResponseSet:
    """One source, several receivers, one sample rate.

    ``ir`` is ``[receiver, sample]``. Several receivers share a file because
    they share a run, a grid and an excitation, and separating them would make
    it possible to compare two responses that were never comparable.
    """

    ir: np.ndarray
    sample_rate_hz: float
    source_position: np.ndarray
    receiver_positions: np.ndarray
    provenance: Provenance
    room_volume_m3: float | None = None
    #: Populated on read, empty on a freshly built set.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ir.ndim != 2:
            raise ValueError(f"ir must be [receiver, sample], got shape {self.ir.shape}")
        if self.receiver_positions.shape != (self.ir.shape[0], 3):
            raise ValueError(
                f"{self.ir.shape[0]} receivers but "
                f"{self.receiver_positions.shape} receiver positions"
            )
        if self.source_position.shape != (3,):
            raise ValueError(f"source position must be (3,), got {self.source_position.shape}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")

    @property
    def duration_s(self) -> float:
        return float(self.ir.shape[1] / self.sample_rate_hz)

    @property
    def receiver_count(self) -> int:
        return int(self.ir.shape[0])


def to_sofa_coordinates(points: np.ndarray) -> np.ndarray:
    """Scene coordinates, Y up, to SOFA cartesian coordinates, Z up.

    A right handed rotation of +90 degrees about X: ``(x, y, z)`` becomes
    ``(x, -z, y)``. Handedness is preserved, so a scene that is not mirrored
    does not become mirrored here.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)


def from_sofa_coordinates(points: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_sofa_coordinates`."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def write_raw(response: ResponseSet, path: Path) -> Path:
    """Write ``response.h5``, the internal record, and return its path.

    Written with no compression. On the projected full dataset the responses
    are of the order of a hundred gigabytes against a voxelisation cache
    approaching a terabyte, so a codec would save cents a month and cost
    exactness. That arithmetic is in ``docs/formats/impulse-response.md``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = RAW_SCHEMA_VERSION
        handle.attrs["sample_rate_hz"] = float(response.sample_rate_hz)
        handle.attrs["provenance_json"] = response.provenance.to_json()
        handle.attrs["created_utc"] = datetime.now(UTC).isoformat()
        if response.room_volume_m3 is not None:
            handle.attrs["room_volume_m3"] = float(response.room_volume_m3)
        handle.create_dataset("ir", data=np.asarray(response.ir, dtype=np.float64))
        handle.create_dataset(
            "source_position", data=np.asarray(response.source_position, dtype=np.float64)
        )
        handle.create_dataset(
            "receiver_positions", data=np.asarray(response.receiver_positions, dtype=np.float64)
        )
    return path


def read_raw(path: Path) -> ResponseSet:
    """Read ``response.h5`` back into a :class:`ResponseSet`."""
    with h5py.File(path, "r") as handle:
        version = int(handle.attrs["schema_version"])
        if version != RAW_SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema version {version}, this reader knows {RAW_SCHEMA_VERSION}"
            )
        volume = handle.attrs.get("room_volume_m3")
        return ResponseSet(
            ir=np.asarray(handle["ir"][...], dtype=np.float64),
            sample_rate_hz=float(handle.attrs["sample_rate_hz"]),
            source_position=np.asarray(handle["source_position"][...], dtype=np.float64),
            receiver_positions=np.asarray(handle["receiver_positions"][...], dtype=np.float64),
            provenance=Provenance.from_json(str(handle.attrs["provenance_json"])),
            room_volume_m3=None if volume is None else float(volume),
            extras={"created_utc": str(handle.attrs["created_utc"])},
        )


def write_sofa(response: ResponseSet, path: Path, *, title: str, licence: str) -> Path:
    """Write ``response.sofa``, the portable artefact, and return its path.

    Each receiver becomes one SOFA measurement with a single receiver at the
    listener's origin, rather than one measurement with several receivers. The
    receivers here are independent points in the room, not the elements of one
    rigid array, and the SOFA convention's ``ReceiverPosition`` is relative to
    the listener: describing six loose points as six elements of one listener
    would claim a rigid geometry that does not exist.
    """
    import sofar

    sofa = sofar.Sofa(SOFA_CONVENTION)
    count = response.receiver_count
    sofa.Data_IR = response.ir[:, np.newaxis, :]
    sofa.Data_SamplingRate = float(response.sample_rate_hz)
    sofa.Data_Delay = np.zeros((1, 1))
    sofa.ListenerPosition = to_sofa_coordinates(response.receiver_positions)
    sofa.ListenerView = np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
    sofa.ListenerUp = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    sofa.ReceiverPosition = np.zeros((1, 3, 1))
    sofa.SourcePosition = np.tile(to_sofa_coordinates(response.source_position), (count, 1))
    sofa.EmitterPosition = np.zeros((1, 3, 1))
    sofa.MeasurementDate = np.zeros(count)
    if response.room_volume_m3 is not None:
        sofa.RoomVolume = float(response.room_volume_m3)
    sofa.GLOBAL_Title = title
    sofa.GLOBAL_License = licence
    sofa.GLOBAL_ApplicationName = "reverberate"
    sofa.GLOBAL_RoomType = "reverberant"
    sofa.GLOBAL_Comment = response.provenance.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    sofar.write_sofa(str(path), sofa)
    return path
