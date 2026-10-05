"""The scene pack of ``docs/formats/scene-pack.md``: its types, its file, its invariants.

The single implementation of the format. The trace stage builds a
:class:`ScenePack` (or feeds a :class:`PackWriter` one source at a time, a
source's low band being half a gigabyte) and the signal engine reads one
with :func:`read_pack`. :func:`validate` checks what the document lets a
reader assume, and both the writer and the reader call it, so a pack that
exists on disk is a pack the engine may trust.

A pack read from disk keeps its three large tables (``low/ir``,
``tail/energy``, ``tail/moments``) in the file and reads them a row at a
time; one built in memory holds them as arrays. Both are indexed the same.

:func:`synthetic_free_field` builds the ``synthetic-free-field`` profile,
levels A and B, whose render is known in closed form.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import h5py
import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.audio import Atmosphere
from reverberate.metrics import band_centres
from reverberate.mirror.directivity import Directivity, omni
from reverberate.mirror.hybrid import Crossover
from reverberate.spatial.sh import channel_count
from reverberate.spatial.translate import (
    EXACT_UNDER_M,
    FUSE_WITHIN_M,
    QUADRATURE_DEGREE,
    REGULARISATION,
    SOURCE_SHARE,
    SURFACE_SHARE,
    TRANSLATE_WITHIN_M,
)

__all__ = [
    "KIND_DIFFRACTED",
    "KIND_DIFFRACTED_REFLECTED",
    "KIND_DIRECT",
    "KIND_SPECULAR",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STEP_S",
    "Air",
    "Cells",
    "Directivity",
    "Early",
    "Header",
    "Level",
    "Listener",
    "Low",
    "Mirror",
    "PackError",
    "PackWriter",
    "ScenePack",
    "Source",
    "Tail",
    "band_map",
    "default_fusion",
    "path_id",
    "read_pack",
    "synthetic_free_field",
    "tail_seed",
    "validate",
    "write_pack",
]

SCHEMA = "reverberate.scene-pack"
SCHEMA_VERSION = 1
STEP_S = 0.05
PROFILES = ("trace", "synthetic-free-field", "synthetic-density")
ORDERING = "ACN"
NORMALISATION = "N3D"
#: A jump of a path's apparent source beyond this between two steps is two paths.
JUMP_M = 0.30
#: Unit vectors are unit to this.
UNIT_TOLERANCE = 1e-6


class PackError(ValueError):
    """A pack that breaks an invariant of the format; the message names it."""


#: The ``kind`` of a row of ``early``.
KIND_DIRECT = 0
KIND_SPECULAR = 1
KIND_DIFFRACTED = 2
KIND_DIFFRACTED_REFLECTED = 3

_IDS: dict[bytes, int] = {}


def path_id(
    kind: int,
    facets: Iterable[int] = (),
    edges: Iterable[int] | None = None,
    *,
    rank: int | None = None,
) -> int:
    """A path's identity, the same at every step it exists: the format's one definition.

    The first eight bytes, little endian, of the SHA-256 of: the kind as one
    byte; the facets in bounce order as little endian ``int32``; and, for a
    diffracted path, ``-1`` then the edges it bends round in order from the
    source, the ``-1`` keeping a facet from being read as an edge. A
    diffracted path whose corners are not all on edges has no such name: it
    gives ``rank``, its rank by delay among the step's such paths, and is
    named by ``-1, -1, rank`` in place of the edges. The trace
    (:mod:`reverberate.mirror.moving`) names its rows with this function.
    """
    words = [int(f) for f in facets]
    if rank is not None:
        words += [-1, -1, int(rank)]
    elif edges is not None:
        words += [-1, *(int(e) for e in edges)]
    key = bytes([int(kind)]) + np.asarray(words, dtype="<i4").tobytes()
    found = _IDS.get(key)
    if found is None:
        found = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        _IDS[key] = found
    return found


def tail_seed(seed: int | str, source_id: str) -> int:
    """A source's tail seed: eight bytes, little endian, of ``sha256("<seed>:tail:<id>")``."""
    digest = hashlib.sha256(f"{seed}:tail:{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def default_fusion() -> dict[str, float]:
    """``fusion_json`` as the library's constants give it: the estimator, and which cells serve.

    The quadrature and the regularisation are what the engine's translation
    reads; the two distances and the two shares are the rule the trace chose
    the cells by (:func:`reverberate.spatial.translate.choose_cells`), kept
    with the pack so a reader knows what its ``mode`` means.
    """
    return {
        "quadrature_degree": QUADRATURE_DEGREE,
        "lambda": REGULARISATION,
        "exact_under_m": EXACT_UNDER_M,
        "translate_within_m": TRANSLATE_WITHIN_M,
        "fuse_within_m": FUSE_WITHIN_M,
        "source_share": SOURCE_SHARE,
        "surface_share": SURFACE_SHARE,
    }


def band_map(bands_hz: Iterable[float], bank_bands_hz: Iterable[float]) -> np.ndarray:
    """Which band of ``bands_hz`` each bank band reads: the nearest to its centre."""
    bands = np.asarray(list(bands_hz), dtype=float)
    return np.array([int(np.argmin(np.abs(bands - c))) for c in bank_bands_hz], dtype=int)


# --------------------------------------------------------------------------
# the types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Header:
    """The root attributes."""

    profile: str
    recipe_sha256: str
    dwelling: str
    scene_id: str
    duration_s: float
    steps: int
    step_s: float = STEP_S
    sample_rate_hz: float = 48000.0
    order: int = 7
    sound_speed_m_s: float = 343.2
    bands_hz: tuple[int, ...] = OCTAVE_BANDS
    bank_bands_hz: tuple[int, ...] = ()
    has_low: bool = False
    has_tail: bool = False
    low_sample_rate_hz: float = 4000.0
    low_samples: int = 4800
    fusion: Mapping[str, float] = field(default_factory=default_fusion)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return channel_count(self.order)

    @property
    def step_samples(self) -> int:
        return int(round(self.step_s * self.sample_rate_hz))

    @property
    def low_step_samples(self) -> int:
        return int(round(self.step_s * self.low_sample_rate_hz))

    @property
    def samples(self) -> int:
        """Output samples of the whole scene: its intervals, each one step long."""
        return (self.steps - 1) * self.step_samples

    @property
    def bank(self) -> tuple[int, ...]:
        return self.bank_bands_hz or band_centres(int(round(self.sample_rate_hz)))


@dataclass(frozen=True)
class Listener:
    position: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True)
class Cells:
    position: np.ndarray
    kind: np.ndarray
    lattice_index: np.ndarray
    clearance_m: np.ndarray
    room: tuple[str, ...]
    grid_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grid_step_m: tuple[float, float, float] = (0.4, 0.0, 0.4)
    layers_y_m: tuple[float, ...] = ()

    @classmethod
    def none(cls) -> Cells:
        return cls(
            np.zeros((0, 3)),
            np.zeros(0, np.uint8),
            np.zeros((0, 3), np.int32),
            np.zeros(0, np.float32),
            (),
        )


@dataclass(frozen=True)
class Early:
    """The arrivals above the crossover, every step's rows one after the other."""

    offsets: np.ndarray
    path_id: np.ndarray
    delay_s: np.ndarray
    arrival: np.ndarray
    departure: np.ndarray
    gain: np.ndarray
    order: np.ndarray
    kind: np.ndarray

    def rows(self, step: int) -> slice:
        return slice(int(self.offsets[step]), int(self.offsets[step + 1]))


@dataclass(frozen=True)
class Low:
    """The band under the crossover: responses by pair, and which a step reads."""

    #: ``[pair, channel, sample]``; an array, or the file's dataset.
    ir: Any
    pair_position: np.ndarray
    pair_cell: np.ndarray
    pair_key: np.ndarray
    seam_db: np.ndarray
    onset_s: np.ndarray
    pair: np.ndarray
    position_weight: np.ndarray
    cell: np.ndarray
    mode: np.ndarray
    #: More than two source positions a step, when the pack was traced so:
    #: ``[step, slot, 2]`` rows of ``ir`` and ``[step, slot, knot]`` weights at
    #: ``slot_knots_hz``. ``None`` in a pack whose steps read ``pair`` alone.
    slot_pair: np.ndarray | None = None
    slot_weight: np.ndarray | None = None
    slot_knots_hz: np.ndarray | None = None


@dataclass(frozen=True)
class Tail:
    """What the late part is made from: histograms, and which a step weighs."""

    #: ``[hist, bin, band]`` and ``[hist, bin, band, 16]``; arrays, or the file's datasets.
    energy: Any
    moments: Any
    scale: np.ndarray
    hist_position: np.ndarray
    hist_cell: np.ndarray
    hist: np.ndarray
    position_weight: np.ndarray
    cell_weight: np.ndarray


@dataclass(frozen=True)
class Level:
    high_gain_db: np.ndarray
    onset_s: np.ndarray


@dataclass(frozen=True)
class Source:
    id: str
    kind: str
    position: np.ndarray
    yaw_deg: np.ndarray
    audible: np.ndarray
    early: Early
    level: Level
    low: Low | None = None
    tail: Tail | None = None
    subtype: str = ""
    directivity_model: str = "omni"
    directivity_enabled: bool = False
    gain_db: float = 0.0
    tail_seed: int = 0


@dataclass(frozen=True)
class Mirror:
    """``/mirror``: the clock and scale of the wave field, and the tail's settings."""

    signature: np.ndarray = field(default_factory=lambda: np.ones(1))
    lead_s: float = 0.0
    alignment_gain: float = 1.0
    lowcut_hz: float = 0.0
    lowcut_order: int = 8
    tail_from_s: float = 0.010
    tail_bursts: int = 24
    tail_gain_db: tuple[float, ...] = (0.0,) * len(OCTAVE_BANDS)
    histogram_bin_s: float = 0.002
    histogram_order: int = 3
    receiver_radius_m: float = 0.0
    settings_json: str = "{}"


@dataclass(frozen=True)
class Air:
    """``/atmosphere``: the air, and whether the engine applies it."""

    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    enabled: bool = False


@dataclass
class ScenePack:
    """One pack. Read from disk it holds the file open: use it as a context, or close it."""

    header: Header
    recipe: bytes
    listener: Listener
    cells: Cells
    sources: dict[str, Source]
    mirror: Mirror = field(default_factory=Mirror)
    crossover: Crossover = field(default_factory=Crossover)
    air: Air = field(default_factory=Air)
    directivity: dict[str, Directivity] = field(default_factory=dict)
    _file: Any = None

    def recipe_json(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.recipe.decode("utf-8"))
        return loaded

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> ScenePack:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


# --------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------

#: Every dataset of a group: its dtype and its shape in the format's own words.
_LISTENER = {"position": ("<f8", "step,3"), "orientation": ("<f4", "step,3")}
_CELLS = {
    "position": ("<f8", "cell,3"),
    "kind": ("u1", "cell"),
    "lattice_index": ("<i4", "cell,3"),
    "clearance_m": ("<f4", "cell"),
}
_SOURCE = {"position": ("<f8", "step,3"), "yaw_deg": ("<f4", "step"), "audible": ("?", "step")}
_EARLY = {
    "offsets": ("<i8", "step+1"),
    "path_id": ("<u8", "row"),
    "delay_s": ("<f8", "row"),
    "arrival": ("<f4", "row,3"),
    "departure": ("<f4", "row,3"),
    "gain": ("<f4", "row,band"),
    "order": ("u1", "row"),
    "kind": ("u1", "row"),
}
_LOW = {
    "ir": ("<f4", "pair,channel,low_sample"),
    "pair_position": ("<f8", "pair,3"),
    "pair_cell": ("<i4", "pair"),
    "pair_key": ("S64", "pair"),
    "seam_db": ("<f4", "pair"),
    "onset_s": ("<f8", "pair"),
    "pair": ("<i4", "step,2,2"),
    "position_weight": ("<f4", "step"),
    "cell": ("<i4", "step,2"),
    "mode": ("u1", "step"),
}
#: The optional tables of ``low``: all three or none.
_LOW_SLOTS = {
    "slot_pair": ("<i4", "step,slot,2"),
    "slot_weight": ("<f4", "step,slot,knot"),
    "slot_knots_hz": ("<f8", "knot"),
}
#: A slot's weight is a share of a response: the weights of a step may pass one where
#: the source is past the last position of its rail, never by this much.
SLOT_WEIGHT_LIMIT = 16.0
_TAIL = {
    "energy": ("<f4", "hist,bin,band"),
    "moments": ("<f4", "hist,bin,band,moment"),
    "scale": ("<f8", "hist,bank"),
    "hist_position": ("<f8", "hist,3"),
    "hist_cell": ("<i4", "hist"),
    "hist": ("<i4", "step,2,2"),
    "position_weight": ("<f4", "step"),
    "cell_weight": ("<f4", "step"),
}
_LEVEL = {"high_gain_db": ("<f4", "step"), "onset_s": ("<f8", "step")}
#: Kept in the file when a pack is read; everything else is loaded.
_LAZY = {"ir", "energy", "moments"}


def _write_group(group: Any, spec: Mapping[str, tuple[str, str]], value: Any) -> None:
    for name, (dtype, _) in spec.items():
        data = getattr(value, name)
        if name in _LAZY and not isinstance(data, np.ndarray):
            # A table still in another file: copied a row at a time.
            target = group.create_dataset(name, shape=data.shape, dtype=dtype)
            for row in range(data.shape[0]):
                target[row] = data[row]
        else:
            group.create_dataset(name, data=np.asarray(data).astype(dtype, copy=False))


def _read_group(group: Any, spec: Mapping[str, tuple[str, str]]) -> dict[str, Any]:
    return {name: group[name] if name in _LAZY else np.asarray(group[name][...]) for name in spec}


class PackWriter:
    """Writes ``pack.h5`` a source at a time; closing it validates what was written."""

    def __init__(
        self,
        target: Path,
        header: Header,
        recipe: bytes,
        listener: Listener,
        cells: Cells,
        *,
        mirror: Mirror | None = None,
        crossover: Crossover | None = None,
        air: Air | None = None,
        directivity: Mapping[str, Directivity] | None = None,
    ) -> None:
        self.target = Path(target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.header = header
        if hashlib.sha256(recipe).hexdigest() != header.recipe_sha256:
            raise PackError("recipe_sha256 is not the digest of the recipe's bytes")
        mirror = mirror or Mirror()
        crossover = crossover or Crossover()
        air = air or Air()
        self._file = h5py.File(self.target, "w")
        f = self._file
        f.attrs["schema"] = SCHEMA
        f.attrs["schema_version"] = SCHEMA_VERSION
        for name in ("profile", "recipe_sha256", "dwelling", "scene_id"):
            f.attrs[name] = getattr(header, name)
        for name in ("duration_s", "step_s", "sample_rate_hz", "sound_speed_m_s"):
            f.attrs[name] = float(getattr(header, name))
        f.attrs["low_sample_rate_hz"] = float(header.low_sample_rate_hz)
        f.attrs["steps"] = int(header.steps)
        f.attrs["order"] = int(header.order)
        f.attrs["low_samples"] = int(header.low_samples)
        f.attrs["ordering"] = ORDERING
        f.attrs["normalisation"] = NORMALISATION
        f.attrs["bands_hz"] = np.asarray(header.bands_hz, dtype=np.int64)
        f.attrs["bank_bands_hz"] = np.asarray(header.bank, dtype=np.int64)
        f.attrs["has_low"] = bool(header.has_low)
        f.attrs["has_tail"] = bool(header.has_tail)
        f.attrs["fusion_json"] = json.dumps(dict(header.fusion), sort_keys=True)
        f.attrs["provenance_json"] = json.dumps(dict(header.provenance), sort_keys=True)
        f.create_dataset("recipe", data=np.frombuffer(recipe, dtype=np.uint8))
        _write_group(f.create_group("listener"), _LISTENER, listener)
        group = f.create_group("cells")
        _write_group(group, _CELLS, cells)
        group.create_dataset("room", data=list(cells.room), dtype=h5py.string_dtype())
        group.attrs["grid_origin_m"] = np.asarray(cells.grid_origin_m, dtype=float)
        group.attrs["grid_step_m"] = np.asarray(cells.grid_step_m, dtype=float)
        group.attrs["layers_y_m"] = np.asarray(cells.layers_y_m, dtype=float)
        group = f.create_group("mirror")
        group.create_dataset("signature", data=np.asarray(mirror.signature, dtype="<f8"))
        for name in ("lead_s", "alignment_gain", "lowcut_hz", "tail_from_s", "histogram_bin_s"):
            group.attrs[name] = float(getattr(mirror, name))
        group.attrs["receiver_radius_m"] = float(mirror.receiver_radius_m)
        for name in ("lowcut_order", "tail_bursts", "histogram_order"):
            group.attrs[name] = int(getattr(mirror, name))
        group.attrs["tail_gain_db"] = np.asarray(mirror.tail_gain_db, dtype=float)
        group.attrs["settings_json"] = mirror.settings_json
        group = f.create_group("crossover")
        for name, value in crossover.record().items():
            group.attrs[name] = float(value)
        group = f.create_group("atmosphere")
        group.attrs["temperature_c"] = float(air.atmosphere.temperature_c)
        group.attrs["humidity_percent"] = float(air.atmosphere.humidity_percent)
        group.attrs["pressure_kpa"] = float(air.atmosphere.pressure_kpa)
        group.attrs["enabled"] = bool(air.enabled)
        group = f.create_group("directivity")
        for model, pattern in (directivity or {}).items():
            sub = group.create_group(model)
            sub.create_dataset("gain_db", data=np.asarray(pattern.gain_db, dtype="<f4"))
            sub.attrs["angles_deg"] = np.asarray(pattern.angles_deg, dtype=float)
        f.create_group("sources")

    def add_source(self, source: Source) -> None:
        """One source's tables. Its large ones may be arrays or another file's datasets."""
        group = self._file["sources"].create_group(source.id)
        group.attrs["kind"] = source.kind
        group.attrs["subtype"] = source.subtype
        group.attrs["directivity_model"] = source.directivity_model
        group.attrs["directivity_enabled"] = bool(source.directivity_enabled)
        group.attrs["gain_db"] = float(source.gain_db)
        group.attrs["tail_seed"] = np.uint64(source.tail_seed)
        _write_group(group, _SOURCE, source)
        _write_group(group.create_group("early"), _EARLY, source.early)
        _write_group(group.create_group("level"), _LEVEL, source.level)
        if source.low is not None:
            low = group.create_group("low")
            _write_group(low, _LOW, source.low)
            if source.low.slot_pair is not None:
                # Most steps are at rest and hold one row and a weight of one: deflated.
                for name, (dtype, _) in _LOW_SLOTS.items():
                    data = np.asarray(getattr(source.low, name)).astype(dtype, copy=False)
                    packed = {} if data.ndim == 1 else {"compression": "gzip", "shuffle": True}
                    low.create_dataset(name, data=data, **packed)
        if source.tail is not None:
            _write_group(group.create_group("tail"), _TAIL, source.tail)

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        with read_pack(self.target):
            pass

    def __enter__(self) -> PackWriter:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if kind is None:
            self.close()
        elif self._file is not None:
            self._file.close()
            self._file = None


def write_pack(target: Path, pack: ScenePack) -> Path:
    """``pack`` as one file, validated before a byte is written and after the last."""
    validate(pack)
    with PackWriter(
        target,
        pack.header,
        pack.recipe,
        pack.listener,
        pack.cells,
        mirror=pack.mirror,
        crossover=pack.crossover,
        air=pack.air,
        directivity=pack.directivity,
    ) as writer:
        for source in pack.sources.values():
            writer.add_source(source)
    return Path(target)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_pack(path: Path, *, check: bool = True, deep: bool = False) -> ScenePack:
    """The pack at ``path``, its file left open for the large tables.

    ``check`` validates it; ``deep`` also reads every low band response to
    check its band limit, which costs a transform per pair.
    """
    f = h5py.File(Path(path), "r")
    try:
        a = f.attrs
        if _text(a.get("schema", "")) != SCHEMA:
            raise PackError(f"{path} is not a scene pack: schema {a.get('schema')!r}")
        if int(a["schema_version"]) != SCHEMA_VERSION:
            raise PackError(f"scene pack version {int(a['schema_version'])} is not read")
        if _text(a["ordering"]) != ORDERING or _text(a["normalisation"]) != NORMALISATION:
            raise PackError("a scene pack is ACN and N3D")
        header = Header(
            profile=_text(a["profile"]),
            recipe_sha256=_text(a["recipe_sha256"]),
            dwelling=_text(a["dwelling"]),
            scene_id=_text(a["scene_id"]),
            duration_s=float(a["duration_s"]),
            steps=int(a["steps"]),
            step_s=float(a["step_s"]),
            sample_rate_hz=float(a["sample_rate_hz"]),
            order=int(a["order"]),
            sound_speed_m_s=float(a["sound_speed_m_s"]),
            bands_hz=tuple(int(v) for v in a["bands_hz"]),
            bank_bands_hz=tuple(int(v) for v in a["bank_bands_hz"]),
            has_low=bool(a["has_low"]),
            has_tail=bool(a["has_tail"]),
            low_sample_rate_hz=float(a["low_sample_rate_hz"]),
            low_samples=int(a["low_samples"]),
            fusion=json.loads(_text(a["fusion_json"])),
            provenance=json.loads(_text(a["provenance_json"])),
        )
        cells_group = f["cells"]
        cells = Cells(
            **_read_group(cells_group, _CELLS),
            room=tuple(_text(v) for v in cells_group["room"][...]),
            grid_origin_m=tuple(float(v) for v in cells_group.attrs["grid_origin_m"]),  # type: ignore[arg-type]
            grid_step_m=tuple(float(v) for v in cells_group.attrs["grid_step_m"]),  # type: ignore[arg-type]
            layers_y_m=tuple(float(v) for v in cells_group.attrs["layers_y_m"]),
        )
        m = f["mirror"]
        mirror = Mirror(
            signature=np.asarray(m["signature"][...], dtype=float),
            lead_s=float(m.attrs["lead_s"]),
            alignment_gain=float(m.attrs["alignment_gain"]),
            lowcut_hz=float(m.attrs["lowcut_hz"]),
            lowcut_order=int(m.attrs["lowcut_order"]),
            tail_from_s=float(m.attrs["tail_from_s"]),
            tail_bursts=int(m.attrs["tail_bursts"]),
            tail_gain_db=tuple(float(v) for v in m.attrs["tail_gain_db"]),
            histogram_bin_s=float(m.attrs["histogram_bin_s"]),
            histogram_order=int(m.attrs["histogram_order"]),
            receiver_radius_m=float(m.attrs["receiver_radius_m"]),
            settings_json=_text(m.attrs["settings_json"]),
        )
        c = f["crossover"].attrs
        crossover = Crossover(
            cutoff_hz=float(c["cutoff_hz"]),
            width_octaves=float(c["width_octaves"]),
            coherent_s=float(c["coherent_s"]),
            coherent_fade_s=float(c["coherent_fade_s"]),
        )
        t = f["atmosphere"].attrs
        air = Air(
            Atmosphere(
                float(t["temperature_c"]), float(t["humidity_percent"]), float(t["pressure_kpa"])
            ),
            bool(t["enabled"]),
        )
        directivity = {
            model: Directivity(
                name=model,
                gain_db=np.asarray(group["gain_db"][...]),
                angles_deg=np.asarray(group.attrs["angles_deg"], float),
                bands_hz=header.bands_hz,
            )
            for model, group in f["directivity"].items()
        }
        sources: dict[str, Source] = {}
        for name, group in f["sources"].items():
            sources[name] = Source(
                id=name,
                kind=_text(group.attrs["kind"]),
                subtype=_text(group.attrs["subtype"]),
                directivity_model=_text(group.attrs["directivity_model"]),
                directivity_enabled=bool(group.attrs["directivity_enabled"]),
                gain_db=float(group.attrs["gain_db"]),
                tail_seed=int(group.attrs["tail_seed"]),
                **_read_group(group, _SOURCE),
                early=Early(**_read_group(group["early"], _EARLY)),
                level=Level(**_read_group(group["level"], _LEVEL)),
                low=Low(
                    **_read_group(group["low"], _LOW),
                    **(
                        _read_group(group["low"], _LOW_SLOTS) if "slot_pair" in group["low"] else {}
                    ),
                )
                if "low" in group
                else None,
                tail=Tail(**_read_group(group["tail"], _TAIL)) if "tail" in group else None,
            )
        pack = ScenePack(
            header=header,
            recipe=bytes(np.asarray(f["recipe"][...], dtype=np.uint8)),
            listener=Listener(**_read_group(f["listener"], _LISTENER)),
            cells=cells,
            sources=sources,
            mirror=mirror,
            crossover=crossover,
            air=air,
            directivity=directivity,
            _file=f,
        )
        if check:
            validate(pack, deep=deep)
        return pack
    except BaseException:
        f.close()
        raise


# --------------------------------------------------------------------------
# the invariants
# --------------------------------------------------------------------------


def _need(condition: Any, message: str) -> None:
    if not bool(condition):
        raise PackError(message)


def _shapes(
    where: str, spec: Mapping[str, tuple[str, str]], value: Any, sizes: dict[str, int]
) -> None:
    """Every dataset of ``value`` has the shape the format gives it; free sizes are learnt."""
    for name, (dtype, dims) in spec.items():
        data = getattr(value, name)
        shape = tuple(int(v) for v in data.shape)
        names = dims.split(",")
        _need(len(shape) == len(names), f"{where}/{name} has shape {shape}, expected [{dims}]")
        for axis, dim in enumerate(names):
            if dim.isdigit():
                want = int(dim)
            elif dim == "step+1":
                want = sizes["step"] + 1
            else:
                want = sizes.setdefault(f"{where}:{dim}" if dim in _FREE else dim, shape[axis])
            _need(shape[axis] == want, f"{where}/{name} has shape {shape}, expected [{dims}]")
        kind = np.dtype(dtype).kind
        have = np.dtype(data.dtype).kind
        _need(
            have == kind or (kind == "f" and have == "f"),
            f"{where}/{name} is {data.dtype}, expected {dtype}",
        )


#: Sizes that belong to one group of one source, not to the pack.
_FREE = {"row", "pair", "hist", "bin", "slot", "knot"}


def _unit(where: str, vectors: np.ndarray) -> None:
    if vectors.size:
        norms = np.linalg.norm(np.asarray(vectors, dtype=float), axis=-1)
        _need(
            np.all(np.abs(norms - 1.0) <= UNIT_TOLERANCE),
            f"{where} is not unit to {UNIT_TOLERANCE}: |v| reaches {norms.max():.9f}",
        )


def _band_limit_hz(crossover: Crossover) -> float:
    return crossover.band_hz()[1]


def validate(pack: ScenePack, *, deep: bool = False) -> None:
    """What ``scene-pack.md`` lets a reader assume, checked; :class:`PackError` otherwise."""
    h = pack.header
    _need(h.profile in PROFILES, f"profile {h.profile!r} is not one of {PROFILES}")
    _need(h.steps >= 2, "a pack has at least two steps")
    _need(
        h.steps == int(round(h.duration_s / h.step_s)) + 1,
        f"steps {h.steps} is not round(duration_s / step_s) + 1",
    )
    _need(
        abs(h.step_s * h.sample_rate_hz - h.step_samples) < 1e-6
        and abs(h.step_s * h.low_sample_rate_hz - h.low_step_samples) < 1e-6,
        "a step is not a whole number of samples at both rates",
    )
    _need(
        abs(
            h.sample_rate_hz / h.low_sample_rate_hz - round(h.sample_rate_hz / h.low_sample_rate_hz)
        )
        < 1e-9,
        "the output rate is not a multiple of the low band's",
    )
    _need(
        hashlib.sha256(pack.recipe).hexdigest() == h.recipe_sha256,
        "recipe_sha256 is not the digest of /recipe",
    )
    _need(len(h.bank) >= 1 and len(h.bands_hz) >= 1, "a pack names its bands")
    sizes: dict[str, int] = {
        "step": h.steps,
        "band": len(h.bands_hz),
        "bank": len(h.bank),
        "channel": h.channels,
        "low_sample": h.low_samples,
    }
    _shapes("listener", _LISTENER, pack.listener, sizes)
    _shapes("cells", _CELLS, pack.cells, sizes)
    cells = sizes["cell"]
    _need(len(pack.cells.room) == cells, "cells/room has not one name per cell")
    _need(np.all(np.isfinite(pack.listener.position)), "listener/position is not finite")
    _need(pack.mirror.signature.ndim == 1 and pack.mirror.signature.size >= 1, "no signature")
    _need(
        len(pack.mirror.tail_gain_db) == len(h.bands_hz),
        "mirror/tail_gain_db has not one value per band",
    )
    for model, pattern in pack.directivity.items():
        _need(
            pattern.gain_db.shape == (len(h.bands_hz), pattern.angles_deg.size),
            f"directivity/{model}/gain_db is not [band, angle]",
        )
        _need(
            pattern.angles_deg[0] == 0.0
            and pattern.angles_deg[-1] == 180.0
            and np.all(np.diff(pattern.angles_deg) > 0.0),
            f"directivity/{model} does not run from 0 to 180 degrees",
        )
    window_s = h.low_samples / h.low_sample_rate_hz
    for name, source in pack.sources.items():
        where = f"sources/{name}"
        _need(source.id == name, f"{where} holds the source {source.id!r}")
        local = dict(sizes)
        _shapes(where, _SOURCE, source, local)
        _shapes(f"{where}/early", _EARLY, source.early, local)
        _shapes(f"{where}/level", _LEVEL, source.level, local)
        _need((source.low is not None) == h.has_low, f"{where}: has_low and the low group disagree")
        _need(
            (source.tail is not None) == h.has_tail,
            f"{where}: has_tail and the tail group disagree",
        )
        if source.directivity_enabled:
            _need(
                source.directivity_model in pack.directivity,
                f"{where} names the directivity {source.directivity_model!r}, which is not held",
            )
        audible = np.asarray(source.audible, dtype=bool)
        early = source.early
        offsets = np.asarray(early.offsets)
        rows = int(early.path_id.shape[0])
        _need(
            offsets[0] == 0 and offsets[-1] == rows and np.all(np.diff(offsets) >= 0),
            f"{where}/early/offsets does not run from 0 to the row count without decreasing",
        )
        counts = np.diff(offsets)
        _need(np.all(counts[~audible] == 0), f"{where}: a step that is not audible has rows")
        if rows:
            step_of = np.repeat(np.arange(h.steps), counts)
            same = step_of[1:] == step_of[:-1]
            _need(
                np.all(early.path_id[1:][same] > early.path_id[:-1][same]),
                f"{where}/early: a step's rows are not sorted by path_id without repeats",
            )
            _unit(f"{where}/early/arrival", early.arrival)
            _unit(f"{where}/early/departure", early.departure)
            _need(
                np.all(np.isfinite(early.gain)) and np.all(early.gain >= 0.0),
                f"{where}/early/gain is not finite and non negative",
            )
            _need(
                np.all(early.delay_s > 0.0) and np.all(early.delay_s < window_s),
                f"{where}/early/delay_s is not positive and under {window_s} s",
            )
            _need(np.all(early.kind <= 3), f"{where}/early/kind is not 0 to 3")
        _need(np.all(np.isfinite(source.level.high_gain_db)), f"{where}/level is not finite")
        if source.low is not None:
            _check_low(where, source.low, audible, local, cells, pack, deep)
        if source.tail is not None:
            _check_tail(where, source.tail, audible, local, cells)


def _slots(
    where: str,
    table: np.ndarray,
    audible: np.ndarray,
    second_source: np.ndarray,
    second_cell: np.ndarray,
    count: int,
) -> None:
    """Invariant 4: which of the four slots of a step hold a row, and that they exist."""
    table = np.asarray(table)
    _need(np.all(table[~audible] == -1), f"{where}: a step that is not audible holds an index")
    _need(np.all((table >= -1) & (table < count)), f"{where} points outside its table")
    live = table[audible] >= 0
    _need(np.all(live[:, 0, 0]), f"{where}[k, 0, 0] is missing at an audible step")
    source_slot = second_source[audible]
    cell_slot = second_cell[audible]
    _need(
        np.all(live[:, 1, 0] == source_slot),
        f"{where}[k, 1, :] is not valid exactly when position_weight is positive",
    )
    _need(
        np.all(live[:, 0, 1] == cell_slot) and np.all(live[:, 1, 1] == (source_slot & cell_slot)),
        f"{where}[k, :, 1] is not valid exactly when the second cell is used",
    )


def _weights(where: str, values: np.ndarray) -> None:
    _need(np.all((values >= 0.0) & (values <= 1.0)), f"{where} is not in [0, 1]")


def _check_low(
    where: str,
    low: Low,
    audible: np.ndarray,
    sizes: dict[str, int],
    cells: int,
    pack: ScenePack,
    deep: bool,
) -> None:
    where = f"{where}/low"
    _shapes(where, _LOW, low, sizes)
    pairs = int(low.pair_cell.shape[0])
    mode = np.asarray(low.mode)
    _need(np.all(mode <= 3), f"{where}/mode is not 0 to 3")
    _need(np.all(mode[~audible] == 0), f"{where}/mode is not 0 where the source is not audible")
    _need(np.all(mode[audible] >= 1), f"{where}/mode is 0 at an audible step")
    _weights(f"{where}/position_weight", low.position_weight)
    _slots(f"{where}/pair", low.pair, audible, low.position_weight > 0.0, mode == 3, pairs)
    cell = np.asarray(low.cell)
    _need(np.all(cell[~audible] == -1), f"{where}/cell holds an index where not audible")
    _need(np.all((cell >= -1) & (cell < cells)), f"{where}/cell points outside /cells")
    _need(np.all(cell[audible][:, 0] >= 0), f"{where}/cell[k, 0] is missing at an audible step")
    _need(
        np.all((cell[audible][:, 1] >= 0) == (mode[audible] == 3)),
        f"{where}/cell[k, 1] is not valid exactly when mode is 3",
    )
    _need(
        np.all((low.pair_cell >= 0) & (low.pair_cell < cells)),
        f"{where}/pair_cell points outside /cells",
    )
    # Invariant 5: a step's pair was solved at the step's cell.
    for b in range(2):
        for a in range(2):
            rows = np.asarray(low.pair)[:, a, b]
            held = rows >= 0
            _need(
                np.all(low.pair_cell[rows[held]] == cell[held, b]),
                f"{where}: pair[k, {a}, {b}] was not solved at cell[k, {b}]",
            )
    given = [getattr(low, name) is not None for name in _LOW_SLOTS]
    _need(all(given) or not any(given), f"{where} holds some of {sorted(_LOW_SLOTS)}, not all")
    if all(given):
        _check_slots(where, low, audible, sizes, pairs)
    exact = float(pack.header.fusion.get("exact_under_m", 0.001))
    is_exact = audible & (mode == 1)
    if is_exact.any():
        away = np.linalg.norm(
            pack.listener.position[is_exact] - pack.cells.position[cell[is_exact, 0]], axis=1
        )
        _need(
            np.all(away <= exact + 1e-9),
            f"{where}: mode 1 with the head {away.max():.4f} m from its cell",
        )
    if pairs:
        limit = _band_limit_hz(pack.crossover)
        freqs = np.fft.rfftfreq(pack.header.low_samples, 1.0 / pack.header.low_sample_rate_hz)
        above = freqs > limit * 1.001
        for row in range(pairs) if deep else sorted({0, pairs - 1}):
            spectrum = np.abs(np.fft.rfft(np.asarray(low.ir[row], dtype=float), axis=-1))
            peak = float(spectrum.max())
            _need(np.isfinite(peak), f"{where}/ir[{row}] is not finite")
            _need(
                peak == 0.0 or float(spectrum[:, above].max()) <= 1e-4 * peak,
                f"{where}/ir[{row}] is not zero above {limit:.0f} Hz",
            )


def _check_slots(
    where: str, low: Low, audible: np.ndarray, sizes: dict[str, int], pairs: int
) -> None:
    """Invariant 13: the slots of a step past the first two, and every slot's weights."""
    _shapes(where, _LOW_SLOTS, low, sizes)
    slots = np.asarray(low.slot_pair)
    weights = np.asarray(low.slot_weight, dtype=float)
    knots = np.asarray(low.slot_knots_hz, dtype=float)
    _need(slots.shape[1] >= 2, f"{where}/slot_pair holds fewer than two slots")
    _need(
        knots.size >= 2 and knots[0] == 0.0 and np.all(np.diff(knots) > 0.0),
        f"{where}/slot_knots_hz does not rise from zero",
    )
    _need(
        np.allclose(np.diff(knots), knots[1] - knots[0]),
        f"{where}/slot_knots_hz is not evenly spaced",
    )
    _need(np.all((slots >= -1) & (slots < pairs)), f"{where}/slot_pair points outside its table")
    _need(
        np.array_equal(slots[:, :2, :], np.asarray(low.pair)),
        f"{where}/slot_pair[k, :2] is not pair[k]",
    )
    cell = np.asarray(low.cell)
    for b in range(2):
        rows = slots[:, :, b]
        held = rows >= 0
        _need(
            np.all(
                low.pair_cell[rows[held]] == np.broadcast_to(cell[:, b : b + 1], rows.shape)[held]
            ),
            f"{where}: slot_pair[k, :, {b}] was not solved at cell[k, {b}]",
        )
        if b:
            _need(
                np.all(held == (held[:, :1] & (slots[:, :, 0] >= 0))),
                f"{where}/slot_pair[k, :, 1] is not valid exactly where the second cell is used",
            )
    _need(np.all(np.isfinite(weights)), f"{where}/slot_weight is not finite")
    _need(
        np.all(np.abs(weights) <= SLOT_WEIGHT_LIMIT),
        f"{where}/slot_weight passes {SLOT_WEIGHT_LIMIT}",
    )
    _need(
        np.all(weights[slots[:, :, 0] < 0] == 0.0),
        f"{where}/slot_weight is not zero where a slot holds no row",
    )
    alone = audible & (slots[:, 1, 0] < 0)
    _need(
        np.all(weights[alone, 0, :] == 1.0),
        f"{where}/slot_weight is not one where a step reads one position",
    )


def _check_tail(
    where: str, tail: Tail, audible: np.ndarray, sizes: dict[str, int], cells: int
) -> None:
    where = f"{where}/tail"
    sizes = dict(sizes)
    sizes["moment"] = int(tail.moments.shape[-1])
    _shapes(where, _TAIL, tail, sizes)
    order = int(round(np.sqrt(sizes["moment"]))) - 1
    _need(channel_count(order) == sizes["moment"], f"{where}/moments is not a whole order")
    hists = int(tail.hist_cell.shape[0])
    _weights(f"{where}/position_weight", tail.position_weight)
    _weights(f"{where}/cell_weight", tail.cell_weight)
    _slots(
        f"{where}/hist",
        tail.hist,
        audible,
        tail.position_weight > 0.0,
        tail.cell_weight > 0.0,
        hists,
    )
    _need(
        np.all((tail.hist_cell >= 0) & (tail.hist_cell < max(cells, 1))),
        f"{where}/hist_cell points outside /cells",
    )
    _need(
        np.all(np.isfinite(tail.scale)) and np.all(tail.scale >= 0.0),
        f"{where}/scale is not finite and non negative",
    )


# --------------------------------------------------------------------------
# the synthetic profile
# --------------------------------------------------------------------------

#: Under this the synthetic low band is faded out: see :func:`synthetic_free_field`.
SYNTHETIC_HIGHPASS_HZ = (80.0, 160.0)


def synthetic_recipe(duration_s: float, source_ids: Iterable[str]) -> bytes:
    """A recipe's canonical bytes holding only what the engine reads: sources and no activity."""
    recipe = {
        "schema": "reverberate.scene-recipe",
        "duration_s": float(duration_s),
        "sources": [{"id": name, "gain_db": 0.0, "activity": []} for name in source_ids],
    }
    text = json.dumps(
        recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def monopole_low_response(
    offset_scene: np.ndarray,
    header: Header,
    crossover: Crossover,
    *,
    highpass_hz: tuple[float, float] = SYNTHETIC_HIGHPASS_HZ,
) -> np.ndarray:
    """A free field monopole seen from a point, as the low band holds it: ``[channel, sample]``.

    ``offset_scene`` is the source seen from the point. The interior
    expansion of ``e^{-ikR} / R`` divided by ``i^n`` per degree, through the
    crossover's low pressure mask (a single arrival lies wholly in the onset
    window) and a raised cosine high pass between ``highpass_hz``. On the
    scale of ``low/ir``: the samples of the 48 kHz response, which
    :func:`reverberate.spatial.lowband.from_stored` gives back.
    """
    from reverberate.spatial.field import monopole_coefficients
    from reverberate.spatial.sh import degrees_of, scene_to_ambisonic

    samples, rate = header.low_samples, header.low_sample_rate_hz
    freqs = np.fft.rfftfreq(samples, 1.0 / rate)
    spectrum = np.zeros((freqs.size, header.channels), dtype=complex)
    lo, hi = highpass_hz
    keep = freqs > lo
    k = 2.0 * np.pi * freqs[keep] / header.sound_speed_m_s
    source = scene_to_ambisonic(np.asarray(offset_scene, dtype=float)[None, :])[0]
    spectrum[keep] = (
        4.0
        * np.pi
        * monopole_coefficients(source, k, header.order)
        / (1j ** degrees_of(header.order))[None, :]
    )
    ramp = np.clip((freqs - lo) / (hi - lo), 0.0, 1.0)
    spectrum *= (0.5 - 0.5 * np.cos(np.pi * ramp))[:, None]
    # The masks are those of the 48 kHz transform, read at the bins the low rate keeps.
    low_mask, _ = crossover.masks(samples, rate, power=False)
    spectrum *= low_mask[:, None]
    # The spectrum is the response's own: its 48 kHz samples are this transform
    # over the ratio of the two rates.
    scale = rate / header.sample_rate_hz
    return np.asarray(np.fft.irfft(spectrum.T, n=samples, axis=-1) * scale, dtype=np.float32)


def synthetic_free_field(
    *,
    level: str = "A",
    source: Iterable[float] = (2.0, 1.5, 0.0),
    listener_start: Iterable[float] = (0.0, 1.5, 0.0),
    listener_end: Iterable[float] | None = None,
    duration_s: float = 1.0,
    source_id: str = "s1",
    cell_origin: Iterable[float] | None = None,
    fuse: bool = False,
    audible: np.ndarray | None = None,
    seed: int = 0,
) -> ScenePack:
    """The ``synthetic-free-field`` profile: one omnidirectional source and no room.

    The listener rests at ``listener_start`` or walks the straight line to
    ``listener_end`` at constant speed. ``level`` ``"A"`` has no low band and
    no crossover; ``"B"`` has two cells 0.40 m apart on the listener's line,
    the first at ``cell_origin`` (the listener's start by default), each
    holding the monopole's expansion. A head within ``exact_under_m`` of a
    cell is ``mode`` 1; elsewhere 2, or 3 with ``fuse``. ``audible`` marks the
    steps that have rows (all of them by default), which is how a test makes
    a birth and a death.
    """
    if level not in ("A", "B"):
        raise ValueError(f"level is 'A' or 'B', got {level!r}")
    steps = int(round(duration_s / STEP_S)) + 1
    has_low = level == "B"
    recipe = synthetic_recipe(duration_s, [source_id])
    header = Header(
        profile="synthetic-free-field",
        recipe_sha256=hashlib.sha256(recipe).hexdigest(),
        dwelling="none",
        scene_id="synthetic",
        duration_s=float(duration_s),
        steps=steps,
        bank_bands_hz=band_centres(48000),
        has_low=has_low,
        has_tail=False,
        provenance={
            "recipe_sha256": hashlib.sha256(recipe).hexdigest(),
            "code_version": "test",
            "cost": [],
        },
    )
    start = np.asarray(list(listener_start), dtype=float)
    end = start if listener_end is None else np.asarray(list(listener_end), dtype=float)
    u = np.linspace(0.0, 1.0, steps)[:, None]
    listener = start[None, :] * (1.0 - u) + end[None, :] * u
    position = np.asarray(list(source), dtype=float)
    heard = np.ones(steps, dtype=bool) if audible is None else np.asarray(audible, dtype=bool)
    to_source = position[None, :] - listener
    distance = np.linalg.norm(to_source, axis=1)
    arrival = (to_source / distance[:, None])[heard]
    rows = int(heard.sum())
    early = Early(
        offsets=np.concatenate([[0], np.cumsum(heard)]).astype(np.int64),
        path_id=np.full(rows, path_id(0), dtype=np.uint64),
        delay_s=(distance / header.sound_speed_m_s)[heard],
        arrival=arrival.astype(np.float32),
        departure=(-arrival).astype(np.float32),
        gain=np.repeat((1.0 / distance[heard])[:, None], len(header.bands_hz), 1).astype(
            np.float32
        ),
        order=np.zeros(rows, dtype=np.uint8),
        kind=np.zeros(rows, dtype=np.uint8),
    )
    crossover = Crossover()
    cells = Cells.none()
    low = None
    if has_low:
        first = start if cell_origin is None else np.asarray(list(cell_origin), dtype=float)
        line = end - start
        along = line / np.linalg.norm(line) if np.linalg.norm(line) > 0 else np.array([1.0, 0, 0])
        centres = np.stack([first, first + 0.40 * along])
        cells = Cells(
            position=centres,
            kind=np.zeros(2, dtype=np.uint8),
            lattice_index=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
            clearance_m=np.full(2, 10.0, dtype=np.float32),
            room=("free", "free"),
            grid_origin_m=(float(first[0]), float(first[1]), float(first[2])),
            layers_y_m=(float(first[1]),),
        )
        away = np.linalg.norm(listener[:, None, :] - centres[None, :, :], axis=2)  # [step, cell]
        nearer = np.argmin(away, axis=1)
        exact = away[np.arange(steps), nearer] <= float(header.fusion["exact_under_m"])
        mode = np.where(exact, 1, 3 if fuse else 2).astype(np.uint8)
        mode[~heard] = 0
        cell = np.full((steps, 2), -1, dtype=np.int32)
        cell[heard, 0] = nearer[heard]
        fused = mode == 3
        cell[fused, 1] = 1 - nearer[fused]
        pair = np.full((steps, 2, 2), -1, dtype=np.int32)
        pair[:, 0, :] = cell  # one source position: the pair's row is its cell's
        ir = np.stack(
            [monopole_low_response(position - centre, header, crossover) for centre in centres]
        )
        direct_s = np.linalg.norm(position[None, :] - centres, axis=1) / header.sound_speed_m_s
        low = Low(
            ir=ir,
            pair_position=np.repeat(position[None, :], 2, 0),
            pair_cell=np.arange(2, dtype=np.int32),
            pair_key=np.array(
                [hashlib.sha256(f"synthetic:{seed}:{i}".encode()).hexdigest() for i in range(2)],
                dtype="S64",
            ),
            seam_db=np.zeros(2, dtype=np.float32),
            onset_s=direct_s,
            pair=pair,
            position_weight=np.zeros(steps, dtype=np.float32),
            cell=cell,
            mode=mode,
        )
    made = Source(
        id=source_id,
        kind="noise",
        position=np.repeat(position[None, :], steps, 0),
        yaw_deg=np.zeros(steps, dtype=np.float32),
        audible=heard,
        early=early,
        level=Level(
            high_gain_db=np.zeros(steps, dtype=np.float32),
            onset_s=distance / header.sound_speed_m_s,
        ),
        low=low,
        tail_seed=tail_seed(seed, source_id),
    )
    pack = ScenePack(
        header=header,
        recipe=recipe,
        listener=Listener(listener, np.zeros((steps, 3), dtype=np.float32)),
        cells=cells,
        sources={source_id: made},
        mirror=Mirror(),
        crossover=crossover,
        air=Air(enabled=False),
        directivity={"omni": omni()},
    )
    validate(pack)
    return pack


def sources_of(pack: ScenePack, chosen: Iterable[str] | None = None) -> Iterator[Source]:
    """The chosen sources, in the pack's order; every one when ``chosen`` is ``None``."""
    names = list(pack.sources) if chosen is None else list(chosen)
    for name in names:
        if name not in pack.sources:
            raise KeyError(f"the pack holds no source {name!r}; it holds {sorted(pack.sources)}")
        yield pack.sources[name]
