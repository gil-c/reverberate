"""The scene recipe as typed values: parsing, the canonical form, the identity.

The format is ``docs/formats/scene-recipe.md`` and this module is its reader
and its writer, of version 1 and of version 2: a recipe keeps the version it
was written in, and a version 1 recipe has the bytes and the identity it
always had. Version 2 adds who talks with whom and what the wearer says
himself, the small movements that cost nothing, the levels, and the sources
that are fixed to an object; every key it adds is refused in a version 1
recipe. Three things are decided here and nowhere else:

- **Parsing is strict.** A key the format does not name, a missing key or a
  value of another type is refused with rule 1. A float written as an integer
  is such a value: ``12`` and ``12.0`` would give two identities to one scene.
- **The canonical form** is the sorted, compact JSON of the format followed by
  one line feed, and **the identity** is the SHA-256 of those bytes.
- **Numbers are quantised by the name of their key**: the suffix says the unit,
  the unit says the step. :func:`quantise` rounds, rule 11 of
  :mod:`reverberate.scenes.validate` checks.

Nothing here touches a file except :func:`load_recipe` and :func:`save_recipe`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "Activity",
    "Assets",
    "Atmosphere",
    "Attach",
    "Clip",
    "Directivity",
    "Dwell",
    "Dwelling",
    "Facing",
    "Gaze",
    "GeneratorRecord",
    "Heights",
    "Keyframe",
    "Listener",
    "Membership",
    "Opening",
    "Output",
    "Rail",
    "Recipe",
    "RecipeError",
    "Rise",
    "Role",
    "Scene",
    "Segment",
    "Source",
    "Station",
    "Sway",
    "Travel",
    "canonical_bytes",
    "decimals_of",
    "load_recipe",
    "parse_recipe",
    "quantise",
    "quantise_tree",
    "recipe_sha256",
    "save_recipe",
]

SCHEMA = "reverberate.scene-recipe"
#: The version a recipe is written in today, and the versions that are read.
SCHEMA_VERSION = 2
SCHEMA_VERSIONS = (1, 2)

STATION_KINDS = ("stand", "seat", "waypoint")
HEIGHTS = ("standing", "seated")
SOURCE_KINDS = ("near_voice", "far_voice", "noise")
NOISE_SUBTYPES = ("appliance", "television", "music", "water", "street", "other")

# Version 2. A ``fixture`` is where a fixed source stands, on or by an object, at a
# height of its own; nobody walks to it.
STATION_KINDS_V2 = (*STATION_KINDS, "fixture")
HEIGHTS_V2 = (*HEIGHTS, "fixed")
#: Distance is no role: a voice is a voice, and what it is to the listener is said by
#: interval (:class:`Role`). The wearer's own voice and a programme's are kinds apart.
SOURCE_KINDS_V2 = ("voice", "own_voice", "media_voice", "noise")
VOICE_KINDS_V2 = ("voice", "own_voice")
SUBTYPES_V2 = {
    "media_voice": ("television", "radio"),
    "noise": ("appliance", "music", "water", "other", "body", "steps", "outside"),
}
ROLES = ("conversation", "outside")
#: Vocal efforts, quietest first (ISO 9921, annex A, and a whisper under them).
EFFORTS = ("whisper", "relaxed", "normal", "raised", "loud")
EVENTS = ("turn", "backchannel", "laughter")
GAZE_MODES = ("talker", "glance", "reading", "away", "event", "walk")
SWAY_AXES = ("x", "y", "z", "yaw")
ATTACH_POINTS = ("mouth", "floor")
OPENING_STATES = ("open", "closed")
#: The name an attachment gives the listener; no source may take it.
LISTENER = "listener"
PROFILES = ("constant", "smoothstep")
FACING_MODES = ("fixed", "travel", "listener")
STATION_ID = re.compile(r"[a-z0-9_]+")


class RecipeError(ValueError):
    """A recipe that breaks a rule of the format, named by the rule's number."""

    def __init__(self, rule: int, message: str) -> None:
        super().__init__(f"rule {rule}: {message}")
        self.rule = rule
        self.message = message


# --------------------------------------------------------------------------
# strict readers
# --------------------------------------------------------------------------


def _shape(message: str) -> RecipeError:
    return RecipeError(1, message)


def _keys(value: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> Any:
    if not isinstance(value, dict):
        raise _shape(f"{where} is not an object")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise _shape(f"{where} has unknown key {unknown[0]!r}")
    missing = [key for key in required if key not in value]
    if missing:
        raise _shape(f"{where} lacks {missing[0]!r}")
    return value


def _float(value: Any, where: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise _shape(f"{where} is {value!r}, expected a finite float written with its point")
    return value


def _int(value: Any, where: str) -> int:
    if type(value) is not int:
        raise _shape(f"{where} is {value!r}, expected an integer")
    return value


def _str(value: Any, where: str, among: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str):
        raise _shape(f"{where} is {value!r}, expected a string")
    if among is not None and value not in among:
        raise _shape(f"{where} is {value!r}, expected one of {', '.join(among)}")
    return value


def _bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise _shape(f"{where} is {value!r}, expected true or false")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise _shape(f"{where} is not an array")
    return value


def _vector(value: Any, where: str, size: int) -> tuple[float, ...]:
    items = _list(value, where)
    if len(items) != size:
        raise _shape(f"{where} has {len(items)} numbers, expected {size}")
    return tuple(_float(item, where) for item in items)


# --------------------------------------------------------------------------
# the values
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dwelling:
    name: str
    scene_id: str
    floor_y_m: float

    @classmethod
    def from_dict(cls, value: Any) -> Dwelling:
        d = _keys(value, "dwelling", ("name", "scene_id", "floor_y_m"))
        return cls(
            _str(d["name"], "dwelling.name"),
            _str(d["scene_id"], "dwelling.scene_id"),
            _float(d["floor_y_m"], "dwelling.floor_y_m"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "scene_id": self.scene_id, "floor_y_m": self.floor_y_m}


@dataclass(frozen=True)
class Assets:
    export_sha256: str
    voxel_low_key: str
    mirror_scene_key: str
    calibration_key: str
    directivity: dict[str, str]
    rooms_rule: str

    _KEYS = (
        "export_sha256",
        "voxel_low_key",
        "mirror_scene_key",
        "calibration_key",
        "directivity",
        "rooms_rule",
    )

    @classmethod
    def from_dict(cls, value: Any) -> Assets:
        d = _keys(value, "assets", cls._KEYS)
        tables = d["directivity"]
        if not isinstance(tables, dict):
            raise _shape("assets.directivity is not an object")
        return cls(
            _str(d["export_sha256"], "assets.export_sha256"),
            _str(d["voxel_low_key"], "assets.voxel_low_key"),
            _str(d["mirror_scene_key"], "assets.mirror_scene_key"),
            _str(d["calibration_key"], "assets.calibration_key"),
            {
                _str(model, "assets.directivity"): _str(digest, f"assets.directivity.{model}")
                for model, digest in tables.items()
            },
            _str(d["rooms_rule"], "assets.rooms_rule"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_sha256": self.export_sha256,
            "voxel_low_key": self.voxel_low_key,
            "mirror_scene_key": self.mirror_scene_key,
            "calibration_key": self.calibration_key,
            "directivity": dict(self.directivity),
            "rooms_rule": self.rooms_rule,
        }


@dataclass(frozen=True)
class Output:
    order: int = 7
    sample_rate_hz: int = 48000

    @classmethod
    def from_dict(cls, value: Any) -> Output:
        d = _keys(value, "output", ("order", "sample_rate_hz"))
        return cls(_int(d["order"], "output.order"), _int(d["sample_rate_hz"], "output.rate"))

    def to_dict(self) -> dict[str, Any]:
        return {"order": self.order, "sample_rate_hz": self.sample_rate_hz}


@dataclass(frozen=True)
class Atmosphere:
    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    pressure_kpa: float = 101.325

    @classmethod
    def from_dict(cls, value: Any) -> Atmosphere:
        d = _keys(value, "atmosphere", ("temperature_c", "humidity_percent", "pressure_kpa"))
        return cls(*(_float(d[key], f"atmosphere.{key}") for key in d_order(cls)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "pressure_kpa": self.pressure_kpa,
        }


def d_order(cls: Any) -> tuple[str, ...]:
    """The fields of a dataclass in their order."""
    return tuple(cls.__dataclass_fields__)


@dataclass(frozen=True)
class Heights:
    standing_m: float = 1.7
    seated_m: float = 1.2

    @classmethod
    def from_dict(cls, value: Any) -> Heights:
        d = _keys(value, "heights", ("standing_m", "seated_m"))
        return cls(
            _float(d["standing_m"], "heights.standing_m"), _float(d["seated_m"], "heights.seated_m")
        )

    def to_dict(self) -> dict[str, Any]:
        return {"standing_m": self.standing_m, "seated_m": self.seated_m}

    def of(self, height: str) -> float:
        return self.seated_m if height == "seated" else self.standing_m


@dataclass(frozen=True)
class Station:
    id: str
    kind: str
    position: tuple[float, float, float]
    height: str
    room: str
    facing_yaw_deg: float
    object: str | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str, version: int = 1) -> Station:
        d = _keys(
            value,
            where,
            ("id", "kind", "position", "height", "room", "facing_yaw_deg"),
            ("object",),
        )
        ident = _str(d["id"], f"{where}.id")
        if not STATION_ID.fullmatch(ident):
            raise _shape(f"{where}.id {ident!r} is not [a-z0-9_]+")
        x, y, z = _vector(d["position"], f"{where}.position", 3)
        return cls(
            ident,
            _str(d["kind"], f"{where}.kind", STATION_KINDS if version == 1 else STATION_KINDS_V2),
            (x, y, z),
            _str(d["height"], f"{where}.height", HEIGHTS if version == 1 else HEIGHTS_V2),
            _str(d["room"], f"{where}.room"),
            _float(d["facing_yaw_deg"], f"{where}.facing_yaw_deg"),
            _str(d["object"], f"{where}.object") if "object" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "position": list(self.position),
            "height": self.height,
            "room": self.room,
            "facing_yaw_deg": self.facing_yaw_deg,
        }
        if self.object is not None:
            out["object"] = self.object
        return out

    @property
    def xz(self) -> tuple[float, float]:
        return (self.position[0], self.position[2])


@dataclass(frozen=True)
class Rail:
    id: str
    a: str
    b: str
    points: tuple[tuple[float, float], ...]
    pitch_m: float = 0.08

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Rail:
        d = _keys(value, where, ("id", "a", "b", "points", "pitch_m"))
        points = _list(d["points"], f"{where}.points")
        if len(points) < 2:
            raise _shape(f"{where}.points has fewer than two points")
        pairs = []
        for point in points:
            x, z = _vector(point, f"{where}.points", 2)
            pairs.append((x, z))
        return cls(
            _str(d["id"], f"{where}.id"),
            _str(d["a"], f"{where}.a"),
            _str(d["b"], f"{where}.b"),
            tuple(pairs),
            _float(d["pitch_m"], f"{where}.pitch_m"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "a": self.a,
            "b": self.b,
            "points": [list(point) for point in self.points],
            "pitch_m": self.pitch_m,
        }


@dataclass(frozen=True)
class Facing:
    mode: str
    yaw_deg: float | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Facing:
        if not isinstance(value, dict):
            raise _shape(f"{where} is not an object")
        mode = _str(value.get("mode"), f"{where}.mode", FACING_MODES)
        if mode == "fixed":
            d = _keys(value, where, ("mode", "yaw_deg"))
            return cls(mode, _float(d["yaw_deg"], f"{where}.yaw_deg"))
        _keys(value, where, ("mode",))
        return cls(mode)

    def to_dict(self) -> dict[str, Any]:
        if self.mode == "fixed":
            return {"mode": self.mode, "yaw_deg": self.yaw_deg}
        return {"mode": self.mode}


@dataclass(frozen=True)
class Dwell:
    station: str
    height: str
    start_s: float
    end_s: float
    facing: Facing
    type: str = "dwell"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "dwell",
            "station": self.station,
            "height": self.height,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "facing": self.facing.to_dict(),
        }


@dataclass(frozen=True)
class Travel:
    rail: str
    origin: str
    to: str
    profile: str
    start_s: float
    end_s: float
    facing: Facing
    type: str = "travel"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "travel",
            "rail": self.rail,
            "from": self.origin,
            "to": self.to,
            "profile": self.profile,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "facing": self.facing.to_dict(),
        }


@dataclass(frozen=True)
class Rise:
    station: str
    to: str
    start_s: float
    end_s: float
    type: str = "rise"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "rise",
            "station": self.station,
            "to": self.to,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


Segment = Dwell | Travel | Rise


def _segment(value: Any, where: str, version: int = 1) -> Segment:
    if not isinstance(value, dict):
        raise _shape(f"{where} is not an object")
    kind = _str(value.get("type"), f"{where}.type", ("dwell", "travel", "rise"))
    times = ("start_s", "end_s")
    if kind == "dwell":
        d = _keys(value, where, ("type", "station", "height", *times, "facing"))
        facing = Facing.from_dict(d["facing"], f"{where}.facing")
        if facing.mode == "travel":
            raise _shape(f"{where}.facing follows a rail's tangent, and a dwell is on none")
        return Dwell(
            _str(d["station"], f"{where}.station"),
            _str(d["height"], f"{where}.height", HEIGHTS if version == 1 else HEIGHTS_V2),
            _float(d["start_s"], f"{where}.start_s"),
            _float(d["end_s"], f"{where}.end_s"),
            facing,
        )
    if kind == "travel":
        d = _keys(value, where, ("type", "rail", "from", "to", "profile", *times, "facing"))
        return Travel(
            _str(d["rail"], f"{where}.rail"),
            _str(d["from"], f"{where}.from"),
            _str(d["to"], f"{where}.to"),
            _str(d["profile"], f"{where}.profile", PROFILES),
            _float(d["start_s"], f"{where}.start_s"),
            _float(d["end_s"], f"{where}.end_s"),
            Facing.from_dict(d["facing"], f"{where}.facing"),
        )
    d = _keys(value, where, ("type", "station", "to", *times))
    return Rise(
        _str(d["station"], f"{where}.station"),
        _str(d["to"], f"{where}.to", HEIGHTS),
        _float(d["start_s"], f"{where}.start_s"),
        _float(d["end_s"], f"{where}.end_s"),
    )


@dataclass(frozen=True)
class Clip:
    library: str
    name: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"library": self.library, "name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class Activity:
    start_s: float
    end_s: float
    clip: Clip
    clip_offset_s: float
    gain_db: float
    #: Version 2, a voice's interval only: the vocal effort the turn is spoken at, and
    #: what the interval is, a turn, an acknowledgement or a laugh.
    effort: str | None = None
    event: str | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str, voiced: bool = False) -> Activity:
        """``voiced``: a voice's interval of a version 2 recipe, which says effort and event."""
        said = ("effort", "event") if voiced else ()
        d = _keys(value, where, ("start_s", "end_s", "clip", "clip_offset_s", "gain_db", *said))
        c = _keys(d["clip"], f"{where}.clip", ("library", "name", "sha256"))
        return cls(
            _float(d["start_s"], f"{where}.start_s"),
            _float(d["end_s"], f"{where}.end_s"),
            Clip(
                _str(c["library"], f"{where}.clip.library"),
                _str(c["name"], f"{where}.clip.name"),
                _str(c["sha256"], f"{where}.clip.sha256"),
            ),
            _float(d["clip_offset_s"], f"{where}.clip_offset_s"),
            _float(d["gain_db"], f"{where}.gain_db"),
            _str(d["effort"], f"{where}.effort", EFFORTS) if voiced else None,
            _str(d["event"], f"{where}.event", EVENTS) if voiced else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "clip": self.clip.to_dict(),
            "clip_offset_s": self.clip_offset_s,
            "gain_db": self.gain_db,
        }
        if self.effort is not None:
            out["effort"] = self.effort
        if self.event is not None:
            out["event"] = self.event
        return out


@dataclass(frozen=True)
class Directivity:
    model: str
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "enabled": self.enabled}


@dataclass(frozen=True)
class Sway:
    """One sinusoid of a small movement about where somebody is.

    ``amplitude * sin(2 pi t / period_s + phase)`` along a scene axis, in
    metres, or about the up axis, in degrees for ``yaw``. A sum of a few of
    them whose periods share no multiple never repeats and swells and fades,
    which is what a body at rest does. It costs nothing under the crossover:
    the low band is read from where the source is without it.
    """

    axis: str
    amplitude: float
    period_s: float
    phase_deg: float

    @classmethod
    def from_dict(cls, value: Any, where: str, axes: tuple[str, ...] = SWAY_AXES) -> Sway:
        if not isinstance(value, dict):
            raise _shape(f"{where} is not an object")
        axis = _str(value.get("axis"), f"{where}.axis", axes)
        size = "amplitude_deg" if axis == "yaw" else "amplitude_m"
        d = _keys(value, where, ("axis", size, "period_s", "phase_deg"))
        return cls(
            axis,
            _float(d[size], f"{where}.{size}"),
            _float(d["period_s"], f"{where}.period_s"),
            _float(d["phase_deg"], f"{where}.phase_deg"),
        )

    def to_dict(self) -> dict[str, Any]:
        size = "amplitude_deg" if self.axis == "yaw" else "amplitude_m"
        return {
            "axis": self.axis,
            size: self.amplitude,
            "period_s": self.period_s,
            "phase_deg": self.phase_deg,
        }


def _sways(value: Any, where: str, axes: tuple[str, ...] = SWAY_AXES) -> tuple[Sway, ...]:
    return tuple(
        Sway.from_dict(item, f"{where}[{index}]", axes)
        for index, item in enumerate(_list(value, where))
    )


@dataclass(frozen=True)
class Role:
    """What a voice is to the listener over an interval: the training label.

    ``conversation``: a member of the group the listener talks with.
    ``outside``: anybody else. ``group`` names the group the voice talks in,
    whichever the role; left out, the voice is in none.
    """

    start_s: float
    end_s: float
    role: str
    group: str | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Role:
        d = _keys(value, where, ("start_s", "end_s", "role"), ("group",))
        return cls(
            _float(d["start_s"], f"{where}.start_s"),
            _float(d["end_s"], f"{where}.end_s"),
            _str(d["role"], f"{where}.role", ROLES),
            _str(d["group"], f"{where}.group") if "group" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start_s": self.start_s, "end_s": self.end_s, "role": self.role}
        if self.group is not None:
            out["group"] = self.group
        return out


@dataclass(frozen=True)
class Attach:
    """A source that is carried: by the listener, or by another source.

    ``at`` ``"mouth"``: where its carrier's mouth is, or, on the listener,
    ``offset_m`` from the centre of the head in the head's own frame
    (front, left, up). ``"floor"``: under its carrier, 5 cm above the floor,
    at the footfall nearest to it, the footfalls standing every ``stride_m``
    along the way walked. Its yaw is its carrier's turned by
    ``yaw_offset_deg``. A carried source has no segments of its own.
    """

    to: str
    at: str
    yaw_offset_deg: float = 0.0
    offset_m: tuple[float, float, float] | None = None
    stride_m: float | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Attach:
        if not isinstance(value, dict):
            raise _shape(f"{where} is not an object")
        to = _str(value.get("to"), f"{where}.to")
        at = _str(value.get("at"), f"{where}.at", ATTACH_POINTS)
        extra: tuple[str, ...] = ()
        if at == "floor":
            extra = ("stride_m",)
        elif to == LISTENER:
            extra = ("offset_m",)
        d = _keys(value, where, ("to", "at", "yaw_offset_deg", *extra))
        offset = None
        if "offset_m" in d:
            x, y, z = _vector(d["offset_m"], f"{where}.offset_m", 3)
            offset = (x, y, z)
        return cls(
            to,
            at,
            _float(d["yaw_offset_deg"], f"{where}.yaw_offset_deg"),
            offset,
            _float(d["stride_m"], f"{where}.stride_m") if "stride_m" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"to": self.to, "at": self.at, "yaw_offset_deg": self.yaw_offset_deg}
        if self.offset_m is not None:
            out["offset_m"] = list(self.offset_m)
        if self.stride_m is not None:
            out["stride_m"] = self.stride_m
        return out


@dataclass(frozen=True)
class Opening:
    """The window or outer door a noise of outside comes in by, open or closed."""

    object: str
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {"object": self.object, "state": self.state}


@dataclass(frozen=True)
class Source:
    id: str
    kind: str
    directivity: Directivity
    gain_db: float
    turn_rate_deg_s: float
    segments: tuple[Segment, ...]
    activity: tuple[Activity, ...]
    subtype: str | None = None
    # Version 2.
    #: The level at 1 m, in dB SPL, of what the source emits at an interval gain of
    #: zero, the source's ``gain_db`` included: for a voice its active speech level.
    level_spl_1m_db: float | None = None
    sway: tuple[Sway, ...] = ()
    roles: tuple[Role, ...] = ()
    attach: Attach | None = None
    opening: Opening | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str, version: int = 1) -> Source:
        if version != 1:
            return cls._from_dict_v2(value, where)
        d = _keys(
            value,
            where,
            ("id", "kind", "directivity", "gain_db", "turn_rate_deg_s", "segments", "activity"),
            ("subtype",),
        )
        kind = _str(d["kind"], f"{where}.kind", SOURCE_KINDS)
        subtype = None
        if "subtype" in d:
            if kind != "noise":
                raise _shape(f"{where}.subtype is given for a {kind}; only a noise has one")
            subtype = _str(d["subtype"], f"{where}.subtype", NOISE_SUBTYPES)
        pattern = _keys(d["directivity"], f"{where}.directivity", ("model", "enabled"))
        return cls(
            id=_str(d["id"], f"{where}.id"),
            kind=kind,
            directivity=Directivity(
                _str(pattern["model"], f"{where}.directivity.model"),
                _bool(pattern["enabled"], f"{where}.directivity.enabled"),
            ),
            gain_db=_float(d["gain_db"], f"{where}.gain_db"),
            turn_rate_deg_s=_float(d["turn_rate_deg_s"], f"{where}.turn_rate_deg_s"),
            segments=tuple(
                _segment(item, f"{where}.segments[{index}]")
                for index, item in enumerate(_list(d["segments"], f"{where}.segments"))
            ),
            activity=tuple(
                Activity.from_dict(item, f"{where}.activity[{index}]")
                for index, item in enumerate(_list(d["activity"], f"{where}.activity"))
            ),
            subtype=subtype,
        )

    @classmethod
    def _from_dict_v2(cls, value: Any, where: str) -> Source:
        d = _keys(
            value,
            where,
            (
                "id",
                "kind",
                "directivity",
                "gain_db",
                "level_spl_1m_db",
                "turn_rate_deg_s",
                "segments",
                "activity",
                "sway",
            ),
            ("subtype", "roles", "attach", "opening"),
        )
        kind = _str(d["kind"], f"{where}.kind", SOURCE_KINDS_V2)
        subtype = None
        if kind in SUBTYPES_V2:
            if "subtype" not in d:
                raise _shape(f"{where} is a {kind} and lacks 'subtype'")
            subtype = _str(d["subtype"], f"{where}.subtype", SUBTYPES_V2[kind])
        elif "subtype" in d:
            raise _shape(f"{where}.subtype is given for a {kind}, which has none")
        if ("roles" in d) != (kind == "voice"):
            raise _shape(f"{where}: a voice says its roles, and no other kind does")
        pattern = _keys(d["directivity"], f"{where}.directivity", ("model", "enabled"))
        opening = None
        if "opening" in d:
            o = _keys(d["opening"], f"{where}.opening", ("object", "state"))
            opening = Opening(
                _str(o["object"], f"{where}.opening.object"),
                _str(o["state"], f"{where}.opening.state", OPENING_STATES),
            )
        voiced = kind in VOICE_KINDS_V2
        return cls(
            id=_str(d["id"], f"{where}.id"),
            kind=kind,
            directivity=Directivity(
                _str(pattern["model"], f"{where}.directivity.model"),
                _bool(pattern["enabled"], f"{where}.directivity.enabled"),
            ),
            gain_db=_float(d["gain_db"], f"{where}.gain_db"),
            turn_rate_deg_s=_float(d["turn_rate_deg_s"], f"{where}.turn_rate_deg_s"),
            segments=tuple(
                _segment(item, f"{where}.segments[{index}]", 2)
                for index, item in enumerate(_list(d["segments"], f"{where}.segments"))
            ),
            activity=tuple(
                Activity.from_dict(item, f"{where}.activity[{index}]", voiced)
                for index, item in enumerate(_list(d["activity"], f"{where}.activity"))
            ),
            subtype=subtype,
            level_spl_1m_db=_float(d["level_spl_1m_db"], f"{where}.level_spl_1m_db"),
            sway=_sways(d["sway"], f"{where}.sway"),
            roles=tuple(
                Role.from_dict(item, f"{where}.roles[{index}]")
                for index, item in enumerate(_list(d.get("roles", []), f"{where}.roles"))
            ),
            attach=Attach.from_dict(d["attach"], f"{where}.attach") if "attach" in d else None,
            opening=opening,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "directivity": self.directivity.to_dict(),
            "gain_db": self.gain_db,
            "turn_rate_deg_s": self.turn_rate_deg_s,
            "segments": [segment.to_dict() for segment in self.segments],
            "activity": [interval.to_dict() for interval in self.activity],
        }
        if self.subtype is not None:
            out["subtype"] = self.subtype
        if self.level_spl_1m_db is not None:
            # Version 2: every key of it, said or empty.
            out["level_spl_1m_db"] = self.level_spl_1m_db
            out["sway"] = [component.to_dict() for component in self.sway]
            if self.kind == "voice":
                out["roles"] = [role.to_dict() for role in self.roles]
            if self.attach is not None:
                out["attach"] = self.attach.to_dict()
            if self.opening is not None:
                out["opening"] = self.opening.to_dict()
        return out


@dataclass(frozen=True)
class Keyframe:
    t_s: float
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    station: str | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Keyframe:
        d = _keys(
            value, where, ("t_s", "position", "yaw_deg", "pitch_deg", "roll_deg"), ("station",)
        )
        x, y, z = _vector(d["position"], f"{where}.position", 3)
        return cls(
            _float(d["t_s"], f"{where}.t_s"),
            (x, y, z),
            _float(d["yaw_deg"], f"{where}.yaw_deg"),
            _float(d["pitch_deg"], f"{where}.pitch_deg"),
            _float(d["roll_deg"], f"{where}.roll_deg"),
            _str(d["station"], f"{where}.station") if "station" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "t_s": self.t_s,
            "position": list(self.position),
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
        }
        if self.station is not None:
            out["station"] = self.station
        return out


@dataclass(frozen=True)
class Membership:
    """The group the listener talks in over an interval; none where ``group`` is left out."""

    start_s: float
    end_s: float
    group: str | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Membership:
        d = _keys(value, where, ("start_s", "end_s"), ("group",))
        return cls(
            _float(d["start_s"], f"{where}.start_s"),
            _float(d["end_s"], f"{where}.end_s"),
            _str(d["group"], f"{where}.group") if "group" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start_s": self.start_s, "end_s": self.end_s}
        if self.group is not None:
            out["group"] = self.group
        return out


@dataclass(frozen=True)
class Gaze:
    """What the listener's head was turned to over an interval, as the generator meant it.

    A record for the audit, not a rule of the movement: the head is the
    keyframes'. ``target`` is a source's id where the mode has one.
    """

    start_s: float
    end_s: float
    mode: str
    target: str | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> Gaze:
        d = _keys(value, where, ("start_s", "end_s", "mode"), ("target",))
        return cls(
            _float(d["start_s"], f"{where}.start_s"),
            _float(d["end_s"], f"{where}.end_s"),
            _str(d["mode"], f"{where}.mode", GAZE_MODES),
            _str(d["target"], f"{where}.target") if "target" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start_s": self.start_s, "end_s": self.end_s, "mode": self.mode}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(frozen=True)
class Listener:
    keyframes: tuple[Keyframe, ...]
    interpolation: str = "linear"
    # Version 2; ``None`` in a version 1 recipe, which has none of the three.
    sway: tuple[Sway, ...] | None = None
    conversation: tuple[Membership, ...] = ()
    gaze: tuple[Gaze, ...] = ()

    @classmethod
    def from_dict(cls, value: Any, version: int = 1) -> Listener:
        more = () if version == 1 else ("sway", "conversation", "gaze")
        d = _keys(value, "listener", ("interpolation", "keyframes", *more))
        frames = tuple(
            Keyframe.from_dict(item, f"listener.keyframes[{index}]")
            for index, item in enumerate(_list(d["keyframes"], "listener.keyframes"))
        )
        interpolation = _str(d["interpolation"], "listener.interpolation", ("linear",))
        if version == 1:
            return cls(frames, interpolation)
        return cls(
            frames,
            interpolation,
            _sways(d["sway"], "listener.sway", ("x", "y", "z")),
            tuple(
                Membership.from_dict(item, f"listener.conversation[{index}]")
                for index, item in enumerate(_list(d["conversation"], "listener.conversation"))
            ),
            tuple(
                Gaze.from_dict(item, f"listener.gaze[{index}]")
                for index, item in enumerate(_list(d["gaze"], "listener.gaze"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "interpolation": self.interpolation,
            "keyframes": [keyframe.to_dict() for keyframe in self.keyframes],
        }
        if self.sway is not None:
            out["sway"] = [component.to_dict() for component in self.sway]
            out["conversation"] = [interval.to_dict() for interval in self.conversation]
            out["gaze"] = [interval.to_dict() for interval in self.gaze]
        return out


@dataclass(frozen=True)
class Scene:
    """Version 2: how calm the scene is, and the floor its speech is held above.

    ``calmness`` runs from 0, as lively as a home gets, to 1, nearly silent.
    ``snr_floor_db`` is the least speech to noise ratio, in free field at
    the listener, any turn of the listener's conversation may have: rule 16.
    """

    calmness: float
    snr_floor_db: float

    @classmethod
    def from_dict(cls, value: Any) -> Scene:
        d = _keys(value, "scene", ("calmness", "snr_floor_db"))
        return cls(_float(d["calmness"], "scene.calmness"), _float(d["snr_floor_db"], "scene.snr"))

    def to_dict(self) -> dict[str, Any]:
        return {"calmness": self.calmness, "snr_floor_db": self.snr_floor_db}


@dataclass(frozen=True)
class GeneratorRecord:
    name: str
    version: str
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> GeneratorRecord:
        d = _keys(value, "generator", ("name", "version", "parameters"))
        if not isinstance(d["parameters"], dict):
            raise _shape("generator.parameters is not an object")
        return cls(
            _str(d["name"], "generator.name"),
            _str(d["version"], "generator.version"),
            d["parameters"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "parameters": self.parameters}


_TOP = (
    "schema",
    "schema_version",
    "dwelling",
    "assets",
    "seed",
    "duration_s",
    "output",
    "atmosphere",
    "heights",
    "stations",
    "rails",
    "sources",
    "listener",
    "generator",
)


@dataclass(frozen=True)
class Recipe:
    """One moving scene. Everything a trace or a page needs, and no audio."""

    dwelling: Dwelling
    assets: Assets
    seed: int
    duration_s: float
    stations: tuple[Station, ...]
    rails: tuple[Rail, ...]
    sources: tuple[Source, ...]
    listener: Listener
    generator: GeneratorRecord | None = None
    output: Output = Output()
    atmosphere: Atmosphere = Atmosphere()
    heights: Heights = Heights()
    #: The version the recipe is written in. A recipe made without saying is of the
    #: first, so that what made version 1 recipes makes them still.
    schema_version: int = 1
    #: Version 2 only.
    scene: Scene | None = None

    @classmethod
    def from_dict(cls, value: Any) -> Recipe:
        if not isinstance(value, dict):
            raise _shape("the recipe is not an object")
        version = value.get("schema_version")
        if type(version) is not int or version not in SCHEMA_VERSIONS:
            read = " or ".join(str(v) for v in SCHEMA_VERSIONS)
            raise _shape(f"schema_version is {version!r}, expected {read}")
        d = _keys(value, "the recipe", _TOP if version == 1 else (*_TOP, "scene"))
        if d["schema"] != SCHEMA:
            raise _shape(f"schema is {d['schema']!r}, expected {SCHEMA!r}")
        seed = _int(d["seed"], "seed")
        if not 0 <= seed < 2**53:
            raise _shape(f"seed {seed} is outside 0 <= seed < 2^53")
        return cls(
            dwelling=Dwelling.from_dict(d["dwelling"]),
            assets=Assets.from_dict(d["assets"]),
            seed=seed,
            duration_s=_float(d["duration_s"], "duration_s"),
            stations=tuple(
                Station.from_dict(item, f"stations[{index}]", version)
                for index, item in enumerate(_list(d["stations"], "stations"))
            ),
            rails=tuple(
                Rail.from_dict(item, f"rails[{index}]")
                for index, item in enumerate(_list(d["rails"], "rails"))
            ),
            sources=tuple(
                Source.from_dict(item, f"sources[{index}]", version)
                for index, item in enumerate(_list(d["sources"], "sources"))
            ),
            listener=Listener.from_dict(d["listener"], version),
            generator=None if d["generator"] is None else GeneratorRecord.from_dict(d["generator"]),
            output=Output.from_dict(d["output"]),
            atmosphere=Atmosphere.from_dict(d["atmosphere"]),
            heights=Heights.from_dict(d["heights"]),
            schema_version=version,
            scene=None if version == 1 else Scene.from_dict(d["scene"]),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version == 1:
            return self._tree()
        if self.scene is None or self.listener.sway is None:
            raise _shape("a version 2 recipe says its scene and its listener's sway")
        return {**self._tree(), "scene": self.scene.to_dict()}

    def _tree(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": self.schema_version,
            "dwelling": self.dwelling.to_dict(),
            "assets": self.assets.to_dict(),
            "seed": self.seed,
            "duration_s": self.duration_s,
            "output": self.output.to_dict(),
            "atmosphere": self.atmosphere.to_dict(),
            "heights": self.heights.to_dict(),
            "stations": [station.to_dict() for station in self.stations],
            "rails": [rail.to_dict() for rail in self.rails],
            "sources": [source.to_dict() for source in self.sources],
            "listener": self.listener.to_dict(),
            "generator": None if self.generator is None else self.generator.to_dict(),
        }

    # Lookups by id. The first of a repeated id wins; rule 3 refuses the repeat.
    @cached_property
    def _stations(self) -> dict[str, Station]:
        return {station.id: station for station in reversed(self.stations)}

    @cached_property
    def _rails(self) -> dict[str, Rail]:
        return {rail.id: rail for rail in reversed(self.rails)}

    def station(self, ident: str) -> Station:
        return self._stations[ident]

    def rail(self, ident: str) -> Rail:
        return self._rails[ident]

    def source(self, ident: str) -> Source:
        for source in self.sources:
            if source.id == ident:
                return source
        raise KeyError(f"no source {ident!r}; the recipe has {[s.id for s in self.sources]}")


# --------------------------------------------------------------------------
# quantisation, the canonical form, the identity
# --------------------------------------------------------------------------

#: Decimals kept per unit, the unit being the key's suffix. ``_deg_s`` and
#: ``_m_s`` are read before ``_s``.
_DECIMALS = (
    ("_deg_s", 2),
    ("_m_s", 3),
    ("_deg", 2),
    ("_db", 2),
    ("_m", 3),
    ("_s", 3),
    ("_c", 3),
    ("_percent", 3),
    ("_kpa", 3),
)
_DECIMALS_BY_KEY = {"position": 3, "points": 3}


def decimals_of(key: str) -> int | None:
    """How many decimals a number under this key keeps, or ``None`` for no step."""
    if key in _DECIMALS_BY_KEY:
        return _DECIMALS_BY_KEY[key]
    for suffix, decimals in _DECIMALS:
        if key.endswith(suffix):
            return decimals
    return None


def quantise_tree(value: Any, key: str = "") -> Any:
    """A recipe's JSON tree with every float rounded to its key's step.

    The ``generator`` block is left as it is: its leaves are what the draws
    were taken from, not quantities of the scene.
    """
    if isinstance(value, dict):
        return {
            name: item if name == "generator" else quantise_tree(item, name)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [quantise_tree(item, key) for item in value]
    if type(value) is float:
        decimals = decimals_of(key)
        # Adding zero turns a negative zero into the zero it is written as.
        return value if decimals is None else round(value, decimals) + 0.0
    return value


def quantise(recipe: Recipe) -> Recipe:
    """The recipe with every number on its step."""
    return Recipe.from_dict(quantise_tree(recipe.to_dict()))


def canonical_bytes(recipe: Recipe) -> bytes:
    """The canonical form: sorted compact JSON, UTF-8, one line feed."""
    text = json.dumps(
        recipe.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return text.encode("utf-8") + b"\n"


def recipe_sha256(recipe: Recipe) -> str:
    """The recipe's identity."""
    return hashlib.sha256(canonical_bytes(recipe)).hexdigest()


def parse_recipe(text: str | bytes) -> Recipe:
    """A recipe from its JSON, indented or canonical. Refuses anything rule 1 refuses."""
    try:
        tree = json.loads(text)
    except json.JSONDecodeError as error:
        raise _shape(f"not JSON: {error}") from error
    return Recipe.from_dict(tree)


def load_recipe(path: Path) -> Recipe:
    return parse_recipe(Path(path).read_bytes())


def save_recipe(recipe: Recipe, path: Path) -> str:
    """Write the canonical form and return the identity."""
    payload = canonical_bytes(recipe)
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
