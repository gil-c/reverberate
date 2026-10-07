"""A recipe drawn from ranges and a seed: who walks where, who speaks when.

``generate(layout, parameters, seed)`` is deterministic: the same layout,
parameters, seed, assets and generator version give the same canonical bytes
in two processes. Every draw comes from a stream of its own,
``PCG64(first eight bytes of sha256("<seed>:<label>"))``, the label naming what
is drawn; nothing reads the global generator, a set's order or a hash.

**The order things are placed in is the order of who yields to whom.**

1. The noises, which do not move, on standing spots that leave the rail graph
   and the walkable floor in one piece.
2. The listener, free on the floor: rests at seats and at points of the floor,
   and walks between them by the shortest way that keeps clear of the noises.
3. The near voices, then the far ones, each on stations and rails. A voice
   takes a station only for as long as nobody placed before it comes within
   the clearance, and a rail only while it is clear; when no move is clear it
   waits, leaves earlier, or is drawn again.
4. Who speaks: the near voices and the listener take turns in one
   conversation; a far voice talks to somebody who is not in the scene; a
   noise is steady or comes and goes.
5. The listener's head, which turns towards the way it walks and, at rest,
   towards the near voice that starts to speak, give or take.

The result is validated against every rule of the format; a draw that fails is
drawn again under the next attempt's labels, and the attempt that passed is
recorded with the parameters.

**Clips.** A voice or a noise names clips of a :class:`ClipLibrary`, read
from a manifest of :mod:`reverberate.scenes.clips` (``--clips``). Where the
library says where a voice's utterances are, **a talk spurt is whole
utterances**: as many in a row as come nearest the length drawn, so no word
is cut, and the spurt lasts what they last. A noise reads on through its
clip and, where the clip loops, round it again; a noise keeps one looping
clip for the whole scene (a tap does not become a shower), and walks through
clips that do not loop (a programme, a playlist). The kinds of the noises
are dealt from the kinds the library holds, without one coming twice before
all have come once. Without a library, :func:`placeholder_clips` stands in,
**only when asked for by name**: its names say ``placeholder`` and its
digests are of those names, not of audio.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point

from reverberate.scenes import kinematics
from reverberate.scenes.layout import Layout, Navigator
from reverberate.scenes.recipe import (
    NOISE_SUBTYPES,
    Activity,
    Assets,
    Clip,
    Directivity,
    Dwell,
    Facing,
    GeneratorRecord,
    Keyframe,
    Listener,
    Recipe,
    Rise,
    Segment,
    Source,
    Station,
    Travel,
    quantise,
)
from reverberate.scenes.validate import HEAD_CLEARANCE_M, SOURCE_CLEARANCE_M, validate

__all__ = [
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "ClipEntry",
    "ClipLibrary",
    "GenerationError",
    "Parameters",
    "generate",
    "load_clip_library",
    "placeholder_assets",
    "placeholder_clips",
    "stream",
]

GENERATOR_NAME = "reverberate.scenes"
#: A change of the output for one seed changes this.
GENERATOR_VERSION = "0.2.0"

#: What the generator keeps beyond the format's clearances, so that a position
#: rounded to its step, or one between two samples, still has them.
MARGIN_M = 0.08

#: No station is taken for less than this.
MIN_STAY_S = 3.0

STEP_S = kinematics.YAW_STEP_S

Range = tuple[float, float]


class GenerationError(RuntimeError):
    """No valid recipe came out of the attempts allowed."""


class _Stuck(Exception):
    """A source found no clear move; it is drawn again."""


def stream(seed: int, label: str) -> np.random.Generator:
    """The stream of draws named ``label`` under ``seed``."""
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "big")))


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameters:
    """The ranges a scene is drawn from. A two element value is ``(min, max)``.

    The defaults are the first scene's: twenty minutes, fourteen sources, and
    **a domestic scene**: people mostly stay where they are, sitting or
    standing, talk, and move a few times. A far voice keeps a station for
    ``dwell_s``, minutes; the listener rests for ``listener_rest_s``, minutes
    too, and the near voices, who follow the listener, move when the listener
    does. Some speech still falls on a walk, a small share of it. What the
    ranges cost is the number of source positions the band under the crossover
    is solved from, one wave solve each: every metre of rail a source is heard
    on is 12.5 of them (``describe`` prints the count). With ``dwell_s`` of 20
    to 180 s and rests of 15 to 120 s, near voices walked 160 to 200 m each
    and the first scene on hssd_0076 asked for 1646 audible positions; with
    these defaults, 375.
    """

    duration_s: float = 1200.0
    #: How many of each. Three people talk with the listener, six elsewhere in
    #: the dwelling, five things make noise.
    near_voice_count: tuple[int, int] = (3, 3)
    far_voice_count: tuple[int, int] = (6, 6)
    noise_count: tuple[int, int] = (5, 5)
    #: What near and far mean: the distance from the station a voice takes to
    #: where the listener rests, measured on the floor plan.
    near_distance_m: Range = (0.6, 2.0)
    far_distance_m: Range = (2.5, 15.0)
    #: A voice's walking speed, the time it stays at a station, its level.
    speed_m_s: Range = (0.5, 1.1)
    dwell_s: Range = (400.0, 1500.0)
    gain_db: Range = (-6.0, 0.0)
    #: A noise's level about the one its clip is stored at, which is its source's at 1 m
    #: (``clip-library.md``): a television, a shower, a hood as loud as they are, 6 dB
    #: either way. The first scene drew -18 to -6 and its five noises together stood 5 to
    #: 9 dB under a near voice in the median and 20 dB under at their quietest: a scene of
    #: voices, with nothing to separate speech from. Centred on the stored level, the
    #: noises stand about where the far voices do, and a near voice 0 to 5 dB over the rest
    #: in the median (``docs/open-questions/first-scene-defects.md``).
    noise_gain_db: Range = (-6.0, 6.0)
    #: The share of a voice's stations that are seats.
    seated_share: Range = (0.5, 0.9)
    #: A talk spurt and the pause after it.
    speech_s: Range = (1.5, 12.0)
    pause_s: Range = (0.5, 6.0)
    #: How often, in the conversation, the next voice starts before the last ends.
    overlap_share: float = 0.1
    turn_rate_deg_s: Range = (90.0, 240.0)
    #: The time to sit down or to stand up.
    rise_s: Range = (1.2, 2.5)
    #: A noise that comes and goes: how long on, how long off; and the share
    #: of noises that never stop.
    noise_on_s: Range = (30.0, 300.0)
    noise_off_s: Range = (10.0, 120.0)
    noise_steady_share: float = 0.4
    listener_speed_m_s: Range = (0.4, 1.0)
    listener_rest_s: Range = (240.0, 720.0)
    listener_turn_rate_deg_s: Range = (40.0, 160.0)
    listener_seated_share: Range = (0.5, 0.8)
    listener_pitch_deg: Range = (-20.0, 10.0)
    #: Standard deviation of where the head stops short of, or past, a talker.
    listener_gaze_jitter_deg: float = 12.0
    #: The share of a near voice's talk spurts the head turns to.
    listener_attend_share: float = 0.8
    rail_pitch_m: float = 0.08
    rail_max_length_m: float = 12.0

    def record(self) -> dict[str, Any]:
        """The tree the recipe's ``generator.parameters`` holds."""

        def pair(value: tuple[float, float]) -> list[float]:
            return [value[0], value[1]]

        return {
            "duration_s": self.duration_s,
            "sources": {
                "near_voice": {
                    "count": pair(self.near_voice_count),
                    "distance_m": pair(self.near_distance_m),
                },
                "far_voice": {
                    "count": pair(self.far_voice_count),
                    "distance_m": pair(self.far_distance_m),
                },
                "noise": {
                    "count": pair(self.noise_count),
                    "gain_db": pair(self.noise_gain_db),
                    "on_s": pair(self.noise_on_s),
                    "off_s": pair(self.noise_off_s),
                    "steady_share": self.noise_steady_share,
                },
                "speed_m_s": pair(self.speed_m_s),
                "dwell_s": pair(self.dwell_s),
                "gain_db": pair(self.gain_db),
                "seated_share": pair(self.seated_share),
                "speech_s": pair(self.speech_s),
                "pause_s": pair(self.pause_s),
                "overlap_share": self.overlap_share,
                "turn_rate_deg_s": pair(self.turn_rate_deg_s),
                "rise_s": pair(self.rise_s),
            },
            "listener": {
                "speed_m_s": pair(self.listener_speed_m_s),
                "rest_s": pair(self.listener_rest_s),
                "turn_rate_deg_s": pair(self.listener_turn_rate_deg_s),
                "seated_share": pair(self.listener_seated_share),
                "pitch_deg": pair(self.listener_pitch_deg),
                "gaze_jitter_deg": self.listener_gaze_jitter_deg,
                "attend_share": self.listener_attend_share,
            },
            "rails": {"pitch_m": self.rail_pitch_m, "max_length_m": self.rail_max_length_m},
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Parameters:
        """The inverse of :meth:`record`; keys it does not write are ignored."""
        sources, listener, rails = record["sources"], record["listener"], record["rails"]

        def pair(value: Any) -> tuple[float, float]:
            return (float(value[0]), float(value[1]))

        def count(value: Any) -> tuple[int, int]:
            return (int(value[0]), int(value[1]))

        return cls(
            duration_s=float(record["duration_s"]),
            near_voice_count=count(sources["near_voice"]["count"]),
            far_voice_count=count(sources["far_voice"]["count"]),
            noise_count=count(sources["noise"]["count"]),
            near_distance_m=pair(sources["near_voice"]["distance_m"]),
            far_distance_m=pair(sources["far_voice"]["distance_m"]),
            speed_m_s=pair(sources["speed_m_s"]),
            dwell_s=pair(sources["dwell_s"]),
            gain_db=pair(sources["gain_db"]),
            noise_gain_db=pair(sources["noise"]["gain_db"]),
            seated_share=pair(sources["seated_share"]),
            speech_s=pair(sources["speech_s"]),
            pause_s=pair(sources["pause_s"]),
            overlap_share=float(sources["overlap_share"]),
            turn_rate_deg_s=pair(sources["turn_rate_deg_s"]),
            rise_s=pair(sources["rise_s"]),
            noise_on_s=pair(sources["noise"]["on_s"]),
            noise_off_s=pair(sources["noise"]["off_s"]),
            noise_steady_share=float(sources["noise"]["steady_share"]),
            listener_speed_m_s=pair(listener["speed_m_s"]),
            listener_rest_s=pair(listener["rest_s"]),
            listener_turn_rate_deg_s=pair(listener["turn_rate_deg_s"]),
            listener_seated_share=pair(listener["seated_share"]),
            listener_pitch_deg=pair(listener["pitch_deg"]),
            listener_gaze_jitter_deg=float(listener["gaze_jitter_deg"]),
            listener_attend_share=float(listener["attend_share"]),
            rail_pitch_m=float(rails["pitch_m"]),
            rail_max_length_m=float(rails["max_length_m"]),
        )


# --------------------------------------------------------------------------
# clips and assets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipEntry:
    """One clip a recipe may name: its library, name, digest and length."""

    library: str
    name: str
    sha256: str
    duration_s: float
    #: ``"voice"`` or a noise's subtype.
    kind: str = "voice"
    #: Who speaks; a voice of the scene keeps to one speaker.
    speaker: str = ""
    #: Where a voice's clip may be cut: ``(start_s, end_s)`` of each
    #: utterance, in order, none overlapping. Empty: nothing is known.
    utterances: tuple[tuple[float, float], ...] = ()
    #: Whether the clip's end runs into its start without a seam.
    loop: bool = False
    #: What the second generator reads (:mod:`reverberate.scenes.social`). The level at
    #: 1 m the file holds, the manifest's ``level.spl_1m_db``; ``None``: not said. The
    #: vocal effort a voice's clip was spoken at, and what it is: a ``turn``, a
    #: ``backchannel`` or ``laughter``. A library that says neither holds read speech
    #: at a normal effort, which is what ``clarify_v1`` is.
    spl_1m_db: float | None = None
    effort: str = "normal"
    event: str = "turn"


@dataclass(frozen=True)
class ClipLibrary:
    entries: tuple[ClipEntry, ...]
    placeholder: bool = False

    def speakers(self) -> list[list[ClipEntry]]:
        """The voices' clips, one list per speaker, in the speakers' order."""
        groups: dict[str, list[ClipEntry]] = {}
        for entry in self.entries:
            if entry.kind == "voice":
                groups.setdefault(entry.speaker, []).append(entry)
        return [sorted(groups[name], key=lambda e: e.name) for name in sorted(groups)]

    def noises(self, subtype: str) -> list[ClipEntry]:
        return sorted((e for e in self.entries if e.kind == subtype), key=lambda e: e.name)


def placeholder_clips(duration_s: float, speakers: int = 32) -> ClipLibrary:
    """Clip references that name no audio: one per speaker and per kind of noise.

    Each is as long as the scene, so no interval outruns its clip. The digest
    is that of the placeholder's name. A trace must refuse them.
    """

    def entry(name: str, kind: str, speaker: str = "") -> ClipEntry:
        digest = hashlib.sha256(f"placeholder:{name}".encode()).hexdigest()
        return ClipEntry("placeholder", name, digest, float(duration_s), kind, speaker)

    voices = [entry(f"voice_{n:02d}", "voice", f"{n:02d}") for n in range(1, speakers + 1)]
    noises = [entry(f"noise_{subtype}", subtype) for subtype in NOISE_SUBTYPES]
    return ClipLibrary(tuple(voices + noises), placeholder=True)


def load_clip_library(path: Path) -> ClipLibrary:
    """A library from a manifest (``docs/formats/clip-library.md``).

    Also read: a bare JSON list of
    ``{library, name, sha256, duration_s, kind, speaker}``, each optionally
    with ``utterances`` and ``loop``, where ``kind`` is ``"voice"`` or a
    noise's subtype.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if data.get("schema") != "reverberate.clip-library" or data.get("schema_version") != 1:
            raise ValueError(f"{path} is not a clip library this generator reads")
        items = [
            {
                **item,
                "library": data["library"],
                "kind": "voice" if item["kind"] == "voice" else item["subtype"],
            }
            for item in data["clips"]
        ]
    else:
        items = data
    return ClipLibrary(
        tuple(
            ClipEntry(
                str(item["library"]),
                str(item["name"]),
                str(item["sha256"]),
                float(item["duration_s"]),
                str(item.get("kind", "voice")),
                str(item.get("speaker", "")),
                tuple((float(a), float(b)) for a, b in item.get("utterances", ())),
                bool(item.get("loop", False)),
                _stored_level(item),
                str(item.get("effort", "normal")),
                str(item.get("event", "turn")),
            )
            for item in items
        )
    )


def _stored_level(item: dict[str, Any]) -> float | None:
    """The level at 1 m a clip's entry states, in dB SPL, where it states one."""
    level = item.get("level")
    stated = level.get("spl_1m_db") if isinstance(level, dict) else item.get("spl_1m_db")
    return None if stated is None else float(stated)


class _Voice:
    """A speaker's clips, read on from one talk spurt to the next."""

    def __init__(self, shelf: list[ClipEntry]) -> None:
        self.shelf = shelf
        self.which = 0
        self.offset = 0.0
        self.next = 0

    def take(self, start: float, wanted: float, limit: float) -> Activity | None:
        """The spurt that starts at ``start``, about ``wanted`` long, over by ``limit``.

        ``None`` when the scene ends before the next utterance would.
        """
        start = round(start, 3)
        for _ in range(len(self.shelf)):
            spoken = self.shelf[self.which].utterances
            if not spoken or self.next < len(spoken):
                break
            self.which, self.offset, self.next = (self.which + 1) % len(self.shelf), 0.0, 0
        entry = self.shelf[self.which]
        clip = Clip(entry.library, entry.name, entry.sha256)
        if not entry.utterances:
            # Nothing is known of the clip: the spurt is the length drawn.
            end = round(min(start + wanted, limit), 3)
            span = end - start
            if span <= 0:
                return None
            for _ in range(len(self.shelf)):
                if self.offset + span <= self.shelf[self.which].duration_s:
                    break
                self.which, self.offset = (self.which + 1) % len(self.shelf), 0.0
            entry = self.shelf[self.which]
            clip = Clip(entry.library, entry.name, entry.sha256)
            if span > entry.duration_s:
                # No clip is that long: the interval ends with the clip.
                end = round(start + math.floor(entry.duration_s * 1000) / 1000, 3)
                span = end - start
            activity = Activity(start, end, clip, round(self.offset, 3), 0.0)
            self.offset = round(self.offset + span, 3)
            return activity
        spoken = entry.utterances
        first, last = spoken[self.next][0], self.next
        while (
            last + 1 < len(spoken)
            and abs(spoken[last + 1][1] - first - wanted) < abs(spoken[last][1] - first - wanted)
            and start + spoken[last + 1][1] - first <= limit
        ):
            last += 1
        span = round(spoken[last][1] - first, 3)
        if round(start + span, 3) > limit:
            return None
        self.next = last + 1
        return Activity(start, round(start + span, 3), clip, round(first, 3), 0.0)


def placeholder_assets() -> Assets:
    """Asset keys that match nothing, for a recipe no trace will read yet."""
    return Assets(
        export_sha256="0" * 64,
        voxel_low_key="0" * 32,
        mirror_scene_key="0" * 32,
        calibration_key="0" * 16,
        directivity={"voice_v1": "0" * 64},
        rooms_rule="adr-0010",
    )


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


def generate(
    layout: Layout,
    parameters: Parameters | None = None,
    seed: int = 0,
    *,
    assets: Assets | None = None,
    clips: ClipLibrary | None = None,
    allow_placeholder_clips: bool = False,
    allow_placeholder_assets: bool = False,
    attempts: int = 8,
) -> Recipe:
    """A valid recipe on this layout, drawn from ``parameters`` under ``seed``.

    ``clips`` is the library voices and noises take their clips from; without
    one, ``allow_placeholder_clips`` must be set and the recipe then names
    :func:`placeholder_clips`. The same holds for ``assets``.
    """
    parameters = parameters or Parameters()
    if clips is None:
        if not allow_placeholder_clips:
            raise ValueError("no clip library was given; pass one or allow_placeholder_clips=True")
        clips = placeholder_clips(parameters.duration_s)
    if assets is None:
        if not allow_placeholder_assets:
            raise ValueError("no assets were given; pass them or allow_placeholder_assets=True")
        assets = placeholder_assets()
    if not 0 <= seed < 2**53:
        raise ValueError(f"seed {seed} is outside 0 <= seed < 2^53")
    reason = "no attempt was made"
    for attempt in range(attempts):
        try:
            recipe = _Scene(layout, parameters, seed, assets, clips, attempt).build()
        except _Stuck as stuck:
            reason = str(stuck)
            continue
        found = validate(recipe, layout.floor)
        if not found:
            return recipe
        reason = str(found[0])
    raise GenerationError(f"no valid recipe in {attempts} attempts; the last failed on: {reason}")


@dataclass
class _Rest:
    start_s: float
    end_s: float
    xz: tuple[float, float]
    seat: str | None


class _Scene:
    """One attempt at a recipe."""

    def __init__(
        self,
        layout: Layout,
        parameters: Parameters,
        seed: int,
        assets: Assets,
        clips: ClipLibrary,
        attempt: int,
    ) -> None:
        self.layout = layout
        self.p = parameters
        self.seed = seed
        self.assets = assets
        self.clips = clips
        self.attempt = attempt
        self.duration = round(float(parameters.duration_s), 3)
        self.heights = layout.heights
        self.floor_y = layout.dwelling.floor_y_m
        #: A recipe with every station and rail and nobody in it: what the
        #: kinematics are asked while the scene is being planned.
        self.shell = Recipe(
            dwelling=layout.dwelling,
            assets=assets,
            seed=seed,
            duration_s=self.duration,
            stations=layout.stations,
            rails=layout.rails,
            sources=(),
            listener=Listener(()),
            heights=layout.heights,
        )
        self.times = kinematics.sample_times(self.shell)
        self.stations = {station.id: station for station in layout.stations}
        self.order = [station.id for station in layout.stations]
        self.index = {ident: n for n, ident in enumerate(self.order)}
        self.station_pos = np.array([station.position for station in layout.stations])
        self.station_blocked = np.zeros((len(self.order), self.times.size), dtype=bool)
        self.rail_length = {rail.id: kinematics.rail_length(rail) for rail in layout.rails}
        self.rail_index = {rail.id: n for n, rail in enumerate(layout.rails)}
        self.rail_blocked = np.zeros((len(layout.rails), self.times.size), dtype=bool)
        standing_y = self.floor_y + self.heights.standing_m
        self.probes = []
        for rail in layout.rails:
            points = np.asarray(rail.points, dtype=float)
            length = self.rail_length[rail.id]
            arcs = np.linspace(0.0, length, max(2, int(math.ceil(length / 0.25)) + 1))
            xz = kinematics._along(points, arcs)
            self.probes.append(np.column_stack([xz[:, 0], np.full(len(xz), standing_y), xz[:, 1]]))
        self.links: dict[str, list[tuple[str, str]]] = {ident: [] for ident in self.order}
        for rail in layout.rails:
            self.links[rail.a].append((rail.id, rail.b))
            self.links[rail.b].append((rail.id, rail.a))
        self.committed: list[tuple[np.ndarray, float]] = []
        self.rests: list[_Rest] = []

    # -- draws --------------------------------------------------------------

    def rng(self, label: str) -> np.random.Generator:
        return stream(self.seed, label if self.attempt == 0 else f"{label}#{self.attempt}")

    @staticmethod
    def draw(rng: np.random.Generator, span: tuple[float, float]) -> float:
        return float(rng.uniform(span[0], span[1])) if span[1] > span[0] else float(span[0])

    # -- occupancy ----------------------------------------------------------

    def tick(self, t: float, up: bool = False) -> int:
        """The grid index at or before ``t``; at or after it with ``up``."""
        index = math.ceil(t / STEP_S - 1e-9) if up else math.floor(t / STEP_S + 1e-9)
        return int(min(max(index, 0), self.times.size - 1))

    def commit(self, track: np.ndarray, clearance: float) -> None:
        """Nobody placed later comes within ``clearance`` of this track."""
        self.committed.append((track, clearance))
        gap = np.linalg.norm(self.station_pos[:, None, :] - track[None, :, :], axis=2)
        self.station_blocked |= gap < clearance
        reach = clearance + 0.1
        for number, probes in enumerate(self.probes):
            low, high = probes.min(axis=0) - reach, probes.max(axis=0) + reach
            near = np.flatnonzero(np.all((track >= low) & (track <= high), axis=1))
            if near.size:
                gap = np.linalg.norm(probes[:, None, :] - track[None, near, :], axis=2)
                self.rail_blocked[number, near] |= (gap < reach).any(axis=0)

    def clear(self, track: np.ndarray, first: int) -> bool:
        """Whether a stretch of track starting at grid index ``first`` keeps every clearance."""
        last = first + len(track)
        for other, clearance in self.committed:
            if (np.linalg.norm(other[first:last] - track, axis=1) < clearance).any():
                return False
        return True

    def free_until(self, station: str, t: float) -> float:
        """Until when the station stays clear from ``t``; ``t`` itself when it is not."""
        row = self.station_blocked[self.index[station], self.tick(t) :]
        if not row.size or row[0]:
            return t
        taken = np.flatnonzero(row)
        if not taken.size:
            return math.inf
        return float(self.times[self.tick(t) + int(taken[0])])

    def track_of(
        self, segments: list[Segment], start_s: float, end_s: float
    ) -> tuple[np.ndarray, int]:
        """The positions of these segments on the grid, and the first grid index."""
        first, last = self.tick(start_s, up=True), self.tick(end_s)
        source = Source("_", "far_voice", Directivity("omni", False), 0.0, 0.0, tuple(segments), ())
        position, *_ = kinematics._place(self.shell, source, self.times[first : last + 1], None)
        return position, first

    # -- the build ----------------------------------------------------------

    def build(self) -> Recipe:
        counts = self.rng("counts")
        near = int(counts.integers(self.p.near_voice_count[0], self.p.near_voice_count[1] + 1))
        far = int(counts.integers(self.p.far_voice_count[0], self.p.far_voice_count[1] + 1))
        noise = int(counts.integers(self.p.noise_count[0], self.p.noise_count[1] + 1))

        noises, navigator = self.place_noises(noise)
        frames = self.walk_listener(navigator)
        head = replace(self.shell, listener=Listener(tuple(frames)))
        self.head_track = kinematics.listener_state(head, self.times).position
        self.commit(self.head_track, HEAD_CLEARANCE_M + MARGIN_M)

        voices: list[Source] = []
        for kind, count in (("near_voice", near), ("far_voice", far)):
            for number in range(1, count + 1):
                voices.append(self.place_voice(f"{kind.split('_')[0]}_{number}", kind))

        sources = self.speak(voices, noises)
        frames = self.turn_head(frames, sources)
        used = self.used(sources, frames)
        record = self.p.record()
        record["clips"] = {"placeholder": self.clips.placeholder}
        record["attempt"] = self.attempt
        return quantise(
            Recipe(
                dwelling=self.layout.dwelling,
                assets=self.assets,
                seed=self.seed,
                duration_s=self.duration,
                stations=tuple(s for s in self.layout.stations if s.id in used[0]),
                # The layout's rails at the pitch asked for: where a source walks is the
                # same at every pitch, and only the positions it is solved at change.
                rails=tuple(
                    replace(r, pitch_m=self.p.rail_pitch_m)
                    for r in self.layout.rails
                    if r.id in used[1]
                ),
                sources=tuple(sources),
                listener=Listener(tuple(frames)),
                generator=GeneratorRecord(GENERATOR_NAME, GENERATOR_VERSION, record),
                heights=self.heights,
            )
        )

    def used(self, sources: list[Source], frames: list[Keyframe]) -> tuple[set[str], set[str]]:
        rails = {s.rail for source in sources for s in source.segments if isinstance(s, Travel)}
        stations = {frame.station for frame in frames if frame.station is not None}
        for source in sources:
            stations |= {s.station for s in source.segments if not isinstance(s, Travel)}
        for track in self.layout.rails:
            if track.id in rails:
                stations |= {track.a, track.b}
        return stations, rails

    # -- noises -------------------------------------------------------------

    def place_noises(self, count: int) -> tuple[list[Source], Navigator]:
        """Noises on standing spots, and the floor the listener is left with."""
        rng = self.rng("noise:stations")
        stands = [s for s in self.layout.stations if s.kind == "stand"]
        if count > len(stands):
            raise GenerationError(f"{count} noises and only {len(stands)} standing spots")
        walk = self.layout.walk_area
        keep_out = HEAD_CLEARANCE_M + MARGIN_M + 0.07
        whole = Navigator(walk, self.layout.settings.route_pitch_m)
        for _ in range(60):
            picked = [stands[int(i)] for i in sorted(rng.choice(len(stands), count, replace=False))]
            if count and not self.graph_holds({s.id for s in picked}):
                continue
            area = walk
            for station in picked:
                area = area.difference(Point(station.xz).buffer(keep_out))
            navigator = Navigator(area, self.layout.settings.route_pitch_m) if count else whole
            if navigator.reachable.sum() >= 0.85 * whole.reachable.sum():
                break
        else:
            raise _Stuck("no placement of the noises leaves the floor in one piece")
        # The kinds are dealt, not drawn each on its own: five noises drawn
        # one by one are three taps and two streets as readily as a home.
        held = [s for s in NOISE_SUBTYPES if self.clips.noises(s)] or list(NOISE_SUBTYPES)
        deal = [held[int(i)] for i in self.rng("noise:subtypes").permutation(len(held))]
        sources = []
        for number, station in enumerate(picked, start=1):
            draws = self.rng(f"source:noise_{number}")
            subtype = deal[(number - 1) % len(deal)]
            yaw = round(float(draws.uniform(-180.0, 180.0)), 2)
            segment = Dwell(station.id, "standing", 0.0, self.duration, Facing("fixed", yaw))
            track = np.tile(np.asarray(station.position), (self.times.size, 1))
            self.commit(track, SOURCE_CLEARANCE_M + MARGIN_M)
            sources.append(
                Source(
                    id=f"noise_{number}",
                    kind="noise",
                    directivity=Directivity("omni", False),
                    gain_db=self.draw(draws, self.p.noise_gain_db),
                    turn_rate_deg_s=180.0,
                    segments=(segment,),
                    activity=(),
                    subtype=subtype,
                )
            )
        return sources, navigator

    def graph_holds(self, removed: set[str]) -> bool:
        """Whether every other hub still reaches every other without these stations."""
        hubs = [s.id for s in self.layout.stations if s.kind != "seat" and s.id not in removed]
        if not hubs:
            return False
        positions = np.array([self.stations[ident].position for ident in removed])
        reach = SOURCE_CLEARANCE_M + MARGIN_M + 0.1
        open_rails = set()
        for rail in self.layout.rails:
            probes = self.probes[self.rail_index[rail.id]]
            gap = np.linalg.norm(probes[:, None, :] - positions[None, :, :], axis=2)
            if not (gap < reach).any():
                open_rails.add(rail.id)
        seen, queue = {hubs[0]}, [hubs[0]]
        while queue:
            here = queue.pop()
            for link, other in self.links[here]:
                if link in open_rails and other not in seen and self.stations[other].kind != "seat":
                    seen.add(other)
                    queue.append(other)
        return all(hub in seen for hub in hubs)

    # -- the listener's walk ------------------------------------------------

    def walk_listener(self, navigator: Navigator) -> list[Keyframe]:
        """Position keyframes: rests at seats and on the floor, walks between."""
        rng = self.rng("listener")
        share = self.draw(rng, self.p.listener_seated_share)
        standing_y = round(self.floor_y + self.heights.standing_m, 3)
        seats = [
            station
            for station in self.layout.stations
            if station.kind == "seat"
            and navigator.node_near(self.layout.access[station.id]) is not None
        ]
        nodes = np.flatnonzero(navigator.reachable)

        def target(avoid: tuple[float, float] | None) -> tuple[tuple[float, float], Station | None]:
            for _ in range(40):
                if seats and rng.uniform() < share:
                    seat = seats[int(rng.integers(len(seats)))]
                    spot, found = self.layout.access[seat.id], seat
                else:
                    node = navigator.nodes[int(nodes[int(rng.integers(nodes.size))])]
                    spot, found = (round(float(node[0]), 3), round(float(node[1]), 3)), None
                if avoid is None or math.dist(spot, avoid) >= 1.0:
                    return spot, found
            raise _Stuck("the listener found nowhere to go")

        frames: list[Keyframe] = []

        def key(t: float, xz: tuple[float, float], y: float, seat: str | None = None) -> None:
            frames.append(Keyframe(round(t, 3), (xz[0], y, xz[1]), 0.0, 0.0, 0.0, seat))

        def seat_frame(t: float, seat: Station) -> None:
            key(t, seat.xz, seat.position[1], seat.id)

        here, seat = target(None)
        t = 0.0
        while True:
            rest = self.draw(rng, self.p.listener_rest_s)
            there, next_seat = target(here)
            speed = self.draw(rng, self.p.listener_speed_m_s)
            sit = [self.draw(rng, self.p.rise_s) for _ in range(2)]
            path = navigator.path(here, there)
            if path is None:
                raise _Stuck("the listener cannot reach where it meant to go")
            path = np.round(path, 3)
            legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
            walk = float(legs.sum()) / speed
            leave = t + rest
            arrive = leave + walk + (sit[0] if seat else 0.0) + (sit[1] if next_seat else 0.0)
            last = arrive + MIN_STAY_S > self.duration
            end = self.duration if last else round(leave, 3)
            if frames and frames[-1].t_s >= t:
                # The walk's last keyframe is the rest's first.
                frames.pop()
            if seat is not None:
                seat_frame(t, seat)
                seat_frame(end, seat)
                self.rests.append(_Rest(t, end, seat.xz, seat.id))
            else:
                key(t, here, standing_y)
                key(end, here, standing_y)
                self.rests.append(_Rest(t, end, here, None))
            if last:
                return frames
            t = end
            if seat is not None:
                t += sit[0]
                key(t, here, standing_y)
            for point, leg in zip(path[1:], legs, strict=True):
                if leg < 1e-3:
                    continue
                t += max(leg / speed, 0.002)
                key(t, (float(point[0]), float(point[1])), standing_y)
            if next_seat is not None:
                t += sit[1]
            t = round(t, 3)
            here, seat = there, next_seat

    # -- voices -------------------------------------------------------------

    def rest_at(self, t: float) -> _Rest:
        """The listener's rest at ``t``, or the next one."""
        for rest in self.rests:
            if rest.end_s > t + 1.0:
                return rest
        return self.rests[-1]

    def place_voice(self, ident: str, kind: str) -> Source:
        reason = ""
        for retry in range(8):
            rng = self.rng(f"source:{ident}" + (f":retry{retry}" if retry else ""))
            try:
                source = self.plan_voice(ident, kind, rng)
            except _Stuck as stuck:
                reason = str(stuck)
                continue
            track, _ = self.track_of(list(source.segments), 0.0, self.duration)
            self.commit(track, SOURCE_CLEARANCE_M + MARGIN_M)
            return source
        raise _Stuck(f"{ident} found no clear way through the scene ({reason})")

    def candidates(self, kind: str, rest: _Rest, t: float, current: str | None) -> list[Station]:
        """Stations of the voice's distance class, as seen from where the listener rests."""
        span = self.p.near_distance_m if kind == "near_voice" else self.p.far_distance_m
        head = self.head_track[self.tick(t)]
        scored = []
        for station in self.layout.stations:
            if station.kind == "waypoint" or station.id == current:
                continue
            away = math.dist(station.xz, rest.xz)
            if kind == "far_voice":
                away = min(away, math.dist(station.xz, (head[0], head[2])))
            scored.append((away, station))
        inside = [station for away, station in scored if span[0] <= away <= span[1]]
        if inside:
            return inside
        # Nothing in the class: the stations nearest to it, on its far side.
        beyond = sorted(
            (abs(away - (span[0] if away < span[0] else span[1])), station.id, station)
            for away, station in scored
            if away >= min(span[0], HEAD_CLEARANCE_M + MARGIN_M)
        )
        return [station for _, _, station in beyond[:6]]

    def fallback(self, kind: str, rest: _Rest, current: str) -> list[Station]:
        """Every station but this one, those nearest the voice's class first."""
        span = self.p.near_distance_m if kind == "near_voice" else self.p.far_distance_m
        middle = 0.5 * (span[0] + span[1])
        pool = [s for s in self.layout.stations if s.kind != "waypoint" and s.id != current]
        return sorted(pool, key=lambda s: (abs(math.dist(s.xz, rest.xz) - middle), s.id))

    def pick(
        self, rng: np.random.Generator, pool: list[Station], seated_share: float
    ) -> list[Station]:
        """The pool in the order it is tried: seats first for a seated draw."""
        shuffled = [pool[int(i)] for i in rng.permutation(len(pool))]
        want_seat = rng.uniform() < seated_share
        first = [s for s in shuffled if (s.kind == "seat") == want_seat]
        rest = [s for s in shuffled if (s.kind == "seat") != want_seat]
        return (first + rest)[:10]

    def plan_voice(self, ident: str, kind: str, rng: np.random.Generator) -> Source:
        near = kind == "near_voice"
        share = self.draw(rng, self.p.seated_share)
        turn = self.draw(rng, self.p.turn_rate_deg_s)
        gain = self.draw(rng, self.p.gain_db)

        def facing(station: Station) -> Facing:
            if near:
                return Facing("listener")
            if station.kind == "seat":
                return Facing("fixed", station.facing_yaw_deg)
            return Facing("fixed", round(float(rng.uniform(-180.0, 180.0)), 2))

        def stay(station: Station, since: float, rest: _Rest) -> float:
            """When the voice means to leave a station it reached at ``since``."""
            if near:
                until = rest.end_s + float(rng.uniform(1.0, 5.0))
                # The listener's next rests may be within reach of this station too.
                for later in self.rests:
                    if later.start_s >= rest.end_s and later.start_s <= until + 30.0:
                        away = math.dist(station.xz, later.xz)
                        if self.p.near_distance_m[0] <= away <= self.p.near_distance_m[1]:
                            until = later.end_s + float(rng.uniform(1.0, 5.0))
            else:
                until = since + self.draw(rng, self.p.dwell_s)
            return max(until, since + MIN_STAY_S)

        # Where the voice is when the scene starts.
        rest = self.rest_at(0.0)
        start = None
        for station in self.pick(rng, self.candidates(kind, rest, 0.0, None), share):
            if self.free_until(station.id, 0.0) >= MIN_STAY_S + 1.0:
                start = station
                break
        if start is None:
            raise _Stuck(f"{ident} has nowhere to start")
        here, since = start, 0.0
        posture = "seated" if start.kind == "seat" else "standing"
        look = facing(here)
        segments: list[Segment] = []
        while True:
            limit = self.free_until(here.id, since) - 1.0
            end = min(stay(here, since, rest), limit)
            if end >= self.duration - MIN_STAY_S and limit >= self.duration:
                segments.append(Dwell(here.id, posture, since, self.duration, look))
                break
            # Leave when meant to; failing that later, while the station stays
            # clear; failing that earlier.
            later = np.arange(end, min(limit, self.duration - MIN_STAY_S), 2.0)
            earlier = np.arange(end - 2.0, since + 0.5, -2.0)
            trip = None
            moments = [
                round(float(leave), 3)
                for leave in [*later.tolist()[:30], *earlier.tolist()[:30]]
                if leave > since
            ]
            # First a station of the voice's class; failing that at every
            # moment, any station at all, the nearest to the class first.
            for anywhere in (False, True):
                for leave in moments:
                    rest = self.rest_at(leave)
                    if anywhere:
                        pool = self.fallback(kind, rest, here.id)
                    else:
                        pool = self.pick(rng, self.candidates(kind, rest, leave, here.id), share)
                    for station in pool:
                        trip = self.trip(rng, here, posture, station, leave)
                        if trip is not None:
                            break
                    if trip is not None:
                        break
                if trip is not None:
                    break
            if trip is None:
                if limit >= self.duration:
                    segments.append(Dwell(here.id, posture, since, self.duration, look))
                    break
                raise _Stuck(f"{ident} cannot leave {here.id} after {since:.1f} s")
            moves, arrival, station = trip
            segments.append(Dwell(here.id, posture, since, moves[0].start_s, look))
            segments += moves
            here, since = station, arrival
            posture = "seated" if station.kind == "seat" else "standing"
            look = facing(here)
        return Source(
            id=ident,
            kind=kind,
            directivity=Directivity("voice_v1", True),
            gain_db=gain,
            turn_rate_deg_s=turn,
            segments=tuple(segments),
            activity=(),
        )

    def trip(
        self, rng: np.random.Generator, here: Station, posture: str, there: Station, leave: float
    ) -> tuple[list[Segment], float, Station] | None:
        """Segments from one station to another leaving at ``leave``, if the way is clear."""
        speed = self.draw(rng, self.p.speed_m_s)
        rises = [round(self.draw(rng, self.p.rise_s), 3) for _ in range(2)]
        t = leave
        moves: list[Segment] = []
        if posture == "seated":
            moves.append(Rise(here.id, "standing", t, round(t + rises[0], 3)))
            t = moves[-1].end_s
        route = self.route(here.id, there.id, t, speed)
        if route is None:
            return None
        for rail, origin, to in route:
            length = self.rail_length[rail]
            factor = 1.5 if len(route) == 1 else 1.0
            span = max(round(factor * length / speed, 3), 0.05)
            profile = "smoothstep" if len(route) == 1 else "constant"
            moves.append(Travel(rail, origin, to, profile, t, round(t + span, 3), Facing("travel")))
            t = moves[-1].end_s
        if there.kind == "seat":
            moves.append(Rise(there.id, "seated", t, round(t + rises[1], 3)))
            t = moves[-1].end_s
        if t + MIN_STAY_S > self.duration:
            return None
        if self.free_until(there.id, t) < t + MIN_STAY_S + 1.0:
            return None
        track, first = self.track_of(moves, leave, t)
        if not self.clear(track, first):
            return None
        return moves, t, there

    def route(
        self, start: str, goal: str, t: float, speed: float
    ) -> list[tuple[str, str, str]] | None:
        """Rails from one station to another that are clear when they are walked."""
        best = {start: t}
        before: dict[str, tuple[str, str]] = {}
        queue: list[tuple[float, str]] = [(t, start)]
        while queue:
            now, here = heapq.heappop(queue)
            if here == goal:
                break
            if now > best.get(here, math.inf):
                continue
            for rail, other in self.links[here]:
                if other != goal and self.stations[other].kind == "seat":
                    continue
                then = now + 1.5 * self.rail_length[rail] / speed
                if then >= self.duration:
                    continue
                window = self.rail_blocked[
                    self.rail_index[rail], self.tick(now) : self.tick(then) + 2
                ]
                if window.any():
                    continue
                arrival = now + self.rail_length[rail] / speed
                if arrival < best.get(other, math.inf):
                    best[other] = arrival
                    before[other] = (rail, here)
                    heapq.heappush(queue, (arrival, other))
        if goal not in before:
            return None
        route = []
        here = goal
        while here != start:
            rail, origin = before[here]
            route.append((rail, origin, here))
            here = origin
        return route[::-1]

    # -- who speaks ---------------------------------------------------------

    def speak(self, voices: list[Source], noises: list[Source]) -> list[Source]:
        duration = self.duration
        spurts: dict[str, list[tuple[float, float]]] = {n.id: [] for n in noises}
        spoken: dict[str, list[Activity]] = {v.id: [] for v in voices}
        speakers = self.clips.speakers()
        if voices and not speakers:
            raise GenerationError("the clip library holds no voice")
        readers = {
            voice.id: _Voice(speakers[number % len(speakers)])
            for number, voice in enumerate(voices)
        }

        # One conversation: the near voices and the listener take turns.
        rng = self.rng("conversation")
        near = [voice.id for voice in voices if voice.kind == "near_voice"]
        table = [*near, "listener"]
        last_end = {ident: -1.0 for ident in table}
        t = float(rng.uniform(0.0, 3.0))
        previous = None
        while near and t < duration:
            others = [ident for ident in table if ident != previous] or table
            speaker = others[int(rng.integers(len(others)))]
            start = max(t, last_end[speaker] + 0.2)
            wanted = self.draw(rng, self.p.speech_s)
            end = min(start + wanted, duration)
            if speaker != "listener":
                spurt = readers[speaker].take(start, wanted, duration)
                end = start if spurt is None else spurt.end_s
                if spurt is not None and end - start >= 0.3:
                    spoken[speaker].append(spurt)
            if end - start >= 0.3:
                last_end[speaker] = end
                previous = speaker
            if end >= duration:
                break
            if rng.uniform() < self.p.overlap_share:
                t = end - float(rng.uniform(0.2, 1.0))
            else:
                t = end + self.draw(rng, self.p.pause_s)

        # A far voice talks to somebody the scene does not hold.
        for voice in voices:
            if voice.kind != "far_voice":
                continue
            rng = self.rng(f"activity:{voice.id}")
            t = float(rng.uniform(0.0, 10.0))
            while t < duration:
                spurt = readers[voice.id].take(t, self.draw(rng, self.p.speech_s), duration)
                end = t if spurt is None else spurt.end_s
                spell = end - t
                if spurt is not None and spell >= 0.3:
                    spoken[voice.id].append(spurt)
                t = end + self.draw(rng, self.p.pause_s) + spell * float(rng.uniform(0.5, 2.0))

        for noise in noises:
            rng = self.rng(f"activity:{noise.id}")
            if rng.uniform() < self.p.noise_steady_share:
                spurts[noise.id].append((0.0, duration))
                continue
            t = 0.0 if rng.uniform() < 0.5 else self.draw(rng, self.p.noise_off_s)
            while t < duration:
                end = min(t + self.draw(rng, self.p.noise_on_s), duration)
                if end - t >= 0.3:
                    spurts[noise.id].append((t, end))
                t = end + self.draw(rng, self.p.noise_off_s)

        out = [replace(voice, activity=tuple(spoken[voice.id])) for voice in voices]
        first: dict[str, int] = {}
        for noise in noises:
            kind = noise.subtype or "other"
            shelf = self.clips.noises(kind)
            if not shelf:
                raise GenerationError(f"the clip library holds nothing for {noise.id}")
            # Which clip a kind starts on is drawn; two noises of one kind
            # start on two clips, while the shelf has two.
            if kind not in first:
                first[kind] = int(self.rng(f"clips:{kind}").integers(len(shelf)))
            out.append(
                replace(noise, activity=tuple(self.cut(spurts[noise.id], shelf, first[kind])))
            )
            first[kind] += 1
        return out

    @staticmethod
    def cut(
        spurts: list[tuple[float, float]], shelf: list[ClipEntry], first: int = 0
    ) -> list[Activity]:
        """A noise's intervals: its clip read on, looped, or followed by the next.

        An interval that outruns its clip is cut there and goes on, with no
        gap, at the start of the same clip when it loops and of the shelf's
        next when it does not. A placeholder is as long as the scene.
        """
        intervals = []
        which, offset = first % len(shelf), 0.0
        for start, end in spurts:
            t, end = round(start, 3), round(end, 3)
            while end - t > 5e-4:
                entry = shelf[which]
                length = math.floor(entry.duration_s * 1000 + 1e-6) / 1000
                left = round(length - offset, 3)
                if left <= 0:
                    which = which if entry.loop else (which + 1) % len(shelf)
                    offset = 0.0
                    continue
                span = min(round(end - t, 3), left)
                clip = Clip(entry.library, entry.name, entry.sha256)
                intervals.append(Activity(t, round(t + span, 3), clip, round(offset, 3), 0.0))
                t, offset = round(t + span, 3), round(offset + span, 3)
        return intervals

    # -- the head -----------------------------------------------------------

    def turn_head(self, frames: list[Keyframe], sources: list[Source]) -> list[Keyframe]:
        """The walk's keyframes with the head's yaw and pitch, and the knots they need."""
        rng = self.rng("listener:head")
        low, high = self.p.listener_pitch_deg
        knots = np.array([frame.t_s for frame in frames])
        places = np.array([frame.position for frame in frames])

        def head_at(t: float) -> np.ndarray:
            return np.array([np.interp(t, knots, places[:, axis]) for axis in range(3)])

        # What the head turns to, and when: (time, yaw, pitch).
        wanted: list[tuple[float, float, float]] = []
        for here, there in zip(frames[:-1], frames[1:], strict=True):
            step = (there.position[0] - here.position[0], there.position[2] - here.position[2])
            if math.hypot(*step) > 0.05 and here.station is None and there.station is None:
                wanted.append((here.t_s, float(kinematics.yaw_of_direction(*step)), 0.0))
            elif there.station is not None and here.station is None:
                front = self.stations[there.station].facing_yaw_deg
                wanted.append((there.t_s, front, 0.0))
        # While sitting down or getting up the head keeps still: a keyframe
        # between the floor and the seat would be on neither.
        moving = [
            (here.t_s, there.t_s)
            for here, there in zip(frames[:-1], frames[1:], strict=True)
            if (here.station is None) != (there.station is None)
        ]

        def settle(t: float) -> float:
            return next((b for a, b in moving if a < t < b), t)

        shell = replace(self.shell, sources=tuple(sources), listener=Listener(tuple(frames)))
        for source in sources:
            if source.kind != "near_voice":
                continue
            starts = np.array([interval.start_s for interval in source.activity])
            mouths, *_ = kinematics._place(shell, source, starts, None)
            for interval, mouth in zip(source.activity, mouths, strict=True):
                rest = next(
                    (r for r in self.rests if r.start_s <= interval.start_s < r.end_s - 0.5), None
                )
                attends = rng.uniform() < self.p.listener_attend_share
                jitter = float(rng.normal(0.0, self.p.listener_gaze_jitter_deg))
                if rest is None or not attends:
                    continue
                gaze = mouth - head_at(interval.start_s)
                flat = math.hypot(gaze[0], gaze[2])
                bearing = float(kinematics.yaw_of_direction(gaze[0], gaze[2])) + jitter
                rise = math.degrees(math.atan2(gaze[1], max(flat, 1e-6)))
                wanted.append((interval.start_s, bearing, min(max(rise, low), high)))
        events = sorted({settle(round(t, 3)): (y, p) for t, y, p in wanted}.items())

        # Each turn runs at its own rate until done, or until the next begins.
        yaw: float = events[0][1][0] if events else 0.0
        pitch: float = 0.0
        turns: list[tuple[float, float, float]] = [(0.0, yaw, pitch)]
        motion: tuple[float, float, float, float, float, float] | None = None

        def value(t: float) -> tuple[float, float]:
            if motion is None:
                return yaw, pitch
            t0, t1, y0, y1, p0, p1 = motion
            share = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
            return y0 + (y1 - y0) * share, p0 + (p1 - p0) * share

        for number, (t, (aim, tilt)) in enumerate(events):
            if t >= self.duration - 0.25:
                break
            yaw, pitch = value(t)
            motion = None
            if t > turns[-1][0]:
                turns.append((t, yaw, pitch))
            aim = yaw + (aim - yaw + 180.0) % 360.0 - 180.0
            rate = self.draw(rng, self.p.listener_turn_rate_deg_s)
            span = max(abs(aim - yaw), abs(tilt - pitch)) / rate
            done = settle(round(t + max(span, 0.2), 3))
            motion = (t, done, yaw, aim, pitch, tilt)
            following = events[number + 1][0] if number + 1 < len(events) else math.inf
            if done < min(following, self.duration):
                turns.append((done, aim, tilt))
                yaw, pitch, motion = aim, tilt, None
        final = value(self.duration)
        turns.append((self.duration, final[0], final[1]))

        turn_t = np.array([entry[0] for entry in turns])
        turn_yaw = np.array([entry[1] for entry in turns])
        turn_pitch = np.array([entry[2] for entry in turns])
        moments = sorted({*knots.tolist(), *turn_t.tolist()})
        seat_of = [(rest.start_s, rest.end_s, rest.seat) for rest in self.rests if rest.seat]
        by_time = {frame.t_s: frame for frame in frames}
        out = []
        for t in moments:
            given = by_time.get(t)
            seat = next((name for a, b, name in seat_of if a <= t <= b), None)
            position = given.position if given is not None else tuple(head_at(t).tolist())
            out.append(
                Keyframe(
                    t,
                    (position[0], position[1], position[2]),
                    float(np.interp(t, turn_t, turn_yaw)),
                    float(np.interp(t, turn_t, turn_pitch)),
                    0.0,
                    seat,
                )
            )
        return out
