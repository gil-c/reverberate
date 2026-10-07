"""The second generator: who talks with whom, drawn so that realism is nearly free.

``generate_social(layout, parameters, seed)`` writes a recipe of version 2
(``docs/formats/scene-recipe.md``), deterministic as the first generator is:
every draw comes from :func:`reverberate.scenes.generate.stream` under a
label of its own. The first generator is untouched and still writes
version 1.

**What a scene costs decides every rule here.** A scene's price is the
number of distinct places a source is heard from under the crossover, one
wave solve each, and then the cells along the listener's way
(``docs/open-questions/recipes-v2.md``). So:

- **People stay at stations.** The first scene spent 1774 of its 1919 low
  band positions on rails. Here a person is at a station for the whole
  scene but for the few times membership changes, and what moves all the
  time, a sway of centimetres, a head that turns, is written as quantities
  the low band does not read (:class:`~.recipe.Sway`, the yaw, the
  listener's head).
- **Nobody is audible on the move but by footsteps.** A turn, with the 1.2 s
  its sound rings, ends before its talker leaves, and the breaths and
  clothes of a walker are silent while it walks: a rail walked in silence
  costs nothing. A footstep is a source of its own at the floor, at the
  footfall, one position every stride and not every 8 cm.
- **A person's noises are at that person's mouth**, the same position as
  the voice: no solve the voice has not paid.
- **Fixed sources are fixed**: one station for the whole scene, by the
  object they belong to where the dwelling labels one.
- **The listener walks the rails too**, between stations, so that the cells
  along its way are places other recipes of the dwelling walk as well.

**Who talks.** The listener and one to three partners are one conversation,
placed as people place themselves to talk (an F-formation, Kendon 1990):
near one another, in one room, each turned towards the others. Other groups
talk among themselves, with turns of their own. Membership changes rarely:
somebody joins, somebody leaves, or the listener walks to another group; a
voice's ``roles`` say, by interval, whether it is of the listener's
conversation, which is the label training reads. The wearer's own voice is a
source of its own, at the mouth.

**Turns.** A turn is whole utterances of the talker's clips. The next
talker starts a floor transfer offset after the last stops, drawn from a
normal law of mean 0.2 s and standard deviation 0.45 s: a gap two times in
three, an overlap one time in three (Heldner and Edlund 2010; Levinson and
Torreira 2015). Acknowledgements fall inside another's turn and laughter
after one, **where the library holds such clips**; ``clarify_v1`` holds
read sentences only, and the generator then schedules neither and says so
under ``generator.parameters.left_out``.

**Levels.** No gain sets a role apart. A talker has a level of its own,
within a few decibels of the normal effort, and raises it with the noise
where it stands (:func:`~.levels.lombard_level_db`); a quiet scene is spoken
at a relaxed effort and now and then whispered. A noise has the level its
class has at 1 m (:data:`NOISE_LEVELS_DB`), higher in a livelier scene. The
recipe is then held to rule 16: the loudest masker is turned down within its
class, then switched off, until no turn of the listener's conversation is
under the scene's floor of speech to noise in free field.

**The head.** The listener looks at the active talker of its conversation
about ``gaze_share`` of the time that somebody else talks, 0.6 by the
owner's rule, with a reaction delay, and otherwise glances at somebody
else, reads, looks away or turns to something that started; the head never
quite stops, by an amount that swells and fades. Talkers look at who
speaks, and at who they speak to.
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from reverberate.scenes import kinematics
from reverberate.scenes.generate import (
    GENERATOR_NAME,
    ClipEntry,
    ClipLibrary,
    GenerationError,
    _Scene,
    _Stuck,
    _Voice,
    placeholder_assets,
    stream,
)
from reverberate.scenes.layout import Layout
from reverberate.scenes.levels import (
    EFFORT_LEVEL_DB,
    VOICE_REFERENCE_DB,
    conversation_snr,
    effort_of,
    effort_range_db,
    level_at,
    lombard_level_db,
    masker_level_at,
)
from reverberate.scenes.recipe import (
    LISTENER,
    Activity,
    Assets,
    Attach,
    Directivity,
    Dwell,
    Facing,
    Gaze,
    GeneratorRecord,
    Keyframe,
    Listener,
    Membership,
    Opening,
    Recipe,
    Rise,
    Role,
    Scene,
    Segment,
    Source,
    Station,
    Sway,
    Travel,
    quantise,
)
from reverberate.scenes.validate import HEAD_CLEARANCE_M, validate

__all__ = [
    "FIXTURE_USES",
    "NOISE_LEVELS_DB",
    "SOCIAL_GENERATOR_VERSION",
    "SocialParameters",
    "generate_social",
    "placeholder_social_clips",
]

#: The second generator's own version; a change of its output for one seed changes it.
SOCIAL_GENERATOR_VERSION = "1.0.0"

#: The level at 1 m, dB SPL, a class of noise is drawn from: the quiet end in a calm
#: scene, the loud end in a lively one. Where they come from is in
#: ``docs/open-questions/recipes-v2.md``; in short, the sound power a machine's
#: European energy label declares less 8 dB (a hemisphere at 1 m), a programme
#: listened to at 55 to 65 dB at 3 m, and for the outside a street's level at the
#: wall less what an open or a closed window takes.
NOISE_LEVELS_DB: dict[str, tuple[float, float]] = {
    "television": (58.0, 70.0),
    "radio": (55.0, 66.0),
    "appliance": (42.0, 68.0),
    "water": (50.0, 66.0),
    "other": (48.0, 64.0),
    "music": (50.0, 68.0),
    "outside:open": (43.0, 60.0),
    "outside:closed": (27.0, 44.0),
    "body": (25.0, 40.0),
    "steps": (40.0, 55.0),
}

#: Which fixtures a class of fixed source may belong to, the likeliest first
#: (:data:`reverberate.scenes.layout.FIXTURE_CATEGORIES`).
FIXTURE_USES: dict[str, tuple[str, ...]] = {
    "television": ("tv",),
    "radio": ("kitchen_counter", "counter", "shelf", "nightstand", "desk", "table"),
    "appliance": ("washing_machine_and_dryer", "kitchen_counter", "counter"),
    "water": ("shower", "kitchen_counter", "toilet"),
    "other": ("kitchen_counter", "counter"),
    "music": ("piano", "shelf", "desk", "table"),
    "outside": ("window", "outer_door"),
}

#: The library's kinds a class of fixed source takes its clips from.
_SHELVES: dict[str, tuple[str, ...]] = {"outside": ("outside", "street")}

#: What the generator keeps beyond the format's clearances: the two sways, and a rounding.
_KEEP_M = HEAD_CLEARANCE_M + 0.14
#: Two people who talk are at least this far apart on the plan.
_APART_M = 0.75
#: A turn ends this long before its talker moves: the tail its sound rings for, and a step.
_GUARD_S = kinematics.AUDIBLE_TAIL_S + 0.1
_STEP_S = kinematics.YAW_STEP_S
#: Nobody arrives with less than this left of the scene.
_STAY_S = 4.0

Range = tuple[float, float]


@dataclass(frozen=True)
class SocialParameters:
    """What a scene of version 2 is drawn from. A two element value is ``(min, max)``.

    The defaults are a development scene: three minutes, of middling calm.
    :meth:`preset` gives the three the owner asked for.
    """

    duration_s: float = 180.0
    #: From 0, as lively as a home gets, to 1, nearly silent. It sets how many other
    #: groups and noises there are and how loud, unless the counts below say.
    calmness: float = 0.5
    #: Rule 16: the least speech to noise ratio, in free field at the listener, of any
    #: turn of the listener's conversation.
    snr_floor_db: float = 0.0
    #: The talkers of the listener's conversation, the listener apart.
    partners: tuple[int, int] = (1, 3)
    #: Left ``None``, these three follow the calmness.
    other_groups: tuple[int, int] | None = None
    noises: tuple[int, int] | None = None
    media: tuple[int, int] | None = None
    other_group_size: tuple[int, int] = (2, 3)
    #: How many times, in the scene, somebody joins or leaves the listener's
    #: conversation or the listener walks to another.
    membership_changes: tuple[int, int] = (1, 1)
    #: Whether the wearer's own voice, and the noises of people, are in the scene.
    own_voice: bool = True
    body_noises: bool = True
    outside: bool = True
    seated_share: float = 0.6
    #: How far apart, on the plan, two members of a group are.
    member_distance_m: Range = (_APART_M, 2.2)
    speed_m_s: Range = (0.6, 1.0)
    rise_s: Range = (1.2, 2.5)
    turn_rate_deg_s: Range = (90.0, 240.0)
    #: A turn: its median length, and its least and its most.
    turn_median_s: float = 2.5
    turn_s: Range = (0.8, 10.0)
    #: The floor transfer offset: mean and standard deviation of a normal law.
    floor_transfer_mean_s: float = 0.2
    floor_transfer_sd_s: float = 0.45
    #: One acknowledgement about every this many seconds of somebody else's turn.
    backchannel_every_s: float = 8.0
    #: The share of turns that somebody laughs after.
    laughter_share: float = 0.06
    #: The share of the time somebody else of the conversation talks that the listener
    #: looks at that talker, and how late the head starts towards a new one.
    gaze_share: float = 0.6
    reaction_s: Range = (0.2, 0.8)
    listener_turn_rate_deg_s: Range = (80.0, 220.0)
    #: How far a body sways from where it is, the sum of its sinusoids; and a talker's yaw.
    sway_m: Range = (0.01, 0.03)
    sway_yaw_deg: Range = (2.0, 8.0)
    #: Standard deviation of a talker's own level about the normal effort.
    talker_level_sd_db: float = 2.5
    #: A noise that comes and goes, as the first generator's.
    noise_on_s: Range = (30.0, 300.0)
    noise_off_s: Range = (10.0, 120.0)
    noise_steady_share: float = 0.5
    rail_pitch_m: float = 0.08
    stride_m: float = 0.64

    @classmethod
    def preset(cls, name: str, duration_s: float = 180.0) -> SocialParameters:
        """``quiet``, ``medium`` or ``lively``: the three scenes of the design note."""
        calm = {"quiet": 0.9, "medium": 0.5, "lively": 0.15}
        if name not in calm:
            raise ValueError(f"no preset {name!r}; there are {', '.join(calm)}")
        partners = {"quiet": (1, 2), "medium": (2, 3), "lively": (3, 3)}[name]
        return cls(duration_s=float(duration_s), calmness=calm[name], partners=partners)

    def record(self) -> dict[str, Any]:
        """The tree the recipe's ``generator.parameters`` holds."""
        tree: dict[str, Any] = {}
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            tree[item.name] = list(value) if isinstance(value, tuple) else value
        return tree

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SocialParameters:
        """The inverse of :meth:`record`; keys it does not write are ignored."""
        known = {item.name for item in dataclasses.fields(cls)}
        given: dict[str, Any] = {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in record.items()
            if name in known
        }
        return cls(**given)


def placeholder_social_clips(duration_s: float, speakers: int = 16) -> ClipLibrary:
    """Clip references that name no audio, for every kind version 2 schedules.

    Beside each speaker's voice, an acknowledgement and a laugh; a clip for
    each class of noise, a person's noises among them, at the middle of its
    class's level. A trace must refuse them.
    """

    def entry(name: str, kind: str, speaker: str = "", event: str = "turn") -> ClipEntry:
        digest = hashlib.sha256(f"placeholder:{name}".encode()).hexdigest()
        span = NOISE_LEVELS_DB.get(kind if kind != "outside" else "outside:open")
        level = VOICE_REFERENCE_DB if span is None else 0.5 * (span[0] + span[1])
        return ClipEntry(
            "placeholder", name, digest, float(duration_s), kind, speaker, (), False, level,
            "normal", event,
        )  # fmt: skip

    voices = []
    for n in range(1, speakers + 1):
        voices.append(entry(f"voice_{n:02d}", "voice", f"{n:02d}"))
        voices.append(entry(f"voice_{n:02d}_backchannel", "voice", f"{n:02d}", "backchannel"))
        voices.append(entry(f"voice_{n:02d}_laughter", "voice", f"{n:02d}", "laughter"))
    kinds = ("television", "radio", "appliance", "water", "other", "music", "outside", "body")
    noises = [entry(f"noise_{kind}", kind) for kind in (*kinds, "steps")]
    return ClipLibrary(tuple(voices + noises), placeholder=True)


def generate_social(
    layout: Layout,
    parameters: SocialParameters | None = None,
    seed: int = 0,
    *,
    assets: Assets | None = None,
    clips: ClipLibrary | None = None,
    allow_placeholder_clips: bool = False,
    allow_placeholder_assets: bool = False,
    attempts: int = 8,
) -> Recipe:
    """A valid recipe of version 2 on this layout, drawn from ``parameters`` under ``seed``."""
    parameters = parameters or SocialParameters()
    if clips is None:
        if not allow_placeholder_clips:
            raise ValueError("no clip library was given; pass one or allow_placeholder_clips=True")
        clips = placeholder_social_clips(parameters.duration_s)
    if assets is None:
        if not allow_placeholder_assets:
            raise ValueError("no assets were given; pass them or allow_placeholder_assets=True")
        assets = placeholder_assets()
    if not 0 <= seed < 2**53:
        raise ValueError(f"seed {seed} is outside 0 <= seed < 2^53")
    if not 0.0 <= parameters.calmness <= 1.0:
        raise ValueError(f"calmness {parameters.calmness} is outside 0 to 1")
    reason = "no attempt was made"
    for attempt in range(attempts):
        try:
            recipe = _Social(layout, parameters, seed, assets, clips, attempt).build()
        except _Stuck as stuck:
            reason = str(stuck)
            continue
        found = validate(recipe, layout.floor)
        if not found:
            return recipe
        reason = str(found[0])
    raise GenerationError(f"no valid recipe in {attempts} attempts; the last failed on: {reason}")


# --------------------------------------------------------------------------
# what a scene is made of while it is planned
# --------------------------------------------------------------------------


@dataclass
class _Block:
    """A stay at a station."""

    start_s: float
    end_s: float
    station: str
    posture: str


@dataclass
class _Person:
    id: str
    blocks: list[_Block]
    #: The segments walked between block ``i`` and block ``i + 1``.
    moves: list[list[Segment]] = field(default_factory=list)
    #: ``(time, group)``: the group from that time on; ``None`` is no group.
    member: list[tuple[float, str | None]] = field(default_factory=list)
    level_db: float = VOICE_REFERENCE_DB
    turn_rate: float = 180.0
    shelves: dict[str, _Voice] = field(default_factory=dict)
    spoken: list[Activity] = field(default_factory=list)
    #: The listener's own way, where the person is the listener: ``(t, x, y, z, station)``.
    frames: list[tuple[float, float, float, float, str | None]] = field(default_factory=list)

    def block_at(self, t: float) -> _Block | None:
        return next((b for b in self.blocks if b.start_s <= t < b.end_s), None)

    def free_from(self, t: float, span: float = 0.0) -> float:
        """The first instant from ``t`` at which this voice is silent for ``span`` and a breath."""
        moved = True
        while moved:
            moved = False
            for said in self.spoken:
                if said.start_s - 0.2 < t + span and t < said.end_s + 0.2:
                    t, moved = said.end_s + 0.2, True
        return t

    def group_at(self, t: float) -> str | None:
        group = None
        for since, name in self.member:
            if since <= t:
                group = name
        return group


@dataclass
class _Fixed:
    """A fixed source: where it is, what it is, how loud, and when it is on."""

    id: str
    kind: str
    subtype: str
    station: Station
    level_db: float
    span_db: Range
    spurts: list[tuple[float, float]]
    shelf: list[ClipEntry]
    first: int
    opening: Opening | None = None
    off: bool = False


class _Social:
    """One attempt at a recipe of version 2."""

    def __init__(
        self,
        layout: Layout,
        parameters: SocialParameters,
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
        self.calm = round(float(parameters.calmness), 2)
        self.floor_y = layout.dwelling.floor_y_m
        self.heights = layout.heights
        self.rails = tuple(replace(r, pitch_m=parameters.rail_pitch_m) for r in layout.rails)
        self.stations = {s.id: s for s in (*layout.stations, *layout.fixtures)}
        self.rail_of = {rail.id: rail for rail in self.rails}
        self.length = {rail.id: kinematics.rail_length(rail) for rail in self.rails}
        self.links: dict[str, list[tuple[str, str]]] = {s.id: [] for s in layout.stations}
        for rail in self.rails:
            self.links[rail.a].append((rail.id, rail.b))
            self.links[rail.b].append((rail.id, rail.a))
        self.probes = {}
        for rail in self.rails:
            points = np.asarray(rail.points, dtype=float)
            arcs = np.linspace(0.0, self.length[rail.id], int(self.length[rail.id] / 0.1) + 2)
            self.probes[rail.id] = kinematics._along(points, arcs)
        self.people: list[_Person] = []
        self.fixed: list[_Fixed] = []
        self.left_out: set[str] = set()
        self.voice = "voice_v1" if "voice_v1" in assets.directivity else "omni"

    # -- draws and small helpers ---------------------------------------------

    def rng(self, label: str) -> np.random.Generator:
        return stream(self.seed, label if self.attempt == 0 else f"{label}#{self.attempt}")

    @staticmethod
    def draw(rng: np.random.Generator, span: Range) -> float:
        return float(rng.uniform(span[0], span[1])) if span[1] > span[0] else float(span[0])

    def count(self, rng: np.random.Generator, given: tuple[int, int] | None, most: float) -> int:
        """A count: the range given, or what the calmness says, ``most`` at its liveliest."""
        if given is not None:
            return int(rng.integers(given[0], given[1] + 1))
        return int(np.clip(round((1.0 - self.calm) * most + rng.uniform(-0.4, 0.4)), 0, most))

    def at(self, station: str, posture: str = "standing") -> np.ndarray:
        """Where a mouth is at a station."""
        held = self.stations[station]
        if held.kind == "fixture":
            return np.asarray(held.position, dtype=float)
        return np.array([held.xz[0], self.floor_y + self.heights.of(posture), held.xz[1]])

    def place_of(self, person: _Person, t: float) -> np.ndarray:
        """Where somebody rests at ``t``, or rested last."""
        block = person.block_at(t) or max(
            (b for b in person.blocks if b.start_s <= t), key=lambda b: b.start_s
        )
        return self.at(block.station, block.posture)

    def shell(self, sources: tuple[Source, ...] = (), frames: tuple[Keyframe, ...] = ()) -> Recipe:
        """A recipe with every station and rail, to ask the kinematics while planning."""
        if not frames and self.people and self.people[0].frames:
            frames = tuple(self.frames_at_rest())
        return Recipe(
            dwelling=self.layout.dwelling,
            assets=self.assets,
            seed=self.seed,
            duration_s=self.duration,
            stations=tuple(self.stations.values()),
            rails=self.rails,
            sources=sources,
            listener=Listener(frames, sway=(), conversation=(), gaze=()),
            heights=self.heights,
            schema_version=2,
            scene=Scene(self.calm, self.p.snr_floor_db),
        )

    # -- the build -------------------------------------------------------------

    def build(self) -> Recipe:
        self.place_fixed()
        self.place_people()
        self.change_membership()
        self.listener_way()
        turned = self.sways()
        for _ in range(4 * len(self.fixed) + 2):
            self.talk()
            sources = self.sources(turned)
            recipe = self.recipe(sources, self.frames_at_rest())
            if not self.quieten(recipe):
                break
        else:
            raise _Stuck("the noise could not be brought under the conversation")
        sources += self.bodies(recipe)
        frames, gaze = self.head(recipe)
        return quantise(self.recipe(sources, frames, gaze, turned[LISTENER]))

    def recipe(
        self,
        sources: list[Source],
        frames: list[Keyframe],
        gaze: tuple[Gaze, ...] = (),
        sway: tuple[Sway, ...] = (),
    ) -> Recipe:
        used = {frame.station for frame in frames if frame.station is not None}
        rails = set()
        for source in sources:
            used |= {s.station for s in source.segments if not isinstance(s, Travel)}
            rails |= {s.rail for s in source.segments if isinstance(s, Travel)}
        for rail in self.rails:
            if rail.id in rails:
                used |= {rail.a, rail.b}
        me = self.people[0]
        edges = [*me.member, (self.duration, None)]
        talk = tuple(
            Membership(since, until, group)
            for (since, group), (until, _) in zip(edges[:-1], edges[1:], strict=True)
            if until > since
        )
        record = self.p.record()
        record["clips"] = {"placeholder": self.clips.placeholder}
        record["left_out"] = {name: True for name in sorted(self.left_out)}
        record["attempt"] = self.attempt
        return replace(
            self.shell(tuple(sources)),
            stations=tuple(s for s in self.stations.values() if s.id in used),
            rails=tuple(rail for rail in self.rails if rail.id in rails),
            listener=Listener(tuple(frames), sway=sway, conversation=talk, gaze=gaze),
            generator=GeneratorRecord(GENERATOR_NAME, SOCIAL_GENERATOR_VERSION, record),
        )

    # -- fixed sources ---------------------------------------------------------

    def shelf(self, subtype: str) -> list[ClipEntry]:
        found: list[ClipEntry] = []
        for kind in _SHELVES.get(subtype, (subtype,)):
            found += self.clips.noises(kind)
        if not found:
            self.left_out.add(subtype)
        return found

    def place_fixed(self) -> None:
        """Programmes and noises, each by the object it belongs to, on or off over the scene."""
        rng = self.rng("fixed")
        lively = 1.0 - self.calm
        wanted: list[tuple[str, str]] = []
        media = self.count(rng, self.p.media, 1.0)
        kinds = ["television", "radio"] if rng.uniform() < 0.7 else ["radio", "television"]
        wanted += [("media_voice", kind) for kind in kinds[:media]]
        deal = [("appliance", "water", "other", "music")[int(i)] for i in rng.permutation(4)]
        wanted += [("noise", kind) for kind in deal[: self.count(rng, self.p.noises, 3.0)]]
        if self.p.outside and rng.uniform() < 0.3 + 0.6 * lively:
            wanted.append(("noise", "outside"))
        taken: set[str] = set()
        stands = [s for s in self.layout.stations if s.kind == "stand"]
        for number, (kind, subtype) in enumerate(wanted, start=1):
            draws = self.rng(f"fixed:{subtype}")
            shelf = self.shelf(subtype)
            if not shelf:
                continue
            station = None
            for category in FIXTURE_USES[subtype]:
                pool = [
                    s
                    for s in self.layout.fixtures
                    if s.object is not None
                    and s.object.rsplit("_", 1)[0] == category
                    and s.id not in taken
                ]
                if pool:
                    station = pool[int(draws.integers(len(pool)))]
                    break
            opening = None
            if subtype == "outside":
                if station is None or station.object is None:
                    # No opening is labelled: nothing comes in by one.
                    self.left_out.add("outside")
                    continue
                state = "open" if draws.uniform() < 0.2 + 0.6 * lively else "closed"
                opening = Opening(station.object, state)
            if station is None:
                # The dwelling labels no such object: a standing spot stands in for it.
                free = [s for s in stands if s.id not in taken]
                if not free:
                    continue
                station = free[int(draws.integers(len(free)))]
            taken.add(station.id)
            span = NOISE_LEVELS_DB[subtype if opening is None else f"outside:{opening.state}"]
            share = float(np.clip(lively + draws.uniform(-0.2, 0.2), 0.0, 1.0))
            spurts = [(0.0, self.duration)]
            if draws.uniform() >= self.p.noise_steady_share and opening is None:
                spurts, t = [], 0.0 if draws.uniform() < 0.6 else self.draw(draws, (5.0, 40.0))
                while t < self.duration:
                    end = min(t + self.draw(draws, self.p.noise_on_s), self.duration)
                    if end - t >= 1.0:
                        spurts.append((round(t, 3), round(end, 3)))
                    t = end + self.draw(draws, self.p.noise_off_s)
            self.fixed.append(
                _Fixed(
                    id=f"{subtype}_{number}",
                    kind=kind,
                    subtype=subtype,
                    station=station,
                    level_db=round(span[0] + share * (span[1] - span[0]), 2),
                    span_db=span,
                    spurts=spurts,
                    shelf=shelf,
                    first=int(draws.integers(len(shelf))),
                    opening=opening,
                )
            )

    def fixed_sources(self) -> list[Source]:
        out = []
        for item in self.fixed:
            if item.off or not item.spurts:
                continue
            stored = [e.spl_1m_db if e.spl_1m_db is not None else item.level_db for e in item.shelf]
            by_name = {e.name: level for e, level in zip(item.shelf, stored, strict=True)}
            gain = round(item.level_db - stored[item.first % len(stored)], 2)
            activity = tuple(
                # Each clip is brought to the source's level, whatever its own.
                replace(a, gain_db=round(item.level_db - by_name[a.clip.name] - gain, 2) + 0.0)
                for a in _Scene.cut(item.spurts, item.shelf, item.first)
            )
            station = item.station
            fixture = station.kind == "fixture"
            faces = item.subtype == "television" and self.voice != "omni"
            out.append(
                Source(
                    id=item.id,
                    kind=item.kind,
                    directivity=Directivity(self.voice if faces else "omni", faces),
                    gain_db=gain,
                    turn_rate_deg_s=0.0,
                    segments=(
                        Dwell(
                            station.id,
                            "fixed" if fixture else "standing",
                            0.0,
                            self.duration,
                            Facing("fixed", station.facing_yaw_deg),
                        ),
                    ),
                    activity=activity,
                    subtype=item.subtype,
                    level_spl_1m_db=item.level_db,
                    opening=item.opening,
                )
            )
        return out

    def quieten(self, recipe: Recipe) -> bool:
        """Turn down what masks the conversation's worst turn; whether anything was changed."""
        worst = min(conversation_snr(recipe), key=lambda row: row[2], default=None)
        # Half a decibel is kept over the floor, for the levels' rounding.
        if worst is None or worst[2] >= self.p.snr_floor_db + 0.5:
            return False
        interval = recipe.source(worst[0]).activity[worst[1]]
        span = interval.end_s - interval.start_s
        times = interval.start_s + span * np.array([0.0, 0.5, 0.999])
        head = kinematics.listener_state(recipe, times).position
        loudest, level = None, -math.inf
        for item in self.fixed:
            if item.off or not item.spurts:
                continue
            here = float(level_at(recipe, recipe.source(item.id), times, head).max())
            if here > level:
                loudest, level = item, here
        speech = level_at(recipe, recipe.source(worst[0]), times, head)
        voices = replace(recipe, sources=tuple(s for s in recipe.sources if s.kind == "voice"))
        babble = masker_level_at(voices, times, head)
        if loudest is None or float((speech - babble).min()) < self.p.snr_floor_db + 0.5:
            raise _Stuck("voices outside the conversation mask it, and no noise can be turned down")
        need = self.p.snr_floor_db + 1.5 - worst[2]
        if loudest.level_db - need >= loudest.span_db[0]:
            loudest.level_db = round(loudest.level_db - need, 2)
        elif loudest.level_db > loudest.span_db[0]:
            loudest.level_db = loudest.span_db[0]
        else:
            # As quiet as its class goes: somebody switches it off.
            loudest.off = True
        return True

    # -- people ----------------------------------------------------------------

    def keep_out(self) -> list[tuple[float, float]]:
        """Where nobody may stand: by a fixed source that stands on the floor."""
        return [item.station.xz for item in self.fixed if item.station.kind != "fixture"]

    def circle(
        self,
        rng: np.random.Generator,
        size: int,
        taken: list[Station],
        apart_m: float,
    ) -> list[Station] | None:
        """Stations for a group: in one room, near one another, away from everybody else."""
        blocked = self.keep_out()
        pool = [
            s
            for s in self.layout.stations
            if s.kind != "waypoint"
            and all(math.dist(s.xz, other.xz) >= apart_m for other in taken)
            and all(math.dist(s.xz, spot) >= _KEEP_M for spot in blocked)
        ]
        low, high = self.p.member_distance_m
        for _ in range(60):
            if not pool:
                return None
            want_seat = rng.uniform() < self.p.seated_share
            anchors = [s for s in pool if (s.kind == "seat") == want_seat] or pool
            chosen = [anchors[int(rng.integers(len(anchors)))]]
            near = [
                s
                for s in pool
                if s.room == chosen[0].room and low <= math.dist(s.xz, chosen[0].xz) <= high
            ]
            for index in rng.permutation(len(near)):
                if len(chosen) == size:
                    break
                station = near[int(index)]
                if all(low <= math.dist(station.xz, c.xz) <= high for c in chosen):
                    chosen.append(station)
            if len(chosen) == size:
                return chosen
        return None

    def place_people(self) -> None:
        """The listener's conversation, the other groups, and somebody alone who may join."""
        rng = self.rng("people")
        partners = int(rng.integers(self.p.partners[0], self.p.partners[1] + 1))
        others = self.count(rng, self.p.other_groups, 2.0)
        changes = int(rng.integers(self.p.membership_changes[0], self.p.membership_changes[1] + 1))
        self.changes = changes
        speakers = self.speakers()
        if not speakers:
            raise GenerationError("the clip library holds no voice")
        order = [int(i) for i in self.rng("speakers").permutation(len(speakers))]
        taken: list[Station] = []
        shift = -6.0 * float(np.clip((self.calm - 0.6) / 0.4, 0.0, 1.0))

        def person(ident: str, station: Station, group: str | None) -> _Person:
            draws = self.rng(f"person:{ident}")
            posture = "seated" if station.kind == "seat" else "standing"
            level = (
                VOICE_REFERENCE_DB
                + shift
                + float(np.clip(draws.normal(0.0, self.p.talker_level_sd_db), -5.0, 5.0))
            )
            made = _Person(
                id=ident,
                blocks=[_Block(0.0, self.duration, station.id, posture)],
                member=[(0.0, group)],
                level_db=round(level, 2),
                turn_rate=self.draw(draws, self.p.turn_rate_deg_s),
                # A reader of its own: two people given one speaker do not share a place in it.
                shelves={
                    key: _Voice(reader.shelf)
                    for key, reader in speakers[order[len(self.people) % len(order)]].items()
                },
            )
            self.people.append(made)
            taken.append(station)
            return made

        first = self.circle(rng, partners + 1, taken, 0.0)
        if first is None:
            raise _Stuck("the dwelling has no place for the listener's conversation")
        person(LISTENER, first[0], "g1")
        for number, station in enumerate(first[1:], start=1):
            person(f"talker_{number}", station, "g1")
        number = partners
        # Another group talks at twice the listener's reach to its own, or further:
        # 6 dB of distance between a partner's voice and a stranger's.
        reach = max(math.dist(first[0].xz, station.xz) for station in first[1:])
        for group in range(2, others + 2):
            size = int(rng.integers(self.p.other_group_size[0], self.p.other_group_size[1] + 1))
            found = self.circle(rng, size, taken, max(2.5, 2.0 * reach))
            if found is None:
                continue
            for station in found:
                number += 1
                person(f"talker_{number}", station, f"g{group}")
        if changes and len({p.member[0][1] for p in self.people}) == 1:
            # Nobody else is in the dwelling: somebody alone, who may come and join.
            alone = self.circle(rng, 1, taken, 3.0) or self.circle(rng, 1, taken, 1.5)
            if alone is not None:
                person(f"talker_{number + 1}", alone[0], None)

    def speakers(self) -> list[dict[str, _Voice]]:
        """Per speaker of the library, a reader for each effort's turns and each event."""
        groups: dict[str, dict[str, list[ClipEntry]]] = {}
        for entry in sorted(self.clips.entries, key=lambda e: e.name):
            if entry.kind == "voice":
                key = entry.effort if entry.event == "turn" else entry.event
                groups.setdefault(entry.speaker, {}).setdefault(key, []).append(entry)
        held = [groups[name] for name in sorted(groups) if "normal" in groups[name]]
        for event in ("backchannel", "laughter"):
            if not any(event in shelves for shelves in held):
                self.left_out.add(event)
        return [{key: _Voice(shelf) for key, shelf in shelves.items()} for shelves in held]

    # -- moves -----------------------------------------------------------------

    def route(
        self, start: str, goal: str, occupied: list[tuple[float, float, float, float]]
    ) -> list[tuple[str, str, str]] | None:
        """Rails from one station to another that pass clear of everybody who stays.

        ``occupied`` is ``(x, z, clearance, drop)`` of each of them, the drop
        being how far under a walker's mouth theirs is.
        """
        ends = (self.stations[start].xz, self.stations[goal].xz)

        def clear(rail: str) -> bool:
            probes = self.probes[rail]
            # At its own two ends a way is as near as the stations are.
            far = np.ones(len(probes), dtype=bool)
            for end in ends:
                far &= np.linalg.norm(probes - np.asarray(end), axis=1) > 0.3
            for x, z, keep, drop in occupied:
                flat = np.linalg.norm(probes[far] - np.array([x, z]), axis=1)
                if (np.hypot(flat, drop) < keep).any():
                    return False
            return True

        best = {start: 0.0}
        before: dict[str, tuple[str, str]] = {}
        queue: list[tuple[float, str]] = [(0.0, start)]
        checked: dict[str, bool] = {}
        while queue:
            so_far, here = heapq.heappop(queue)
            if here == goal:
                break
            if so_far > best.get(here, math.inf):
                continue
            for rail, other in sorted(self.links[here]):
                if other != goal and self.stations[other].kind == "seat":
                    continue
                if rail not in checked:
                    checked[rail] = clear(rail)
                if not checked[rail]:
                    continue
                reach = so_far + self.length[rail]
                if reach < best.get(other, math.inf):
                    best[other] = reach
                    before[other] = (rail, here)
                    heapq.heappush(queue, (reach, other))
        if goal not in before:
            return None
        way, here = [], goal
        while here != start:
            rail, origin = before[here]
            way.append((rail, origin, here))
            here = origin
        return way[::-1]

    def move(
        self, rng: np.random.Generator, person: _Person, leave: float, goal: Station
    ) -> float | None:
        """Walk somebody to another station from ``leave``; when it arrives, or ``None``."""
        block = person.blocks[-1]
        me = self.people[0]
        others = []
        for other in self.people:
            if other is not person:
                spot = self.place_of(other, leave)
                # The head and a mouth are kept further apart than two mouths.
                keep = _KEEP_M if me in (person, other) else _KEEP_M - 0.1
                drop = self.floor_y + self.heights.standing_m - float(spot[1])
                others.append((float(spot[0]), float(spot[2]), keep, drop))
        others += [(x, z, _KEEP_M, 0.0) for x, z in self.keep_out()]
        way = self.route(block.station, goal.id, others)
        if way is None:
            return None
        speed = self.draw(rng, self.p.speed_m_s)
        rises = [round(self.draw(rng, self.p.rise_s), 3) for _ in range(2)]
        t = round(leave, 3)
        moves: list[Segment] = []
        if block.posture == "seated":
            moves.append(Rise(block.station, "standing", t, round(t + rises[0], 3)))
            t = moves[-1].end_s
        for rail, origin, to in way:
            alone = len(way) == 1
            span = max(round((1.5 if alone else 1.0) * self.length[rail] / speed, 3), 0.05)
            profile = "smoothstep" if alone else "constant"
            moves.append(Travel(rail, origin, to, profile, t, round(t + span, 3), Facing("travel")))
            t = moves[-1].end_s
        posture = "standing"
        if goal.kind == "seat":
            moves.append(Rise(goal.id, "seated", t, round(t + rises[1], 3)))
            t, posture = moves[-1].end_s, "seated"
        if t + _STAY_S > self.duration:
            return None
        block.end_s = round(leave, 3)
        person.moves.append(moves)
        person.blocks.append(_Block(t, self.duration, goal.id, posture))
        return t

    def near_group(self, rng: np.random.Generator, group: str, t: float) -> list[Station]:
        """Free stations a newcomer to a group may take, in the order they are tried."""
        members = [
            self.stations[p.blocks[-1].station] for p in self.people if p.group_at(t) == group
        ]
        everybody = [self.stations[p.blocks[-1].station] for p in self.people]
        blocked = self.keep_out()
        # A newcomer keeps a metre from everybody: the way in passes clear of them all.
        low, high = max(self.p.member_distance_m[0], 1.0), max(self.p.member_distance_m[1], 1.6)
        pool = [
            s
            for s in self.layout.stations
            if s.kind != "waypoint"
            and members
            and s.room == members[0].room
            and all(math.dist(s.xz, other.xz) >= low for other in everybody)
            and all(math.dist(s.xz, spot) >= _KEEP_M for spot in blocked)
            and min(math.dist(s.xz, m.xz) for m in members) <= high
        ]
        return [pool[int(i)] for i in rng.permutation(len(pool))]

    def change_membership(self) -> None:
        """Somebody joins, somebody leaves, or the listener walks to another group: rarely."""
        rng = self.rng("membership")
        me = self.people[0]
        times = sorted(float(rng.uniform(0.2, 0.75)) * self.duration for _ in range(self.changes))
        last = 0.0
        for when in times:
            when = round(max(when, last + 25.0), 2)
            if when > self.duration - 20.0:
                break
            mine = me.group_at(when)
            partners = [p for p in self.people[1:] if p.group_at(when) == mine]
            rest = [p for p in self.people[1:] if p.group_at(when) != mine]
            groups = sorted({g for p in rest if (g := p.group_at(when)) is not None})
            kinds = []
            if rest and mine is not None:
                kinds.append("join")
            if len(partners) >= 2:
                kinds.append("leave")
            if groups:
                kinds.append("walk")
            arrived = None
            for kind in (kinds[int(i)] for i in rng.permutation(len(kinds))):
                target: str | None
                if kind == "join":
                    mover = rest[int(rng.integers(len(rest)))]
                    assert mine is not None
                    target, goals = mine, self.near_group(rng, mine, when)
                elif kind == "leave":
                    mover = partners[int(rng.integers(len(partners)))]
                    target = groups[int(rng.integers(len(groups)))] if groups else None
                    goals = (
                        self.near_group(rng, target, when)
                        if target is not None
                        else self.far_from(rng, when)
                    )
                else:
                    mover = me
                    target = groups[int(rng.integers(len(groups)))]
                    goals = self.near_group(rng, target, when)
                for goal in goals[:8]:
                    arrived = self.move(rng, mover, when, goal)
                    if arrived is not None:
                        mover.member += [(round(when, 3), None), (arrived, target)]
                        break
                if arrived is not None:
                    break
            last = when if arrived is None else arrived

    def far_from(self, rng: np.random.Generator, t: float) -> list[Station]:
        """Free stations away from everybody, for somebody who leaves and joins nobody."""
        everybody = [self.stations[p.blocks[-1].station] for p in self.people]
        pool = [
            s
            for s in self.layout.stations
            if s.kind != "waypoint"
            and all(math.dist(s.xz, other.xz) >= 2.5 for other in everybody)
            and all(math.dist(s.xz, spot) >= _KEEP_M for spot in self.keep_out())
        ]
        return [pool[int(i)] for i in rng.permutation(len(pool))]

    # -- the listener's way ----------------------------------------------------

    def listener_way(self) -> None:
        """The listener's stays and walks as keyframes of position: ``(t, x, y, z, station)``."""
        me = self.people[0]
        standing = round(self.floor_y + self.heights.standing_m, 3)
        frames: list[tuple[float, float, float, float, str | None]] = []
        for number, block in enumerate(me.blocks):
            spot = self.at(block.station, block.posture)
            x, y, z = (round(float(v), 3) for v in spot)
            frames += [
                (block.start_s, x, y, z, block.station),
                (block.end_s, x, y, z, block.station),
            ]
            if number == len(me.blocks) - 1:
                break
            # The walk to the next stay: along the rails, in the time a source takes over
            # them; and between a seat and the floor in one step, the reach a rail has
            # towards a seat, in the time a source takes to rise or to sit.
            moves = me.moves[number]
            travels = [s for s in moves if isinstance(s, Travel)]
            for segment in travels:
                points = list(self.rail_of[segment.rail].points)
                if segment.origin != self.rail_of[segment.rail].a:
                    points = points[::-1]
                up = segment is travels[0] and block.posture == "seated"
                down = segment is travels[-1] and me.blocks[number + 1].posture == "seated"
                inner = points[(1 if up else 0) : (len(points) - 1 if down else len(points))]
                legs = [math.dist(a, b) for a, b in zip(inner[:-1], inner[1:], strict=True)]
                whole, walked = sum(legs), 0.0
                span = segment.end_s - segment.start_s
                frames.append((segment.start_s, inner[0][0], standing, inner[0][1], None))
                for point, leg in zip(inner[1:], legs, strict=True):
                    walked += leg
                    t = round(segment.start_s + span * walked / max(whole, 1e-9), 3)
                    frames.append((t, point[0], standing, point[1], None))
                frames.append((segment.end_s, inner[-1][0], standing, inner[-1][1], None))
        tidy: list[tuple[float, float, float, float, str | None]] = []
        for frame in frames:
            if tidy and frame[0] <= tidy[-1][0]:
                # Two keyframes of one instant are one, the one that names its station.
                if frame[4] is not None:
                    tidy[-1] = (tidy[-1][0], *frame[1:])
                continue
            tidy.append(frame)
        me.frames = tidy

    def frames_at_rest(self) -> list[Keyframe]:
        """The listener's keyframes with a head that does not turn: the way alone."""
        return [
            Keyframe(t, (x, y, z), 0.0, 0.0, 0.0, station)
            for t, x, y, z, station in self.people[0].frames
        ]

    # -- who speaks ------------------------------------------------------------

    def present(self, group: str, t: float) -> list[_Person]:
        """Who is of a group and at rest at ``t``, and stays so for a moment."""
        out = []
        for person in self.people:
            if person is self.people[0] and not self.p.own_voice:
                continue
            block = person.block_at(t)
            if (
                person.group_at(t) == group
                and block is not None
                and block.end_s - t > 1.0
                and (person.group_at(block.end_s - 1e-3) == group or block.end_s >= self.duration)
            ):
                out.append(person)
        return out

    def noise_at(self, person: _Person, t: float) -> float:
        """The fixed sources' level where somebody is, dB SPL in free field."""
        times = np.array([round(t, 3)])
        level = masker_level_at(self.noise, times, self.place_of(person, t)[None, :], every=True)
        return float(level[0])

    def say(
        self,
        rng: np.random.Generator,
        person: _Person,
        start: float,
        wanted: float,
        limit: float,
        event: str = "turn",
        under_db: float = 0.0,
    ) -> Activity | None:
        """One interval of somebody's voice, at the effort the noise there asks for."""
        level = lombard_level_db(person.level_db, self.noise_at(person, start)) - under_db
        if event == "turn" and self.hushed and rng.uniform() < 0.15:
            # A whisper: in a calm dwelling where nobody else talks.
            level = EFFORT_LEVEL_DB["whisper"] + (person.level_db - VOICE_REFERENCE_DB) / 2.0
        effort = effort_of(level)
        low, high = effort_range_db(effort)
        # Kept off the edge between two efforts, which a rounding would cross.
        level = float(np.clip(level, low + 0.1, high - 0.1))
        key = event if event != "turn" else effort
        reader = person.shelves.get(key) or (person.shelves["normal"] if event == "turn" else None)
        if reader is None:
            return None
        spurt = reader.take(start, wanted, limit)
        if spurt is None or spurt.end_s - spurt.start_s < 0.2:
            return None
        said = replace(
            spurt, gain_db=round(level - person.level_db, 2) + 0.0, effort=effort, event=event
        )
        person.spoken.append(said)
        return said

    def talk(self) -> None:
        """Every group's turns, each group by itself."""
        self.noise = self.shell(tuple(self.fixed_sources()))
        named = {group for person in self.people for _, group in person.member}
        self.hushed = self.calm >= 0.85 and len(named - {None}) == 1
        for person in self.people:
            person.spoken = []
            for reader in person.shelves.values():
                reader.which, reader.offset, reader.next = 0, 0.0, 0
        groups = sorted({g for p in self.people for _, g in p.member if g is not None})
        for group in groups:
            self.converse(group)
        for person in self.people:
            kept: list[Activity] = []
            for said in sorted(person.spoken, key=lambda a: a.start_s):
                if not kept or said.start_s >= kept[-1].end_s:
                    kept.append(said)
            person.spoken = kept

    def converse(self, group: str) -> None:
        rng = self.rng(f"conversation:{group}")
        p = self.p
        t = float(rng.uniform(0.3, 2.5))
        previous: _Person | None = None
        while t < self.duration - 1.0:
            here = self.present(group, t)
            if len(here) < 2:
                t, previous = t + 1.0, None
                continue
            others = [person for person in here if person is not previous] or here
            holds = previous in here and rng.uniform() < 0.25
            speaker = (
                previous
                if holds and previous is not None
                else others[int(rng.integers(len(others)))]
            )
            start = round(speaker.free_from(t), 3)
            block = speaker.block_at(start)
            wanted = float(np.clip(rng.lognormal(math.log(p.turn_median_s), 0.7), *p.turn_s))
            said = None
            if block is not None and speaker.group_at(start) == group:
                limit = self.duration if block.end_s >= self.duration else block.end_s - _GUARD_S
                said = self.say(rng, speaker, start, wanted, limit)
            if said is None:
                t, previous = t + 1.0, None
                continue
            span = said.end_s - said.start_s
            for other in here:
                # An acknowledgement inside the turn, from somebody who listens.
                if other is speaker or span < 2.0 or "backchannel" not in other.shelves:
                    continue
                for _ in range(int(rng.poisson(span / p.backchannel_every_s))):
                    when = float(rng.uniform(said.start_s + 0.8, said.end_s - 0.2))
                    if other.free_from(when, 0.7) == when and other.block_at(when + 2.0):
                        self.say(
                            rng, other, round(when, 3), float(rng.uniform(0.3, 0.7)),
                            said.end_s + 0.5, "backchannel", 3.0,
                        )  # fmt: skip
            if rng.uniform() < p.laughter_share * (1.5 - self.calm):
                for other in here:
                    when = said.end_s + float(rng.uniform(-0.2, 0.3))
                    if (
                        "laughter" in other.shelves
                        and rng.uniform() < 0.6
                        and other.free_from(when, 1.8) == when
                        and other.block_at(when + 3.2) is not None
                    ):
                        self.say(
                            rng, other, round(when, 3), float(rng.uniform(0.6, 1.8)),
                            when + 2.0, "laughter",
                        )  # fmt: skip
            gap = float(
                np.clip(rng.normal(p.floor_transfer_mean_s, p.floor_transfer_sd_s), -1.0, 2.0)
            )
            if rng.uniform() < 0.04 + 0.1 * self.calm:
                # A lapse: nobody takes the floor for a while, longer in a calm scene.
                gap += float(rng.uniform(2.0, 8.0))
            t = max(said.end_s + max(gap, -0.5 * span), said.start_s + 0.3)
            previous = speaker

    # -- sources ---------------------------------------------------------------

    def sways(self) -> dict[str, tuple[Sway, ...]]:
        """Everybody's small movements: sinusoids whose sum never repeats."""
        out = {}
        for person in self.people:
            rng = self.rng(f"sway:{person.id}")
            reach = self.draw(rng, self.p.sway_m)
            parts = []
            for axis, share, period in (
                ("x", 0.30, (11.0, 37.0)),
                ("z", 0.25, (13.0, 41.0)),
                ("x", 0.15, (3.0, 8.0)),
                ("z", 0.15, (2.5, 7.0)),
                ("y", 0.15, (2.5, 6.0)),
            ):
                parts.append(
                    Sway(
                        axis, reach * share, self.draw(rng, period), float(rng.uniform(0.0, 360.0))
                    )
                )
            if person.id != LISTENER:
                turn = self.draw(rng, self.p.sway_yaw_deg)
                for share, period in ((0.65, (9.0, 30.0)), (0.35, (2.0, 6.0))):
                    parts.append(
                        Sway(
                            "yaw", turn * share, self.draw(rng, period), float(rng.uniform(0, 360))
                        )
                    )
            out[person.id] = tuple(parts)
        return out

    def turns_of(self, group: str | None) -> list[tuple[float, float, _Person]]:
        """The turns spoken in a group, in the order they start."""
        if group is None:
            return []
        found = [
            (said.start_s, said.end_s, person)
            for person in self.people
            for said in person.spoken
            if said.event == "turn" and person.group_at(said.start_s) == group
        ]
        return sorted(found, key=lambda turn: (turn[0], turn[2].id))

    def neutral(self, person: _Person, block: _Block) -> float:
        """The way a body faces at a stay: a seat's front, or the others of its group."""
        station = self.stations[block.station]
        if station.kind == "seat":
            return station.facing_yaw_deg
        middle = 0.5 * (block.start_s + block.end_s)
        group = person.group_at(middle)
        others = [
            self.place_of(p, middle)
            for p in self.people
            if p is not person and group is not None and p.group_at(middle) == group
        ]
        if not others:
            return station.facing_yaw_deg
        centre = np.mean(others, axis=0)
        return float(
            kinematics.yaw_of_direction(centre[0] - station.xz[0], centre[2] - station.xz[1])
        )

    def look(self, person: _Person, block: _Block, target: np.ndarray, share: float) -> float:
        """The yaw of a head turned towards a point: most of the way, the eyes do the rest."""
        here = self.at(block.station, block.posture)
        bearing = float(kinematics.yaw_of_direction(target[0] - here[0], target[2] - here[2]))
        neutral = self.neutral(person, block)
        turn = (bearing - neutral + 180.0) % 360.0 - 180.0
        return neutral + float(np.clip(share * turn, -80.0, 80.0))

    def facings(self, person: _Person, block: _Block) -> list[tuple[float, float]]:
        """When a talker's head turns during a stay, and to what yaw: who speaks, who is heard."""
        rng = self.rng(f"facing:{person.id}:{block.start_s:.3f}")
        out = [(block.start_s, self.neutral(person, block))]
        for start, _, speaker in self.turns_of(person.group_at(block.start_s + 0.5)):
            if not block.start_s + 0.3 < start < block.end_s - 0.5:
                continue
            share = float(rng.uniform(0.7, 0.95))
            if speaker is person:
                # A talker looks at somebody it speaks to.
                group = person.group_at(start)
                others = [p for p in self.people if p is not person and p.group_at(start) == group]
                if not others:
                    continue
                target = self.place_of(others[int(rng.integers(len(others)))], start)
                when = start - float(rng.uniform(0.0, 0.3))
            elif rng.uniform() < 0.75:
                target = self.place_of(speaker, start)
                when = start + self.draw(rng, self.p.reaction_s)
            else:
                continue
            yaw = self.look(person, block, target, share) + float(rng.normal(0.0, 4.0))
            if (
                when - out[-1][0] >= 0.6
                and when < block.end_s - 0.5
                and abs(yaw - out[-1][1]) >= 8.0
            ):
                out.append((round(when, 3), round(yaw, 2)))
        return out

    def sources(self, sway: dict[str, tuple[Sway, ...]]) -> list[Source]:
        """The voices, the wearer's own, and the fixed sources."""
        me = self.people[0]
        edges = sorted({t for t, _ in me.member} | {0.0, self.duration})
        out = []
        for person in self.people[1:]:
            segments: list[Segment] = []
            for number, block in enumerate(person.blocks):
                turns = self.facings(person, block)
                ends = [t for t, _ in turns[1:]] + [block.end_s]
                for (since, yaw), until in zip(turns, ends, strict=True):
                    facing = Facing("fixed", round(yaw, 2) + 0.0)
                    segments.append(Dwell(block.station, block.posture, since, until, facing))
                if number < len(person.moves):
                    segments += person.moves[number]
            cuts = sorted({*edges, *(t for t, _ in person.member)})
            roles: list[Role] = []
            for since, until in zip(cuts[:-1], cuts[1:], strict=True):
                middle = 0.5 * (since + until)
                group = person.group_at(middle)
                role = (
                    "conversation"
                    if group is not None and group == me.group_at(middle)
                    else "outside"
                )
                if roles and (roles[-1].role, roles[-1].group) == (role, group):
                    roles[-1] = replace(roles[-1], end_s=until)
                else:
                    roles.append(Role(since, until, role, group))
            out.append(
                Source(
                    id=person.id,
                    kind="voice",
                    directivity=Directivity(self.voice, self.voice != "omni"),
                    gain_db=round(person.level_db - VOICE_REFERENCE_DB, 2) + 0.0,
                    turn_rate_deg_s=person.turn_rate,
                    segments=tuple(segments),
                    activity=tuple(person.spoken),
                    level_spl_1m_db=person.level_db,
                    sway=sway[person.id],
                    roles=tuple(roles),
                )
            )
        if self.p.own_voice:
            out.append(
                Source(
                    id="own_voice",
                    kind="own_voice",
                    directivity=Directivity(self.voice, self.voice != "omni"),
                    gain_db=round(me.level_db - VOICE_REFERENCE_DB, 2) + 0.0,
                    turn_rate_deg_s=0.0,
                    segments=(),
                    activity=tuple(me.spoken),
                    level_spl_1m_db=me.level_db,
                    # The mouth: 9 cm in front of the centre of the head and 5 cm under it.
                    attach=Attach(LISTENER, "mouth", 0.0, (0.09, 0.0, -0.05)),
                )
            )
        return out + self.fixed_sources()

    def bodies(self, recipe: Recipe) -> list[Source]:
        """Everybody's breaths, clothes and chair at the mouth, and footsteps under a walk."""
        if not self.p.body_noises:
            return []
        out = []
        for person in self.people:
            rng = self.rng(f"body:{person.id}")
            mine = person.id == LISTENER
            shelf = self.shelf("body")
            spurts, t = [], float(rng.uniform(1.0, 8.0))
            while shelf and t < self.duration - 2.0:
                block = person.block_at(t)
                span = float(rng.uniform(0.3, 1.5))
                # Silent on the move, and for as long before it as a sound rings.
                if block is not None and (
                    block.end_s >= self.duration or t + span + _GUARD_S < block.end_s
                ):
                    spurts.append((round(t, 3), round(min(t + span, self.duration), 3)))
                t += span + float(rng.uniform(4.0, 16.0))
            turned = float(rng.uniform(-180.0, 180.0))
            if spurts:
                level = round(self.draw(rng, NOISE_LEVELS_DB["body"]), 2)
                first = int(rng.integers(len(shelf)))
                stored = shelf[first].spl_1m_db
                out.append(
                    Source(
                        id=f"{person.id}_body",
                        kind="noise",
                        subtype="body",
                        # A pattern turned any way: it costs nothing, and no two are alike.
                        directivity=Directivity(self.voice, self.voice != "omni"),
                        gain_db=round(level - (level if stored is None else stored), 2) + 0.0,
                        turn_rate_deg_s=0.0,
                        segments=(),
                        activity=tuple(_Scene.cut(spurts, [shelf[first]])),
                        level_spl_1m_db=level,
                        attach=Attach(LISTENER, "mouth", turned, (0.0, 0.0, -0.2))
                        if mine
                        else Attach(person.id, "mouth", turned),
                    )
                )
            walks = [
                (b.end_s, n.start_s)
                for b, n in zip(person.blocks[:-1], person.blocks[1:], strict=True)
            ]
            steps = self.shelf("steps") if walks else []
            if not steps:
                continue
            level = round(self.draw(rng, NOISE_LEVELS_DB["steps"]), 2)
            stored = steps[0].spl_1m_db
            source = Source(
                id=f"{person.id}_steps",
                kind="noise",
                subtype="steps",
                directivity=Directivity("omni", False),
                gain_db=round(level - (level if stored is None else stored), 2) + 0.0,
                turn_rate_deg_s=0.0,
                segments=(),
                activity=(),
                level_spl_1m_db=level,
                attach=Attach(LISTENER if mine else person.id, "floor", 0.0, None, self.p.stride_m),
            )
            # A footfall sounds when the walker comes to it: where the source hops.
            falls = []
            probe = replace(recipe, sources=(*recipe.sources, source))
            for leave, arrive in walks:
                times = np.arange(leave, arrive + _STEP_S, _STEP_S)
                where = kinematics.source_state(probe, source.id, times).position
                hops = np.flatnonzero(np.linalg.norm(np.diff(where, axis=0), axis=1) > 1e-6) + 1
                starts = [round(float(times[i]), 3) for i in hops]
                for start, following in zip(starts, [*starts[1:], arrive + 0.3], strict=True):
                    falls.append((start, round(min(start + 0.3, following - 0.01), 3)))
            falls = [(a, b) for a, b in falls if b - a > 0.05 and b <= self.duration]
            if falls:
                out.append(replace(source, activity=tuple(_Scene.cut(falls, steps))))
        return out

    # -- the listener's head ---------------------------------------------------

    def gaze(self, recipe: Recipe, attend: float) -> tuple[list[Gaze], float]:
        """What the listener looks at over the scene, and the share it is the active talker."""
        rng = self.rng("listener:gaze")
        me = self.people[0]
        grid = np.arange(0.0, self.duration, _STEP_S)
        # Who of the listener's conversation holds the floor at each step: the last to start.
        active = np.full(grid.size, -1, dtype=np.int64)
        index = {person.id: n for n, person in enumerate(self.people)}
        for block in me.blocks:
            for start, end, person in self.turns_of(me.group_at(block.start_s + 0.5)):
                if person is not me and block.start_s <= start < block.end_s:
                    active[(grid >= start) & (grid < min(end, block.end_s))] = index[person.id]
        heard = [s.id for s in recipe.sources if s.attach is None and s.activity]
        onsets = sorted(
            (a.start_s, s.id)
            for s in recipe.sources
            if s.kind in ("noise", "media_voice") and s.attach is None
            for a in s.activity[:1]
            if a.start_s > 1.0
        )
        out: list[Gaze] = []

        def add(start: float, end: float, mode: str, target: str | None = None) -> None:
            start, end = round(start, 3), round(end, 3)
            if end - start >= 0.1:
                out.append(Gaze(start, end, mode, target))

        for number, block in enumerate(me.blocks):
            group = me.group_at(block.start_s + 0.5)
            partners = [
                p.id
                for p in self.people[1:]
                if group is not None and p.group_at(block.start_s + 0.5) == group
            ]
            t, last = block.start_s, None
            while t < block.end_s - 0.2:
                u, long = rng.uniform(), rng.uniform(6.0, 20.0)
                short, kind = rng.uniform(1.5, 6.0), int(rng.integers(3))
                started = next((who for when, who in onsets if t - 1.0 <= when <= t + 0.5), None)
                if started is not None and not (out and out[-1].target == started):
                    end = min(t + short, block.end_s)
                    add(t, end, "event", started)
                elif partners and u < attend:
                    end = min(t + long, block.end_s)
                    # The head follows who holds the floor, a reaction late.
                    span = (grid >= t) & (grid < end)
                    holder = active[span]
                    current = last or partners[int(rng.integers(len(partners)))]
                    since = t
                    for step in np.flatnonzero(np.diff(holder) != 0) + 1:
                        who = int(holder[step])
                        if who < 0 or self.people[who].id == current:
                            continue
                        when = float(grid[span][step]) + self.draw(rng, self.p.reaction_s)
                        when = max(when, since + 0.1)
                        if when >= end - 0.1:
                            break
                        add(since, when, "talker", current)
                        current, since = self.people[who].id, when
                    add(since, end, "talker", current)
                    last = current
                else:
                    end = min(t + short, block.end_s)
                    choice = [h for h in heard if h != last]
                    if kind == 0 and choice:
                        add(t, end, "glance", choice[int(rng.integers(len(choice)))])
                    else:
                        add(t, end, "reading" if kind == 1 else "away")
                t = end
            if number < len(me.blocks) - 1:
                add(block.end_s, me.blocks[number + 1].start_s, "walk")
        # The share: of the steps where somebody else holds the floor, those looked at.
        looked = np.full(grid.size, -2, dtype=np.int64)
        for interval in out:
            if interval.mode == "talker" and interval.target is not None:
                span = (grid >= interval.start_s) & (grid < interval.end_s)
                looked[span] = index[interval.target]
        talking = active >= 0
        share = float((looked[talking] == active[talking]).mean()) if talking.any() else 1.0
        return out, share

    def head(self, recipe: Recipe) -> tuple[list[Keyframe], tuple[Gaze, ...]]:
        """The listener's keyframes with a head that follows the gaze and never quite stops."""
        me = self.people[0]
        # The share of the attentive spells that gives the share of the time asked for.
        low, high, best = 0.0, 1.0, None
        for _ in range(7):
            attend = 0.5 * (low + high)
            gaze, share = self.gaze(recipe, attend)
            if best is None or abs(share - self.p.gaze_share) < abs(best[1] - self.p.gaze_share):
                best = (gaze, share)
            low, high = (attend, high) if share < self.p.gaze_share else (low, attend)
        assert best is not None
        gaze = best[0]
        rng = self.rng("listener:head")
        knots = np.array([frame[0] for frame in me.frames])
        places = np.array([frame[1:4] for frame in me.frames], dtype=float)

        def place(t: float) -> np.ndarray:
            return np.array([np.interp(t, knots, places[:, axis]) for axis in range(3)])

        by_id = {person.id: person for person in self.people}
        slow, phase = float(rng.uniform(17.0, 43.0)), float(rng.uniform(0.0, 2.0 * math.pi))
        yaw, pitch = self.neutral(me, me.blocks[0]), 0.0
        turns: list[tuple[float, float, float, float]] = [(0.0, yaw, pitch, 0.0)]

        def turn(t: float, aim: float, tilt: float, roll: float) -> None:
            """A knot of the head, no further from the last than the head turns in the time."""
            nonlocal yaw, pitch
            reach = 300.0 * (t - turns[-1][0])
            if reach <= 0.0:
                return
            yaw += float(np.clip(aim - yaw, -reach, reach))
            pitch += float(np.clip(tilt - pitch, -reach, reach))
            turns.append((t, yaw, pitch, float(np.clip(roll, -reach, reach))))

        for interval in gaze:
            block = me.block_at(interval.start_s + 0.05)
            here = place(interval.start_s)
            aim, tilt = yaw, 0.0
            if interval.mode == "walk" or block is None:
                # Along the way, leg after leg, with a look to the side now and then.
                last = interval.start_s
                for when in knots[(knots >= interval.start_s) & (knots < interval.end_s)]:
                    ahead = place(min(float(when) + 0.6, interval.end_s)) - place(float(when))
                    if math.hypot(ahead[0], ahead[2]) < 0.05 or when - last < 0.4:
                        continue
                    aim = float(kinematics.yaw_of_direction(ahead[0], ahead[2]))
                    aim = yaw + (aim - yaw + 180.0) % 360.0 - 180.0
                    turn(
                        float(when) + 0.4,
                        aim + float(rng.normal(0.0, 6.0)),
                        float(rng.normal(-5.0, 4.0)),
                        0.0,
                    )
                    last = float(when)
                continue
            if interval.target is not None:
                if interval.target in by_id:
                    target = self.place_of(by_id[interval.target], interval.start_s)
                else:
                    target = kinematics.source_state(
                        recipe, interval.target, np.array([interval.start_s])
                    ).position[0]
                aim = self.look(me, block, target, float(rng.uniform(0.7, 0.95)))
                flat = math.hypot(target[0] - here[0], target[2] - here[2])
                tilt = 0.6 * math.degrees(math.atan2(target[1] - here[1], max(flat, 1e-6)))
                aim += float(rng.normal(0.0, 5.0))
            elif interval.mode == "reading":
                aim = self.neutral(me, block) + float(rng.uniform(-15.0, 15.0))
                tilt = float(rng.uniform(-45.0, -25.0))
            else:
                aim = self.neutral(me, block) + float(rng.uniform(-70.0, 70.0))
                tilt = float(rng.uniform(-10.0, 15.0))
            aim = yaw + (aim - yaw + 180.0) % 360.0 - 180.0
            tilt = float(np.clip(tilt, -50.0, 30.0))
            rate = self.draw(rng, self.p.listener_turn_rate_deg_s)
            span = interval.end_s - interval.start_s
            took = min(max(abs(aim - yaw), abs(tilt - pitch)) / rate + 0.2, 0.8 * span)
            turn(interval.start_s, yaw, pitch, turns[-1][3])
            turn(interval.start_s + took, aim, tilt, 0.0)
            # At rest on its target the head still moves, by an amount that swells and fades.
            t = interval.start_s + took + float(rng.uniform(0.6, 1.6))
            while t < interval.end_s - 0.3:
                size = 1.0 + 3.0 * abs(math.sin(2.0 * math.pi * t / slow + phase))
                wander = rng.normal(0.0, 1.0, 3) * (size, size / 2.0, size / 3.0)
                turn(t, aim + float(wander[0]), tilt + float(wander[1]), float(wander[2]))
                t += float(rng.uniform(0.6, 1.6))
        if self.duration > turns[-1][0]:
            turns.append((self.duration, yaw, pitch, 0.0))
        # No keyframe between a seat and the floor: it would be on neither.
        sitting = [
            (a[0], b[0])
            for a, b in zip(me.frames[:-1], me.frames[1:], strict=True)
            if a[4] != b[4] and "seat" in {self.stations[s].kind for s in (a[4], b[4]) if s}
        ]
        rows = sorted({round(t, 3): (y, p, r) for t, y, p, r in turns}.items())
        turn_t = np.array([t for t, _ in rows])
        angles = np.array([values for _, values in rows], dtype=float)
        moments = sorted(
            {
                *knots.tolist(),
                *(t for t in turn_t.tolist() if not any(a < t < b for a, b in sitting)),
            }
        )
        named = {frame[0]: frame[4] for frame in me.frames}
        frames = []
        for t in moments:
            block = next((b for b in me.blocks if b.start_s <= t <= b.end_s), None)
            station = named.get(t) if t in named else (block.station if block is not None else None)
            x, y, z = (round(float(v), 3) for v in place(t))
            frames.append(
                Keyframe(
                    t,
                    (x, y, z),
                    float(np.interp(t, turn_t, angles[:, 0])),
                    float(np.interp(t, turn_t, angles[:, 1])),
                    float(np.interp(t, turn_t, angles[:, 2])),
                    station,
                )
            )
        return frames, tuple(gaze)
