"""Levels in free field: what a scene's sources are at a point, before any room.

A version 2 recipe says what every source is at 1 m (``level_spl_1m_db``)
and what each interval adds (``gain_db``), on the clip library's scale. From
those and the distances, this module gives the level of a source at a point
with nothing but the ``1 / d`` of free air: no wall, no reverberation, no
directivity. It is what the generator sets a talker's vocal effort from and
what rule 16 of the format holds the listener's conversation to. A room adds
to both the speech and the noise; a wall between a noise and the listener
takes from the noise alone, so the estimate errs towards more noise than
there is behind a wall, and says nothing of the room's own reverberation.

**Vocal effort** is ISO 9921's ladder, A-weighted speech level at 1 m in
front of the mouth: relaxed 54 dB, normal 60, raised 66, loud 72 (annex A),
with a whisper put at 45 dB under them. The clip library stores a voice at
the normal effort, 60 dB (``docs/formats/clip-library.md``). **The Lombard
rule**: a talker raises the voice by about half a decibel for each decibel
of noise above about 45 dB at the talker (Lazarus 1986, the review ISO 9921
takes its rule from: 0.3 to 0.6 dB a decibel), up to the loud effort.
"""

from __future__ import annotations

import numpy as np

from reverberate.scenes.kinematics import anchor_at, listener_state
from reverberate.scenes.recipe import LISTENER, Recipe, Source

__all__ = [
    "EFFORT_LEVEL_DB",
    "LOMBARD_ONSET_DB",
    "LOMBARD_SLOPE",
    "NEAREST_M",
    "VOICE_REFERENCE_DB",
    "conversation_snr",
    "effort_of",
    "effort_range_db",
    "in_conversation",
    "level_at",
    "lombard_level_db",
    "masker_level_at",
]

#: A-weighted speech level at 1 m, dB, by vocal effort: ISO 9921:2003, annex A, and a
#: whisper, which the standard does not rank, 15 dB under the normal effort.
EFFORT_LEVEL_DB = {"whisper": 45.0, "relaxed": 54.0, "normal": 60.0, "raised": 66.0, "loud": 72.0}
#: What the clip library stores a voice at: the normal effort.
VOICE_REFERENCE_DB = 60.0
#: The noise at the talker above which the voice rises, and by how much a decibel.
LOMBARD_ONSET_DB = 45.0
LOMBARD_SLOPE = 0.5
#: No source is taken nearer than this: the ``1 / d`` law is a far field's.
NEAREST_M = 0.3

_LADDER = tuple(EFFORT_LEVEL_DB.items())


def effort_range_db(effort: str) -> tuple[float, float]:
    """The levels at 1 m an effort names: half way to the efforts either side."""
    names = [name for name, _ in _LADDER]
    index = names.index(effort)
    level = _LADDER[index][1]
    low = -np.inf if index == 0 else 0.5 * (level + _LADDER[index - 1][1])
    high = np.inf if index == len(_LADDER) - 1 else 0.5 * (level + _LADDER[index + 1][1])
    return float(low), float(high)


def effort_of(level_db: float) -> str:
    """The effort a speech level at 1 m is spoken at: the nearest step of the ladder."""
    for name, _ in _LADDER:
        if level_db < effort_range_db(name)[1]:
            return name
    return _LADDER[-1][0]


def lombard_level_db(base_db: float, noise_db: float) -> float:
    """The speech level at 1 m of a talker whose own level is ``base_db``, in this noise."""
    raised = base_db + LOMBARD_SLOPE * max(0.0, noise_db - LOMBARD_ONSET_DB)
    return float(min(raised, max(base_db, EFFORT_LEVEL_DB["loud"])))


def _gain_at(source: Source, times: np.ndarray) -> np.ndarray:
    """The interval gain in force at each time, dB; NaN where the source is silent."""
    gain = np.full(times.size, np.nan)
    for interval in source.activity:
        gain[(times >= interval.start_s) & (times < interval.end_s)] = interval.gain_db
    return gain


def level_at(recipe: Recipe, source: Source, times: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The source's level at ``points`` and ``times``, dB SPL in free field; ``-inf`` if silent."""
    if source.level_spl_1m_db is None:
        raise ValueError(f"source {source.id} says no level: the recipe is not of version 2")
    gain = _gain_at(source, times)
    away = np.linalg.norm(anchor_at(recipe, source.id, times) - points, axis=1)
    level = source.level_spl_1m_db + gain - 20.0 * np.log10(np.maximum(away, NEAREST_M))
    out: np.ndarray = np.where(np.isnan(gain), -np.inf, level)
    return out


def in_conversation(source: Source, times: np.ndarray) -> np.ndarray:
    """Whether a voice is of the listener's conversation at each time."""
    inside = np.zeros(times.size, dtype=bool)
    for role in source.roles:
        if role.role == "conversation":
            inside |= (times >= role.start_s) & (times < role.end_s)
    return inside


def _masks(source: Source, times: np.ndarray) -> np.ndarray:
    """Whether the source is a masker of the listener's conversation at each time."""
    if source.attach is not None and source.attach.to == LISTENER:
        return np.zeros(times.size, dtype=bool)
    if source.kind == "own_voice":
        return np.zeros(times.size, dtype=bool)
    if source.kind == "voice":
        return ~in_conversation(source, times)
    return np.ones(times.size, dtype=bool)


def masker_level_at(
    recipe: Recipe, times: np.ndarray, points: np.ndarray, *, every: bool = False
) -> np.ndarray:
    """The power sum, in dB SPL, of everything that masks the conversation, at ``points``.

    The noises, the programmes and the voices outside the listener's
    conversation; not the listener's own voice and body. ``every`` counts
    every source the recipe holds, whatever it is: what a talker of a
    recipe of noises alone raises the voice against.
    """
    power = np.zeros(times.size)
    for source in recipe.sources:
        masks = np.ones(times.size, dtype=bool) if every else _masks(source, times)
        if not masks.any() or not source.activity:
            continue
        level = level_at(recipe, source, times, points)
        power += np.where(masks, 10.0 ** (level / 10.0), 0.0)
    with np.errstate(divide="ignore"):
        out: np.ndarray = 10.0 * np.log10(power)
    return out


def conversation_snr(recipe: Recipe) -> list[tuple[str, int, float]]:
    """Per turn of the listener's conversation: the source, the interval, its least SNR in dB.

    A turn is an interval whose ``event`` is ``turn``, of a voice that is of
    the conversation when it starts. The ratio is read at the turn's start,
    middle and end, at the listener's head, and the least is kept; an
    instant at which the voice is no longer of the conversation, the
    listener having left in the middle of the turn, is not read.
    """
    rows: list[tuple[str, int, float]] = []
    for source in recipe.sources:
        if source.kind != "voice" or not source.activity:
            continue
        starts = np.array([interval.start_s for interval in source.activity])
        ends = np.array([interval.end_s for interval in source.activity])
        mine = in_conversation(source, starts) & np.array(
            [interval.event == "turn" for interval in source.activity]
        )
        if not mine.any():
            continue
        which = np.flatnonzero(mine)
        span = ends[which] - starts[which]
        times = np.concatenate([starts[which] + share * span for share in (0.0, 0.5, 0.999)])
        head = listener_state(recipe, times).position
        ratio = level_at(recipe, source, times, head) - masker_level_at(recipe, times, head)
        ratio = np.where(in_conversation(source, times), ratio, np.inf)
        least = ratio.reshape(3, which.size).min(axis=0)
        rows += [(source.id, int(n), float(v)) for n, v in zip(which, least, strict=True)]
    return rows
