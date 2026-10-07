"""What a recipe of version 2 says of a source, as the trace writes it into the pack.

A version 2 recipe (``docs/formats/scene-recipe.md``) holds sources no wave
solve answers, and says things of a source the first format did not. This
module is where the trace turns them into what the pack holds
(``docs/formats/scene-pack.md``, "What version 2 adds to a source"):

- :func:`said` gives the attributes of a source's group: that the mirror
  renders it alone and why, whether its direct sound is in the pack, its
  level at 1 m, what carries it, the opening it comes in by;
- :func:`without_direct` takes the direct sound out of the arrivals of a
  source at the listener's own mouth;
- :func:`pane_gain_db` and :func:`coloured` lay a closed window's
  transmission on a source, octave band by octave band.

**The wearer's own voice.** The mouth is 0.10 m from where the field is
taken: its direct sound at the device is the path round the head and through
the bone, which no room simulation without a head has. So the pack holds
the room's answer alone, the reflections and the late part from a source at
the mouth, and the direct path's row is kept **at no gain**: the late part
still starts 10 ms after it, as every source's does. The stem of an
``own_voice`` is therefore what the room gives back, and the device stage
adds the dry clip through a mouth to microphone response
(``docs/open-questions/recipes-v2.md``, section 4).

**A closed window** is its pane radiating. The recipe's level is what
comes in, at 1 m from the fixture, 0.25 m inside the pane; the trace
renders it as one source there, by the mirror alone, with the glazing's
sound reduction index laid on it a band at a time about the single number
the recipe's level was made with. The figures are EN 12758's generic
ones for float glass, and ISO 717-1's traffic spectrum; **both are quoted
from memory of the standards and are to be checked against them before
anything is published from them** (they agree with each other: the
spectrum through the indices gives the single numbers the standard prints,
25 and 26 dB). What this leaves out is in the format's document.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.render.pack import KIND_DIRECT
from reverberate.scenes import Recipe, wave_band
from reverberate.scenes.recipe import LISTENER, Source

__all__ = [
    "CARRIED",
    "CLOSED",
    "GLAZING",
    "GLAZING_R_DB",
    "NEAR",
    "TRAFFIC_DB",
    "coloured",
    "has_direct",
    "pane_gain_db",
    "said",
    "traffic_reduction_db",
    "without_direct",
]

#: Why the mirror renders a source alone, as the pack says it.
CARRIED = "carried"
CLOSED = "closed opening"
NEAR = "near a surface"

#: The sound reduction index of a glazing, dB, in the octave bands 125 Hz to 4 kHz: EN
#: 12758:2011, generic values for float glass. ``single``: one pane of 4 mm, ``R_w`` 29
#: (C -2, C_tr -3). ``double``: 4 mm, a cavity of 6 to 16 mm, 4 mm, ``R_w`` 29 (C -1,
#: C_tr -4): two panes are no better than one in that single number, and worse at
#: 250 Hz, where the cavity resonates. The band at 8 kHz takes the value at 4 kHz.
GLAZING_R_DB: dict[str, tuple[float, ...]] = {
    "single": (17.0, 20.0, 26.0, 32.0, 33.0, 26.0),
    "double": (21.0, 17.0, 25.0, 35.0, 37.0, 31.0),
}
#: What a closed window is taken to be unless the bundle says (``trace.glazing``):
#: the glazing the recipes' levels were drawn for (``recipes-v2.md``, section 3).
GLAZING = "double"
#: ISO 717-1, spectrum No. 2: urban road traffic, A-weighted, in the octave bands 125 Hz
#: to 2 kHz, dB re the whole. What ``C_tr`` is computed against.
TRAFFIC_DB = (-14.0, -10.0, -7.0, -4.0, -6.0)


def traffic_reduction_db(glazing: str = GLAZING) -> float:
    """``R_w + C_tr`` of a glazing: what it takes from road traffic, A-weighted, in dB.

    From its indices and :data:`TRAFFIC_DB`, as ISO 717-1 defines it, and
    not rounded: 25.1 dB for the double glazing, 25.7 for the single.
    """
    index = np.asarray(_index(glazing)[: len(TRAFFIC_DB)])
    through = 10.0 ** ((np.asarray(TRAFFIC_DB) - index) / 10.0)
    return float(-10.0 * np.log10(through.sum()))


def _index(glazing: str) -> tuple[float, ...]:
    if glazing not in GLAZING_R_DB:
        raise ValueError(f"a glazing is one of {sorted(GLAZING_R_DB)}, not {glazing!r}")
    return GLAZING_R_DB[glazing]


def pane_gain_db(glazing: str = GLAZING) -> tuple[float, ...]:
    """What a closed window lays on its noise, dB, a band of the pack's octave bands.

    ``(R_w + C_tr) - R`` in each band: **the level the recipe says is
    kept for road traffic** (its ``level_spl_1m_db`` is the street's less
    one number for the glazing), and each band departs from it by what the
    glazing takes there more or less than that number. Positive under
    500 Hz, where a pane holds little back, and negative above: +4, +8, 0,
    -10, -12, -6, -6 dB for the double glazing.
    """
    index = _index(glazing)
    whole = traffic_reduction_db(glazing)
    made = [whole - index[min(band, len(index) - 1)] for band in range(len(OCTAVE_BANDS))]
    return tuple(round(float(value), 2) for value in made)


def has_direct(source: Source) -> bool:
    """Whether the pack holds the direct sound of ``source``: not of one at the listener's mouth."""
    attach = source.attach
    return not (attach is not None and attach.to == LISTENER and attach.at == "mouth")


def without_direct(early: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The arrivals with the direct path at no gain: its row stays, and its delay.

    ``early`` is the datasets of a source's ``early`` group
    (:meth:`reverberate.mirror.moving.EarlyTable.pack`). The row is kept so
    that the late part starts where it does for every source, 10 ms after
    the first arrival, and not at its histogram's first bin, which holds
    every ray as it leaves a mouth that is inside the receiver's sphere.
    """
    gain = np.array(early["gain"], copy=True)
    gain[np.asarray(early["kind"]) == KIND_DIRECT] = 0.0
    return {**early, "gain": gain}


def coloured(
    early: dict[str, np.ndarray],
    tail: dict[str, np.ndarray],
    gain_db: tuple[float, ...],
    bank_band: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A source's arrivals and late part with ``gain_db`` laid on them, a band at a time.

    ``early/gain`` is a pressure a band of the pack's octave bands;
    ``tail/scale`` is an energy a band of the bank, ``bank_band`` saying
    which octave band each reads (:func:`reverberate.render.pack.band_map`).
    The engine then needs to know nothing of it.
    """
    colour = np.asarray(gain_db, dtype=float)
    pressure = (10.0 ** (colour / 20.0)).astype(np.float32)
    power = 10.0 ** (colour[np.asarray(bank_band, dtype=int)] / 10.0)
    return (
        {**early, "gain": (np.asarray(early["gain"]) * pressure[None, :]).astype(np.float32)},
        {**tail, "scale": np.asarray(tail["scale"], dtype=np.float64) * power[None, :]},
    )


def said(
    recipe: Recipe,
    source: Source,
    *,
    near: tuple[str, ...] = (),
    glazing: str = GLAZING,
) -> dict[str, Any]:
    """The attributes of a source's group that version 2 adds: keywords of ``pack.Source``.

    ``near`` names the sources the plan gave to the mirror for standing too
    near a surface (:attr:`reverberate.trace.plan.Profile.mirror_only`).
    A source of a version 1 recipe says nothing: the empty mapping.
    """
    made: dict[str, Any] = {}
    if recipe.schema_version < 2:
        if source.id in near:
            made.update(mirror_only=True, mirror_only_because=NEAR)
        return made
    if source.level_spl_1m_db is not None:
        made["level_spl_1m_db"] = float(source.level_spl_1m_db)
    closed = source.opening is not None and source.opening.state == "closed"
    if source.attach is not None:
        made.update(carried_by=source.attach.to, carried_at=source.attach.at)
    if source.opening is not None:
        made.update(opening_object=source.opening.object, opening_state=source.opening.state)
    if not wave_band(source) or source.id in near:
        why = CARRIED if source.attach is not None else CLOSED if closed else NEAR
        made.update(mirror_only=True, mirror_only_because=why)
    if not has_direct(source):
        made["direct"] = False
    if closed:
        made.update(
            band_gain_db=pane_gain_db(glazing),
            band_gain_of=f"a closed window's {glazing} glazing, EN 12758, about its R_w + C_tr",
        )
    return made
