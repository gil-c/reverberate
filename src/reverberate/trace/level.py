"""The levelling at the crossover, per pair and per step: the three scalars of a pack.

:func:`reverberate.mirror.hybrid.blend` joins one wave response and one
mirror response at a point: it levels the mirror onto the wave field by one
scalar read over the crossover's octave (``seam_db``), and it anchors on
the wave response's loudest sample the window inside which the two are
joined in pressure. A pack holds a step's responses apart, so the same
levelling is written as three numbers the engine applies:

- ``low/seam_db`` of a pair (:func:`pair_seam_db`): ``seam_db`` between the
  pair's wave response, with its air and before its masks, as ``blend``
  reads it, and the mirror rendered at the same pair
  (:func:`mirror_omni`): the pair's own paths and the tail the pack gives
  that pair, through :func:`reverberate.mirror.render.render_point`, on the
  wave field's clock and scale;
- ``low/onset_s`` of a pair (:func:`pair_low`): the loudest sample of that
  wave response's channel 0, read at 48 kHz;
- ``level/high_gain_db`` and ``level/onset_s`` of a step
  (:func:`step_levels`): the alignment's gain and the step's pairs' seams,
  and the step's first arrival plus what its pairs' onsets trail their own
  first arrival by, both weighted as the engine weighs the pairs.

At rest, the source on a solved position and the head on a cell, a step
reads one pair with weight one and the three are ``blend``'s own.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.audio import Atmosphere
from reverberate.mirror.hybrid import Crossover, seam_db
from reverberate.mirror.moving import EarlyTable
from reverberate.mirror.rays import Histogram
from reverberate.mirror.render import _band_map, band_pulse_energy, render_point
from reverberate.mirror.tails import TailTable
from reverberate.spatial.lowband import (
    LOW_DURATION_S,
    LOW_RATE_HZ,
    decimate,
    low_side,
    onset_s,
    with_air,
)
from reverberate.spatial.translate import MODE_FUSED
from reverberate.trace.assets import FIELD_RATE_HZ, MirrorAssets

__all__ = [
    "first_arrival_s",
    "mirror_omni",
    "pair_low",
    "pair_seam_db",
    "step_levels",
]


def pair_low(
    cached: Any,
    crossover: Crossover,
    atmosphere: Atmosphere,
    *,
    sound_speed_m_s: float,
    xp: Any = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """A pair's response in the cache form as the pack keeps it.

    Returns ``low/ir`` (``[channel, 4800]`` float32), ``low/onset_s``, and
    channel 0 of the response with its air and before its masks, at 4 kHz:
    what the seam is read on.
    """
    from reverberate.compute import to_numpy

    aired = with_air(cached, LOW_RATE_HZ, atmosphere, sound_speed_m_s=sound_speed_m_s, xp=xp)
    onset = onset_s(aired[0], LOW_RATE_HZ, xp=xp)
    stored = low_side(aired, LOW_RATE_HZ, crossover, xp=xp)
    return (
        np.asarray(to_numpy(stored), dtype=np.float32),
        float(onset),
        np.asarray(to_numpy(aired[0]), dtype=float),
    )


def first_arrival_s(early: EarlyTable, row: int) -> float:
    """The smallest delay of a step's rows; the straight line's where it has none."""
    rows = early.rows(row)
    if rows.stop > rows.start:
        return float(early.delay_s[rows].min())
    straight = float(np.linalg.norm(early.listener[row] - early.source[row]))
    return straight / early.sound_speed_m_s


def mirror_omni(
    early: EarlyTable,
    row: int,
    tail: TailTable | None,
    tail_row: int,
    assets: MirrorAssets,
    *,
    seed: int,
    xp: Any = np,
) -> np.ndarray:
    """Channel 0 of the mirror at one pair, on the wave field's clock and scale: 1.2 s at 48 kHz.

    ``early`` and ``tail`` are tables whose "steps" are pairs, the source on
    the pair's solved position and the head on its cell; the pair is step
    ``row`` of the first and ``tail_row`` of the second. The response is
    :func:`reverberate.mirror.render.render_point` at order 0, whose channel
    0 is that of any order: the pair's image paths and its diffracted onset,
    the tail of the histograms the pack gives the pair, each on its own
    scale and summed in energy, then the air, the low cut and the
    signature; shifted by the alignment's lead and multiplied by its gain,
    as :func:`reverberate.mirror.files.write_field` writes a field.
    """
    settings = assets.settings
    render = replace(
        settings.render, order=0, sample_rate_hz=FIELD_RATE_HZ, duration_s=LOW_DURATION_S
    )
    c = settings.sound_speed_m_s
    radius = settings.rays.receiver_radius_m
    images, onset = early.paths(row)
    direct = images.order == 0
    bins = int(np.ceil(settings.rays.duration_s / settings.rays.bin_s))
    channels = (settings.rays.order + 1) ** 2
    energy = np.zeros((bins, len(OCTAVE_BANDS)))
    moments = np.zeros((bins, len(OCTAVE_BANDS), channels))
    scale: np.ndarray | None = None
    if tail is not None:
        rows, share = tail.weights(tail_row)
        if rows.size:
            # A histogram is multiplied by its own scale before the weighted sum. The
            # renderer applies one scale: the others are folded into the energies.
            if bool(direct.any()):
                _, picks = _band_map(FIELD_RATE_HZ)
                distance = float(images.length_m[direct][0])
                expected = radius**2 / (4.0 * max(distance, 1.05 * radius) ** 2)
                amplitude = images.gain[np.flatnonzero(direct)[0]][picks]
                own = np.asarray(amplitude**2 * band_pulse_energy(FIELD_RATE_HZ) / expected)
            else:
                own = np.asarray(tail.scale[rows[0]], dtype=float)
            if float(own[0]) > 0.0:
                factor = share * tail.scale[rows][:, 0] / float(own[0])
                energy = np.tensordot(factor, tail.energy[rows], axes=1)
                moments = np.tensordot(share, tail.moments[rows], axes=1)
                scale = own
    heard = float(energy.sum()) > 0.0
    histogram = Histogram(
        energy=energy[None],
        moments=moments[None],
        hits=np.full((1, bins), 1 if heard else 0, dtype=np.int64),
        bin_s=settings.rays.bin_s,
        bands_hz=tuple(OCTAVE_BANDS),
        order=settings.rays.order,
        rays=settings.rays.rays,
    )
    straight = float(np.linalg.norm(early.listener[row] - early.source[row])) / c
    response, _ = render_point(
        0,
        images,
        histogram,
        render,
        tail_gain_db=np.asarray(settings.parameters.tail_gain_db, dtype=float),
        receiver_radius_m=radius,
        sound_speed_m_s=c,
        seed=seed,
        fallback=None if bool(direct.any()) else (scale, straight, onset),
        signature=assets.signature,
        xp=xp,
    )
    signal = np.asarray(response.signals[0], dtype=float)
    lead = assets.lead_samples
    shifted = np.zeros_like(signal)
    if lead >= 0:
        shifted[lead:] = signal[: signal.size - lead]
    else:
        shifted[:lead] = signal[-lead:]
    return shifted * assets.gain


def pair_seam_db(aired_omni: np.ndarray, mirror: np.ndarray, crossover: Crossover) -> float:
    """``mirror.hybrid.seam_db`` of a pair: the wave side at 4 kHz, the mirror's at 48 kHz.

    The seam is a ratio of energies over the crossover's octave, 707 to
    1414 Hz, whose bins the cache form holds exactly
    (:func:`reverberate.spatial.lowband.decimate`): the mirror's response is
    brought to the same form and the two are read at 4 kHz.
    """
    high = np.asarray(decimate(np.asarray(mirror, dtype=float)[None, :], FIELD_RATE_HZ)[0], float)
    return seam_db(np.asarray(aired_omni, dtype=float), high, LOW_RATE_HZ, crossover)


def step_levels(
    *,
    audible: np.ndarray,
    pair: np.ndarray,
    position_weight: np.ndarray,
    cell: np.ndarray,
    mode: np.ndarray,
    listener: np.ndarray,
    cells: np.ndarray,
    seam_db_of: np.ndarray,
    trail_s_of: np.ndarray,
    early: EarlyTable,
    alignment_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``level/high_gain_db`` and ``level/onset_s`` of one source, ``[step]`` each.

    ``pair``, ``position_weight``, ``cell`` and ``mode`` are the source's
    ``low`` group; ``seam_db_of`` and ``trail_s_of`` are per row of
    ``low/ir``: the pair's seam, and what its onset trails the mirror's
    first arrival at the pair by. The pairs of a step are weighted by
    ``position_weight`` across the source's two positions and by inverse
    distance across the two cells.
    """
    steps = int(audible.shape[0])
    high_gain_db = np.zeros(steps, dtype=np.float32)
    onset = np.zeros(steps, dtype=np.float64)
    base = 20.0 * np.log10(alignment_gain)
    for step in np.flatnonzero(audible):
        across = np.array([1.0 - float(position_weight[step]), float(position_weight[step])])
        between = np.array([1.0, 0.0])
        if int(mode[step]) == MODE_FUSED:
            away = np.linalg.norm(cells[cell[step]] - listener[step][None, :], axis=1)
            total = float(away.sum())
            if total > 0.0:
                between = np.array([away[1], away[0]]) / total
        seam = 0.0
        trail = 0.0
        for a in range(2):
            for b in range(2):
                weight = across[a] * between[b]
                row = int(pair[step, a, b])
                if weight > 0.0 and row >= 0:
                    seam += weight * float(seam_db_of[row])
                    trail += weight * float(trail_s_of[row])
        high_gain_db[step] = base + seam
        onset[step] = first_arrival_s(early, int(step)) + trail
    return high_gain_db, onset
