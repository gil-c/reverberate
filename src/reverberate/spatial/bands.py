"""Three solves on three grids, one ambisonic response.

The economy of roadmap section 4.2 runs each band on its own grid, and the
array of :mod:`reverberate.spatial.array` is built in node index space, so the
same array cannot exist on three grids: on the 32.7 mm low grid a 12 mm shell
holds no nodes at all. ``docs/open-questions/three-band-ambisonic-array.md``
lists the nine consequences. This module takes the one road that is open by
construction: **each band is encoded with its own array on its own grid, and
the bands are put together in the spherical harmonic domain**, where the
representation is shared and nothing depends on which node was which.

What that buys, and what it costs, stated rather than assumed:

- The recombination filters of :mod:`reverberate.bands` are linear and the
  same for every channel, so a direction, which is nothing but the ratios
  between channels, passes through them unchanged away from the crossovers.
- Two solves are not on one scale. The level ratio is measured on the ``W``
  channel, which is the omnidirectional pressure at the centre and the one
  quantity every band's array measures identically, and it is applied to
  every channel of that band. The ratio predicted from the two source
  bandwidths is checked against the measurement, exactly as
  :mod:`reverberate.experiments.w37_three_band` does.
- Each band's centre is the node nearest the requested point **on its own
  grid**, so the three expansions are about three points up to half a cell
  diagonal apart. :func:`centre_offsets` reports the distances so the record
  can carry them. At 10.5 points per wavelength that is at most about 0.3 rad
  of phase at each band's own top, inside a crossover whose two sides W3
  already measured as decorrelated by -6 dB; it is a stated approximation,
  not a hidden one.
- The synthetic tail is drawn **per channel**. In a diffuse field the N3D
  channels carry equal expected energy and no correlation between them, so
  independent noise per channel, each calibrated on its own computed level
  over the crossfade, is the diffuse covariance, and the binaural decode then
  gives the interaural coherence of section 5.5 for nothing. The decay rate is
  read once from the ``W`` channel, because the diffuse decay is one number
  per band whatever the direction.
- Each band's effective order is whatever its own conditioning measurement
  says; a band that supports less is zero padded to the assembled order, which
  is the physical truncation and not an invention.

Air absorption is applied to each band before it is windowed and extended,
as the W37 experiments do, so that a window measured on a wet response and a
tail calibrated on it agree with each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate import bands as band_split
from reverberate.audio import Atmosphere
from reverberate.metrics import band_centres, rt60_per_band
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count
from reverberate.tail import local_decay_s, synthesise, transpose

__all__ = [
    "BandSolve",
    "assemble",
    "calibration_bands_hz",
    "centre_offsets",
    "extend",
    "pad_order",
]


@dataclass(frozen=True)
class BandSolve:
    """One band's encoded response, and the solve it came from."""

    name: str
    ambisonic: Ambisonic
    fmax_hz: float
    grid_step_m: float

    @property
    def omni(self) -> np.ndarray:
        """The ``W`` channel: the pressure at the centre, order zero."""
        return np.asarray(self.ambisonic.signals[0], dtype=float)


def pad_order(ambisonic: Ambisonic, order: int) -> Ambisonic:
    """The same response carried at a higher order, the new channels silent.

    A band whose array supports order 3 holds nothing above it, and saying so
    with zeros is the truncation the physics already applied; inventing the
    channels would be the thing that is not allowed.
    """
    if order < ambisonic.order:
        raise ValueError(f"cannot pad order {ambisonic.order} down to {order}")
    if order == ambisonic.order:
        return ambisonic
    signals = np.zeros((channel_count(order), ambisonic.signals.shape[1]))
    signals[: ambisonic.signals.shape[0]] = ambisonic.signals
    return Ambisonic(
        signals=signals,
        sample_rate_hz=ambisonic.sample_rate_hz,
        order=order,
        centre=ambisonic.centre,
        normalisation=ambisonic.normalisation,
        ordering=ambisonic.ordering,
    )


def centre_offsets(solves: list[BandSolve], requested: np.ndarray) -> list[dict[str, Any]]:
    """How far each band's expansion centre sits from the point that was asked for.

    Each is the nearest node on that band's grid, so the bound is half that
    grid's cell diagonal, and the phase it costs at the band's own top is
    ``k d``: about the same fraction of a wavelength for every band, since the
    step scales with the wavelength.
    """
    rows = []
    for solve in solves:
        centre = np.asarray(solve.ambisonic.centre, dtype=float)
        distance = float(np.linalg.norm(centre - np.asarray(requested, dtype=float)))
        bound = float(np.sqrt(3.0) * solve.grid_step_m / 2.0)
        rows.append(
            {
                "band": solve.name,
                "centre": [round(float(v), 6) for v in centre],
                "offset_m": round(distance, 6),
                "bound_m": round(bound, 6),
                "phase_at_fmax_rad": round(2.0 * np.pi * solve.fmax_hz / 343.0 * distance, 4),
            }
        )
    return rows


def calibration_bands_hz(lower_fmax_hz: float, sample_rate_hz: int) -> tuple[float, float]:
    """The octave bands two adjacent solves can be levelled on.

    Both must resolve them: the top usable octave of the lower solve sits an
    octave under its ``fmax``, where its source pulse still has energy and
    the band-limiting low pass has not started, and three octaves is enough
    for a median that one bad band cannot set.
    """
    centres = np.asarray(band_centres(sample_rate_hz), dtype=float)
    top = lower_fmax_hz / 2.0
    usable = centres[(centres <= top * 1.01) & (centres >= top / 8.0)]
    if len(usable) < 1:
        raise ValueError(f"no octave band sits under {lower_fmax_hz:g} Hz at this rate")
    return float(usable[0]), float(usable[-1])


def _level(
    subject: BandSolve, reference: BandSolve, sample_rate_hz: int
) -> tuple[float, dict[str, Any]]:
    """The gain that puts ``subject`` on ``reference``'s scale, and the check on it."""
    common = min(subject.omni.shape[0], reference.omni.shape[0])
    calibration = calibration_bands_hz(subject.fmax_hz, sample_rate_hz)
    gain = band_split.level_ratio(
        subject.omni[:common],
        reference.omni[:common],
        sample_rate_hz,
        calibration_hz=calibration,
    )
    predicted = subject.fmax_hz / reference.fmax_hz
    measured_db = 20.0 * float(np.log10(gain))
    predicted_db = 20.0 * float(np.log10(predicted))
    if abs(measured_db - predicted_db) > 3.0:
        raise ValueError(
            f"the {subject.name} and {reference.name} solves differ by {measured_db:+.2f} dB "
            f"on {calibration[0]:g} to {calibration[1]:g} Hz but their source bandwidths "
            f"predict {predicted_db:+.2f} dB; they are not the same room under the same "
            "excitation and summing them would put a band at the wrong level"
        )
    return gain, {
        "subject": subject.name,
        "reference": reference.name,
        "calibration_hz": list(calibration),
        "measured_db": round(measured_db, 3),
        "predicted_db": round(predicted_db, 3),
    }


def assemble(
    low: BandSolve,
    mid: BandSolve,
    high: BandSolve,
    *,
    crossovers_hz: tuple[float, float] = band_split.DEFAULT_CROSSOVERS_HZ,
    taps: int = band_split.TAPS,
) -> tuple[Ambisonic, dict[str, Any]]:
    """One ambisonic response from three, each band-limited to its own solve.

    The high band is the reference scale, the mid is levelled on it and the low
    on the mid, so a level defect between two solves is caught where it is
    introduced. The output is at the highest order of the three, centred where
    the high band was expanded, and it carries the filter bank's group delay
    identically in every band and every channel.
    """
    rates = {
        low.ambisonic.sample_rate_hz,
        mid.ambisonic.sample_rate_hz,
        high.ambisonic.sample_rate_hz,
    }
    if len(rates) != 1:
        raise ValueError(f"the bands are delivered at different sample rates: {sorted(rates)}")
    rate = int(round(high.ambisonic.sample_rate_hz))
    if not low.fmax_hz < mid.fmax_hz < high.fmax_hz:
        raise ValueError("the bands must be given lowest first, by their fmax")
    if not low.fmax_hz >= crossovers_hz[0] and mid.fmax_hz >= crossovers_hz[1]:
        raise ValueError(
            f"a solve stops under the crossover it is meant to carry: fmax "
            f"{low.fmax_hz:g}/{mid.fmax_hz:g} against crossovers {crossovers_hz}"
        )

    gain_mid, mid_check = _level(mid, high, rate)
    gain_low_on_mid, low_check = _level(low, mid, rate)
    gains = (gain_low_on_mid * gain_mid, gain_mid, 1.0)

    order = max(low.ambisonic.order, mid.ambisonic.order, high.ambisonic.order)
    parts = [pad_order(solve.ambisonic, order) for solve in (low, mid, high)]
    split = band_split.split_filters(float(rate), crossovers_hz, taps=taps)
    signals = band_split.recombine(
        parts[0].signals, parts[1].signals, parts[2].signals, split, gains=gains
    )
    assembled = Ambisonic(
        signals=np.asarray(signals, dtype=float),
        sample_rate_hz=float(rate),
        order=order,
        centre=high.ambisonic.centre,
    )
    record = {
        "bands": [
            {
                "band": solve.name,
                "fmax_hz": solve.fmax_hz,
                "grid_step_m": solve.grid_step_m,
                "order": solve.ambisonic.order,
                "computed_s": round(solve.ambisonic.duration_s, 4),
                "gain": round(float(gain), 6),
                "gain_db": round(20.0 * float(np.log10(gain)), 3),
            }
            for solve, gain in zip((low, mid, high), gains, strict=True)
        ],
        "level_checks": [low_check, mid_check],
        "split": split.record(),
        "order": order,
        "domain": "spherical harmonic, ACN, N3D; each band encoded on its own grid",
    }
    return assembled, record


def extend(
    solve: BandSolve,
    total_s: float,
    *,
    calibration: BandSolve,
    mean_absorption: np.ndarray,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    seed: int,
    fade_s: float = 0.010,
) -> tuple[BandSolve, dict[str, Any]]:
    """Continue a band's computed response with a diffuse tail to ``total_s``.

    The decay is read on the ``W`` channel: locally, on the last part of what
    was solved, where that fit converges, and otherwise transposed from the
    calibration band's own decay through the room's absorption and the air,
    which is roadmap section 5.5's rule. Every channel then draws its own
    noise at that decay and is levelled on its own computed part over the
    crossfade, which is the diffuse field covariance in N3D.
    """
    rate = int(round(solve.ambisonic.sample_rate_hz))
    window = solve.ambisonic.signals.shape[1]
    total = int(round(total_s * rate))
    if total < window:
        raise ValueError(
            f"{solve.name} was solved for {window / rate:.3f} s and cannot be extended to "
            f"{total_s:.3f} s"
        )
    prediction = transpose(
        rt60_per_band(calibration.omni, rate),
        mean_absorption,
        rate,
        mid_fmax_hz=calibration.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed_m_s,
    )
    transposed = np.asarray(prediction.t60_s, dtype=float)
    local = local_decay_s(solve.omni[None, :], rate, window / rate)
    decay = np.where(np.isfinite(local), local, transposed)
    if total == window:
        extended = solve.ambisonic
    else:
        signals = np.vstack(
            [
                synthesise(
                    channel,
                    rate,
                    decay,
                    total,
                    rng=np.random.default_rng(seed + index),
                    fade_s=fade_s,
                )
                for index, channel in enumerate(solve.ambisonic.signals)
            ]
        )
        extended = Ambisonic(
            signals=signals,
            sample_rate_hz=solve.ambisonic.sample_rate_hz,
            order=solve.ambisonic.order,
            centre=solve.ambisonic.centre,
        )
    record = {
        "band": solve.name,
        "computed_s": round(window / rate, 4),
        "total_s": round(total / rate, 4),
        "fade_s": fade_s,
        "band_centres_hz": [int(c) for c in band_centres(rate)],
        "tail_t60_s": [None if not np.isfinite(v) else round(float(v), 4) for v in decay],
        "from_local_fit": [bool(v) for v in np.isfinite(local)],
        "transposition": prediction.record(),
        "note": (
            "one decay per band read on W, independent noise per channel levelled on "
            "its own computed part: the diffuse covariance in N3D"
        ),
    }
    return BandSolve(solve.name, extended, solve.fmax_hz, solve.grid_step_m), record
