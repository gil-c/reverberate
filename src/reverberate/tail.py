"""The synthetic reverberation tail, and the window that hands over to it.

Roadmap section 5.5. The high band domain is a real room, so it does have a
diffuse field and the tail could in principle be computed. It should still be
synthesised, for cost: the measured bedroom's 0.5 s at 16 kHz is 2.80 h of
solver against 0.34 h for 60 ms. The tail is where the money goes and where the
ear is least discriminating, and solver cost is linear in the window.

**What transposes from the mid band to the high band, and what does not.** The
averaged decay slope, the geometric mixing time, and the arrival time and
direction of a reflection all transpose. The waveform of a reflection does not.
So the tail is rebuilt as noise with the right per-band decay rather than
copied, and the early part is the only thing the solver has to produce.

**Channel-independent noise is physically correct here, not a shortcut.** In a
diffuse field interaural coherence falls to nothing near 1 kHz and stays
negligible above, about 0.04 at 8 kHz. Each receiver therefore draws its own
noise, and :func:`synthesise` takes an explicit generator so that is
reproducible from a seed rather than merely random.

**One defect this module exists to not have.** A first version enveloped white
noise and then summed the bands, which matched band energy to 0.00 dB while
getting T30 wrong by 60 per cent: every band's noise leaked into every other
band, so the recombined signal decayed at an average rate rather than a
per-band one. The noise is band-limited *before* it is enveloped, and
``tests/test_tail.py`` checks T30 and not only energy, because energy alone did
not catch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.audio import Atmosphere
from reverberate.metrics import (
    band_centres,
    energy_decay_curve,
    octave_filter,
    rt60_per_band,
)

__all__ = [
    "DIFFUSE_COHERENCE_LIMIT_HZ",
    "Transposition",
    "local_decay_s",
    "window_for_level_s",
    "effective_mean_free_path_m",
    "eyring_t60_s",
    "mixing_time_s",
    "splice",
    "synthesise",
    "transpose",
]

#: Above this, interaural coherence in a diffuse field is negligible, about
#: 0.04 at 8 kHz, so independent noise per receiver is what the physics says.
#: Below it the tail is correlated between nearby receivers and this module's
#: independence assumption is wrong, which is why the low band is not
#: synthesised at all.
DIFFUSE_COHERENCE_LIMIT_HZ = 1000.0


def mixing_time_s(volume_m3: float) -> float:
    """When the field becomes diffuse, in seconds. Polack's ``sqrt(V)`` in ms.

    Reproduces the roadmap's own figures: 12 ms for the living room's 151.7 m3
    and 7 ms for the kitchen's 45.5. It is the earliest a synthetic tail can
    honestly take over, because before it the response is discrete reflections
    whose arrival times carry the spatial information, and noise has none.
    """
    if volume_m3 <= 0.0:
        raise ValueError(f"volume must be positive, got {volume_m3} m3")
    return float(np.sqrt(volume_m3) / 1000.0)


def local_decay_s(
    ir: np.ndarray,
    sample_rate_hz: int,
    window_s: float,
    *,
    fraction: float = 0.4,
    block_s: float = 0.005,
) -> np.ndarray:
    """T60 per band, fitted on the last part of what was actually solved.

    Roadmap 5.5 asks for the synthetic tail to be "calibrated band by band on
    the last genuinely computed milliseconds". :func:`transpose` supplies a
    decay averaged over the whole response, which is a different number, and
    the difference grows with how deep the splice is.

    **Measured, and it is why a deep splice needs this.** Splicing the mid band
    at -30 dB with a globally fitted T30 made the assembled response decay
    13 per cent fast at 2 kHz. A real decay is not one straight line: the slope
    near -30 dB is not the slope of the -5 to -35 dB average, and the deeper the
    splice the more they differ. A tail that continues the *local* slope joins
    the curve it is actually continuing.

    Fitted on the block level rather than on a Schroeder curve, because
    backward integration of a truncated response collapses at its own end and
    would bias the last fit downwards. Returns NaN for a band with too little
    left to fit, which :func:`synthesise` reads as "do not invent this one" and
    a caller can fall back on :func:`transpose` for.
    """
    signals = np.atleast_2d(np.asarray(ir, dtype=float))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    window = int(round(window_s * sample_rate_hz))
    block = max(1, int(round(block_s * sample_rate_hz)))
    start = max(0, window - int(round(window * fraction)))
    blocks = (window - start) // block
    centres = band_centres(sample_rate_hz)
    out = np.full(len(centres), np.nan)
    if blocks < 4:
        return out

    times = (np.arange(blocks) * block + block / 2.0) / sample_rate_hz
    for band in range(len(centres)):
        slopes = []
        for receiver in range(signals.shape[0]):
            filtered = octave_filter(signals[receiver], sample_rate_hz)[band]
            segment = filtered[start : start + blocks * block].reshape(blocks, block)
            power = (segment**2).mean(axis=1)
            if not np.all(power > 0.0):
                continue
            level = 10.0 * np.log10(power)
            slope = float(np.polyfit(times, level, 1)[0])
            if slope < -1e-6:
                slopes.append(-60.0 / slope)
        if slopes:
            out[band] = float(np.median(slopes))
    return out


def window_for_level_s(
    ir: np.ndarray,
    sample_rate_hz: int,
    level_db: float,
    *,
    bands_hz: tuple[float, float] | None = None,
    floor_s: float = 0.010,
) -> float:
    """How long to solve for, to reach ``level_db`` in every band that matters.

    Replaces a duration typed by an operator. ``duration`` reaches the solver as
    ``Nt`` and nothing derived it, so a run was as long as somebody guessed.
    This measures it instead, on the energy decay curve of a response of the
    same room, and takes the **slowest** band the run contributes, because a
    window that suits the top of a band cuts the bottom of it short.

    **The rule scales itself, which is the point.** Air absorption steepens the
    decay towards the top of the audible range, so one level gives a long
    window low down and a short one high up. Measured on ``w29_16k`` with air,
    a -30 dB window is 223 ms at 125 Hz, 124 ms at 2 kHz and 86 ms at 16 kHz.
    It also reproduces the figure this project arrived at empirically: 60 ms
    for the high band is about -18 dB.

    ``bands_hz`` restricts the search to the octave centres a band run actually
    contributes, as ``(low, high)`` inclusive. Passing ``None`` considers every
    band, which is what a single broadband run wants.

    Returns seconds, never less than ``floor_s``. A band whose decay never
    reaches ``level_db`` inside the response contributes the whole response
    rather than a NaN: the answer is then "at least this long", which is the
    honest reading of a measurement that ran out of signal.
    """
    signals = np.atleast_2d(np.asarray(ir, dtype=float))
    if level_db >= 0.0:
        raise ValueError(f"level must be a decay in decibels below zero, got {level_db}")
    centres = np.asarray(band_centres(sample_rate_hz), dtype=float)
    if bands_hz is not None:
        low, high = bands_hz
        wanted = (centres >= low) & (centres <= high)
    else:
        wanted = np.ones(len(centres), dtype=bool)
    if not wanted.any():
        raise ValueError(f"no octave band of this rate lies in {bands_hz}")

    longest = floor_s
    for receiver in range(signals.shape[0]):
        for band, keep in zip(
            octave_filter(signals[receiver], sample_rate_hz), wanted, strict=True
        ):
            if not keep:
                continue
            curve = energy_decay_curve(band)
            below = np.flatnonzero(curve <= level_db)
            reached = below[0] / sample_rate_hz if len(below) else len(band) / sample_rate_hz
            longest = max(longest, float(reached))
    return longest


def eyring_t60_s(
    mean_free_path_m: float,
    mean_absorption: float,
    *,
    sound_speed_m_s: float,
) -> float:
    """Eyring's reverberation time on a given mean free path, in seconds.

    ``13.8 L / (c ln(1 / (1 - alpha)))``. Written on an explicit mean free path
    rather than on ``4V/S``, because that substitution is the most fragile
    assumption in the formula and :func:`effective_mean_free_path_m` measures it
    instead.
    """
    if not 0.0 < mean_absorption < 1.0:
        raise ValueError(f"mean absorption must be in (0, 1), got {mean_absorption}")
    if mean_free_path_m <= 0.0:
        raise ValueError(f"mean free path must be positive, got {mean_free_path_m} m")
    return float(13.8 * mean_free_path_m / (sound_speed_m_s * -np.log1p(-mean_absorption)))


def effective_mean_free_path_m(
    t60_s: float,
    mean_absorption: float,
    *,
    sound_speed_m_s: float,
    air_db_per_m: float = 0.0,
) -> float:
    """Invert Eyring on a measured decay: the path length the room behaves as.

    This is the point of calibrating on the mid band rather than predicting
    from geometry. ``4V/S`` assumes a convex empty room, and a furnished one is
    neither; the measured bedroom's effective path is 2.66 m against a ``4V/S``
    of 1.20. Taking it from a run that exists removes that assumption instead
    of carrying it into the high band.

    ``air_db_per_m`` is subtracted first, so the returned path describes the
    boundaries alone and can be reused at another frequency where the air term
    is different.
    """
    if not np.isfinite(t60_s) or t60_s <= 0.0:
        raise ValueError(f"reverberation time must be finite and positive, got {t60_s}")
    surface_rate = 60.0 / t60_s - air_db_per_m * sound_speed_m_s
    if surface_rate <= 0.0:
        raise ValueError(
            "air alone already decays faster than the measurement, so the boundaries "
            "cannot be separated from it at this frequency"
        )
    surface_t60 = 60.0 / surface_rate
    return float(surface_t60 * sound_speed_m_s * -np.log1p(-mean_absorption) / 13.8)


@dataclass(frozen=True)
class Transposition:
    """The decay a high band tail should have, and where each band came from."""

    centres_hz: tuple[int, ...]
    t60_s: tuple[float, ...]
    #: True where the value was copied from a band the mid band run resolves,
    #: false where it was extrapolated by Eyring plus air. The distinction is
    #: the whole uncertainty of the method and must not be lost.
    measured: tuple[bool, ...]
    mean_free_path_m: float
    anchor_hz: int

    def record(self) -> dict[str, Any]:
        return {
            "centres_hz": list(self.centres_hz),
            "t60_s": [float(value) for value in self.t60_s],
            "measured": list(self.measured),
            "mean_free_path_m": self.mean_free_path_m,
            "anchor_hz": self.anchor_hz,
        }


def transpose(
    mid_t60_s: np.ndarray,
    mean_absorption: np.ndarray,
    sample_rate_hz: int,
    *,
    mid_fmax_hz: float,
    atmosphere: Atmosphere | None = None,
    sound_speed_m_s: float,
    anchor_hz: float = 2000.0,
) -> Transposition:
    """Predict the high band's per-band decay from a mid band run.

    Two rules, and which one applies is decided by the mid band's own ``fmax``
    rather than by taste.

    **Where the mid band resolves the band, its value is copied.** Measured on
    ``w27_sealed`` against ``w29_16k``, the same room at 4 and 16 kHz, the two
    agree to 1.6 to 3.4 per cent mean absolute error at 1, 2 and 4 kHz, against
    a receiver-to-receiver spread of 3.5 to 4.3 and a 5 per cent just
    noticeable difference. There is nothing to gain by modelling what has been
    measured.

    **Above it, Eyring on the calibrated mean free path plus the air term.**
    The effective path is taken from the mid band at ``anchor_hz``, where the
    air term is still small enough to separate cleanly, and reapplied with each
    band's own absorption.

    **On the bedroom this cannot be told from copying the anchor.** Both give 5
    to 6 per cent at 16 kHz against a 3.1 per cent spread, which is W29's own
    conclusion: that room's absorption is nearly flat with frequency, so no
    rule is distinguishable there. The physically motivated one is used because
    it is no worse and it will separate in a room with contrast, which is why
    W30 chose the kitchen.
    """
    atmosphere = atmosphere or Atmosphere()
    centres = np.asarray(band_centres(sample_rate_hz), dtype=float)
    mid = np.asarray(mid_t60_s, dtype=float)
    alpha = np.asarray(mean_absorption, dtype=float)
    if not (len(centres) == len(mid) == len(alpha)):
        raise ValueError(
            f"{len(centres)} bands, {len(mid)} decay times and {len(alpha)} absorption "
            "coefficients: all three must be on the same band axis"
        )

    air = atmosphere.attenuation_db_per_m(centres)
    anchor = int(np.argmin(np.abs(centres - anchor_hz)))
    if not np.isfinite(mid[anchor]):
        raise ValueError(
            f"the mid band run has no usable decay at {centres[anchor]:g} Hz, so the "
            "effective mean free path cannot be calibrated"
        )
    path = effective_mean_free_path_m(
        float(mid[anchor]),
        float(alpha[anchor]),
        sound_speed_m_s=sound_speed_m_s,
        air_db_per_m=float(air[anchor]),
    )

    predicted: list[float] = []
    measured: list[bool] = []
    for band, centre in enumerate(centres):
        if centre <= mid_fmax_hz and np.isfinite(mid[band]):
            predicted.append(float(mid[band]))
            measured.append(True)
            continue
        surface = eyring_t60_s(path, float(alpha[band]), sound_speed_m_s=sound_speed_m_s)
        predicted.append(float(60.0 / (60.0 / surface + air[band] * sound_speed_m_s)))
        measured.append(False)

    return Transposition(
        centres_hz=tuple(int(round(centre)) for centre in centres),
        t60_s=tuple(predicted),
        measured=tuple(measured),
        mean_free_path_m=path,
        anchor_hz=int(round(centres[anchor])),
    )


def synthesise(
    early: np.ndarray,
    sample_rate_hz: int,
    t60_s: np.ndarray,
    total_samples: int,
    *,
    rng: np.random.Generator,
    fade_s: float = 0.010,
) -> np.ndarray:
    """Extend a truncated response with a noise tail of the given decay.

    ``early`` is ``(samples,)`` or ``(receivers, samples)`` and holds the part
    the solver actually computed. The return is ``(…, total_samples)``: the
    early part unchanged, then a crossfade, then noise decaying at ``t60_s``
    per octave band.

    Each band's noise is **band-limited before it is enveloped**, and its level
    is matched to the computed response's own level over the crossfade window,
    band by band. Matching over the fade rather than at a single sample is what
    stops a discontinuity in level or spectral colour, which is the audible
    trap rather than the seam itself.

    Every receiver draws independently, which is what a diffuse field above
    1 kHz actually looks like.
    """
    signals = np.atleast_2d(np.asarray(early, dtype=float))
    window = signals.shape[1]
    if total_samples < window:
        raise ValueError(
            f"cannot extend {window} samples to {total_samples}: the tail must be longer "
            "than the part that was computed"
        )
    fade = int(round(fade_s * sample_rate_hz))
    if fade < 1 or fade > window:
        raise ValueError(
            f"a {fade_s * 1000:g} ms crossfade is {fade} samples, which does not fit "
            f"inside a {window} sample window"
        )

    centres = np.asarray(band_centres(sample_rate_hz), dtype=float)
    decay = np.asarray(t60_s, dtype=float)
    if len(decay) != len(centres):
        raise ValueError(f"{len(decay)} decay times for {len(centres)} bands")

    times = np.arange(total_samples) / sample_rate_hz
    fade_start = window - fade
    # Linear ramp: the early part is alone before the fade, the tail alone after.
    weight = np.zeros(total_samples)
    weight[fade_start:window] = np.linspace(0.0, 1.0, fade, endpoint=False)
    weight[window:] = 1.0

    out = np.zeros((signals.shape[0], total_samples))
    for receiver in range(signals.shape[0]):
        # The computed part is carried through as the solver wrote it, never
        # rebuilt from its bands. An octave filter bank is not a
        # perfect-reconstruction bank, so summing its outputs returns something
        # close to the input rather than the input, and the early part is the
        # half of the response that carries the spatial information.
        bands = octave_filter(signals[receiver], sample_rate_hz)
        tail = np.zeros(total_samples)
        for band in range(len(centres)):
            if not np.isfinite(decay[band]) or decay[band] <= 0.0:
                # No decay to imitate: the band stops where the solver stopped,
                # rather than being invented.
                continue
            noise = octave_filter(rng.standard_normal(total_samples), sample_rate_hz)[band]
            envelope = 10.0 ** (
                -60.0 * (times - fade_start / sample_rate_hz) / (20.0 * decay[band])
            )
            noise = noise * envelope
            reference = float(np.sqrt((bands[band, fade_start:window] ** 2).mean()))
            level = float(np.sqrt((noise[fade_start:window] ** 2).mean()))
            if level > 0.0:
                noise *= reference / level
            tail += noise

        computed = np.zeros(total_samples)
        computed[:window] = signals[receiver]
        out[receiver] = computed * (1.0 - weight) + tail * weight

    return out[0] if np.ndim(early) == 1 else out


def splice(
    reference: np.ndarray,
    sample_rate_hz: int,
    window_s: float,
    t60_s: np.ndarray,
    *,
    rng: np.random.Generator,
    fade_s: float = 0.010,
) -> np.ndarray:
    """Truncate a response at ``window_s`` and extend it with a synthetic tail.

    The experimental form of :func:`synthesise`. Truncating a stored response
    is exactly equivalent to having stopped the solve at that window, because
    an explicit time-marching scheme cannot revise a sample it has already
    written. So the whole duration lever can be measured on a response that has
    already been bought, at no rental, by splicing here and comparing against
    the untruncated original.
    """
    signals = np.atleast_2d(np.asarray(reference, dtype=float))
    window = int(round(window_s * sample_rate_hz))
    if not 0 < window <= signals.shape[1]:
        raise ValueError(
            f"a {window_s * 1000:g} ms window is {window} samples, outside the "
            f"{signals.shape[1]} the reference holds"
        )
    spliced = synthesise(
        signals[:, :window],
        sample_rate_hz,
        t60_s,
        signals.shape[1],
        rng=rng,
        fade_s=fade_s,
    )
    return spliced[0] if np.ndim(reference) == 1 else spliced


def measured_decay(ir: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """T30 per band, the quantity a spliced tail is judged on.

    A thin alias for :func:`reverberate.metrics.rt60_per_band`, present so a
    caller of this module does not have to know that the project's T60 is a T30
    extrapolated from the -5 to -35 dB span.
    """
    return rt60_per_band(ir, sample_rate_hz)
