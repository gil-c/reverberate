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
    octave_filter,
)

__all__ = [
    "DIFFUSE_COHERENCE_LIMIT_HZ",
    "Transposition",
    "effective_mean_free_path_m",
    "eyring_t60_s",
    "local_decay_s",
    "mean_absorption_of",
    "synthesise",
    "transpose",
]

#: Above this, interaural coherence in a diffuse field is negligible, about
#: 0.04 at 8 kHz, so independent noise per receiver is what the physics says.
#: Below it the tail is correlated between nearby receivers and this module's
#: independence assumption is wrong, which is why the low band is not
#: synthesised at all.
DIFFUSE_COHERENCE_LIMIT_HZ = 1000.0


def mean_absorption_of(report: dict[str, Any], bands: int) -> np.ndarray:
    """Area-weighted absorption per band, from a run's own report.

    Extends the catalogue's top band to the analysis bank's top by the rule of
    roadmap 6.2: each class's own 2 to 4 kHz ratio applied per octave, clipped
    to at most 1 and at least 0.8. The catalogue stops at 8 kHz because that is
    where its measurements stop, and the extension is a judgement labelled as
    one rather than a measurement.
    """
    classes = report["room"]["per_class"]
    area = np.array([entry["area_m2"] for entry in classes], dtype=float)
    alpha = np.array([entry["random_incidence_absorption"] for entry in classes], dtype=float)
    if area.sum() <= 0.0:
        raise ValueError("the report has no surface area to weight absorption by")

    while alpha.shape[1] < bands:
        ratio = np.clip(
            np.where(alpha[:, -3] > 0.0, alpha[:, -2] / np.maximum(alpha[:, -3], 1e-9), 1.0),
            0.8,
            1.0,
        )
        alpha = np.column_stack([alpha, np.clip(alpha[:, -1] * ratio, 1e-4, 0.99)])
    return np.asarray((area[:, None] * alpha[:, :bands]).sum(axis=0) / area.sum())


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
