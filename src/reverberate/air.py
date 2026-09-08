"""Atmospheric absorption, applied to a response the solver has already computed.

The solver has no viscosity term. PFFDTD is pinned at ``aa319f6c`` and its
kernel carries no Stokes filter, so every response this project holds decays
only by its boundaries. At 16 kHz that omits the term that dominates: air
alone gives a T60 of 0.48 s at 20 C and 50 per cent relative humidity, against
0.28 s for the measured bedroom's surfaces, and the two together read 0.19 s.
Leaving it out makes the top of the band decay roughly twice as long as it
should.

**Why this belongs here and not in the kernel.** In an impulse response every
sample arriving at time ``t`` has travelled exactly ``c t`` of path, whatever
geometry it took to get there. Atmospheric absorption is therefore *exactly* a
per-sample, frequency-dependent gain ``exp(-m(f) c t)``. It is a time-varying
filter on a signal that already exists, not an approximation that a diffuse
field assumption has to excuse. So it costs no rental, it applies
retroactively to every response already computed, and it needs no patch to the
pinned solver. See ``docs/adr/0008-air-absorption-as-post-processing.md``.

**Humidity is not a detail and must be declared with every run.** At 16 kHz the
attenuation runs from 0.252 dB/m at 80 per cent relative humidity to 0.466 at
30 per cent, a factor of 1.85. :class:`Atmosphere` exists so that a run records
what it assumed rather than inheriting a default nobody wrote down, and
:meth:`Atmosphere.record` is what a ``report.json`` carries.

The coefficient is ISO 9613-1, the same formulation PFFDTD's own reference
implementation uses. :func:`attenuation_db_per_m` reproduces the five figures
the roadmap quotes at 50 per cent to four decimal places, and those five are
tests rather than comments.

**This module is a duplicate and is meant to be deleted whole.** The W10 branch
implements the same standard in :mod:`reverberate.audio` as
``air_absorption_np_per_m`` and ``apply_air_absorption``, written independently
and at the same time.

**Cross-checked twice, and the second time they agree exactly.** A first
comparison put them 1.27e-5 apart, which turned out to be the truncation in
ISO's own 8.686 against the exact ``20 / ln 10``. With that removed at the
source, converting their nepers with 8.686 reproduces this module's decibels to
**2.2e-16**, from 125 Hz to 16 kHz at 30, 50 and 80 per cent relative humidity.
Two people wrote the standard separately and landed on the same floating-point
number. The applied *filter* still differs by 7e-6 in energy, which is the
short-time transform and not the physics: root-Hann against Hann.

``audio.py`` is the right home: it is where the "air absorption is not
modelled" claim lived, and nepers per metre is what its consumer wants.
:class:`Atmosphere` has already been carried across, so **nothing here needs
saving. When W10 lands, delete this module and import from
:mod:`reverberate.audio`.** Neither branch should merge both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

__all__ = [
    "REFERENCE_PRESSURE_KPA",
    "Atmosphere",
    "apply",
    "attenuation_db_per_m",
    "gain",
]

#: Reference atmospheric pressure, kPa. ISO 9613-1's ``p_r``, one standard
#: atmosphere.
REFERENCE_PRESSURE_KPA = 101.325

#: Reference air temperature, K. ISO 9613-1's ``T_0``, which is 20 C.
_REFERENCE_TEMPERATURE_K = 293.15

#: Triple-point isotherm of water, K. ISO 9613-1's ``T_01``, used only by the
#: saturation-pressure fit.
_TRIPLE_POINT_K = 273.16


@dataclass(frozen=True)
class Atmosphere:
    """The air a response is claimed to have propagated through.

    Defaults are 20 C, 50 per cent relative humidity and one standard
    atmosphere, which is the condition the roadmap's own attenuation table is
    quoted at. They are a stated choice, not a physical constant: at 16 kHz the
    humidity alone moves the attenuation by a factor of 1.85 across the range a
    room plausibly sits in, so a run that does not record this has not recorded
    its own decay.
    """

    temperature_c: float = 20.0
    relative_humidity_pct: float = 50.0
    pressure_kpa: float = REFERENCE_PRESSURE_KPA

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_humidity_pct <= 100.0:
            raise ValueError(
                f"relative humidity must be a percentage in [0, 100], got "
                f"{self.relative_humidity_pct}"
            )
        if self.pressure_kpa <= 0.0:
            raise ValueError(f"pressure must be positive, got {self.pressure_kpa} kPa")
        if self.temperature_c <= -273.15:
            raise ValueError(f"temperature must be above absolute zero, got {self.temperature_c} C")

    @property
    def temperature_k(self) -> float:
        """Absolute temperature, K."""
        return self.temperature_c + 273.15

    def record(self) -> dict[str, Any]:
        """What a ``report.json`` carries, so the decay can be reproduced."""
        return {
            "temperature_c": self.temperature_c,
            "relative_humidity_pct": self.relative_humidity_pct,
            "pressure_kpa": self.pressure_kpa,
            "standard": "ISO 9613-1",
        }


def attenuation_db_per_m(
    frequency_hz: float | np.ndarray,
    atmosphere: Atmosphere | None = None,
) -> np.ndarray:
    """Pure-tone atmospheric attenuation coefficient, dB per metre.

    ISO 9613-1:1993, the classical relaxation model: a small
    classical-plus-rotational term, plus a relaxation term each for oxygen and
    nitrogen, whose relaxation frequencies depend on how much water vapour is
    present. Below about 1 kHz the coefficient is negligible for room-sized
    paths; above 8 kHz it is the dominant absorber in a furnished room.

    ``frequency_hz`` may be a scalar or an array; the return always has the
    shape of the input as an array, so a caller can broadcast it against a
    band axis without special-casing one frequency.
    """
    atmosphere = atmosphere or Atmosphere()
    freq = np.asarray(frequency_hz, dtype=float)
    if np.any(freq < 0.0):
        raise ValueError("frequency must not be negative")

    temperature_k = atmosphere.temperature_k
    relative_temperature = temperature_k / _REFERENCE_TEMPERATURE_K
    relative_pressure = atmosphere.pressure_kpa / REFERENCE_PRESSURE_KPA

    # Saturation vapour pressure over the reference pressure, ISO 9613-1 B.3.
    saturation_ratio = 10.0 ** (-6.8346 * (_TRIPLE_POINT_K / temperature_k) ** 1.261 + 4.6151)
    # Molar concentration of water vapour, per cent.
    water_vapour = atmosphere.relative_humidity_pct * saturation_ratio / relative_pressure

    # Relaxation frequencies of oxygen and nitrogen, Hz.
    oxygen_hz = relative_pressure * (
        24.0 + 4.04e4 * water_vapour * (0.02 + water_vapour) / (0.391 + water_vapour)
    )
    nitrogen_hz = (
        relative_pressure
        * relative_temperature**-0.5
        * (
            9.0
            + 280.0 * water_vapour * np.exp(-4.170 * (relative_temperature ** (-1.0 / 3.0) - 1.0))
        )
    )

    classical = 1.84e-11 / relative_pressure * relative_temperature**0.5
    oxygen = 0.01275 * np.exp(-2239.1 / temperature_k) / (oxygen_hz + freq**2 / oxygen_hz)
    nitrogen = 0.1068 * np.exp(-3352.0 / temperature_k) / (nitrogen_hz + freq**2 / nitrogen_hz)

    coefficient = classical + relative_temperature**-2.5 * (oxygen + nitrogen)
    return np.asarray(8.686 * freq**2 * coefficient, dtype=float)


def gain(
    frequency_hz: float | np.ndarray,
    time_s: float | np.ndarray,
    atmosphere: Atmosphere | None = None,
    *,
    sound_speed_m_s: float,
) -> np.ndarray:
    """Linear gain a tone at ``frequency_hz`` has after ``time_s`` of flight.

    ``exp(-m(f) c t)`` written as a decibel decay, since that is the form the
    coefficient is tabulated in. Frequency and time broadcast against each
    other, so passing a band axis and a sample axis gives the whole gain
    surface in one call.

    ``sound_speed_m_s`` is keyword-only and has no default on purpose: the path
    length a sample has flown is ``c t``, so a wrong speed here scales every
    attenuation, and the solver's own value belongs in its provenance rather
    than in this module.
    """
    if sound_speed_m_s <= 0.0:
        raise ValueError(f"sound speed must be positive, got {sound_speed_m_s} m/s")
    decibels = attenuation_db_per_m(frequency_hz, atmosphere) * sound_speed_m_s
    return np.asarray(10.0 ** (-decibels * np.asarray(time_s, dtype=float) / 20.0), dtype=float)


def apply(
    ir: np.ndarray,
    sample_rate_hz: float,
    atmosphere: Atmosphere | None = None,
    *,
    sound_speed_m_s: float,
    frame: int = 256,
) -> np.ndarray:
    """Apply atmospheric absorption to one or more impulse responses.

    ``ir`` is ``(samples,)`` or ``(receivers, samples)``; the return has the
    same shape and dtype ``float64``. Time is measured from sample zero, which
    is the instant the source fires, so the arithmetic is only correct on a
    response whose origin is the excitation and not on an arbitrary excerpt.

    The gain varies in both time and frequency, so it is applied on a short-time
    spectrum: each analysis frame is multiplied by ``exp(-m(f) c t)`` at that
    frame's centre time and its own bin frequencies, then overlap-added back.
    ``frame`` trades the two errors against each other. At 256 samples and
    48 kHz a frame spans 5.3 ms, over which the 16 kHz gain moves by 0.7 dB,
    while the bin spacing is 187 Hz, which resolves ``m(f)`` everywhere it is
    large.

    **What the frame sets is a dynamic range, not a time limit.** Every frame
    tracks the analytic decay exactly until the signal gets faint enough that
    spectral leakage from the loud part of the spectrum reaches it, and past
    that point the output is numerical residue rather than a filtered tone.
    Measured on a 16 kHz tone, which falls 125 dB per second: the analytic
    level past which the curve never comes back within 6 dB of the line.

    ======== ====================
    frame    exact down to
    ======== ====================
    128      -155 dB
    **256**  **-170 dB**
    512      -190 dB
    1024     -220 dB
    2048     -239 dB
    4096     -240 dB
    ======== ====================

    **The step per doubling is not a constant.** It reads 15, 20, 30, 19 then
    1 dB here, saturating near -240 dB where the transform's own arithmetic
    takes over from leakage.

    **The W10 branch measures the same definition about 50 to 70 dB higher, and
    the cause is the window scheme rather than the measurement.** Reimplementing
    the overlap-add by hand, same gain surface and same three-quarter overlap,
    only the window pair changing, at a 256-sample frame:

    ===================================== ==============
    analysis and synthesis pair           departure
    ===================================== ==============
    root-Hann both sides, W10's scheme    -119 dB
    Hann analysis, no synthesis window    -124 dB
    **Hann both sides, which is scipy's** **-170 dB**
    ===================================== ==============

    The hand-rolled Hann-twice reproduces :func:`scipy.signal.istft` to the
    decimal at every frame, so this is what ``apply`` uses: an effective Hann
    squared, which rolls off far faster in frequency than a cosine window and
    therefore keeps leakage out of a faint bin much longer. The advantage grows
    with the frame, 45 dB at 128 and 70 dB at 1024.

    **This does not make either scheme wrong.** Splitting the modification
    evenly between analysis and synthesis, which is what the root pair buys, is
    a real property that this one does not have; both are exact wherever there
    is signal; and no room response comes within seventy decibels of either
    limit. It is recorded because two docstrings in one repository otherwise
    carry numbers that differ by 70 dB with no reason attached.

    **Duration is only how long it takes to fall that far.** A 16 kHz tone
    reaches the default's departure level at about 1.35 s, which is why a naive
    slope fitted over a longer record reads shallow. That is the fit meeting
    the residue, not the filter mis-applying the gain, and restricting the fit
    to where the tone is still above -100 dB gives 0.08 per cent at every
    duration out to 2 s.

    **A room response never gets there.** Its top band still holds content
    where a tone has gone, because the tail is broadband. Measured against a
    4096-sample frame, on ``w29_16k`` at 1.0 s and ``w27_sealed`` at 1.5 s, six
    receivers each, the default moves per-octave T30 by at most 0.37 per cent
    and per-octave energy by at most 0.104 dB, against a 3.1 per cent noise
    floor.

    So: the default suits any room response this project produces, and a caller
    filtering a tone, an impulse or anything else that empties its top band
    should raise the frame. The W10 branch carries a ``frame_for(duration_s)``
    that picks conservatively from a measured table, deliberately not
    duplicated here because this module is deleted on merge.
    """
    signals = np.atleast_2d(np.asarray(ir, dtype=float))
    if signals.ndim != 2:
        raise ValueError(f"ir must be 1- or 2-dimensional, got shape {np.shape(ir)}")
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz} Hz")
    if frame < 8 or frame % 2:
        raise ValueError(f"frame must be an even length of at least 8, got {frame}")

    atmosphere = atmosphere or Atmosphere()
    overlap = frame - frame // 4
    frequencies, times, spectra = signal.stft(
        signals,
        fs=sample_rate_hz,
        nperseg=frame,
        noverlap=overlap,
        boundary="zeros",
        padded=True,
    )
    surface = gain(
        frequencies[:, None],
        times[None, :],
        atmosphere,
        sound_speed_m_s=sound_speed_m_s,
    )
    _, restored = signal.istft(
        spectra * surface[None, :, :],
        fs=sample_rate_hz,
        nperseg=frame,
        noverlap=overlap,
    )
    out = np.asarray(restored, dtype=float)[:, : signals.shape[1]]
    if out.shape[1] < signals.shape[1]:
        out = np.pad(out, ((0, 0), (0, signals.shape[1] - out.shape[1])))
    return out[0] if np.ndim(ir) == 1 else out
