"""What a simulated response is actually worth, measured per octave band.

Until now a simulation returned one number, `rt60_broadband`. That is not the
quantity this project predicts: section 5.7 defines the target as octave-band
decay envelopes, so the broadband figure could not show whether a prediction
or an approximation was any good. Worse, it hides its own errors: a 10 dB
error at 4 kHz and an opposite one at 125 Hz average out to a broadband number
that looks excellent. Every validation recorded before this module existed is
optimistic for that reason and has to be redone with it.

Everything here works on one impulse response and returns per-band arrays on
``reverberate.acoustics.OCTAVE_BANDS``, so the metrics and the materials speak
the same language.

The measures are the standard ones, and each is here because it answers a
different question about an approximation:

- **RT60 and EDT** say how long the room rings, per band. Frequency-dependent
  absorption is invisible without them.
- **The energy decay curve** is the prediction target itself rather than a
  scalar summary of it, so comparing curves is what actually validates a model.
- **C50 and DRR** are sensitive to where the source and listener stand, and to
  directivity, which a whole-room average is not.
- **Direct sound level and arrival time** are purely geometric, which makes
  them the most sensitive thing to a geometry approximation near the source.
- **Early reflection energy** covers the first 50 ms, where the image source
  model and the real shape of the room matter most.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyroomacoustics as pra

from reverberate.acoustics import OCTAVE_BANDS, SPEED_OF_SOUND

#: Boundary between "early" and "late" energy, in seconds. 50 ms is the
#: convention C50 is defined on, and the window within which reflections fuse
#: with the direct sound rather than being heard as echoes.
EARLY_WINDOW = 0.050


def band_centres(fs: int) -> tuple[int, ...]:
    """The centre frequencies the filter bank actually produces, at this rate.

    Not the same thing as :data:`~reverberate.acoustics.OCTAVE_BANDS`, and the
    difference matters. ``OCTAVE_BANDS`` is the seven bands the material data is
    defined on, 125 Hz to 8 kHz. ``pyroomacoustics`` keeps doubling until it
    reaches Nyquist, so at 48 kHz it returns **eight** bands, the last centred on
    16 kHz. Every per-band array in this module is therefore one longer than
    ``OCTAVE_BANDS``, and labelling those arrays with ``OCTAVE_BANDS`` silently
    dropped the top band and misaligned nothing visibly, which is the worst way
    for it to be wrong. Ask the bank what it produced instead of assuming.
    """
    bank = pra.acoustics.OctaveBandsFactory(fs=fs, base_frequency=OCTAVE_BANDS[0], n_fft=512)
    return tuple(int(round(float(value))) for value in bank.centers)


def octave_filter(rir: np.ndarray, fs: int) -> np.ndarray:
    """Split a response into its octave bands.

    Uses ``pyroomacoustics``' own filter bank so that the bands a response is
    analysed on are exactly the bands its materials were defined on. Returns an
    array shaped ``(bands, samples)``. See :func:`band_centres` for why that
    first axis can be longer than :data:`~reverberate.acoustics.OCTAVE_BANDS`.
    """
    bands = pra.acoustics.OctaveBandsFactory(fs=fs, base_frequency=OCTAVE_BANDS[0], n_fft=512)
    filtered = bands.analysis(np.asarray(rir, dtype=float))
    if filtered.ndim == 1:
        filtered = filtered[:, None]
    return np.asarray(filtered).T


def energy_decay_curve(band: np.ndarray) -> np.ndarray:
    """Schroeder backward integration, normalised to 0 dB, in decibels.

    The decay curve rather than the raw envelope: integrating backwards removes
    the noise-like fluctuation of a single response and gives the smooth curve
    that reverberation times are defined on.
    """
    energy = np.asarray(band, dtype=float) ** 2
    tail = np.cumsum(energy[::-1])[::-1]
    total = tail[0] if len(tail) and tail[0] > 0 else 0.0
    if total <= 0:
        return np.full(len(energy), -np.inf)
    return 10.0 * np.log10(np.maximum(tail / total, 1e-20))


def _decay_time(curve: np.ndarray, fs: int, start_db: float, end_db: float, factor: float) -> float:
    """Seconds to fall ``start_db`` to ``end_db``, extrapolated by ``factor``.

    Returns NaN rather than a guess when the curve never reaches ``end_db``:
    a response too short or too noisy to support the measurement should say so,
    not quietly return a number that looks like a measurement.
    """
    below_start = np.flatnonzero(curve <= start_db)
    below_end = np.flatnonzero(curve <= end_db)
    if len(below_start) == 0 or len(below_end) == 0:
        return float("nan")
    first, last = int(below_start[0]), int(below_end[0])
    if last <= first:
        return float("nan")
    return float((last - first) / fs * factor)


def rt60_per_band(rir: np.ndarray, fs: int) -> np.ndarray:
    """T30 per octave band, in seconds.

    Measured over the -5 to -35 dB span and extrapolated to 60 dB, which is the
    usual practice: a real response rarely has 60 dB of clean decay above its
    noise floor, and the first 5 dB are contaminated by the direct sound.
    """
    return np.array(
        [
            _decay_time(energy_decay_curve(band), fs, -5.0, -35.0, 2.0)
            for band in octave_filter(rir, fs)
        ]
    )


def edt_per_band(rir: np.ndarray, fs: int) -> np.ndarray:
    """Early decay time per band, in seconds: the 0 to -10 dB slope scaled to 60 dB.

    EDT is dominated by the earliest reflections, so it tracks the perceived
    reverberance more closely than RT60 and is far more sensitive to the local
    geometry around the source and the listener. That makes it the better alarm
    when an approximation has flattened something nearby.
    """
    return np.array(
        [
            _decay_time(energy_decay_curve(band), fs, 0.0, -10.0, 6.0)
            for band in octave_filter(rir, fs)
        ]
    )


def direct_sound(rir: np.ndarray, fs: int) -> tuple[float, int]:
    """Level in dB and arrival sample of the direct sound.

    Taken as the response's peak, which for a source with line of sight is the
    direct path. Purely geometric, so it is the first thing to move when the
    geometry near the source changes.
    """
    response = np.asarray(rir, dtype=float)
    if len(response) == 0 or not np.any(response):
        return float("-inf"), 0
    arrival = int(np.argmax(np.abs(response)))
    level = 20.0 * np.log10(max(abs(float(response[arrival])), 1e-20))
    return level, arrival


def _split_at(band: np.ndarray, arrival: int, fs: int) -> tuple[float, float]:
    """Energy before and after the 50 ms boundary, measured from the direct sound."""
    boundary = arrival + int(EARLY_WINDOW * fs)
    energy = np.asarray(band, dtype=float) ** 2
    return float(np.sum(energy[:boundary])), float(np.sum(energy[boundary:]))


def clarity_per_band(rir: np.ndarray, fs: int) -> np.ndarray:
    """C50 per band, in dB: early energy over late energy.

    Rises with intelligibility, and unlike RT60 it depends on where the source
    and the listener actually are, which is what makes it worth reporting
    alongside a reverberation time rather than instead of it.
    """
    _, arrival = direct_sound(rir, fs)
    values = []
    for band in octave_filter(rir, fs):
        early, late = _split_at(band, arrival, fs)
        values.append(10.0 * np.log10(early / late) if late > 0 else float("inf"))
    return np.array(values)


def direct_to_reverberant_per_band(
    rir: np.ndarray, fs: int, direct_window: float = 0.0025
) -> np.ndarray:
    """DRR per band, in dB.

    The direct path is taken as a short window around the peak rather than a
    single sample, because a band-filtered impulse is spread over several
    samples and a one-sample reading would understate it at low frequency.
    """
    _, arrival = direct_sound(rir, fs)
    half = max(int(direct_window * fs), 1)
    values = []
    for band in octave_filter(rir, fs):
        energy = np.asarray(band, dtype=float) ** 2
        start, stop = max(arrival - half, 0), arrival + half
        direct = float(np.sum(energy[start:stop]))
        reverberant = float(np.sum(energy[stop:]))
        values.append(10.0 * np.log10(direct / reverberant) if reverberant > 0 else float("inf"))
    return np.array(values)


def early_energy_per_band(rir: np.ndarray, fs: int) -> np.ndarray:
    """Energy in the first 50 ms per band, in dB relative to the whole response.

    The window where the image source model does the work and where the shape
    of the room is heard rather than averaged, so it is where a badly
    approximated obstacle shows up first.
    """
    _, arrival = direct_sound(rir, fs)
    values = []
    for band in octave_filter(rir, fs):
        early, late = _split_at(band, arrival, fs)
        total = early + late
        values.append(10.0 * np.log10(early / total) if total > 0 else float("-inf"))
    return np.array(values)


@dataclass(frozen=True)
class BandMetrics:
    """Everything measurable about one response, per octave band.

    Kept as one object so that a comparison between two simulations is a
    comparison of all of it, rather than of whichever scalar was to hand.
    """

    bands: tuple[int, ...]
    rt60: np.ndarray
    edt: np.ndarray
    c50: np.ndarray
    drr: np.ndarray
    early_energy: np.ndarray
    direct_level: float
    direct_time: float
    decay_curves: np.ndarray

    def summary(self) -> str:
        parts = [
            f"{band} Hz RT60 {rt60:.2f}s C50 {c50:+.1f}dB"
            for band, rt60, c50 in zip(self.bands, self.rt60, self.c50, strict=True)
        ]
        return "; ".join(parts)


def measure(rir: np.ndarray, fs: int) -> BandMetrics:
    """Every per-band metric for one impulse response, in one pass."""
    level, arrival = direct_sound(rir, fs)
    return BandMetrics(
        bands=band_centres(fs),
        rt60=rt60_per_band(rir, fs),
        edt=edt_per_band(rir, fs),
        c50=clarity_per_band(rir, fs),
        drr=direct_to_reverberant_per_band(rir, fs),
        early_energy=early_energy_per_band(rir, fs),
        direct_level=level,
        direct_time=arrival / fs,
        decay_curves=np.array([energy_decay_curve(band) for band in octave_filter(rir, fs)]),
    )


#: Just noticeable differences, from section 5.11. Acceptance is expressed in
#: these rather than in loss values because "within the JND for reverberation
#: time" is a sentence a reader can evaluate and "MSE 0.0043" is not.
JND_RT60_RELATIVE = 0.05
JND_C50_DB = 1.0
JND_DRR_DB = 2.0


@dataclass(frozen=True)
class MetricComparison:
    """How far one response sits from another, per band and against the JNDs.

    This is what an approximation has to answer to: not "is the broadband RT60
    close" but "is every band within the difference a listener could hear".
    """

    rt60_relative_error: np.ndarray
    c50_error_db: np.ndarray
    drr_error_db: np.ndarray
    direct_level_error_db: float
    worst_band: int

    @property
    def rt60_within_jnd(self) -> bool:
        return bool(np.all(np.nan_to_num(self.rt60_relative_error, nan=0.0) <= JND_RT60_RELATIVE))

    @property
    def c50_within_jnd(self) -> bool:
        return bool(np.all(np.nan_to_num(np.abs(self.c50_error_db), nan=0.0) <= JND_C50_DB))

    @property
    def drr_within_jnd(self) -> bool:
        return bool(np.all(np.nan_to_num(np.abs(self.drr_error_db), nan=0.0) <= JND_DRR_DB))

    @property
    def within_jnd(self) -> bool:
        """Every metric, every band. One band failing is a failure."""
        return self.rt60_within_jnd and self.c50_within_jnd and self.drr_within_jnd

    def summary(self) -> str:
        worst = float(np.nanmax(np.abs(self.rt60_relative_error)))
        return (
            f"worst RT60 error {worst:.1%} at {self.worst_band} Hz, "
            f"RT60 {'ok' if self.rt60_within_jnd else 'OUT'}, "
            f"C50 {'ok' if self.c50_within_jnd else 'OUT'}, "
            f"DRR {'ok' if self.drr_within_jnd else 'OUT'}"
        )


def compare(reference: BandMetrics, candidate: BandMetrics) -> MetricComparison:
    """Measure a candidate against a reference, band by band.

    ``reference`` is the truth being approximated, so errors are relative to it.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rt60_error = np.abs(candidate.rt60 - reference.rt60) / reference.rt60
    worst_index = int(np.nanargmax(np.nan_to_num(rt60_error, nan=-1.0)))
    return MetricComparison(
        rt60_relative_error=rt60_error,
        c50_error_db=candidate.c50 - reference.c50,
        drr_error_db=candidate.drr - reference.drr,
        direct_level_error_db=candidate.direct_level - reference.direct_level,
        worst_band=reference.bands[worst_index],
    )


def _peak_within(
    freqs: np.ndarray, prominence: np.ndarray, centre: float, tolerance: float = 0.015
) -> tuple[float, float] | None:
    """The highest bin within ``tolerance`` of ``centre``, as ``(hz, db)``."""
    window = (freqs > centre * (1 - tolerance)) & (freqs < centre * (1 + tolerance))
    if not window.any():
        return None
    at = int(np.flatnonzero(window)[np.argmax(prominence[window])])
    return float(freqs[at]), float(prominence[at])


def axial_modes(
    rir: np.ndarray,
    fs: int,
    *,
    height_m: float,
    sound_speed_m_s: float = SPEED_OF_SOUND,
    receiver_height_m: float | None = None,
    start_s: float = 0.3,
    band_hz: tuple[float, float] = (300.0, 1200.0),
    search: float = 0.06,
) -> dict[str, Any]:
    """The floor to ceiling axial series in the late response, found and sized.

    Two large parallel surfaces with the least absorption in the room, the
    floor and the ceiling, keep a comb of axial modes at ``n c / 2H`` ringing
    after the diffuse field has decayed. W39 met it as a "line at 701 Hz" that
    a pocket census could not explain: it was ``n = 15`` of a 3.63 m room, and
    the missing ``n = 14`` sat on a pressure node at the listener's height.

    The late part of ``rir`` (from ``start_s``) is transformed, each bin is
    compared with a 60 Hz moving mean of the spectrum, and the comb spacing is
    searched within ``search`` of ``c / 2H`` for the one whose harmonics stand
    highest above that floor. ``prominence_db`` per line and its mean are what
    to read against ``between_lines_db``, the same statistic half way between
    the lines; a ``contrast_db`` above about 6 dB is a flutter a listener hears
    as a pitch in the tail. Nothing here removes it: the report says
    what the model's flat parallel surfaces did.
    """
    signal = np.asarray(rir, dtype=float)
    late = signal[int(round(start_s * fs)) :]
    if late.size < fs // 4:
        return {"found": False, "why": f"under 0.25 s of response after {start_s} s"}
    late = late * np.hanning(late.size)
    n = 1 << int(np.ceil(np.log2(late.size * 8)))
    spectrum = np.abs(np.fft.rfft(late, n=n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    width = max(int(round(60.0 / (freqs[1] - freqs[0]))), 3)
    kernel = np.ones(width) / width
    floor = np.convolve(spectrum, kernel, mode="same")
    prominence = 10.0 * np.log10((spectrum + 1e-30) / (floor + 1e-30))

    nominal = sound_speed_m_s / (2.0 * height_m)
    best: tuple[float, float, list[dict[str, Any]], float, float] | None = None
    for spacing in np.linspace(nominal * (1 - search), nominal * (1 + search), 61):
        orders = np.arange(int(np.ceil(band_hz[0] / spacing)), int(band_hz[1] / spacing) + 1)
        if orders.size < 4:
            continue
        rows, between = [], []
        for order in orders:
            centre = order * spacing
            found = _peak_within(freqs, prominence, centre)
            if found is None:
                continue
            rows.append(
                {
                    "n": int(order),
                    "predicted_hz": round(float(centre), 1),
                    "found_hz": round(float(found[0]), 1),
                    "prominence_db": round(float(found[1]), 1),
                }
            )
            # The same statistic half way to the next line: what a peak picked
            # out of noise alone measures, and what a line is compared with.
            offset = _peak_within(freqs, prominence, centre + 0.5 * spacing)
            if offset is not None:
                between.append(offset[1])
        if len(rows) < 4 or not between:
            continue
        lines_db = float(np.mean([row["prominence_db"] for row in rows]))
        between_db = float(np.mean(between))
        score = lines_db - between_db
        if best is None or score > best[0]:
            best = (score, float(spacing), rows, lines_db, between_db)
    if best is None:
        return {"found": False, "why": "the band holds fewer than four harmonics"}
    score, spacing, rows, lines_db, between_db = best
    fitted_height = sound_speed_m_s / (2.0 * spacing)
    if receiver_height_m is not None:
        for row in rows:
            shape = abs(np.cos(np.pi * row["n"] * receiver_height_m / fitted_height))
            row["at_pressure_node"] = bool(shape < 0.25)
    return {
        "found": True,
        "height_m": round(float(height_m), 3),
        "spacing_hz": round(spacing, 2),
        "height_from_spacing_m": round(fitted_height, 3),
        "late_from_s": start_s,
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "lines": rows,
        "mean_prominence_db": round(lines_db, 2),
        "between_lines_db": round(between_db, 2),
        "contrast_db": round(score, 2),
        "flutter": bool(score > 6.0),
        "note": (
            "axial modes between the floor and the ceiling, n c / 2H; flat parallel "
            "surfaces in the model ring where a real room's scattering breaks them"
        ),
    }


def schroeder_frequency(rt60: float, volume: float) -> float:
    """Above this frequency the room is diffuse, below it modal, in Hz.

    Worth reporting next to the metrics: below it a room does not have a
    reverberation time in the statistical sense at all, so a large low-band
    error may be the measurement losing its meaning rather than the geometry
    being wrong. A useful guard against chasing a defect that is not there.
    """
    if volume <= 0 or rt60 <= 0:
        return float("inf")
    return float(2000.0 * np.sqrt(rt60 / volume))


def critical_distance(volume: float, rt60: float) -> float:
    """Where direct and reverberant energy are equal, in metres.

    Sets the scale for source and listener placement: closer than this the
    direct sound dominates and geometry near the source matters most, further
    away the diffuse field does and the room as a whole matters more.
    """
    if rt60 <= 0 or volume <= 0:
        return float("inf")
    return float(0.057 * np.sqrt(volume / rt60))


def wavelength_of(frequency: float) -> float:
    """Wavelength in metres, for deciding what a feature of a given size does."""
    return SPEED_OF_SOUND / float(frequency)
