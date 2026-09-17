"""Criteria A and B: what a candidate response must reproduce of the reference.

A response is judged in two parts that never trade against each other. Before
the **mixing time** it is a set of discrete reflections, each with a time, a
direction and a level per octave; that part carries localisation. After it the
response is a decorrelated tail whose statistics carry the timbre of the room.
Nothing is compared sample by sample: W3 measured a -6 dB waveform floor
between two grids of the same room, so a sample-wise error would measure the
solver's own noise.

**Part A, the reflections.** The 64 channels are beamformed on a grid of
directions, per octave band, into a spatial echogram (time, direction, band).
A reflection is a peak of the broadband echogram above the local floor and
above a level relative to the direct sound; its attributes are read from the
lobe around the peak. The same detector runs on the reference and on the
candidate, so neither side is favoured, and the two lists are matched by a
gate in time and angle and then an assignment of minimum cost. From the
matching come recall, precision and the median errors of time, direction and
level; from the echograms comes a continuous distance that the calibration
minimises.

**Part B, the tail.** After the mixing time, measured on both responses by
the echo density profile of Abel and Huang: decay times per octave, the
colour of the tail relative to the direct sound, the energy per spherical
harmonic order (flat for a diffuse field), the energy per direction sector
(the anisotropy a corridor or an open door leaves), the interaural coherence
after a binaural decode, and the continuity of level and colour across the
seam where a synthesised tail takes over.

Every function works in the ambisonic frame of :mod:`reverberate.spatial.sh`
(x front, y left, z up), on signals ``[channel, sample]`` in ACN and N3D at
one sample rate, which is what a field of ``docs/formats/ambisonic-field.md``
holds. Directions come back as unit vectors in that frame with azimuth and
elevation beside them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt
from scipy.special import erfc

from reverberate.audio import lowpass
from reverberate.metrics import band_centres, edt_per_band, octave_filter_rows, rt60_per_band
from reverberate.spatial.binaural import (
    BinauralDecoder,
    coherence_floor,
    design_decoder,
    ild_db,
    interaural_coherence,
    itd_s,
    render,
)
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.hrtf import sphere_head
from reverberate.spatial.sh import channel_count, degrees_of, quadrature, real_sh

__all__ = [
    "BandLevels",
    "Criteria",
    "CriteriaSettings",
    "Echogram",
    "Matching",
    "PointReport",
    "Reflection",
    "TailReport",
    "Targets",
    "aggregate",
    "beam_weights",
    "detect_reflections",
    "echogram",
    "echogram_distance_db",
    "judge",
    "match_reflections",
    "mixing_time_s",
    "sector_directions",
    "tail_statistics",
]

#: Octave bands a reflection's level is read on. 125 Hz is left out because an
#: octave filtered pulse there is eight milliseconds wide and says nothing
#: about one reflection; 16 kHz is left out because the reference is solved
#: to 8 kHz and synthesised above.
LEVEL_BANDS_HZ: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000)

#: ``erfc(1 / sqrt 2)``: the share of Gaussian samples beyond one standard
#: deviation, which normalises the echo density of a diffuse tail to one.
_GAUSSIAN_TAIL = float(erfc(1.0 / np.sqrt(2.0)))


@dataclass(frozen=True)
class CriteriaSettings:
    """Every number the judgement depends on that is a choice."""

    #: The response is judged below this frequency: the reference is solved
    #: to 8 kHz and synthesised above, and the synthesis is not the solver.
    band_limit_hz: float = 8000.0
    #: Reflections are detected on the response above this frequency, where
    #: a pulse is short enough to have a time of its own.
    detection_low_hz: float = 500.0
    #: The mirror answers above this frequency only: a wave solve covers the
    #: rest. Decay, colour and seam are judged on the octave bands at and above
    #: it, the echogram too, the reflections and the late field's directions
    #: on the response high passed there. Zero judges every band.
    focus_low_hz: float = 1000.0
    #: Reflections are looked for this long after the direct sound.
    early_s: float = 0.050
    #: The echogram spans this long after the direct sound.
    echogram_s: float = 0.080
    #: Time bin of the echogram.
    time_bin_s: float = 0.001
    #: Quadrature degree of the echogram's direction grid: 16 gives 153
    #: directions, about twenty degrees apart.
    echogram_degree: int = 16
    #: Quadrature degree of the detection grid: 30 gives 496 directions.
    detection_degree: int = 30
    #: Window over which a reflection's energy is integrated, and the
    #: half width of the local maximum test.
    window_s: float = 0.0005
    #: A reflection must rise this far above the local floor, which is the
    #: median of the total energy over ``floor_span_s`` around it.
    prominence_db: float = 6.0
    floor_span_s: float = 0.010
    #: Reflections below this level relative to the direct sound are neither
    #: counted nor expected.
    level_floor_db: float = -15.0
    #: Two peaks closer than this in time and angle are one reflection. Half
    #: a millisecond is the energy window: a coloured pulse read through it
    #: can show two humps that far apart, and two arrivals closer than that
    #: are one arrival at 8 kHz anyway.
    merge_time_s: float = 0.0005
    merge_angle_deg: float = 20.0
    #: Matching gates: a pair farther apart than either is never matched.
    gate_time_s: float = 0.0002
    gate_angle_deg: float = 15.0
    #: The direction of a reflection is the energy centroid of the lobe
    #: within this angle of the strongest direction.
    lobe_deg: float = 25.0
    #: Echo density window of the mixing time, and the value it must reach.
    echo_window_s: float = 0.020
    echo_threshold: float = 0.95
    #: The colour of the tail is read over this long after the mixing time.
    tail_colour_s: float = 0.100
    #: Level and colour are read this long either side of the seam.
    seam_s: float = 0.010
    #: Order of the binaural decode the interaural figures are read on, and
    #: the octave the tail's coherence is read in.
    binaural_order: int = 3
    coherence_band_hz: float = 2000.0
    #: Window after the direct sound the interaural time and level are read on.
    direct_window_s: float = 0.005
    #: The early coherence is read over this window after the direct sound.
    early_coherence_s: float = 0.080

    def record(self) -> dict[str, Any]:
        return asdict(self)


#: The settings every function takes when none are given.
DEFAULT_SETTINGS = CriteriaSettings()


@dataclass(frozen=True)
class Targets:
    """The acceptance thresholds agreed on 2026-09-16, as numbers a report can quote."""

    recall: float = 0.90
    precision: float = 0.90
    time_error_s: float = 0.0001
    direction_error_deg: float = 10.0
    level_error_db: float = 1.0
    itd_error_s: float = 20e-6
    ild_error_db: float = 1.0
    early_coherence_error: float = 0.075
    mixing_time_relative: float = 0.20
    t30_relative: float = 0.05
    edt_relative: float = 0.05
    tail_colour_db: float = 1.0
    order_energy_db: float = 1.0
    sector_energy_db: float = 2.0
    late_coherence_error: float = 0.05
    seam_db: float = 1.0

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BandLevels:
    """Energy per octave band, in decibels relative to the direct sound's."""

    centres_hz: tuple[int, ...]
    levels_db: tuple[float, ...]


@dataclass(frozen=True)
class Reflection:
    """One discrete arrival: when, from where, how strong, and per band."""

    time_s: float
    unit_vector: tuple[float, float, float]
    azimuth_deg: float
    elevation_deg: float
    #: Broadband level relative to the direct sound, in dB.
    level_db: float
    bands: BandLevels
    #: Reflections are numbered in time order; the direct sound is 0.
    index: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time_ms": round(self.time_s * 1000.0, 4),
            "unit_vector": [round(v, 5) for v in self.unit_vector],
            "azimuth_deg": round(self.azimuth_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "level_db": round(self.level_db, 3),
            "bands_hz": list(self.bands.centres_hz),
            "band_levels_db": [round(v, 3) for v in self.bands.levels_db],
        }


@dataclass(frozen=True)
class Echogram:
    """Energy per time bin, direction and band, and what it was read on."""

    #: ``[bin, direction, band]``, linear energy, normalised by the direct
    #: sound's broadband energy.
    energy: np.ndarray
    times_s: np.ndarray
    directions: np.ndarray
    bands_hz: tuple[int, ...]


@dataclass(frozen=True)
class Matching:
    """How two lists of reflections line up."""

    pairs: tuple[tuple[int, int], ...]
    unmatched_reference: tuple[int, ...]
    unmatched_candidate: tuple[int, ...]
    recall: float
    precision: float
    time_errors_s: tuple[float, ...]
    direction_errors_deg: tuple[float, ...]
    level_errors_db: tuple[float, ...]
    band_level_errors_db: tuple[tuple[float, ...], ...]

    def record(self) -> dict[str, Any]:
        return {
            "pairs": [list(p) for p in self.pairs],
            "unmatched_reference": list(self.unmatched_reference),
            "unmatched_candidate": list(self.unmatched_candidate),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "time_error_median_ms": _median_ms(self.time_errors_s),
            "direction_error_median_deg": _median(self.direction_errors_deg),
            "level_error_median_db": _median(self.level_errors_db),
            "band_level_error_median_db": [
                _median([row[k] for row in self.band_level_errors_db])
                for k in range(len(self.band_level_errors_db[0]))
            ]
            if self.band_level_errors_db
            else [],
        }


@dataclass(frozen=True)
class TailReport:
    """Part B of one response."""

    mixing_time_s: float
    bands_hz: tuple[int, ...]
    t30_s: tuple[float, ...]
    edt_s: tuple[float, ...]
    colour_db: tuple[float, ...]
    #: Mean energy per channel of each degree, in dB relative to degree 0.
    order_energy_db: tuple[float, ...]
    #: Energy of each sector beam over the tail, in dB relative to their mean.
    sector_energy_db: tuple[float, ...]
    early_coherence: float
    late_coherence: float
    coherence_floor: float
    direct_itd_s: float
    direct_ild_db: float
    #: Level step across the seam per band: energy after over energy before, dB.
    seam_step_db: tuple[float, ...]

    def record(self) -> dict[str, Any]:
        return {
            "mixing_time_ms": round(self.mixing_time_s * 1000.0, 3),
            "bands_hz": list(self.bands_hz),
            "t30_s": [_round_or_none(v, 4) for v in self.t30_s],
            "edt_s": [_round_or_none(v, 4) for v in self.edt_s],
            "colour_db": [_round_or_none(v, 3) for v in self.colour_db],
            "order_energy_db": [_round_or_none(v, 3) for v in self.order_energy_db],
            "sector_energy_db": [_round_or_none(v, 3) for v in self.sector_energy_db],
            "early_coherence": round(self.early_coherence, 4),
            "late_coherence": round(self.late_coherence, 4),
            "coherence_floor": round(self.coherence_floor, 4),
            "direct_itd_us": round(self.direct_itd_s * 1e6, 2),
            "direct_ild_db": round(self.direct_ild_db, 3),
            "seam_step_db": [_round_or_none(v, 3) for v in self.seam_step_db],
        }


@dataclass(frozen=True)
class PointReport:
    """The whole judgement of one candidate response against its reference."""

    reference: tuple[Reflection, ...]
    candidate: tuple[Reflection, ...]
    matching: Matching
    echogram_distance_db: tuple[float, ...]
    reference_tail: TailReport
    candidate_tail: TailReport
    verdicts: dict[str, bool]
    errors: dict[str, float]

    def record(self) -> dict[str, Any]:
        return {
            "reference_reflections": [r.record() for r in self.reference],
            "candidate_reflections": [r.record() for r in self.candidate],
            "matching": self.matching.record(),
            "echogram_distance_db": [round(v, 3) for v in self.echogram_distance_db],
            "reference_tail": self.reference_tail.record(),
            "candidate_tail": self.candidate_tail.record(),
            "errors": {k: _round_or_none(v, 5) for k, v in self.errors.items()},
            "verdicts": dict(self.verdicts),
            "passed": int(sum(self.verdicts.values())),
            "of": len(self.verdicts),
        }


@dataclass
class Criteria:
    """The settings and targets, and the prepared grids and decoder they imply."""

    settings: CriteriaSettings = field(default_factory=CriteriaSettings)
    targets: Targets = field(default_factory=Targets)

    def record(self) -> dict[str, Any]:
        return {"settings": self.settings.record(), "targets": self.targets.record()}


# --------------------------------------------------------------------------
# beams
# --------------------------------------------------------------------------


def max_re_weights(order: int) -> np.ndarray:
    """Per degree max-rE weights, ``P_n(cos(137.9 deg / (N + 1.51)))``, unit at degree 0."""
    angle = np.radians(137.9 / (order + 1.51))
    return np.asarray(
        [float(np.polynomial.legendre.Legendre.basis(n)(np.cos(angle))) for n in range(order + 1)]
    )


def beam_weights(order: int, unit_vectors: np.ndarray) -> np.ndarray:
    """``[direction, channel]`` weights whose beam answers one to a unit plane wave from its axis.

    Max-rE weighted plane wave decomposition: the harmonics of the direction,
    each degree tapered, divided by the beam's own response on axis, which is
    ``sum_n g_n (2n + 1)`` in N3D. Sidelobes of a plain hypercardioid beam
    at order 7 sit at -8 dB; tapered they fall to about -18 dB, which is what
    keeps a strong reflection from being seen twice.
    """
    per_degree = max_re_weights(order)
    taper = per_degree[degrees_of(order)]
    basis = real_sh(order, unit_vectors) * taper[None, :]
    # sum over channels of g_n Y_nm(u)^2 is sum_n g_n (2n + 1) in N3D.
    on_axis = float(np.sum(per_degree * (2.0 * np.arange(order + 1) + 1.0)))
    return np.asarray(basis / on_axis)


def sector_directions() -> np.ndarray:
    """Twelve directions, the vertices of an icosahedron, in the ambisonic frame."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array(
        [
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ],
        dtype=float,
    )
    return np.asarray(raw / np.linalg.norm(raw, axis=1, keepdims=True))


@lru_cache(maxsize=8)
def _grid(degree: int) -> np.ndarray:
    return quadrature(degree)[0]


@lru_cache(maxsize=16)
def _beams(order: int, degree: int) -> np.ndarray:
    return beam_weights(order, _grid(degree))


@lru_cache(maxsize=8)
def _neighbours(degree: int, within_deg: float) -> tuple[np.ndarray, ...]:
    """For each direction of the grid, the indices of those within ``within_deg`` of it."""
    grid = _grid(degree)
    cosines = np.clip(grid @ grid.T, -1.0, 1.0)
    close = cosines >= np.cos(np.radians(within_deg))
    return tuple(np.flatnonzero(row) for row in close)


@lru_cache(maxsize=4)
def _decoder(order: int, sample_rate_hz: float) -> BinauralDecoder:
    taps = 512
    return design_decoder(
        sphere_head(sample_rate_hz, taps),
        order=order,
        sample_rate_hz=sample_rate_hz,
        filter_length=taps,
    )


def _bandpass(signals: np.ndarray, rate: float, low_hz: float | None, high_hz: float) -> np.ndarray:
    """Zero phase band limiting, so the timing of a pulse is not moved."""
    nyquist = 0.5 * rate
    high = min(high_hz, 0.98 * nyquist)
    if low_hz is not None and low_hz > 0.0:
        sos = butter(4, [low_hz / nyquist, high / nyquist], btype="band", output="sos")
    else:
        sos = butter(4, high / nyquist, btype="low", output="sos")
    return np.asarray(sosfiltfilt(sos, signals, axis=-1), dtype=float)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def _azimuth_elevation(unit: np.ndarray) -> tuple[float, float]:
    return (
        float(np.degrees(np.arctan2(unit[1], unit[0]))),
        float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0)))),
    )


def _median(values: Any) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if array.size == 0 else round(float(np.median(array)), 4)


def _median_ms(values: Any) -> float | None:
    array = np.asarray(list(values), dtype=float) * 1000.0
    return None if array.size == 0 else round(float(np.median(array)), 4)


def _round_or_none(value: float, digits: int) -> float | None:
    return None if not np.isfinite(value) else round(float(value), digits)


# --------------------------------------------------------------------------
# part A: reflections
# --------------------------------------------------------------------------


def direct_arrival(signals: np.ndarray, rate: float, settings: CriteriaSettings) -> int:
    """The sample of the direct sound: the first peak of the band limited omni channel."""
    omni = _bandpass(signals[0:1], rate, settings.detection_low_hz, settings.band_limit_hz)[0]
    envelope = omni**2
    peak = float(envelope.max())
    if peak <= 0.0:
        raise ValueError("a silent response has no direct sound")
    # The first sample within 6 dB of the strongest one, then the local maximum
    # after it: the direct sound is the earliest strong arrival, not the
    # strongest, which in a corner can be a reflection.
    first = int(np.flatnonzero(envelope >= peak * 10 ** (-6.0 / 10.0))[0])
    window = int(round(settings.window_s * rate))
    stop = min(first + window, envelope.size)
    return first + int(np.argmax(envelope[first:stop]))


def _band_signals(
    signals: np.ndarray, rate: float, bands_hz: tuple[int, ...]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """``[band, channel, sample]`` through the project's octave bank."""
    centres = band_centres(int(round(rate)))
    indices = np.array([centres.index(b) for b in bands_hz], dtype=int)
    channels = signals.shape[0]
    rows = np.repeat(signals, len(indices), axis=0)
    row_bands = np.tile(indices, channels)
    filtered = octave_filter_rows(rows, int(round(rate)), row_bands)
    return filtered.reshape(channels, len(indices), -1).transpose(1, 0, 2), bands_hz


def echogram(
    ambisonic: Ambisonic,
    settings: CriteriaSettings = DEFAULT_SETTINGS,
    *,
    bands_hz: tuple[int, ...] = LEVEL_BANDS_HZ,
) -> Echogram:
    """Energy per time bin, direction and octave band after the direct sound.

    Normalised by the broadband energy of the direct sound over one window,
    so two responses of different gain are compared on their structure.
    """
    rate = ambisonic.sample_rate_hz
    start = direct_arrival(ambisonic.signals, rate, settings)
    length = int(round(settings.echogram_s * rate))
    window = int(round(settings.window_s * rate))
    lead = window  # a little before the direct, so its own bin is whole
    begin = max(start - lead, 0)
    # Filtering needs context on both sides of the excerpt.
    pad = 4096
    lo = max(begin - pad, 0)
    hi = min(begin + length + pad, ambisonic.signals.shape[1])
    excerpt = _bandpass(ambisonic.signals[:, lo:hi], rate, None, settings.band_limit_hz)
    per_band, _ = _band_signals(excerpt, rate, bands_hz)
    per_band = per_band[:, :, begin - lo : begin - lo + length]
    beams = _beams(ambisonic.order, settings.echogram_degree)
    bin_samples = int(round(settings.time_bin_s * rate))
    bins = length // bin_samples
    energy = np.zeros((bins, beams.shape[0], len(bands_hz)))
    for b in range(len(bands_hz)):
        steered = beams @ per_band[b]  # [direction, sample]
        squared = steered[:, : bins * bin_samples] ** 2
        energy[:, :, b] = squared.reshape(beams.shape[0], bins, bin_samples).sum(axis=2).T
    direct = _bandpass(ambisonic.signals[0:1, lo:hi], rate, None, settings.band_limit_hz)[0]
    reference = float(np.sum(direct[start - lo : start - lo + window] ** 2))
    times = (np.arange(bins) * bin_samples + begin - start) / rate
    return Echogram(
        energy=energy / max(reference, 1e-30),
        times_s=times,
        directions=_grid(settings.echogram_degree),
        bands_hz=bands_hz,
    )


def echogram_distance_db(a: Echogram, b: Echogram) -> tuple[float, ...]:
    """Per band, the energy weighted mean absolute difference in decibels.

    Weighted by the larger of the two energies in each cell, so the cells that
    hold the reflections decide and the empty ones do not: a log spectral
    distance on the spatial echogram.
    """
    if a.energy.shape != b.energy.shape:
        raise ValueError(f"echograms of different shapes: {a.energy.shape} and {b.energy.shape}")
    floor = 1e-9
    out = []
    for band in range(a.energy.shape[2]):
        ea = a.energy[:, :, band] + floor
        eb = b.energy[:, :, band] + floor
        weight = np.maximum(ea, eb)
        difference = np.abs(10.0 * np.log10(ea / eb))
        out.append(float(np.sum(weight * difference) / np.sum(weight)))
    return tuple(out)


def detect_reflections(
    ambisonic: Ambisonic,
    settings: CriteriaSettings = DEFAULT_SETTINGS,
    *,
    bands_hz: tuple[int, ...] = LEVEL_BANDS_HZ,
) -> tuple[Reflection, ...]:
    """The direct sound and the discrete reflections of the early part.

    The first entry is the direct sound, at index 0 and level 0 dB. The rest
    are the peaks of the broadband detection echogram, merged by proximity,
    above the local floor by ``prominence_db`` and above ``level_floor_db``
    relative to the direct sound, in time order.
    """
    rate = ambisonic.sample_rate_hz
    signals = ambisonic.signals
    start = direct_arrival(signals, rate, settings)
    length = int(round(settings.early_s * rate))
    window = int(round(settings.window_s * rate))
    half = max(window // 2, 1)
    begin = max(start - window, 0)
    pad = 4096
    lo = max(begin - pad, 0)
    hi = min(begin + length + window + pad, signals.shape[1])
    detection = _bandpass(
        signals[:, lo:hi],
        rate,
        max(settings.detection_low_hz, settings.focus_low_hz),
        settings.band_limit_hz,
    )
    limited = _bandpass(signals[:, lo:hi], rate, None, settings.band_limit_hz)
    per_band, _ = _band_signals(limited, rate, bands_hz)
    offset = begin - lo
    span = length + window
    beams = _beams(ambisonic.order, settings.detection_degree)
    grid = _grid(settings.detection_degree)
    steered = beams @ detection[:, offset : offset + span]  # [direction, sample]
    # Energy over a sliding window: the quantity a reflection is a peak of.
    smoothed = uniform_filter1d(steered**2, size=window, axis=1, mode="constant") * window
    total = smoothed.max(axis=0)
    direct_at = start - begin
    direct_energy = float(total[direct_at])
    if direct_energy <= 0.0:
        raise ValueError("no energy at the direct sound")
    floor_span = int(round(settings.floor_span_s * rate))
    threshold = direct_energy * 10 ** (settings.level_floor_db / 10.0)
    prominence = 10 ** (settings.prominence_db / 10.0)
    # The local floor around each sample: the median of the total energy over
    # the floor span, computed once for every sample.
    floor = median_filter(total, size=2 * floor_span + 1, mode="nearest")
    # A peak is the largest value of its own direction within the window, above
    # the level floor and above the local floor by the prominence; the first
    # sample of a plateau is the peak, as the argmax would say.
    local_max = maximum_filter1d(smoothed, size=2 * half + 1, axis=1, mode="constant")
    # And the largest over the directions within the lobe at that sample:
    # every direction inside a beam's main lobe peaks at the same instant,
    # and only the axis of the lobe is the reflection.
    lobe_max = np.empty_like(smoothed)
    for k, near_k in enumerate(_neighbours(settings.detection_degree, settings.lobe_deg)):
        lobe_max[k] = smoothed[near_k].max(axis=0)
    is_peak = (
        (smoothed >= local_max)
        & (smoothed >= lobe_max)
        & (smoothed >= threshold)
        & (smoothed >= floor * prominence)
    )
    is_peak[:, : direct_at + half] = False
    is_peak[:, span - half :] = False
    ks, ns = np.nonzero(is_peak)
    candidates: list[tuple[int, int, float]] = [
        (int(n), int(k), float(smoothed[k, n])) for k, n in zip(ks, ns, strict=True)
    ]
    candidates.sort(key=lambda c: -c[2])
    merge_samples = int(round(settings.merge_time_s * rate))
    kept: list[tuple[int, int, float]] = []
    for n, k, value in candidates:
        near = any(
            abs(n - m) <= merge_samples and _angle_deg(grid[k], grid[j]) <= settings.merge_angle_deg
            for m, j, _ in kept
        )
        if not near:
            kept.append((n, k, value))
    kept.sort(key=lambda c: c[0])

    def describe(n: int, k: int, index: int) -> Reflection:
        # Direction: the energy centroid of the lobe around the strongest direction.
        column = smoothed[:, n]
        within = np.zeros(len(grid), dtype=bool)
        within[_neighbours(settings.detection_degree, settings.lobe_deg)[k]] = True
        weights = np.where(within, np.maximum(column - column[within].min(), 0.0), 0.0)
        if weights.sum() <= 0.0:
            weights = within.astype(float)
        unit = weights @ grid
        norm = float(np.linalg.norm(unit))
        unit = unit / norm if norm > 0 else grid[k]
        # Time: the energy centroid of the beam's own energy over one window
        # either side of the peak. The bank's pulses are symmetric, so the
        # centroid is the arrival; a coloured pulse whose window energy shows
        # two humps still centres where it arrived.
        energy_t = steered[k, max(n - window, 0) : min(n + window + 1, span)] ** 2
        samples_t = np.arange(max(n - window, 0), min(n + window + 1, span))
        refined = (
            float(np.sum(energy_t * samples_t) / np.sum(energy_t))
            if energy_t.sum() > 0.0
            else float(n)
        )
        # Levels: the beam toward the refined direction, energy over the
        # window, broadband on the detection band and per octave. On the
        # refined direction rather than the grid point, because the grid is
        # eleven degrees apart and a beam a grid step off its axis reads a
        # decibel low; the direct sound is read the same way, so the ratio
        # carries no such loss.
        beam = beam_weights(ambisonic.order, unit[None, :])[0]
        centre_n = int(round(refined))
        a, b = max(centre_n - half, 0), min(centre_n + half + 1, span)
        broadband = float(np.sum((beam @ detection[:, offset + a : offset + b]) ** 2))
        levels = []
        for band in range(len(bands_hz)):
            steer = beam @ per_band[band][:, offset + a : offset + b]
            levels.append(float(np.sum(steer**2)))
        azimuth, elevation = _azimuth_elevation(unit)
        return Reflection(
            time_s=(refined + begin - start) / rate,
            unit_vector=(float(unit[0]), float(unit[1]), float(unit[2])),
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            level_db=broadband,
            bands=BandLevels(bands_hz, tuple(levels)),
            index=index,
        )

    direct_direction = int(np.argmax(smoothed[:, direct_at]))
    direct = describe(direct_at, direct_direction, 0)
    reflections = [direct]
    for index, (n, k, _) in enumerate(kept, start=1):
        reflections.append(describe(n, k, index))
    # Every level relative to the direct sound's own, so a gain cancels, and
    # every time relative to the direct sound's refined arrival, so the two
    # responses are read with the same clock.
    reference = np.asarray(direct.bands.levels_db, dtype=float)
    out = []
    for r in reflections:
        levels = np.asarray(r.bands.levels_db, dtype=float)
        relative = 10.0 * np.log10(np.maximum(levels, 1e-30) / np.maximum(reference, 1e-30))
        out.append(
            Reflection(
                r.time_s - direct.time_s,
                r.unit_vector,
                r.azimuth_deg,
                r.elevation_deg,
                float(10.0 * np.log10(max(r.level_db, 1e-30) / max(direct.level_db, 1e-30))),
                BandLevels(bands_hz, tuple(float(v) for v in relative)),
                r.index,
            )
        )
    return tuple(out)


def match_reflections(
    reference: tuple[Reflection, ...],
    candidate: tuple[Reflection, ...],
    settings: CriteriaSettings = DEFAULT_SETTINGS,
) -> Matching:
    """Pair the two lists: a gate in time and angle, then the assignment of least cost.

    The direct sounds (index 0) are paired with each other and left out of
    the counts. Recall is over the reference's reflections above the level
    floor; precision over the candidate's.
    """
    ref = [r for r in reference if r.index != 0]
    cand = [c for c in candidate if c.index != 0]
    big = 1e6
    cost = np.full((len(ref), len(cand)), big)
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            dt = abs(r.time_s - c.time_s)
            angle = _angle_deg(np.asarray(r.unit_vector), np.asarray(c.unit_vector))
            if dt <= settings.gate_time_s and angle <= settings.gate_angle_deg:
                cost[i, j] = (
                    dt / settings.gate_time_s
                    + angle / settings.gate_angle_deg
                    + abs(r.level_db - c.level_db) / 3.0
                )
    pairs: list[tuple[int, int]] = []
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if cost[i, j] < big]
    matched_ref = {i for i, _ in pairs}
    matched_cand = {j for _, j in pairs}
    time_errors = tuple(abs(ref[i].time_s - cand[j].time_s) for i, j in pairs)
    direction_errors = tuple(
        _angle_deg(np.asarray(ref[i].unit_vector), np.asarray(cand[j].unit_vector))
        for i, j in pairs
    )
    level_errors = tuple(abs(ref[i].level_db - cand[j].level_db) for i, j in pairs)
    band_errors = tuple(
        tuple(
            abs(a - b) for a, b in zip(ref[i].bands.levels_db, cand[j].bands.levels_db, strict=True)
        )
        for i, j in pairs
    )
    return Matching(
        pairs=tuple((ref[i].index, cand[j].index) for i, j in pairs),
        unmatched_reference=tuple(r.index for i, r in enumerate(ref) if i not in matched_ref),
        unmatched_candidate=tuple(c.index for j, c in enumerate(cand) if j not in matched_cand),
        recall=len(pairs) / len(ref) if ref else 1.0,
        precision=len(pairs) / len(cand) if cand else 1.0,
        time_errors_s=time_errors,
        direction_errors_deg=direction_errors,
        level_errors_db=level_errors,
        band_level_errors_db=band_errors,
    )


# --------------------------------------------------------------------------
# part B: the tail
# --------------------------------------------------------------------------


def mixing_time_s(
    omni: np.ndarray, rate: float, settings: CriteriaSettings = DEFAULT_SETTINGS
) -> float:
    """Abel and Huang's echo density: when the response first looks Gaussian.

    Over a sliding window the share of samples beyond one standard deviation
    is divided by that share for a Gaussian; a sparse early part reads well
    under one, a diffuse tail reads one. The mixing time is the first window
    centre where the profile reaches ``echo_threshold``, measured from the
    direct sound.
    """
    signal = np.asarray(omni, dtype=float)
    start = direct_arrival(signal[None, :], rate, settings)
    window = max(int(round(settings.echo_window_s * rate)), 16)
    hop = max(window // 8, 1)
    profile = []
    centres = []
    for begin in range(start, signal.size - window + 1, hop):
        block = signal[begin : begin + window]
        sigma = float(np.std(block))
        share = float(np.mean(np.abs(block) > sigma)) if sigma > 0.0 else 0.0
        profile.append(share / _GAUSSIAN_TAIL)
        centres.append(begin + window / 2.0)
    for value, centre in zip(profile, centres, strict=True):
        if value >= settings.echo_threshold:
            return float((centre - start) / rate)
    return float("nan")


def _sector_energy_db(
    signals: np.ndarray, order: int, begin: int, end: int, directions: np.ndarray
) -> tuple[float, ...]:
    use = min(order, 3)
    keep = channel_count(use)
    beams = beam_weights(use, directions)
    steered = beams @ signals[:keep, begin:end]
    energy = np.sum(steered**2, axis=1)
    mean = float(np.mean(energy))
    if mean <= 0.0:
        return tuple(float("nan") for _ in energy)
    return tuple(float(10.0 * np.log10(max(e, 1e-30) / mean)) for e in energy)


def _order_energy_db(signals: np.ndarray, order: int, begin: int, end: int) -> tuple[float, ...]:
    degrees = degrees_of(order)
    per_channel = np.sum(signals[:, begin:end] ** 2, axis=1)
    means = np.array([float(per_channel[degrees == n].mean()) for n in range(order + 1)])
    if means[0] <= 0.0:
        return tuple(float("nan") for _ in means)
    return tuple(float(10.0 * np.log10(max(m, 1e-30) / means[0])) for m in means)


def _coherence(brir: np.ndarray, rate: float, band_hz: float) -> float:
    """The frames' interaural coherence in the band, each frame weighted by its energy.

    :func:`interaural_coherence` gives one value per 20 ms frame; a plain
    mean over a response's late part counts a frame 100 dB down as much as
    the first one, and on 0076 the reference's frames grow coherent only
    below -40 dB, where nothing is heard. The weights are the frames' energy
    at the two ears in the same band, framed as the coherence is.
    """
    _, values = interaural_coherence(brir, rate, band_hz=band_hz)
    if not values.size:
        return float("nan")
    high = min(band_hz * np.sqrt(2.0), 0.49 * rate)
    block = lowpass(np.asarray(brir, dtype=float), rate, high)
    block = block - lowpass(block, rate, band_hz / np.sqrt(2.0))
    frame = max(int(round(0.02 * rate)), 8)
    hop = max(int(round(0.01 * rate)), 1)
    energy = np.array(
        [
            float(np.sum(block[:, start : start + frame] ** 2))
            for start in range(0, block.shape[1] - frame + 1, hop)
        ]
    )[: values.size]
    if float(energy.sum()) <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * energy) / np.sum(energy))


def tail_statistics(
    ambisonic: Ambisonic,
    settings: CriteriaSettings = DEFAULT_SETTINGS,
    *,
    bands_hz: tuple[int, ...] = LEVEL_BANDS_HZ,
    mixing_time: float | None = None,
) -> TailReport:
    """Part B of one response: decay, colour, diffuseness, anisotropy, coherence, seam."""
    rate = ambisonic.sample_rate_hz
    signals = ambisonic.signals
    fs = int(round(rate))
    start = direct_arrival(signals, rate, settings)
    omni = _bandpass(signals[0:1], rate, None, settings.band_limit_hz)[0]
    t_mix = mixing_time if mixing_time is not None else mixing_time_s(omni, rate, settings)
    if not np.isfinite(t_mix):
        t_mix = settings.early_s
    mix_at = start + int(round(t_mix * rate))

    centres = band_centres(fs)
    picks = [centres.index(b) for b in bands_hz]
    t30 = rt60_per_band(omni[start:], fs)[picks]
    edt = edt_per_band(omni[start:], fs)[picks]

    per_band = octave_filter_rows(
        np.repeat(omni[None, :], len(picks), axis=0), fs, np.asarray(picks)
    )
    window = int(round(settings.window_s * rate))
    colour_span = int(round(settings.tail_colour_s * rate))
    seam = int(round(settings.seam_s * rate))
    colour = []
    seam_step = []
    for row in per_band:
        direct = float(np.sum(row[start : start + window] ** 2))
        tail = float(np.sum(row[mix_at : mix_at + colour_span] ** 2))
        colour.append(10.0 * np.log10(max(tail, 1e-30) / max(direct, 1e-30)))
        before = float(np.sum(row[max(mix_at - seam, 0) : mix_at] ** 2))
        after = float(np.sum(row[mix_at : mix_at + seam] ** 2))
        seam_step.append(10.0 * np.log10(max(after, 1e-30) / max(before, 1e-30)))

    end = signals.shape[1]
    limited = _bandpass(signals, rate, settings.focus_low_hz or None, settings.band_limit_hz)
    order_energy = _order_energy_db(limited, ambisonic.order, mix_at, end)
    sectors = _sector_energy_db(limited, ambisonic.order, mix_at, end, sector_directions())

    decoder = _decoder(min(ambisonic.order, settings.binaural_order), rate)
    brir = render(
        Ambisonic(limited[: channel_count(decoder.order)], rate, decoder.order, ambisonic.centre),
        decoder,
    )
    delay = decoder.modelling_delay_samples
    ears_start = start + delay
    early_span = int(round(settings.early_coherence_s * rate))
    direct_span = int(round(settings.direct_window_s * rate))
    head = brir[:, ears_start : ears_start + direct_span]
    itd = itd_s(head, rate)
    ild = ild_db(head)
    early_coherence = _coherence(
        brir[:, ears_start : ears_start + early_span], rate, settings.coherence_band_hz
    )
    late_coherence = _coherence(brir[:, mix_at + delay :], rate, settings.coherence_band_hz)
    floor = coherence_floor(rate, band_hz=settings.coherence_band_hz)
    return TailReport(
        mixing_time_s=float(t_mix),
        bands_hz=bands_hz,
        t30_s=tuple(float(v) for v in t30),
        edt_s=tuple(float(v) for v in edt),
        colour_db=tuple(float(v) for v in colour),
        order_energy_db=order_energy,
        sector_energy_db=sectors,
        early_coherence=early_coherence,
        late_coherence=late_coherence,
        coherence_floor=floor,
        direct_itd_s=itd,
        direct_ild_db=ild,
        seam_step_db=tuple(float(v) for v in seam_step),
    )


# --------------------------------------------------------------------------
# the judgement
# --------------------------------------------------------------------------


def focus_bands(
    settings: CriteriaSettings, bands_hz: tuple[int, ...] = LEVEL_BANDS_HZ
) -> tuple[int, ...]:
    """The octave bands the judgement reads: those at and above the focus."""
    return tuple(b for b in bands_hz if b >= settings.focus_low_hz * 0.99)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    """Largest relative error over the bands where both are finite."""
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask]) / a[mask]))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask])))


def judge(
    reference: Ambisonic,
    candidate: Ambisonic,
    criteria: Criteria | None = None,
) -> PointReport:
    """Everything of section 3 of the plan, for one point, with a verdict per criterion."""
    criteria = criteria or Criteria()
    s, t = criteria.settings, criteria.targets
    if reference.sample_rate_hz != candidate.sample_rate_hz:
        raise ValueError("the two responses must share a sample rate")
    ref_list = detect_reflections(reference, s)
    cand_list = detect_reflections(candidate, s)
    matching = match_reflections(ref_list, cand_list, s)
    judged = focus_bands(s)
    distance = echogram_distance_db(
        echogram(reference, s, bands_hz=judged), echogram(candidate, s, bands_hz=judged)
    )
    ref_tail = tail_statistics(reference, s)
    cand_tail = tail_statistics(candidate, s, mixing_time=ref_tail.mixing_time_s)
    kept = np.array([b in judged for b in ref_tail.bands_hz])

    def focus(values: tuple[float, ...]) -> np.ndarray:
        return np.asarray(np.asarray(values, dtype=float)[kept])

    cand_own_mix = mixing_time_s(
        _bandpass(candidate.signals[0:1], candidate.sample_rate_hz, None, s.band_limit_hz)[0],
        candidate.sample_rate_hz,
        s,
    )

    def median(values: tuple[float, ...]) -> float:
        return float(np.median(values)) if values else float("nan")

    errors = {
        "recall": matching.recall,
        "precision": matching.precision,
        "time_error_s": median(matching.time_errors_s),
        "direction_error_deg": median(matching.direction_errors_deg),
        "level_error_db": median(matching.level_errors_db),
        "itd_error_s": abs(ref_tail.direct_itd_s - cand_tail.direct_itd_s),
        "ild_error_db": abs(ref_tail.direct_ild_db - cand_tail.direct_ild_db),
        "early_coherence_error": abs(ref_tail.early_coherence - cand_tail.early_coherence),
        "mixing_time_relative": abs(cand_own_mix - ref_tail.mixing_time_s) / ref_tail.mixing_time_s
        if np.isfinite(cand_own_mix) and ref_tail.mixing_time_s > 0
        else float("nan"),
        "t30_relative": _relative(focus(ref_tail.t30_s), focus(cand_tail.t30_s)),
        "edt_relative": _relative(focus(ref_tail.edt_s), focus(cand_tail.edt_s)),
        "tail_colour_db": _max_abs(focus(ref_tail.colour_db), focus(cand_tail.colour_db)),
        "order_energy_db": _max_abs(
            np.asarray(ref_tail.order_energy_db), np.asarray(cand_tail.order_energy_db)
        ),
        "sector_energy_db": _max_abs(
            np.asarray(ref_tail.sector_energy_db), np.asarray(cand_tail.sector_energy_db)
        ),
        "late_coherence_error": abs(ref_tail.late_coherence - cand_tail.late_coherence),
        "seam_db": _max_abs(focus(ref_tail.seam_step_db), focus(cand_tail.seam_step_db)),
    }
    limits = {
        "recall": (t.recall, "min"),
        "precision": (t.precision, "min"),
        "time_error_s": (t.time_error_s, "max"),
        "direction_error_deg": (t.direction_error_deg, "max"),
        "level_error_db": (t.level_error_db, "max"),
        "itd_error_s": (t.itd_error_s, "max"),
        "ild_error_db": (t.ild_error_db, "max"),
        "early_coherence_error": (t.early_coherence_error, "max"),
        "mixing_time_relative": (t.mixing_time_relative, "max"),
        "t30_relative": (t.t30_relative, "max"),
        "edt_relative": (t.edt_relative, "max"),
        "tail_colour_db": (t.tail_colour_db, "max"),
        "order_energy_db": (t.order_energy_db, "max"),
        "sector_energy_db": (t.sector_energy_db, "max"),
        "late_coherence_error": (t.late_coherence_error, "max"),
        "seam_db": (t.seam_db, "max"),
    }
    verdicts = {}
    for name, (limit, sense) in limits.items():
        value = errors[name]
        if not np.isfinite(value):
            verdicts[name] = False
        else:
            verdicts[name] = bool(value >= limit) if sense == "min" else bool(value <= limit)
    return PointReport(
        reference=ref_list,
        candidate=cand_list,
        matching=matching,
        echogram_distance_db=distance,
        reference_tail=ref_tail,
        candidate_tail=cand_tail,
        verdicts=verdicts,
        errors=errors,
    )


def aggregate(reports: list[PointReport]) -> dict[str, Any]:
    """Medians of every error and the share of points passing each criterion."""
    if not reports:
        return {"points": 0}
    names = list(reports[0].errors)
    out: dict[str, Any] = {"points": len(reports), "errors": {}, "pass_fraction": {}}
    for name in names:
        values = np.asarray([r.errors[name] for r in reports], dtype=float)
        finite = values[np.isfinite(values)]
        out["errors"][name] = {
            "median": _round_or_none(float(np.median(finite)), 5) if finite.size else None,
            "p90": _round_or_none(float(np.percentile(finite, 90)), 5) if finite.size else None,
        }
        out["pass_fraction"][name] = round(float(np.mean([r.verdicts[name] for r in reports])), 4)
    out["echogram_distance_db"] = [
        _round_or_none(float(np.median([r.echogram_distance_db[k] for r in reports])), 3)
        for k in range(len(reports[0].echogram_distance_db))
    ]
    return out
