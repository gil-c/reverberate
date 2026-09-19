"""The settings, thresholds and records of the judgement of a mirror against the wave field.

A response is judged in two parts that never trade against each other:
before the mixing time a set of discrete reflections (time, direction, level
per octave), after it a tail whose statistics carry the room's timbre. The
same detectors run on both sides; nothing is compared sample by sample.
Signals are ``[channel, sample]`` in ACN and N3D, directions in the frame of
:mod:`reverberate.spatial.sh`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import erfc

from reverberate.mirror.direct import direct_arrival
from reverberate.spatial.binaural import (
    BinauralDecoder,
    design_decoder,
)
from reverberate.spatial.hrtf import sphere_head
from reverberate.spatial.sh import degrees_of, quadrature, real_sh

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
    #: The lowest frequency judged. Zero judges what is heard, the whole band;
    #: a calibration of the mirror alone sets the hybrid's crossover, since
    #: the wave solver supplies everything under it.
    low_hz: float = 0.0
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
    """The acceptance thresholds, as numbers a report can quote.

    These are the differences a listener hears: 5 per cent of a decay time
    (ISO 3382-1), a decibel of level, 20 microseconds of interaural delay,
    0.075 of interaural coherence. They are the thresholds the model has to
    reach, and they do not move because a measurement is noisy. Where one
    point's reading is noisier than the difference asked about, the
    judgement says so by failing; raising the threshold to clear the noise
    would only hide the fact.
    """

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


def arrival(signals: np.ndarray, rate: float, settings: CriteriaSettings) -> int:
    """The direct sound's sample, found in the judgement's own band and window."""
    return direct_arrival(
        signals,
        rate,
        band_hz=(settings.detection_low_hz, settings.band_limit_hz),
        window_s=settings.window_s,
    )
