"""Part A: the reflections before the mixing time, detected and matched."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.optimize import linear_sum_assignment

from reverberate.metrics import band_centres, octave_filter_rows
from reverberate.mirror.calibration.criteria import (
    DEFAULT_SETTINGS,
    LEVEL_BANDS_HZ,
    BandLevels,
    CriteriaSettings,
    Echogram,
    Matching,
    Reflection,
    _angle_deg,
    _azimuth_elevation,
    _beams,
    _grid,
    _neighbours,
    arrival,
    beam_weights,
)
from reverberate.mirror.direct import bandpass as _bandpass
from reverberate.spatial.encode import Ambisonic


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
    start = arrival(ambisonic.signals, rate, settings)
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
    start = arrival(signals, rate, settings)
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
        max(settings.detection_low_hz, settings.low_hz),
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
