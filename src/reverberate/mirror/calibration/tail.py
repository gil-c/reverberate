"""Part B: the tail after the mixing time, as statistics."""

from __future__ import annotations

import numpy as np

from reverberate.audio import lowpass
from reverberate.metrics import band_centres, edt_per_band, octave_filter_rows, rt60_per_band
from reverberate.mirror.calibration.criteria import (
    _GAUSSIAN_TAIL,
    DEFAULT_SETTINGS,
    LEVEL_BANDS_HZ,
    CriteriaSettings,
    TailReport,
    _decoder,
    arrival,
    beam_weights,
    sector_directions,
)
from reverberate.mirror.direct import bandpass as _bandpass
from reverberate.spatial.binaural import (
    coherence_floor,
    ild_db,
    interaural_coherence,
    itd_s,
    render,
)
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count, degrees_of


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
    start = arrival(signal[None, :], rate, settings)
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
    the first one, and a wave field's frames grow coherent far below what is
    heard. The weights are the frames' energy
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
    start = arrival(signals, rate, settings)
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
    limited = _bandpass(signals, rate, settings.low_hz or None, settings.band_limit_hz)
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
