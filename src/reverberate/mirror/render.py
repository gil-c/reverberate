"""From paths to an ambisonic response: the discrete part, and an interim diffuse tail.

Every path is a band limited pulse at its arrival time, from its arrival
direction, at its per band gain. The pulses of one octave band are laid down
as fractional delay kernels on the harmonics of their directions, the band's
signals go through the project's own octave bank (the bank the reference is
analysed with, so a band here is the same band there), and the bands are
summed. That is the early part the criteria's part A reads.

The tail is what the ray tracer will produce; until it exists the tail is
**statistical**: Eyring's decay from the derived scene's own areas and
absorption, at the level Barron and Lee's revised theory gives the
reflected energy relative to the direct sound at the listener's distance,
drawn as independent noise per channel from the mixing time. It is labelled
as such in the record and it is what makes part B measurable before the
rays exist; it is not the mirror's answer for the tail.

No air absorption: the reference field carries none (the solver has no
viscosity term and the assembly applies ADR 0009's filter only to the
transposition of the tail's decay), and the mirror reproduces the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.audio import lowpass
from reverberate.metrics import band_centres, octave_filter_rows
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import Paths
from reverberate.mirror.rays import Histogram
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic

__all__ = [
    "RenderSettings",
    "band_pulse_energy",
    "bank_reading",
    "barron_reflected_ratio",
    "eyring_t60_s",
    "label_areas_m2",
    "render",
    "render_paths",
    "smooth_over_time",
    "storey_volume_m3",
    "tail_from_histogram",
]


@dataclass(frozen=True)
class RenderSettings:
    """What the renderer chooses."""

    sample_rate_hz: float = 48000.0
    order: int = 7
    duration_s: float = 1.2
    #: The response is band limited here; zero keeps the whole band, which is
    #: what the reference carries (its direct pulse reaches 20 kHz at -10 dB).
    band_limit_hz: float = 0.0
    #: Air absorption over the response's own time axis, as the reference's
    #: encoding chain applies it (``accel.encode``): the default atmosphere.
    air_absorption: bool = True
    #: The chain's own low cut, 40 Hz at order 8 (``w40_volume_field.plan``).
    lowcut_hz: float = 40.0
    lowcut_order: int = 8
    #: Half length of the windowed sinc that places a pulse between samples.
    delay_half_taps: int = 16
    #: Where the statistical tail starts and how long it fades in.
    tail_from_s: float = 0.020
    tail_fade_s: float = 0.010
    #: The tail's scale divides the direct energy by what the histogram's
    #: direct bin holds in expectation, ``r^2 / (4 d^2)`` per band for a sphere
    #: of radius r at distance d and rays of energy 1/N. False reads the first
    #: two bins instead, which also hold the floor and ceiling reflections
    #: arriving within 4 ms: +2 dB median, +4.7 dB at p90 on 0076.
    analytic_direct: bool = True
    #: Noise bursts per histogram bin, each from its own sampled direction.
    tail_bursts: int = 6
    #: Cap, in seconds, on the half width of the moving mean that smooths the
    #: histogram's energy and moments over time before the tail is
    #: synthesised. The rays estimate a smooth late decay; what they leave
    #: bin to bin is the estimator's own noise, not the room. On 0076 the
    #: reference's broadband envelope fluctuates 1.9 dB about its decay and
    #: the unsmoothed tail 4.0 dB. Zero keeps the histogram as the rays left
    #: it.
    tail_smooth_s: float = 0.050
    #: The moving mean's half width is this fraction of the time elapsed
    #: since the tail began, capped at ``tail_smooth_s``. A window that grows
    #: keeps the early decay's own slope, which the early decay time reads,
    #: and averages hardest where the rays are thinnest.
    tail_smooth_fraction: float = 0.10
    #: The histogram's moments of degree n are weighted by this to the n
    #: before the burst directions are drawn: 1 keeps the rays' leaning, 0
    #: draws them uniformly. The rays mix directions less than the wave
    #: field does (a specular bounce keeps the elevation), and this is the
    #: host side's lever on it.
    tail_order_weight: float = 1.0
    #: A point with no direct path scales its tail on its own first arrival,
    #: the diffracted onset, instead of on the median the other points gave.
    #: The rays came round the same doorway the onset did, so the two should
    #: be the same sound. Measured on 0076 they are not: the share of
    #: criteria met on 32 shadowed points falls from 0.316 to 0.293 and the
    #: tail's colour error rises from 8.5 to 11.2 dB, because the histogram's
    #: first bin behind a door holds too few crossings to be a scale. Left
    #: off; kept because the measurement is worth repeating on another
    #: storey.
    tail_scale_on_onset: bool = False
    #: The histogram tail's band energies go through the inverse of what the
    #: bank reads of shaped noise (``bank_reading``), so they read as meant.
    bank_corrected: bool = True
    #: The direct sound is placed this long after the start, as the
    #: reference's own chain does; the criteria align on it anyway.
    lead_s: float = 0.0

    def record(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "order": self.order,
            "duration_s": self.duration_s,
            "band_limit_hz": self.band_limit_hz,
            "air_absorption": self.air_absorption,
            "lowcut_hz": self.lowcut_hz,
            "lowcut_order": self.lowcut_order,
            "delay_half_taps": self.delay_half_taps,
            "tail_from_s": self.tail_from_s,
            "tail_fade_s": self.tail_fade_s,
            "lead_s": self.lead_s,
            "analytic_direct": self.analytic_direct,
            "bank_corrected": self.bank_corrected,
            "tail_bursts": self.tail_bursts,
            "tail_order_weight": self.tail_order_weight,
            "tail_smooth_s": self.tail_smooth_s,
            "tail_smooth_fraction": self.tail_smooth_fraction,
            "tail_scale_on_onset": self.tail_scale_on_onset,
        }


def _band_map(rate: float) -> tuple[tuple[int, ...], np.ndarray]:
    """The bank's bands at this rate, and which material band each reads its gain from."""
    centres = band_centres(int(round(rate)))
    material = np.asarray(OCTAVE_BANDS, dtype=float)
    picks = np.array([int(np.argmin(np.abs(material - c))) for c in centres], dtype=int)
    return centres, picks


def _fractional_pulses(
    delays_samples: np.ndarray, amplitudes: np.ndarray, half: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed sinc kernels placing each pulse between samples: taps and their sample indices.

    Both ``[pulse, 2 half + 1]``; the caller accumulates the taps at the indices.
    """
    offsets = np.arange(-half, half + 1)
    base = np.floor(delays_samples).astype(int)
    fraction = delays_samples - base
    taps = offsets[None, :] - fraction[:, None]
    kernel = np.sinc(taps) * np.hanning(2 * half + 3)[1:-1][None, :]
    return np.asarray(kernel * amplitudes[:, None]), np.asarray(base[:, None] + offsets[None, :])


@lru_cache(maxsize=4)
def _bank_kernels(rate: float) -> np.ndarray:
    """The bank's FIR filters, ``[band, taps]``, as :func:`octave_filter_rows` reads them."""
    from reverberate.metrics import octave_bank

    kernels = np.ascontiguousarray(np.asarray(octave_bank(int(round(rate))).filters, float).T)
    kernels.setflags(write=False)
    return kernels


def band_rows(rows: Any, rate: float, bands: np.ndarray, xp: Any = np) -> Any:
    """:func:`octave_filter_rows` on ``xp``: row k through band ``bands[k]``, same length."""
    if xp is np:
        return octave_filter_rows(np.asarray(rows, dtype=float), int(round(rate)), bands)
    from cupyx.scipy.signal import fftconvolve  # type: ignore[import-not-found]

    kernels = xp.asarray(_bank_kernels(rate))[xp.asarray(np.asarray(bands, dtype=int))]
    return fftconvolve(xp.asarray(rows, dtype=xp.float64), kernels, mode="same", axes=1)


@lru_cache(maxsize=8)
def band_pulse_energy(rate: float) -> np.ndarray:
    """The energy per band of a unit pulse through the bank, over its whole response.

    A 1 ms window around the direct pulse keeps only part of it where the
    band filter rings longer than the window: -4.5, -3.9, -3.7 and -0.7 dB at
    125, 250, 500 and 1000 Hz at 48 kHz. The tail's scale reads this instead.
    """
    centres, _ = _band_map(rate)
    length = int(round(0.2 * rate))
    impulse = np.zeros((len(centres), length))
    impulse[:, length // 2] = 1.0
    rows = octave_filter_rows(impulse, int(round(rate)), np.arange(len(centres)))
    energy = np.asarray(np.sum(rows**2, axis=1), dtype=float)
    energy.setflags(write=False)
    return energy


@lru_cache(maxsize=8)
def bank_reading(rate: float) -> np.ndarray:
    """``M[b, k]``: what the bank's band b reads of unit energy synthesised in band k.

    The tail is noise shaped by band k's filter, so a reading through the
    same bank applies that filter twice and leaks into the neighbours: a flat
    synthesis reads 1 dB low in every band and 4.6 dB low at 125 Hz. The
    tail's energies go through the inverse of this matrix first.
    """
    centres, _ = _band_map(rate)
    count = len(centres)
    fs = int(round(rate))
    noise = np.random.default_rng(12345).standard_normal(2 * fs)
    shaped = octave_filter_rows(np.repeat(noise[None, :], count, 0), fs, np.arange(count))
    shaped /= np.sqrt(np.sum(shaped**2, axis=1, keepdims=True))
    reading = np.zeros((count, count))
    for k in range(count):
        read = octave_filter_rows(np.repeat(shaped[k][None, :], count, 0), fs, np.arange(count))
        reading[:, k] = np.sum(read**2, axis=1)
    reading.setflags(write=False)
    return reading


def early_signals(
    paths: Paths, settings: RenderSettings, sound_speed_m_s: float, xp: Any = np
) -> Any:
    """The discrete part as ``[channel, sample]`` on ``xp``: every pulse, every band at once."""
    rate = settings.sample_rate_hz
    length = int(round(settings.duration_s * rate))
    channels = channel_count(settings.order)
    centres, picks = _band_map(rate)
    if paths.count == 0:
        return xp.zeros((channels, length))
    directions = scene_to_ambisonic(paths.direction)
    harmonics = real_sh(settings.order, directions)  # [path, channel]
    delays = (paths.length_m / sound_speed_m_s + settings.lead_s) * rate
    half = settings.delay_half_taps
    # The early part lives in a short buffer, filtered there, then placed.
    latest = int(np.ceil(delays.max())) + half + 2
    span = latest + 1024
    kernels, indices = _fractional_pulses(delays, np.ones(paths.count), half)
    bands = len(centres)
    # value[band, path, tap, channel] = gain[path, band] kernel[path, tap] Y[path, channel],
    # added path after path as slices: the same order of additions on every device.
    weighted = paths.gain[:, picks].T[:, :, None] * kernels[None, :, :]  # [band, path, tap]
    # Laid on the host, where a slice costs nothing, then moved once.
    values = weighted[:, :, :, None] * harmonics[None, :, None, :]
    laid = np.zeros((bands, span, channels))
    for path in range(paths.count):
        first = int(indices[path, 0])
        lo = max(first, 0)
        hi = min(first + indices.shape[1], span)
        if hi <= lo:
            continue
        laid[:, lo:hi, :] += values[:, path, lo - first : hi - first, :]
    rows = xp.asarray(np.ascontiguousarray(laid.transpose(0, 2, 1)).reshape(bands * channels, span))
    filtered = band_rows(rows, rate, np.repeat(np.arange(bands), channels), xp)
    early = filtered.reshape(bands, channels, span).sum(axis=0)
    signals = xp.zeros((channels, length))
    take = min(span, length)
    signals[:, :take] = early[:, :take]
    return signals


def render_paths(paths: Paths, settings: RenderSettings, sound_speed_m_s: float) -> Ambisonic:
    """The discrete part: every path as a band limited pulse on the harmonics of its direction."""
    signals = early_signals(paths, settings, sound_speed_m_s)
    return Ambisonic(signals, settings.sample_rate_hz, settings.order, paths.receiver)


# --------------------------------------------------------------------------
# the interim statistical tail
# --------------------------------------------------------------------------


def storey_volume_m3(scene: DerivedScene) -> float:
    """Floor facet area times the shell's height: the extruded storey's volume."""
    floor = sum(f.area for f in scene.facets if f.kind == "shell_floor")
    height = float(scene.bmax[1] - scene.bmin[1])
    if floor <= 0.0 or height <= 0.0:
        # No floor facet survived the area rule: fall back to the box.
        return float(np.prod(scene.bmax - scene.bmin))
    return float(floor * height)


def label_areas_m2(scene: DerivedScene) -> np.ndarray:
    """Area per label: the census's, or the occluder triangles' when there is no census."""
    census = scene.census.get("labels") if scene.census else None
    if census and all(label in census for label in scene.labels):
        return np.array([float(census[label]["area_m2"]) for label in scene.labels], dtype=float)
    v = scene.occluder_vertices
    areas = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    return np.asarray(np.bincount(scene.occluder_label, weights=areas, minlength=len(scene.labels)))


def eyring_t60_s(scene: DerivedScene, volume_m3: float, sound_speed_m_s: float) -> np.ndarray:
    """Eyring's reverberation time per material band from the label areas and the absorption."""
    areas = label_areas_m2(scene)
    absorption = scene.materials.absorption  # [label, band]
    total = float(areas.sum())
    mean = (areas[:, None] * absorption).sum(axis=0) / max(total, 1e-9)
    mean = np.clip(mean, 1e-4, 0.999)
    return np.asarray(
        24.0 * np.log(10.0) * volume_m3 / (-sound_speed_m_s * total * np.log(1.0 - mean))
    )


def barron_reflected_ratio(distance_m: float, t60_s: np.ndarray, volume_m3: float) -> np.ndarray:
    """Reflected energy over direct energy at ``distance_m``, Barron and Lee (1988), per band.

    Direct energy ``100 / r^2``; reflected ``31200 T / V`` times
    ``exp(-0.04 r / T)``, the revised theory's allowance for the energy that
    has decayed before the direct sound arrives.
    """
    t = np.asarray(t60_s, dtype=float)
    reflected = 31200.0 * t / volume_m3 * np.exp(-0.04 * distance_m / t)
    direct = 100.0 / max(distance_m, 1e-3) ** 2
    return np.asarray(reflected / direct)


def smooth_over_time(
    values: np.ndarray, half: int, *, first: int = 0, fraction: float = 0.0
) -> np.ndarray:
    """A centred moving mean over the first axis, from bin ``first`` on.

    The rays give one Monte Carlo estimate of a smooth late decay per time
    bin. What that estimate leaves bin to bin is its own variance: on 0076 a
    bin of 2 ms holds some sixty ray crossings whose energies are so unequal
    that the bin to bin spread reaches 4 dB where the reference's is 2 dB.
    The mean over ``2 * h + 1`` bins divides that variance by the count.

    The window grows with the time since ``first``: ``h`` is ``fraction`` of
    the bins elapsed, capped at ``half``. The early tail therefore keeps its
    own slope, which the early decay time reads, and the late tail, where
    the estimate is worst and the field is diffuse, is averaged hard. A flat
    window of 20 ms costs 0.05 of early decay time on 0076; the growing one
    costs nothing and removes as much of the noise.

    Bins before ``first`` are left alone, so the direct bin a scale is read
    from stays what the rays wrote. Each window is normalised by how many
    bins it covers, so neither end gains or loses energy.
    """
    if (half <= 0 and fraction <= 0.0) or values.shape[0] <= 1:
        return values
    out = np.array(values, dtype=float, copy=True)
    tail = out[first:]
    if tail.shape[0] <= 1:
        return out
    flat = tail.reshape(tail.shape[0], -1)
    padded = np.zeros((flat.shape[0] + 1, flat.shape[1]), dtype=float)
    np.cumsum(flat, axis=0, out=padded[1:])
    count = flat.shape[0]
    steps = np.arange(count)
    widths = (
        np.full(count, half)
        if fraction <= 0.0
        else np.minimum(np.rint(fraction * steps).astype(int), half if half > 0 else count)
    )
    lo = np.maximum(steps - widths, 0)
    hi = np.minimum(steps + widths + 1, count)
    means = (padded[hi] - padded[lo]) / (hi - lo)[:, None]
    out[first:] = means.reshape(tail.shape)
    return out


def tail_from_histogram(
    histogram: Histogram,
    receiver: int,
    direct_energy: np.ndarray,
    settings: RenderSettings,
    *,
    sound_speed_m_s: float,
    start_s: float,
    seed: int,
    bursts: int = 6,
    band_gain_db: np.ndarray | None = None,
    scale_per_band: np.ndarray | None = None,
    xp: Any = np,
) -> tuple[Any, dict[str, Any]]:
    """The tail as noise shaped by a receiver's histogram, band by band, with its anisotropy.

    Per band and per time bin the histogram gives the energy and its
    directional moments to the histogram's order. The energy is drawn as
    ``bursts`` independent noise bursts per bin, each from a direction
    sampled with the probability the moments give it (the moments'
    expansion, clipped at zero, on a quadrature of the sphere), each encoded
    on the harmonics of its direction at the output order. Channels are
    therefore decorrelated as a diffuse field is, and where the histogram
    is anisotropic the tail is too.

    ``direct_energy`` is the rendered direct pulse's energy per band; the
    histogram's own direct bin is what it is scaled against, so the tail
    sits at the level the rays give it relative to the direct sound.
    ``band_gain_db``, one value per octave band, is the calibration's tail
    gain on top of that. ``scale_per_band`` replaces the reading of the
    direct bin where there is no direct sound to read it from (a room the
    rays reach round a doorway): the scale the other points gave.
    """
    from reverberate.spatial.sh import quadrature

    rate = settings.sample_rate_hz
    length = int(round(settings.duration_s * rate))
    channels = channel_count(settings.order)
    centres, picks = _band_map(rate)
    energy = histogram.energy[receiver]  # [bin, band]
    moments = histogram.moments[receiver]  # [bin, band, channel]
    bins = energy.shape[0]
    bin_samples = int(round(histogram.bin_s * rate))
    rng = np.random.default_rng(seed)
    grid, weights = quadrature(2 * histogram.order + 2)
    basis_low = real_sh(histogram.order, grid)  # [direction, low channel]
    if settings.tail_order_weight != 1.0:
        from reverberate.spatial.sh import degrees_of

        weights_of_degree = settings.tail_order_weight ** degrees_of(histogram.order)
        basis_low = basis_low * weights_of_degree[None, :]
    basis_out = real_sh(settings.order, grid)  # [direction, out channel]
    # The histogram's direct bin: the first bin with energy, its scale.
    first = int(np.argmax(np.any(energy > 0.0, axis=1))) if np.any(energy > 0.0) else -1
    from_bin = int(np.ceil((start_s + settings.tail_from_s) / histogram.bin_s))
    scale = np.zeros(len(picks))
    if scale_per_band is not None:
        scale = np.asarray(scale_per_band, dtype=float)[: len(picks)]
    elif first >= 0:
        for band, pick in enumerate(picks):
            reference = float(np.sum(energy[first : first + 2, pick]))
            scale[band] = direct_energy[band] / reference if reference > 0.0 else 0.0
    half = int(round(settings.tail_smooth_s / histogram.bin_s))
    fraction = settings.tail_smooth_fraction
    if half > 0 or fraction > 0.0:
        energy = smooth_over_time(energy, half, first=from_bin, fraction=fraction)
        moments = smooth_over_time(moments, half, first=from_bin, fraction=fraction)
    band_power = np.ones(len(OCTAVE_BANDS))
    if band_gain_db is not None:
        band_power = 10.0 ** (np.asarray(band_gain_db, dtype=float) / 10.0)
    # The energy each bin should read per band, then what to synthesise so the
    # bank reads it: the inverse of the bank's own reading, clipped at zero.
    wanted = energy[:, picks] * (scale * band_power[picks])[None, :]  # [bin, band]
    if settings.bank_corrected:
        wanted = np.maximum(np.linalg.solve(bank_reading(rate), wanted.T).T, 0.0)
    # Every bin and band at once: the directions drawn from the moments'
    # density on the quadrature, the bursts laid bin by bin, each burst an
    # equal share of its bin's energy, encoded on its direction's harmonics.
    count = len(picks)
    held = min(bins, -(-length // bin_samples))
    energy_in = np.array(wanted[:held], dtype=float)
    energy_in[: min(from_bin, held)] = 0.0
    density = np.maximum(moments[:held][:, picks, :] @ basis_low.T, 0.0) * weights
    empty = density.sum(axis=-1, keepdims=True) <= 0.0
    density = np.where(empty, weights, density)
    cumulative = np.cumsum(density / density.sum(axis=-1, keepdims=True), axis=-1)
    cumulative[..., -1] = np.inf
    drawn = rng.random((held, count, bursts))
    # The same comparisons on either device: the first direction whose
    # cumulative probability passes the draw, [bin, band, burst].
    chosen = xp.argmax(
        xp.asarray(drawn)[..., None] < xp.asarray(cumulative)[:, :, None, :], axis=-1
    )
    span = held * bin_samples
    if xp is np:
        draws = rng.standard_normal((count, bursts, span))
    else:
        draws = xp.random.default_rng(seed).standard_normal((count, bursts, span))
    if span > length:
        draws[:, :, length:] = 0.0
    segments = draws.reshape(count, bursts, held, bin_samples)
    have = xp.sum(segments**2, axis=-1)  # [band, burst, bin]
    share = xp.asarray(energy_in.T / bursts)[:, None, :]  # [band, 1, bin]
    gain = xp.where(share > 0.0, xp.sqrt(share / xp.maximum(have, 1e-30)), 0.0)
    coefficients = xp.asarray(basis_out)[chosen]  # [bin, band, burst, channel]
    coefficients = coefficients * gain.transpose(2, 0, 1)[..., None]
    laid = xp.matmul(
        coefficients.transpose(1, 0, 3, 2),  # [band, bin, channel, burst]
        segments.transpose(0, 2, 1, 3),  # [band, bin, burst, sample]
    )  # [band, bin, channel, sample]
    per_band = xp.zeros((count, channels, length))
    stop = min(span, length)
    per_band[:, :, :stop] = laid.transpose(0, 2, 1, 3).reshape(count, channels, span)[:, :, :stop]
    # Band limit each band's noise to its octave: the bursts were white, so
    # each band's rows go through their own filter and the bands are summed;
    # the filter keeps a fraction of a white burst's energy, restored here
    # band by band on the omni channel so the histogram's energy is kept.
    rows = per_band.reshape(count * channels, length)
    row_bands = np.repeat(np.arange(count), channels)
    shaped = band_rows(rows, rate, row_bands, xp).reshape(count, channels, length)
    wanted_omni = xp.sum(per_band[:, 0] ** 2, axis=1)
    kept_omni = xp.sum(shaped[:, 0] ** 2, axis=1)
    ratio = xp.where(
        (kept_omni > 0.0) & (wanted_omni > 0.0),
        xp.sqrt(wanted_omni / xp.maximum(kept_omni, 1e-300)),
        1.0,
    )
    shaped *= ratio[:, None, None]
    tail = shaped.sum(axis=0)
    record = {
        "kind": "histogram of the rays, noise bursts per bin from sampled directions",
        "bursts_per_bin": bursts,
        "smooth_half_bins": half,
        "smooth_fraction": fraction,
        "scale_per_band": [round(float(v), 6) for v in scale],
        "seed": seed,
    }
    return tail, record


def render(
    paths: Paths,
    scene: DerivedScene,
    settings: RenderSettings,
    *,
    sound_speed_m_s: float,
    seed: int = 0,
    with_tail: bool = True,
) -> tuple[Ambisonic, dict[str, Any]]:
    """The discrete part plus the interim statistical tail, and the record of both."""
    rate = settings.sample_rate_hz
    early = render_paths(paths, settings, sound_speed_m_s)
    signals = early.signals.copy()
    record: dict[str, Any] = {"paths": int(paths.count), "settings": settings.record()}
    direct = paths.order == 0
    distance = float(paths.length_m[direct][0]) if direct.any() else float("nan")
    if with_tail and np.isfinite(distance):
        volume = storey_volume_m3(scene)
        t60 = eyring_t60_s(scene, volume, sound_speed_m_s)
        ratio = barron_reflected_ratio(distance, t60, volume)
        centres, picks = _band_map(rate)
        channels = signals.shape[0]
        length = signals.shape[1]
        start = int(round((distance / sound_speed_m_s + settings.lead_s) * rate))
        tail_at = start + int(round(settings.tail_from_s * rate))
        fade = max(int(round(settings.tail_fade_s * rate)), 1)
        times = np.arange(length - tail_at) / rate
        rng = np.random.default_rng(seed)
        # The direct sound's energy per band, read on the rendered early part
        # over half a millisecond around its arrival, is what the ratio scales.
        window = max(int(round(0.0005 * rate)), 1)
        direct_rows = octave_filter_rows(
            np.repeat(signals[0:1, max(start - window, 0) : start + window + 1], len(centres), 0),
            int(round(rate)),
            np.arange(len(centres)),
        )
        direct_energy = np.sum(direct_rows**2, axis=1)
        ramp = np.ones(length - tail_at)
        ramp[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
        draws = rng.standard_normal((channels, len(centres), length - tail_at))
        tail = np.zeros((channels, length - tail_at))
        for band, pick in enumerate(picks):
            decay = float(t60[pick])
            envelope = np.exp(-3.0 * np.log(10.0) * times / decay)
            # Energy of the tail after the mixing time, relative to the direct.
            remaining = float(
                ratio[pick] * np.exp(-6.0 * np.log(10.0) * settings.tail_from_s / decay)
            )
            noise = octave_filter_rows(draws[:, band, :], int(round(rate)), np.full(channels, band))
            noise = noise * envelope[None, :] * ramp[None, :]
            have = float(np.sum(noise[0] ** 2))
            want = remaining * float(direct_energy[band])
            scale = np.sqrt(want / have) if have > 0.0 else 0.0
            tail += noise * scale
        signals[:, tail_at:] += tail
        record["tail"] = {
            "kind": "statistical, Eyring decay at Barron's level, independent noise per channel",
            "volume_m3": round(volume, 3),
            "t60_s": [round(float(v), 4) for v in t60],
            "reflected_over_direct": [round(float(v), 5) for v in ratio],
            "from_s": settings.tail_from_s,
            "seed": seed,
        }
    else:
        record["tail"] = None
    if settings.lowcut_hz > 0.0:
        sos = butter(settings.lowcut_order, settings.lowcut_hz, btype="high", fs=rate, output="sos")
        signals = np.asarray(sosfilt(sos, signals, axis=-1))
    if settings.band_limit_hz > 0.0:
        signals = lowpass(signals, rate, settings.band_limit_hz)
    return Ambisonic(signals, rate, settings.order, paths.receiver), record
