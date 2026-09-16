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
    "barron_reflected_ratio",
    "eyring_t60_s",
    "label_areas_m2",
    "render",
    "render_paths",
    "storey_volume_m3",
    "tail_from_histogram",
]


@dataclass(frozen=True)
class RenderSettings:
    """What the renderer chooses."""

    sample_rate_hz: float = 48000.0
    order: int = 7
    duration_s: float = 1.2
    #: The response is band limited here, where the reference is solved to.
    band_limit_hz: float = 8000.0
    #: The chain's own low cut, 40 Hz at order 8 (``w40_volume_field.plan``).
    lowcut_hz: float = 40.0
    lowcut_order: int = 8
    #: Half length of the windowed sinc that places a pulse between samples.
    delay_half_taps: int = 16
    #: Where the statistical tail starts and how long it fades in.
    tail_from_s: float = 0.020
    tail_fade_s: float = 0.010
    #: The direct sound is placed this long after the start, as the
    #: reference's own chain does; the criteria align on it anyway.
    lead_s: float = 0.0

    def record(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "order": self.order,
            "duration_s": self.duration_s,
            "band_limit_hz": self.band_limit_hz,
            "lowcut_hz": self.lowcut_hz,
            "lowcut_order": self.lowcut_order,
            "delay_half_taps": self.delay_half_taps,
            "tail_from_s": self.tail_from_s,
            "tail_fade_s": self.tail_fade_s,
            "lead_s": self.lead_s,
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


def render_paths(paths: Paths, settings: RenderSettings, sound_speed_m_s: float) -> Ambisonic:
    """The discrete part: every path as a band limited pulse on the harmonics of its direction."""
    rate = settings.sample_rate_hz
    length = int(round(settings.duration_s * rate))
    channels = channel_count(settings.order)
    centres, picks = _band_map(rate)
    if paths.count == 0:
        return Ambisonic(np.zeros((channels, length)), rate, settings.order, paths.receiver)
    directions = scene_to_ambisonic(paths.direction)
    harmonics = real_sh(settings.order, directions)  # [path, channel]
    delays = (paths.length_m / sound_speed_m_s + settings.lead_s) * rate
    half = settings.delay_half_taps
    # The early part lives in a short buffer, filtered there, then placed.
    latest = int(np.ceil(delays.max())) + half + 2
    buffer = np.zeros((len(centres), channels, latest + 1024))
    kernels, indices = _fractional_pulses(delays, np.ones(paths.count), half)
    valid = (indices >= 0) & (indices < buffer.shape[2])
    for band, pick in enumerate(picks):
        gains = paths.gain[:, pick]  # [path]
        weighted = kernels * gains[:, None]  # [path, tap]
        # signal[channel, sample] += sum over paths of Y[path, channel] * kernel[path, tap]
        for tap in range(kernels.shape[1]):
            ok = valid[:, tap]
            if not ok.any():
                continue
            contribution = harmonics[ok].T * weighted[ok, tap][None, :]  # [channel, path]
            np.add.at(buffer[band].T, indices[ok, tap], contribution.T)
    rows = buffer.reshape(len(centres) * channels, -1)
    row_bands = np.repeat(np.arange(len(centres)), channels)
    filtered = octave_filter_rows(rows, int(round(rate)), row_bands)
    early = filtered.reshape(len(centres), channels, -1).sum(axis=0)
    signals = np.zeros((channels, length))
    take = min(early.shape[1], length)
    signals[:, :take] = early[:, :take]
    return Ambisonic(signals, rate, settings.order, paths.receiver)


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
) -> tuple[np.ndarray, dict[str, Any]]:
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
    basis_out = real_sh(settings.order, grid)  # [direction, out channel]
    # The histogram's direct bin: the first bin with energy, its scale.
    first = int(np.argmax(np.any(energy > 0.0, axis=1))) if np.any(energy > 0.0) else -1
    from_bin = int(np.ceil((start_s + settings.tail_from_s) / histogram.bin_s))
    scale = np.zeros(len(picks))
    if first >= 0:
        for band, pick in enumerate(picks):
            reference = float(np.sum(energy[first : first + 2, pick]))
            scale[band] = direct_energy[band] / reference if reference > 0.0 else 0.0
    draws = rng.standard_normal((len(picks), bursts, length))
    per_band = np.zeros((len(picks), channels, length))
    for b in range(from_bin, bins):
        at = b * bin_samples
        if at >= length:
            break
        stop = min(at + bin_samples, length)
        for band, pick in enumerate(picks):
            total = float(energy[b, pick]) * scale[band]
            if total <= 0.0:
                continue
            density = np.maximum(basis_low @ moments[b, pick], 0.0) * weights
            if density.sum() <= 0.0:
                density = weights.copy()
            probability = density / density.sum()
            chosen = rng.choice(grid.shape[0], size=bursts, p=probability)
            burst = draws[band, :, at:stop]
            # Each burst carries an equal share of the bin's energy.
            have = np.sum(burst**2, axis=1)
            gain = np.sqrt(total / bursts / np.maximum(have, 1e-30))
            per_band[band, :, at:stop] += basis_out[chosen].T @ (burst * gain[:, None])
    # Band limit each band's noise to its octave: the bursts were white, so
    # each band's rows go through their own filter and the bands are summed;
    # the filter keeps a fraction of a white burst's energy, restored here
    # band by band on the omni channel so the histogram's energy is kept.
    rows = per_band.reshape(len(picks) * channels, length)
    row_bands = np.repeat(np.arange(len(picks)), channels)
    shaped = octave_filter_rows(rows, int(round(rate)), row_bands).reshape(
        len(picks), channels, length
    )
    for band in range(len(picks)):
        wanted = float(np.sum(per_band[band, 0] ** 2))
        kept = float(np.sum(shaped[band, 0] ** 2))
        if kept > 0.0 and wanted > 0.0:
            shaped[band] *= np.sqrt(wanted / kept)
    tail = shaped.sum(axis=0)
    record = {
        "kind": "histogram of the rays, noise bursts per bin from sampled directions",
        "bursts_per_bin": bursts,
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
    signals = lowpass(signals, rate, settings.band_limit_hz)
    return Ambisonic(signals, rate, settings.order, paths.receiver), record
