"""Three solves on three grids, one ambisonic response.

The economy of roadmap section 4.2 runs each band on its own grid, and the
array of :mod:`reverberate.spatial.array` is built in node index space, so the
same array cannot exist on three grids: on the 32.7 mm low grid a 12 mm shell
holds no nodes at all. ``docs/open-questions/three-band-ambisonic-array.md``
lists the nine consequences. This module takes the one road that is open by
construction: **each band is encoded with its own array on its own grid, and
the bands are put together in the spherical harmonic domain**, where the
representation is shared and nothing depends on which node was which.

What that buys, and what it costs, stated rather than assumed:

- The recombination filters of :mod:`reverberate.bands` are linear and the
  same for every channel, so a direction, which is nothing but the ratios
  between channels, passes through them unchanged away from the crossovers.
- Two solves are not on one scale. The level ratio is measured on the ``W``
  channel, which is the omnidirectional pressure at the centre and the one
  quantity every band's array measures identically, and it is applied to
  every channel of that band. The ratio predicted from the two source
  bandwidths is checked against the measurement, exactly as
  :mod:`reverberate.experiments.w37_three_band` does.
- Each band's centre is the node nearest the requested point **on its own
  grid**, so the three expansions are about three points up to half a cell
  diagonal apart. :func:`centre_offsets` reports the distances so the record
  can carry them. At 10.5 points per wavelength that is at most about 0.3 rad
  of phase at each band's own top, inside a crossover whose two sides W3
  already measured as decorrelated by -6 dB; it is a stated approximation,
  not a hidden one.
- The synthetic tail is drawn **per channel**. In a diffuse field the N3D
  channels carry equal expected energy and no correlation between them, so
  independent noise per channel, each calibrated on its own computed level
  over the crossfade, is the diffuse covariance, and the binaural decode then
  gives the interaural coherence of section 5.5 for nothing. The decay rate is
  read once from the ``W`` channel, because the diffuse decay is one number
  per band whatever the direction.
- Each band's effective order is whatever its own conditioning measurement
  says; a band that supports less is zero padded to the assembled order, which
  is the physical truncation and not an invention.

Air absorption is applied to each band before it is windowed and extended,
as the W37 experiments do, so that a window measured on a wet response and a
tail calibrated on it agree with each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate import bands as band_split
from reverberate.audio import Atmosphere
from reverberate.metrics import band_centres, rt60_per_band
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count
from reverberate.tail import local_decay_s, synthesise, transpose

__all__ = [
    "CROSSOVER_OF_FMAX",
    "BandSolve",
    "assemble",
    "calibration_bands_hz",
    "centre_offsets",
    "continue_from",
    "decay_from_bands",
    "extend",
    "extend_spectrum",
    "level_gain",
    "pad_order",
]


#: Where a seam sits, as a fraction of the lower solve's ``fmax``: 12 points per
#: wavelength rather than the 10.5 of the edge itself.
CROSSOVER_OF_FMAX = 0.8


@dataclass(frozen=True)
class BandSolve:
    """One band's encoded response, and the solve it came from."""

    name: str
    ambisonic: Ambisonic
    fmax_hz: float
    grid_step_m: float

    @property
    def omni(self) -> np.ndarray:
        """The ``W`` channel: the pressure at the centre, order zero."""
        return np.asarray(self.ambisonic.signals[0], dtype=float)


def pad_order(ambisonic: Ambisonic, order: int) -> Ambisonic:
    """The same response carried at a higher order, the new channels silent.

    A band whose array supports order 3 holds nothing above it, and saying so
    with zeros is the truncation the physics already applied; inventing the
    channels would be the thing that is not allowed.
    """
    if order < ambisonic.order:
        raise ValueError(f"cannot pad order {ambisonic.order} down to {order}")
    if order == ambisonic.order:
        return ambisonic
    signals = np.zeros((channel_count(order), ambisonic.signals.shape[1]))
    signals[: ambisonic.signals.shape[0]] = ambisonic.signals
    return Ambisonic(
        signals=signals,
        sample_rate_hz=ambisonic.sample_rate_hz,
        order=order,
        centre=ambisonic.centre,
        normalisation=ambisonic.normalisation,
        ordering=ambisonic.ordering,
    )


def centre_offsets(solves: list[BandSolve], requested: np.ndarray) -> list[dict[str, Any]]:
    """How far each band's expansion centre sits from the point that was asked for.

    Each is the nearest node on that band's grid, so the bound is half that
    grid's cell diagonal, and the phase it costs at the band's own top is
    ``k d``: about the same fraction of a wavelength for every band, since the
    step scales with the wavelength.
    """
    rows = []
    for solve in solves:
        centre = np.asarray(solve.ambisonic.centre, dtype=float)
        distance = float(np.linalg.norm(centre - np.asarray(requested, dtype=float)))
        bound = float(np.sqrt(3.0) * solve.grid_step_m / 2.0)
        rows.append(
            {
                "band": solve.name,
                "centre": [round(float(v), 6) for v in centre],
                "offset_m": round(distance, 6),
                "bound_m": round(bound, 6),
                "phase_at_fmax_rad": round(2.0 * np.pi * solve.fmax_hz / 343.0 * distance, 4),
            }
        )
    return rows


def calibration_bands_hz(lower_fmax_hz: float, sample_rate_hz: int) -> tuple[float, float]:
    """The octave bands two adjacent solves can be levelled on.

    Both must resolve them: the top usable octave of the lower solve sits an
    octave under its ``fmax``, where its source pulse still has energy and
    the band-limiting low pass has not started, and three octaves is enough
    for a median that one bad band cannot set.
    """
    centres = np.asarray(band_centres(sample_rate_hz), dtype=float)
    top = lower_fmax_hz / 2.0
    usable = centres[(centres <= top * 1.01) & (centres >= top / 8.0)]
    if len(usable) < 1:
        raise ValueError(f"no octave band sits under {lower_fmax_hz:g} Hz at this rate")
    return float(usable[0]), float(usable[-1])


def _level(
    subject: BandSolve, reference: BandSolve, sample_rate_hz: int
) -> tuple[float, dict[str, Any]]:
    """The gain that puts ``subject`` on ``reference``'s scale, and the check on it."""
    common = min(subject.omni.shape[0], reference.omni.shape[0])
    calibration = calibration_bands_hz(subject.fmax_hz, sample_rate_hz)
    gain = band_split.level_ratio(
        subject.omni[:common],
        reference.omni[:common],
        sample_rate_hz,
        calibration_hz=calibration,
    )
    predicted = subject.fmax_hz / reference.fmax_hz
    measured_db = 20.0 * float(np.log10(gain))
    predicted_db = 20.0 * float(np.log10(predicted))
    if abs(measured_db - predicted_db) > 3.0:
        raise ValueError(
            f"the {subject.name} and {reference.name} solves differ by {measured_db:+.2f} dB "
            f"on {calibration[0]:g} to {calibration[1]:g} Hz but their source bandwidths "
            f"predict {predicted_db:+.2f} dB; they are not the same room under the same "
            "excitation and summing them would put a band at the wrong level"
        )
    return gain, {
        "subject": subject.name,
        "reference": reference.name,
        "calibration_hz": list(calibration),
        "measured_db": round(measured_db, 3),
        "predicted_db": round(predicted_db, 3),
    }


def level_gain(subject: BandSolve, reference: BandSolve) -> tuple[float, dict[str, Any]]:
    """The gain that puts ``subject`` on ``reference``'s scale, checked against the bandwidths."""
    return _level(subject, reference, int(round(reference.ambisonic.sample_rate_hz)))


def decay_from_bands(
    low: BandSolve,
    mid: BandSolve,
    *,
    mean_absorption: np.ndarray,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    edge: float = 0.7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One decay per octave band, each read from the solve that can measure it.

    The low band was solved for the whole decay and resolves the bottom
    octaves; the mid band resolves the middle ones over its own window. Above
    the mid band the decay is transposed by Eyring on the calibrated mean free
    path plus the air, which is roadmap 5.5's rule. A band that holds a solve's
    own edge is not read from that solve, since half of it is the low pass
    skirt: ``edge`` is the fraction of ``fmax`` under which a band is trusted.
    """
    rate = int(round(mid.ambisonic.sample_rate_hz))
    centres = np.asarray(band_centres(rate), dtype=float)
    prediction = transpose(
        rt60_per_band(mid.omni, rate),
        mean_absorption,
        rate,
        mid_fmax_hz=edge * mid.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed_m_s,
    )
    decay = np.asarray(prediction.t60_s, dtype=float)
    source = ["mid" if measured else "transposed" for measured in prediction.measured]
    from_low = rt60_per_band(low.omni, int(round(low.ambisonic.sample_rate_hz)))
    for band, centre in enumerate(centres):
        if centre <= edge * low.fmax_hz and np.isfinite(from_low[band]) and from_low[band] > 0:
            decay[band] = from_low[band]
            source[band] = "low"
    record = {
        "band_centres_hz": [int(c) for c in centres],
        "t60_s": [None if not np.isfinite(v) else round(float(v), 4) for v in decay],
        "source": source,
        "transposition": prediction.record(),
    }
    return decay, record


def continue_from(
    high: BandSolve,
    mid: BandSolve,
    *,
    gain: float,
    t60_s: np.ndarray,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    fade_s: float = 0.010,
) -> tuple[BandSolve, dict[str, Any]]:
    """Carry the high band past its window with the mid band's computed structure.

    A short high band solve ends while the room is still throwing discrete
    reflections at the listener: in 296 m3 a far wall answers every thirty
    milliseconds well past a hundred. Noise there is the wrong texture, and
    audibly so. The mid band was solved longer and holds those same
    reflections, at the same instants and from the same directions, an octave
    or two lower; :func:`extend_spectrum` writes them up to the high band's
    ``fmax`` with the decay corrected per band, and the result, put on the high
    band's scale by ``gain``, continues the high band from its last computed
    samples to the end of the mid band's window. The diffuse tail after that
    is :func:`extend`'s business.
    """
    rate = int(round(high.ambisonic.sample_rate_hz))
    if int(round(mid.ambisonic.sample_rate_hz)) != rate:
        raise ValueError("the two bands are delivered at different sample rates")
    window = high.ambisonic.signals.shape[1]
    length = mid.ambisonic.signals.shape[1]
    if length <= window:
        return high, {"continued": False, "why": "the mid band is no longer than the high band"}
    fade = int(round(fade_s * rate))
    if fade < 1 or fade > window:
        raise ValueError(f"a {fade_s * 1000:g} ms crossfade does not fit in the high band's window")
    lifted, extension = extend_spectrum(
        mid.ambisonic,
        fmax_hz=mid.fmax_hz,
        t60_s=t60_s,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed_m_s,
        ceiling_hz=high.fmax_hz,
    )
    order = max(high.ambisonic.order, lifted.order)
    own = pad_order(high.ambisonic, order).signals
    borrowed = pad_order(lifted, order).signals * gain
    weight = np.zeros(length)
    weight[window - fade : window] = np.linspace(0.0, 1.0, fade, endpoint=False)
    weight[window:] = 1.0
    signals = borrowed * weight[None, :]
    signals[:, :window] += own * (1.0 - weight[None, :window])
    continued = Ambisonic(
        signals=np.ascontiguousarray(signals),
        sample_rate_hz=float(rate),
        order=order,
        centre=high.ambisonic.centre,
    )
    record = {
        "continued": True,
        "computed_s": round(window / rate, 4),
        "continued_to_s": round(length / rate, 4),
        "fade_s": fade_s,
        "gain_db": round(20.0 * float(np.log10(gain)), 3),
        "from": f"the mid band's computed structure, lifted {extension['template_hz']} -> "
        f"{extension['synthesised_hz']} Hz",
        "rule": (
            "the discrete reflections the mid band solved are carried up an octave at their "
            "own instants and directions; noise starts only where the mid band's window ends"
        ),
    }
    return BandSolve(high.name, continued, high.fmax_hz, high.grid_step_m), record


def assemble(
    low: BandSolve,
    mid: BandSolve,
    high: BandSolve,
    *,
    crossovers_hz: tuple[float, float] | None = None,
    taps: int = band_split.TAPS,
) -> tuple[Ambisonic, dict[str, Any]]:
    """One ambisonic response from three, each band-limited to its own solve.

    ``crossovers_hz`` defaults to :data:`CROSSOVER_OF_FMAX` times the lower
    solve's ``fmax`` at each seam. A crossover *on* a solve's own edge lets the
    edge through at -6 dB, and the edge is where the scheme's group velocity
    goes to zero at 10.5 points per wavelength: on the 81 m2 living room the
    1 kHz solve left a line at exactly 1000 Hz that never decayed and owned the
    late response between 300 Hz and 3 kHz.

    The high band is the reference scale, the mid is levelled on it and the low
    on the mid, so a level defect between two solves is caught where it is
    introduced. The output is at the highest order of the three, centred where
    the high band was expanded, and it carries the filter bank's group delay
    identically in every band and every channel.
    """
    rates = {
        low.ambisonic.sample_rate_hz,
        mid.ambisonic.sample_rate_hz,
        high.ambisonic.sample_rate_hz,
    }
    if len(rates) != 1:
        raise ValueError(f"the bands are delivered at different sample rates: {sorted(rates)}")
    rate = int(round(high.ambisonic.sample_rate_hz))
    if not low.fmax_hz < mid.fmax_hz < high.fmax_hz:
        raise ValueError("the bands must be given lowest first, by their fmax")
    if crossovers_hz is None:
        crossovers_hz = (CROSSOVER_OF_FMAX * low.fmax_hz, CROSSOVER_OF_FMAX * mid.fmax_hz)
    if not low.fmax_hz >= crossovers_hz[0] and mid.fmax_hz >= crossovers_hz[1]:
        raise ValueError(
            f"a solve stops under the crossover it is meant to carry: fmax "
            f"{low.fmax_hz:g}/{mid.fmax_hz:g} against crossovers {crossovers_hz}"
        )

    gain_mid, mid_check = _level(mid, high, rate)
    gain_low_on_mid, low_check = _level(low, mid, rate)
    gains = (gain_low_on_mid * gain_mid, gain_mid, 1.0)

    order = max(low.ambisonic.order, mid.ambisonic.order, high.ambisonic.order)
    parts = [pad_order(solve.ambisonic, order) for solve in (low, mid, high)]
    split = band_split.split_filters(float(rate), crossovers_hz, taps=taps)
    signals = band_split.recombine(
        parts[0].signals, parts[1].signals, parts[2].signals, split, gains=gains
    )
    assembled = Ambisonic(
        signals=np.asarray(signals, dtype=float),
        sample_rate_hz=float(rate),
        order=order,
        centre=high.ambisonic.centre,
    )
    record = {
        "bands": [
            {
                "band": solve.name,
                "fmax_hz": solve.fmax_hz,
                "grid_step_m": solve.grid_step_m,
                "order": solve.ambisonic.order,
                "computed_s": round(solve.ambisonic.duration_s, 4),
                "gain": round(float(gain), 6),
                "gain_db": round(20.0 * float(np.log10(gain)), 3),
            }
            for solve, gain in zip((low, mid, high), gains, strict=True)
        ],
        "level_checks": [low_check, mid_check],
        "split": split.record(),
        "order": order,
        "domain": "spherical harmonic, ACN, N3D; each band encoded on its own grid",
    }
    return assembled, record


def extend(
    solve: BandSolve,
    total_s: float,
    *,
    calibration: BandSolve,
    mean_absorption: np.ndarray,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    seed: int,
    fade_s: float = 0.010,
    t60_s: np.ndarray | None = None,
) -> tuple[BandSolve, dict[str, Any]]:
    """Continue a band's computed response with a diffuse tail to ``total_s``.

    ``t60_s`` overrides the decay per band when given, which is what
    :func:`decay_from_bands` is for: a fit on the last forty milliseconds of a
    short solve read 2.8 s at 8 kHz on the mid band of a room that decays in
    half a second, and nothing downstream should trust a number like that.

    The decay is read on the ``W`` channel: locally, on the last part of what
    was solved, where that fit converges, and otherwise transposed from the
    calibration band's own decay through the room's absorption and the air,
    which is roadmap section 5.5's rule. Every channel then draws its own
    noise at that decay and is levelled on its own computed part over the
    crossfade, which is the diffuse field covariance in N3D.
    """
    rate = int(round(solve.ambisonic.sample_rate_hz))
    window = solve.ambisonic.signals.shape[1]
    total = int(round(total_s * rate))
    if total < window:
        raise ValueError(
            f"{solve.name} was solved for {window / rate:.3f} s and cannot be extended to "
            f"{total_s:.3f} s"
        )
    prediction = transpose(
        rt60_per_band(calibration.omni, rate),
        mean_absorption,
        rate,
        mid_fmax_hz=calibration.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed_m_s,
    )
    transposed = np.asarray(prediction.t60_s, dtype=float)
    if t60_s is not None:
        decay = np.asarray(t60_s, dtype=float)
        local = np.full_like(decay, np.nan)
    else:
        local = local_decay_s(solve.omni[None, :], rate, window / rate)
        decay = np.where(np.isfinite(local), local, transposed)
    if total == window:
        extended = solve.ambisonic
    else:
        signals = np.vstack(
            [
                synthesise(
                    channel,
                    rate,
                    decay,
                    total,
                    rng=np.random.default_rng(seed + index),
                    fade_s=fade_s,
                )
                for index, channel in enumerate(solve.ambisonic.signals)
            ]
        )
        extended = Ambisonic(
            signals=signals,
            sample_rate_hz=solve.ambisonic.sample_rate_hz,
            order=solve.ambisonic.order,
            centre=solve.ambisonic.centre,
        )
    record = {
        "band": solve.name,
        "computed_s": round(window / rate, 4),
        "total_s": round(total / rate, 4),
        "fade_s": fade_s,
        "band_centres_hz": [int(c) for c in band_centres(rate)],
        "tail_t60_s": [None if not np.isfinite(v) else round(float(v), 4) for v in decay],
        "from_local_fit": [bool(v) for v in np.isfinite(local)],
        "decay_given": t60_s is not None,
        "transposition": prediction.record(),
        "note": (
            "one decay per band read on W, independent noise per channel levelled on "
            "its own computed part: the diffuse covariance in N3D"
        ),
    }
    return BandSolve(solve.name, extended, solve.fmax_hz, solve.grid_step_m), record


def _t60_per_bin(
    frequency_hz: np.ndarray,
    band_centres_hz: np.ndarray,
    t60_s: np.ndarray,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
) -> np.ndarray:
    """Decay time at every bin, the surface term interpolated and the air term exact.

    The per band decay carries both the boundary and the air. The two are
    separated at the band centres, the boundary part is held per octave band
    and past the last band, and the air part is put back at each bin's own
    frequency from ISO 9613-1, so the extension above the last octave the
    catalogue knows about still loses exactly what air takes.
    """
    centres = np.asarray(band_centres_hz, dtype=float)
    decay = np.asarray(t60_s, dtype=float)
    usable = np.isfinite(decay) & (decay > 0.0)
    if usable.sum() < 2:
        raise ValueError("at least two bands need a finite decay to extend from")
    air_band = atmosphere.attenuation_db_per_m(centres[usable]) * sound_speed_m_s
    surface_rate = np.maximum(60.0 / decay[usable] - air_band, 1e-6)
    # Piecewise constant per octave, because that is what the catalogue is: an
    # absorption is a number per band, and a bin belongs to the band whose
    # centre is nearest on a log axis. Past the last band the last one holds.
    nearest = np.abs(
        np.log(np.maximum(frequency_hz, 1.0))[:, None] - np.log(centres[usable])[None, :]
    ).argmin(axis=1)
    rate = np.log(surface_rate)[nearest]
    air = atmosphere.attenuation_db_per_m(frequency_hz) * sound_speed_m_s
    return np.asarray(60.0 / (np.exp(rate) + air), dtype=float)


def extend_spectrum(
    ambisonic: Ambisonic,
    *,
    fmax_hz: float,
    t60_s: np.ndarray,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    ceiling_hz: float | None = None,
    template_top: float = 0.9,
    frame: int = 256,
) -> tuple[Ambisonic, dict[str, Any]]:
    """Synthesise the spectrum above ``fmax_hz`` from the octave the solver did compute.

    **What is copied, and why that is allowed.** Every bin above the solved
    band takes the short time spectrum of the bin an octave (or two, or three)
    below it, in every channel at once. For a discrete arrival the ratios
    between spherical harmonic channels depend on its direction and not on its
    frequency, so the copy carries every early reflection's instant and
    direction up unchanged; for the diffuse tail it carries independent
    per channel noise up, which is what the diffuse field is. What it does not
    carry is the true fine structure of the interference between arrivals at
    those frequencies, which two grids of the same room already fail to share
    (W3's -6 dB floor), and which nothing downstream reads.

    **What is changed is the decay.** Energy at a frequency decays at that
    frequency's own rate, boundary and air together, so the copied bin is
    multiplied by ``10^(-1.5 t (1 / T60(f) - 1 / T60(f_source)))``: unity at the
    direct sound, the difference of the two Eyring lines afterwards. That is
    the reflection count argument in its integrated form, and it is the same
    rule whether the sample sits in an early reflection or in the tail.

    Below ``template_top * fmax_hz`` nothing is touched; above it everything is
    replaced, including the band limiting skirt of the solve itself. Hann
    analysis and synthesis at three quarters overlap reconstruct exactly, so
    the untouched part comes back to rounding.
    """
    from scipy import signal as dsp

    from reverberate.metrics import band_centres

    rate = float(ambisonic.sample_rate_hz)
    ceiling = min(rate / 2.0, ceiling_hz if ceiling_hz is not None else rate / 2.0)
    top = template_top * fmax_hz
    if not 0.0 < top < ceiling:
        raise ValueError(f"nothing to synthesise between {top:g} and {ceiling:g} Hz")
    overlap = frame - frame // 4
    frequency, times, spectra = dsp.stft(
        ambisonic.signals, fs=rate, nperseg=frame, noverlap=overlap, boundary="zeros", padded=True
    )
    decay = _t60_per_bin(
        frequency,
        np.asarray(band_centres(int(round(rate))), dtype=float),
        t60_s,
        atmosphere,
        sound_speed_m_s,
    )
    step = frequency[1] - frequency[0]
    targets = np.flatnonzero((frequency > top) & (frequency <= ceiling))
    octaves = np.ceil(np.log2(frequency[targets] / top)).astype(int)
    sources = np.rint(frequency[targets] / 2.0**octaves / step).astype(int)
    # Amplitude falls 60 dB in one T60, so 10^(-3 t / T60), and the copied bin
    # carries the difference between its own line and its source's.
    gain = 10.0 ** (
        -3.0 * times[None, :] * (1.0 / decay[targets][:, None] - 1.0 / decay[sources][:, None])
    )
    # A bin moved to another frequency must advance its phase at that
    # frequency from frame to frame, or the overlapping frames cancel where
    # they meet: the phase vocoder's rule, and without it the copy lost 8 dB.
    hop = frame - overlap
    frames = np.arange(spectra.shape[2])
    advance = np.exp(2j * np.pi * (targets - sources)[:, None] * hop * frames[None, :] / frame)
    spectra[:, targets, :] = spectra[:, sources, :] * (gain * advance)[None, :, :]
    spectra[:, frequency > ceiling, :] = 0.0
    _, restored = dsp.istft(spectra, fs=rate, nperseg=frame, noverlap=overlap)
    signals = np.asarray(restored, dtype=float)[:, : ambisonic.signals.shape[1]]
    extended = Ambisonic(
        signals=np.ascontiguousarray(signals),
        sample_rate_hz=rate,
        order=ambisonic.order,
        centre=ambisonic.centre,
        normalisation=ambisonic.normalisation,
        ordering=ambisonic.ordering,
    )
    before = float((ambisonic.signals**2).sum())
    after = float((signals**2).sum())
    record = {
        "solved_to_hz": fmax_hz,
        "template_hz": [round(top / 2.0, 1), round(top, 1)],
        "synthesised_hz": [round(top, 1), round(ceiling, 1)],
        "octaves_copied": int(octaves.max()) if len(octaves) else 0,
        "frame": frame,
        "t60_used_s": {
            str(int(c)): (None if not np.isfinite(v) else round(float(v), 4))
            for c, v in zip(band_centres(int(round(rate))), t60_s, strict=True)
        },
        "energy_added_db": round(10.0 * np.log10(max(after, 1e-30) / max(before, 1e-30)), 3),
        "rule": (
            "each bin above the template takes the bin one or more octaves below it in "
            "every channel, times 10^(-3 t (1/T60(f) - 1/T60(f_source))); direction and "
            "timing of every arrival are carried up, the fine structure is not"
        ),
    }
    return extended, record
