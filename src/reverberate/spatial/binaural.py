"""Decoding an ambisonic response to two ears, with the head in the way.

The decode itself is one line: for a field whose plane wave density is
``a(s) = sum b_nm Y_nm(s)`` and a head whose response to a plane wave from
``s`` is ``H(s) = sum H_nm Y_nm(s)``, the ear signal is

    y = integral a(s) H(s) ds = sum_nm b_nm H_nm

exactly, because the harmonics are orthogonal. Everything below is about what
happens when the sum is truncated, which it always is.

**Truncation is not a blur, it is a loss of level and of interaural
difference.** A rigid sphere at 16 kHz has content out to order 45; truncated
at order 7 its own reconstruction is 33 dB down at order 30 and worse below.
Decoding the truncated coefficients against the truncated head gives a signal
that is low passed and whose interaural level difference has collapsed, which
is precisely the cue the consumer of this dataset separates voices with.

**Two published fixes, and both are applied here.**

*Magnitude least squares* (Schorkhuber, Zaunschirm and Holdrich, 2018). Above a
cut-on near the frequency where the interaural phase stops being a usable cue,
about 2 kHz, the ear cannot hear the phase of a single direction but can hear
its magnitude. So above the cut-on the fit matches ``|H|`` and lets the phase
go, taking each bin's target phase from the previous bin's own reconstruction
so the result stays continuous and its impulse response stays compact.

*The diffuse field covariance constraint* (Zaunschirm, Schorkhuber and
Holdrich, 2018). Truncation lowers the energy the decode returns for an
isotropic field, and lowers the interaural coherence with it. A per frequency
two by two correction restores both to the full head's, and the unitary factor
is chosen to stay as close as possible to the uncorrected decode so the fix
does not rotate the ears into each other.

Both are switched on by default and both are recorded, because a binaural
rendering that does not say how it was decoded cannot be compared with another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.signal import fftconvolve

from reverberate.audio import lowpass
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.hrtf import HrtfSet
from reverberate.spatial.sh import channel_count, real_sh, rotate_yaw

#: Which coherence is meant. ``max`` is the interaural cross correlation
#: coefficient, the peak over lags and the established perceptual measure;
#: ``zero`` is the coherence at zero lag, which is the quantity the roadmap's
#: diffuse field prediction is about. They are not the same number.
Lag = Literal["max", "zero"]

__all__ = [
    "BinauralDecoder",
    "Lag",
    "coherence_floor",
    "covariance_correction",
    "design_decoder",
    "ild_db",
    "interaural_coherence",
    "itd_s",
    "render",
]


@dataclass(frozen=True)
class BinauralDecoder:
    """Two filters per ambisonic channel, and how they were arrived at."""

    filters: np.ndarray
    sample_rate_hz: float
    order: int
    modelling_delay_samples: int
    cut_on_hz: float
    covariance_constrained: bool
    head: str

    def __post_init__(self) -> None:
        if self.filters.ndim != 3 or self.filters.shape[0] != 2:
            raise ValueError(f"filters must be [ear, channel, tap], got {self.filters.shape}")
        if self.filters.shape[1] != channel_count(self.order):
            raise ValueError(f"{self.filters.shape[1]} channels do not match order {self.order}")

    @property
    def length(self) -> int:
        return int(self.filters.shape[2])

    def record(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "order": self.order,
            "filter_length": self.length,
            "sample_rate_hz": self.sample_rate_hz,
            "modelling_delay_samples": self.modelling_delay_samples,
            # None rather than a number a JSON reader cannot express: a decoder
            # that never switches to matching magnitude is the plain fit, and
            # writing infinity here produces a file that is not JSON.
            "magls_cut_on_hz": self.cut_on_hz if np.isfinite(self.cut_on_hz) else None,
            "covariance_constrained": self.covariance_constrained,
            "note": (
                "magnitude least squares above the cut-on and a diffuse field "
                "covariance constraint, both after Schorkhuber, Zaunschirm and "
                "Holdrich 2018; a truncated decode without them loses level and "
                "interaural level difference at high frequency"
            ),
        }


def _projector(
    unit_vectors: np.ndarray, weights: np.ndarray | None, order: int, regularisation: float
) -> np.ndarray:
    """The matrix taking sampled directions to coefficients, ``[channel, direction]``."""
    basis = real_sh(order, unit_vectors)
    if weights is None:
        weight = np.full(unit_vectors.shape[0], 4.0 * np.pi / unit_vectors.shape[0])
    else:
        weight = np.asarray(weights, dtype=float)
    normal = basis.T @ (basis * weight[:, None]) + regularisation * np.eye(channel_count(order))
    return np.asarray(np.linalg.solve(normal, basis.T * weight[None, :]), dtype=float)


def covariance_correction(
    reference: np.ndarray, truncated: np.ndarray, *, regularisation: float = 1e-12
) -> np.ndarray:
    """The two by two matrix restoring a truncated decode's diffuse field covariance.

    ``M = L_ref Q L_N^-1`` with ``Q`` the unitary that keeps ``M`` closest to
    the identity in the least squares sense, from the singular value
    decomposition of ``L_ref^H L_N``. Without ``Q`` the constraint is still met
    and the ears can come back rotated into each other, which is audible as a
    collapsed image rather than as a level error.
    """
    eye = np.eye(2)
    l_ref = np.linalg.cholesky(reference + regularisation * np.trace(reference) * eye)
    l_trunc = np.linalg.cholesky(truncated + regularisation * np.trace(truncated) * eye)
    left, _, right = np.linalg.svd(l_ref.conj().T @ l_trunc)
    return np.asarray(l_ref @ (left @ right) @ np.linalg.inv(l_trunc), dtype=complex)


def design_decoder(
    hrtf: HrtfSet,
    *,
    order: int,
    sample_rate_hz: float,
    filter_length: int,
    magls_cut_on_hz: float = 2000.0,
    covariance_constraint: bool = True,
    regularisation: float = 1e-8,
) -> BinauralDecoder:
    """Fit an order ``order`` binaural decoder to a sampled head.

    ``hrtf`` must be sampled on ``numpy.fft.rfftfreq(filter_length,
    1 / sample_rate_hz)``, which is what makes the result a filter rather than
    an interpolation of one.
    """
    frequency = np.fft.rfftfreq(filter_length, 1.0 / sample_rate_hz)
    if hrtf.frequency_hz.size != frequency.size or not np.allclose(hrtf.frequency_hz, frequency):
        raise ValueError(
            "the head must be sampled on the decoder's own frequency grid; "
            f"got {hrtf.frequency_hz.size} frequencies for a {filter_length} tap filter"
        )
    projector = _projector(hrtf.unit_vectors, hrtf.weights, order, regularisation)
    basis = real_sh(order, hrtf.unit_vectors)

    if hrtf.weights is None:
        weight = np.full(hrtf.unit_vectors.shape[0], 4.0 * np.pi / hrtf.unit_vectors.shape[0])
    else:
        weight = np.asarray(hrtf.weights, dtype=float)

    coefficients = np.zeros((2, channel_count(order), frequency.size), dtype=complex)
    above = frequency >= magls_cut_on_hz
    for bin_index in range(frequency.size):
        target = hrtf.responses[:, :, bin_index]  # [ear, direction]
        if above[bin_index] and bin_index > 0:
            reconstructed = coefficients[:, :, bin_index - 1] @ basis.T
            phase = np.exp(1j * np.angle(reconstructed))
            target = np.abs(target) * phase
        fitted = target @ projector.T
        if covariance_constraint and above[bin_index]:
            reference = (target * weight[None, :]) @ target.conj().T / (4.0 * np.pi)
            truncated = fitted @ fitted.conj().T
            fitted = covariance_correction(reference, truncated) @ fitted
        coefficients[:, :, bin_index] = fitted

    taps = np.fft.irfft(coefficients, n=filter_length, axis=-1)
    delay = filter_length // 2
    taps = np.roll(taps, delay, axis=-1)
    window = np.hanning(filter_length)
    # Only the ends are tapered: a full window would eat the filter's own body.
    edge = filter_length // 8
    taper = np.ones(filter_length)
    taper[:edge] = window[:edge]
    taper[-edge:] = window[-edge:]
    return BinauralDecoder(
        filters=np.asarray(taps * taper, dtype=float),
        sample_rate_hz=float(sample_rate_hz),
        order=order,
        modelling_delay_samples=delay,
        cut_on_hz=float(magls_cut_on_hz),
        covariance_constrained=bool(covariance_constraint),
        head=hrtf.description,
    )


def render(
    ambisonic: Ambisonic, decoder: BinauralDecoder, *, field_yaw_rad: float = 0.0
) -> np.ndarray:
    """Two ear signals from an ambisonic response, ``[ear, sample]``.

    ``field_yaw_rad`` rotates the **field**, counter clockwise seen from above.
    A listener who turns their head left by ``psi`` hears the field turn right,
    so pass ``-psi``. The argument is named for the field because that is the
    thing being rotated, and because "yaw" alone is the one place this is
    routinely got backwards.
    """
    if decoder.order > ambisonic.order:
        raise ValueError(
            f"a decoder of order {decoder.order} needs at least that order, "
            f"and the response is order {ambisonic.order}"
        )
    if not np.isclose(decoder.sample_rate_hz, ambisonic.sample_rate_hz):
        raise ValueError(
            f"decoder at {decoder.sample_rate_hz} Hz against a response at "
            f"{ambisonic.sample_rate_hz} Hz"
        )
    keep = channel_count(decoder.order)
    signals = ambisonic.signals[:keep]
    if field_yaw_rad:
        signals = rotate_yaw(signals.T, field_yaw_rad, decoder.order).T
    out = np.zeros((2, signals.shape[1] + decoder.length - 1))
    for ear in range(2):
        for channel in range(keep):
            out[ear] += fftconvolve(signals[channel], decoder.filters[ear, channel])
    return np.asarray(out, dtype=float)


def itd_s(
    brir: np.ndarray,
    sample_rate_hz: float,
    *,
    maximum_s: float = 1.5e-3,
    cut_off_hz: float = 1500.0,
) -> float:
    """Interaural time difference by cross correlation, positive when the left ear leads.

    **Band limited, and that is not a refinement.** Above about 1.5 kHz half a
    wavelength is shorter than the head, so the interaural phase is ambiguous
    and no longer a cue a listener uses; a broadband correlation of a real head
    response is dominated by exactly that region and returns a fraction of the
    true delay. It is also the band a magnitude least squares decoder
    deliberately abandons the phase in, so measuring it there would be
    measuring the decoder's own arbitrary choice.
    """
    block = np.asarray(brir, dtype=float)
    if block.shape[0] != 2:
        raise ValueError(f"expected two ears, got shape {block.shape}")
    if cut_off_hz > 0.0:
        block = lowpass(block, sample_rate_hz, cut_off_hz)
    limit = int(round(maximum_s * sample_rate_hz))
    correlation = fftconvolve(block[0], block[1][::-1], mode="full")
    centre = block.shape[1] - 1
    window = correlation[centre - limit : centre + limit + 1]
    return float((int(np.argmax(window)) - limit) / sample_rate_hz) * -1.0


def ild_db(brir: np.ndarray) -> float:
    """Broadband interaural level difference in dB, positive when the left ear is louder."""
    block = np.asarray(brir, dtype=float)
    energy = np.sum(block**2, axis=1)
    if np.any(energy <= 0.0):
        return float("nan")
    return float(10.0 * np.log10(energy[0] / energy[1]))


def interaural_coherence(
    brir: np.ndarray,
    sample_rate_hz: float,
    *,
    frame_s: float = 0.02,
    hop_s: float = 0.01,
    band_hz: float | None = None,
    lag: Lag = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Coherence between the ears against time, as ``(times, coherence)``.

    Roadmap section 5.5 makes a prediction this measures: in a diffuse field
    the interaural coherence falls to nothing near 1 kHz and stays negligible
    above, about 0.04 at 8 kHz, which is why channel independent noise is
    physically correct in a high frequency tail. A rendering whose tail stays
    coherent has not decoded a diffuse field.

    **``band_hz`` is what makes it test that claim.** Measured broadband it does
    not: a room response's energy is dominated by the octaves below 1 kHz, where
    the two ears are genuinely coherent because the head is small against the
    wavelength, so a broadband figure of 0.7 is correct and says nothing about
    the prediction. The claim is about the octaves above.

    **``lag`` decides which of two different quantities this is, and only one of
    them tests the roadmap.** The peak over lags is heavily biased upwards on a
    short window: on two genuinely independent ears, where the true value is
    zero, it reads 0.59 at 250 Hz and 0.19 at 8 kHz over 20 ms frames. So a
    measured 0.33 at 8 kHz is not eight times the roadmap's 0.04; it is a number
    whose own floor is 0.19. Quote :func:`coherence_floor` beside it.
    """
    block = np.asarray(brir, dtype=float)
    if band_hz is not None:
        high = min(band_hz * np.sqrt(2.0), 0.49 * sample_rate_hz)
        block = lowpass(block, sample_rate_hz, high)
        block = block - lowpass(block, sample_rate_hz, band_hz / np.sqrt(2.0))
    frame = max(int(round(frame_s * sample_rate_hz)), 8)
    hop = max(int(round(hop_s * sample_rate_hz)), 1)
    times, values = [], []
    for start in range(0, block.shape[1] - frame + 1, hop):
        left = block[0, start : start + frame]
        right = block[1, start : start + frame]
        norm = np.sqrt(np.sum(left**2) * np.sum(right**2))
        times.append((start + frame / 2.0) / sample_rate_hz)
        if norm <= 0:
            values.append(0.0)
        elif lag == "zero":
            values.append(float(np.sum(left * right) / norm))
        else:
            values.append(float(np.max(np.abs(fftconvolve(left, right[::-1]))) / norm))
    return np.asarray(times), np.asarray(values)


def coherence_floor(
    sample_rate_hz: float,
    *,
    frame_s: float = 0.02,
    hop_s: float = 0.01,
    band_hz: float | None = None,
    lag: Lag = "max",
    seconds: float = 1.0,
    seed: int = 0,
) -> float:
    """What :func:`interaural_coherence` reads on ears that share nothing.

    The true answer is zero and the measurement does not give it, because the
    peak over lags of a finite correlation is positive by construction. Quoting
    a coherence without this floor invites a reader to compare it against a
    theoretical value it cannot reach: on 20 ms frames the floor is 0.59 at
    250 Hz and 0.19 at 8 kHz.
    """
    rng = np.random.default_rng(seed)
    ears = rng.standard_normal((2, int(seconds * sample_rate_hz)))
    _, values = interaural_coherence(
        ears, sample_rate_hz, frame_s=frame_s, hop_s=hop_s, band_hz=band_hz, lag=lag
    )
    return float(np.mean(values)) if values.size else 0.0
