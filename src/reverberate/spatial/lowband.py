"""The band under the crossover as a pack keeps it: 4 kHz, 4800 samples, the low masks taken.

``docs/formats/scene-pack.md`` stores each wave response between a source
position and a listening cell as ``low/ir``: the response with the
crossover's low side already taken as :func:`reverberate.mirror.hybrid.blend`
takes it, at 4 kHz. Its spectrum is zero above the end of the crossover's
ramp, 1414 Hz, so nothing is lost by keeping the bins under 2 kHz alone, and
one response is 1.23 MB instead of 14.7.

Three forms, and the functions between them:

- the **response** at the delivery rate, 48 kHz, ``[channel, sample]``;
- the **cache form**: the same decimated to 4 kHz and 4800 samples by
  :func:`decimate`, its masks and its air not yet taken. It is what the
  dwelling's pair cache holds, because a pair's key
  (:func:`pair_key`) names neither a crossover nor an atmosphere;
- the **stored form**: :func:`low_side` of either, which is ``low/ir``.

:func:`to_stored` goes from the first to the last and :func:`from_stored`
back to 48 kHz. The decimation is exact for whatever lies under 2 kHz: the
bins of the transform are kept, not filtered. Everything runs in the array
namespace of its argument, ``numpy`` or ``cupy``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.fft import next_fast_len

from reverberate.audio import Atmosphere, frame_for
from reverberate.mirror.hybrid import Crossover
from reverberate.spatial.translate import SOUND_SPEED_M_S, namespace_of

__all__ = [
    "FIELD_UNIT_AT_1M",
    "HEADROOM",
    "LOWCUT_HZ",
    "LOWCUT_ORDER",
    "LOW_DURATION_S",
    "LOW_RATE_HZ",
    "LOW_SAMPLES",
    "decimate",
    "delayed",
    "from_stored",
    "low_side",
    "lowcut_response",
    "onset_s",
    "pair_key",
    "solve_fmax_hz",
    "to_stored",
    "with_air",
]

#: The rate, the length and the window of a stored low band response.
LOW_RATE_HZ = 4000.0
LOW_SAMPLES = 4800
LOW_DURATION_S = LOW_SAMPLES / LOW_RATE_HZ

#: A low only solve's ``fmax`` over the top of the crossover's ramp: 1500 Hz over 1414.
HEADROOM = 1.5 / 2.0**0.5

#: The rate an onset is located at, whatever rate the response is kept at.
_ONSET_RATE_HZ = 48000.0

#: The scale of a field and of the pair cache: what the direct sound of a
#: source of unit gain reads at 1 m in free air, as the amplitude of an
#: impulse at 48 kHz. The solver's impulse is one step of its grid, heard as
#: the free space ``1 / (4 pi d)``, and a field is on the scale of its solve
#: to 8 kHz at 10.5 points per wavelength, whose grid runs at
#: ``sqrt(3) * 10.5 * 8000`` Hz: 0.02625. Measured on the validated field of
#: hssd_0076 (S1, eight lattice points 0.34 to 0.72 m away, under 1.5 kHz):
#: 0.0258 to 0.0264, and on the first three pairs solved to 1500 Hz, before
#: their first reflection: 0.0235 to 0.0255. Dividing by it is what makes a
#: pack physical (``docs/formats/scene-pack.md``).
FIELD_UNIT_AT_1M: float = 48000.0 / (4.0 * float(np.pi) * float(np.sqrt(3.0)) * 10.5 * 8000.0)


#: Where the band under the crossover starts: the low cut of every solve's fit
#: (``reverberate.accel.pairs.LOWCUT_HZ`` and ``LOWCUT_ORDER``, a Butterworth high pass
#: of order 8 at 40 Hz). -3 dB at 40 Hz, -1 dB at 43.7 Hz, -0.14 dB at 50 Hz, nothing
#: from 63 Hz up: a pack is valid from 45 Hz (``docs/formats/scene-pack.md``).
LOWCUT_HZ, LOWCUT_ORDER = 40.0, 8


def lowcut_response(
    freqs_hz: np.ndarray, cut_hz: float = LOWCUT_HZ, order: int = LOWCUT_ORDER
) -> np.ndarray:
    """The fit's low cut as a spectrum, complex and causal: what a stand-in for a solve applies.

    The analogue Butterworth high pass the fit designs
    (:func:`reverberate.accel.dsp.lowcut_sos`), read at ``freqs_hz``; zero
    at 0 Hz. A stand-in that cut higher than a solve does made a synthetic
    pack say that nothing is rendered under 80 Hz
    (``docs/open-questions/chain-audit.md``, D11).
    """
    from scipy.signal import butter, freqs_zpk

    zeros, poles, gain = butter(
        order, 2.0 * np.pi * cut_hz, btype="high", analog=True, output="zpk"
    )
    _, response = freqs_zpk(zeros, poles, gain, worN=2.0 * np.pi * np.asarray(freqs_hz, float))
    return np.asarray(response, dtype=complex)


def solve_fmax_hz(crossover: Crossover | None = None, *, headroom: float = HEADROOM) -> float:
    """The ``fmax`` a solve that carries the low side alone is given: 1500 Hz at 1 kHz.

    The low masks reach zero at the top of the ramp,
    ``cutoff_hz * 2^(width_octaves / 2)``, 1414 Hz, and the solve must be
    valid up to there: 10.5 points per wavelength or more, fitted by the
    encoder, and inside the solve's own band limit. A field's low band is not:
    it is solved to 1000 Hz, band limited there by a filter that is 6 dB down
    at 1000 Hz, and left unfitted above it, because a field takes its wave
    data over 800 Hz from the next grid up. Alone, it would leave the low side
    2.75 dB short over the crossover's octave.

    ``headroom`` is how far over the ramp's top the solve's ``fmax`` is put.
    What the band limiting filter then takes from the low side:

    | ``fmax`` | at 1000 Hz | at 1189 Hz | energy, 707 to 1414 Hz | cost |
    | --- | --- | --- | --- | --- |
    | 1414 | -0.53 dB | -1.94 dB | -0.32 dB | 4.0 |
    | 1500 | -0.33 dB | -1.26 dB | -0.21 dB | 5.1 |
    | 1768 | -0.09 dB | -0.36 dB | -0.06 dB | 9.8 |

    the cost being the stencil's against a solve to 1 kHz, the fourth power
    of the ratio. At 1189 Hz the power mask is itself at -8.4 dB. 1500 Hz is
    the least round figure over the ramp's top; 1768 Hz is the ramp's top at
    0.8 of ``fmax``, where a field puts its seams
    (:data:`reverberate.spatial.bands.CROSSOVER_OF_FMAX`), at twice the price.
    """
    _, top = (crossover or Crossover()).band_hz()
    return float(top * headroom)


def _resized(ir: Any, samples: int, xp: Any) -> Any:
    """``ir`` cut or padded with zeros to ``samples`` along its last axis."""
    have = int(ir.shape[-1])
    if have >= samples:
        return ir[..., :samples]
    out = xp.zeros((*ir.shape[:-1], samples), dtype=ir.dtype)
    out[..., :have] = ir
    return out


def decimate(ir: Any, rate_hz: float, *, duration_s: float = LOW_DURATION_S, xp: Any = None) -> Any:
    """A response at ``rate_hz`` as ``duration_s`` at 4 kHz: its bins under 2 kHz, kept.

    ``ir`` is ``[..., sample]``; it is cut or padded to ``duration_s`` first,
    the engine running a step or three over the window it was asked for.
    Returns float32. Exact when nothing lies over 2 kHz; what does is dropped,
    not folded.
    """
    xp = namespace_of(ir, xp=xp)
    samples = round(duration_s * rate_hz)
    low = round(duration_s * LOW_RATE_HZ)
    if abs(samples - duration_s * rate_hz) > 1e-6 or samples < low:
        raise ValueError(f"{duration_s} s at {rate_hz} Hz is not a whole number of samples")
    block = _resized(xp.asarray(ir, dtype=xp.float64), samples, xp)
    spectrum = xp.fft.rfft(block, axis=-1)[..., : low // 2 + 1]
    # The bin at the new Nyquist would be read twice by the shorter transform.
    spectrum[..., -1] = 0.0
    out: Any = (xp.fft.irfft(spectrum, n=low, axis=-1) * (low / samples)).astype(xp.float32)
    return out


def from_stored(low: Any, rate_hz: float = 48000.0, *, xp: Any = None) -> Any:
    """A stored or cached response brought to ``rate_hz``: the inverse of :func:`decimate`.

    ``[..., 4800]`` at 4 kHz in, ``[..., 57600]`` at 48 kHz out, float64. No
    filter is designed: the spectrum is carried over and is zero above 2 kHz.
    """
    xp = namespace_of(low, xp=xp)
    have = int(low.shape[-1])
    samples = round(have * rate_hz / LOW_RATE_HZ)
    spectrum = xp.fft.rfft(xp.asarray(low, dtype=xp.float64), axis=-1)
    full = xp.zeros((*spectrum.shape[:-1], samples // 2 + 1), dtype=xp.complex128)
    full[..., : spectrum.shape[-1]] = spectrum
    out: Any = xp.fft.irfft(full, n=samples, axis=-1) * (samples / have)
    return out


def delayed(ir: Any, rate_hz: float, delay_s: float, *, xp: Any = None) -> Any:
    """``ir`` later by ``delay_s``, at its length: what leaves its end is dropped, nothing wraps.

    The pair cache is on the geometric clock, a pack on that clock plus the
    mirror's lead, which is seldom a whole number of samples at 4 kHz (512
    at 48 kHz are 42.67). The delay is a phase, taken on a transform long
    enough that what it pushes past the end does not come back at the start.
    Returns float64.
    """
    xp = namespace_of(ir, xp=xp)
    block = xp.asarray(ir, dtype=xp.float64)
    if delay_s == 0.0:
        return block
    if delay_s < 0.0:
        raise ValueError("a response is delayed, never advanced: its start would be lost")
    samples = int(block.shape[-1])
    moved = int(np.ceil(delay_s * rate_hz))
    padded = int(next_fast_len(samples + moved + 64, real=True))
    freqs = xp.fft.rfftfreq(padded, 1.0 / rate_hz)
    spectrum = xp.fft.rfft(block, n=padded, axis=-1) * xp.exp(-2j * np.pi * freqs * delay_s)
    out: Any = xp.fft.irfft(spectrum, n=padded, axis=-1)[..., :samples]
    # The response now ends where it was not cut to end. Its last ``delay_s`` fall to
    # zero, as long as what it lost: a response cut on a sample rings in every transform
    # that reads it afterwards, the crossover's masks first.
    fall = min(moved, samples)
    out[..., samples - fall :] *= xp.asarray(
        0.5 + 0.5 * np.cos(np.pi * (np.arange(fall) + 1.0) / fall)
    )
    return out


def onset_s(omni: Any, rate_hz: float, *, xp: Any = None) -> float:
    """When the loudest sample of channel 0 falls, read at 48 kHz: the pack's ``low/onset_s``.

    A response kept at 4 kHz is brought to 48 kHz first, so the answer does
    not depend on the rate the response happens to be kept at.
    """
    xp = namespace_of(omni, xp=xp)
    signal = xp.asarray(omni, dtype=xp.float64)
    if rate_hz != _ONSET_RATE_HZ:
        if rate_hz != LOW_RATE_HZ:
            raise ValueError(f"an onset is read at 48 kHz or at 4 kHz, not at {rate_hz} Hz")
        signal = from_stored(signal, _ONSET_RATE_HZ, xp=xp)
    return float(int(xp.argmax(xp.abs(signal)))) / _ONSET_RATE_HZ


def _onset_window(onset: float, samples: int, rate_hz: float, crossover: Crossover) -> np.ndarray:
    """:meth:`Crossover.onset_window` sampled at ``rate_hz``, its onset given in seconds.

    At 48 kHz, and with the onset on a sample, this is that method's window
    to rounding; at 4 kHz it is the same window read every twelfth sample.
    """
    if crossover.coherent_s <= 0.0:
        return np.zeros(samples)
    fade = max(crossover.coherent_fade_s, 1.0 / _ONSET_RATE_HZ)
    after = (np.arange(samples) / rate_hz - (onset + crossover.coherent_s)) / fade
    window: np.ndarray = 0.5 * (1.0 + np.cos(np.pi * np.clip(after, 0.0, 1.0)))
    return window


def with_air(
    ir: Any,
    rate_hz: float,
    atmosphere: Atmosphere,
    *,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    xp: Any = None,
) -> Any:
    """A wave response with the air's absorption, at the rate it comes at: float64.

    What :func:`low_side` takes its masks of, and what the onset and the seam
    of a pair are read on (:mod:`reverberate.trace.level`): the cache holds a
    response before its air, and ``blend`` joins one that carries it.
    """
    from reverberate.accel.dsp import air_absorption

    xp = namespace_of(ir, xp=xp)
    block = xp.asarray(ir, dtype=xp.float64)
    samples = int(block.shape[-1])
    # The frame of :func:`frame_for` is in samples at 48 kHz; the same
    # span of time at this rate, a multiple of four.
    frame = max(32, 4 * round(frame_for(samples / rate_hz) * rate_hz / 48000.0 / 4))
    return air_absorption(
        block,
        rate_hz,
        xp,
        sound_speed_m_s=sound_speed_m_s,
        atmosphere=atmosphere,
        frame=frame,
    )


def low_side(
    ir: Any,
    rate_hz: float,
    crossover: Crossover | None = None,
    *,
    atmosphere: Atmosphere | None = None,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    onset: float | None = None,
    xp: Any = None,
) -> Any:
    """The crossover's low side of a wave response, as ``blend`` takes it.

    ``ir`` is ``[channel, sample]`` at ``rate_hz``, 48 kHz or the cache
    form's 4 kHz. The part within the onset window of channel 0 goes through
    the pressure mask and the rest through the power mask
    (:meth:`Crossover.masks`), so the result is zero above the ramp's top.
    The window is anchored on ``onset``, in seconds, and on the loudest
    sample of channel 0 where none is given, which is ``blend``'s own rule
    and is right where the loudest sample is the direct sound.
    With ``atmosphere`` the air's absorption is applied first
    (:func:`reverberate.audio.apply_air_absorption`); without, the response
    is taken as it is, which is right for one that already carries its air.
    Returns float64, the shape and the rate of ``ir``.
    """
    xp = namespace_of(ir, xp=xp)
    crossover = crossover or Crossover()
    block = xp.asarray(ir, dtype=xp.float64)
    samples = int(block.shape[-1])
    if atmosphere is not None:
        block = with_air(block, rate_hz, atmosphere, sound_speed_m_s=sound_speed_m_s, xp=xp)
    anchor = onset_s(block[0], rate_hz, xp=xp) if onset is None else float(onset)
    together = xp.asarray(_onset_window(anchor, samples, rate_hz, crossover))
    power, _ = crossover.masks(samples, rate_hz, power=True)
    pressure, _ = crossover.masks(samples, rate_hz, power=False)
    spectrum = xp.fft.rfft(block * (1.0 - together), axis=-1) * xp.asarray(power)
    spectrum = spectrum + xp.fft.rfft(block * together, axis=-1) * xp.asarray(pressure)
    out: Any = xp.fft.irfft(spectrum, n=samples, axis=-1)
    return out


def to_stored(
    ir: Any,
    rate_hz: float,
    crossover: Crossover | None = None,
    *,
    atmosphere: Atmosphere | None = None,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
    xp: Any = None,
) -> Any:
    """A wave response at ``rate_hz`` as the pack's ``low/ir``: ``[channel, 4800]`` float32.

    The low side is taken at the rate the response comes at, then decimated,
    so from a field's 48 kHz response this is the low part of
    :func:`reverberate.mirror.hybrid.blend` exactly; from the cache form it is
    :func:`low_side` and nothing else.
    """
    xp = namespace_of(ir, xp=xp)
    if rate_hz != LOW_RATE_HZ:
        samples = round(LOW_DURATION_S * rate_hz)
        ir = _resized(xp.asarray(ir, dtype=xp.float64), samples, xp)
    masked = low_side(
        ir, rate_hz, crossover, atmosphere=atmosphere, sound_speed_m_s=sound_speed_m_s, xp=xp
    )
    if rate_hz == LOW_RATE_HZ:
        out: Any = masked.astype(xp.float32)
        return out
    return decimate(masked, rate_hz, xp=xp)


def pair_key(
    voxel_low_key: str,
    source_m: Any,
    cell_m: Any,
    *,
    encoder: dict[str, Any],
    solver: str,
    window_s: float = LOW_DURATION_S,
) -> str:
    """The key of one (source position, listening cell) pair in the dwelling's cache.

    The SHA-256, 64 hexadecimal characters, of the canonical JSON of the
    grid's key, the two positions in whole millimetres, the encoder's
    settings, the solver's version and the window. No recipe, no crossover
    and no atmosphere enter it: the cache holds the response before its masks
    and its air, so every recipe on the dwelling that passes through the pair
    finds it.
    """

    def millimetres(position: Any) -> list[int]:
        values = np.asarray(position, dtype=float).reshape(3)
        return [round(float(v) * 1000.0) for v in values]

    record = {
        "voxel_low_key": str(voxel_low_key),
        "source_mm": millimetres(source_m),
        "cell_mm": millimetres(cell_m),
        "encoder": encoder,
        "solver": str(solver),
        "window_s": float(window_s),
    }
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()
