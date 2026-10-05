"""A pack's low band responses in fewer bytes: the bins that hold something, in 16 bits.

``low/ir`` is 1.23 MB a pair, ``[64, 4800]`` float32 at 4 kHz, and a scene
of twenty minutes has seventeen thousand pairs. Four things of a response
are stored that nothing reads, and each is a lever here
(``docs/open-questions/low-band-compact.md`` has what each was measured to
cost):

- **the bins over the crossover's ramp** (``bins``). The response is zero
  above 1414 Hz and is kept as 4800 samples of time that reach 2 kHz. Its
  transform's bins up to the ramp's top are the same response, 1699 complex
  numbers for 4800 real ones;
- **the high degrees where they hold no field** (``degree``). Within a
  radius ``R`` of the cell, the ``2n + 1`` channels of degree ``n`` hold
  ``(2n + 1) j_n(k R)^2`` of a plane wave's energy, which is under
  ``(2n + 1) ((k R)^n / (2n + 1)!!)^2``, and the engine reads a cell
  within :data:`REACH_M` of its centre. Degree ``n`` is kept from the
  frequency where that bound reaches a stated level (:func:`pass_hz`,
  :func:`cut_hz`), through a raised cosine :data:`RAMP_HZ` wide under it;
- **the end of a response that has decayed** (``decay``). Each degree is cut
  where what it gives within reach has fallen a stated number of decibels
  under the loudest moment of the pressure, faded over :data:`FADE_S`, and
  its transform taken at that length: a response half as long has half the
  bins;
- **sixteen bits of thirty two** (``int16``, with one scale for each channel
  and each :data:`BLOCK_HZ` of a pair). The samples themselves in 16 bits
  are not offered: their noise is as loud above the ramp as under it
  (``float16`` of the synthetic pack is over the 1e-4 a reader may assume
  there) and as loud at a response's end as at its arrival.

A pack whose responses are kept this way has ``low/compact`` in place of
``low/ir`` (``docs/formats/scene-pack.md``), and :class:`CompactIr` gives
the engine what the dataset gave it: ``ir[row]`` is ``[channel, sample]``
float32, decoded when asked for. Nothing of the engine's mathematics
changes. :func:`compact_pack` rewrites a pack that exists, so the same trace
is heard both ways; :class:`Levers` given to
:class:`reverberate.render.pack.PackWriter` writes one so from the start.
With no lever a pack is written as it always was, byte for byte.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.spatial.sh import degrees_of

__all__ = [
    "BLOCK_HZ",
    "FADE_S",
    "FORMAT",
    "LADDER",
    "RAMP_HZ",
    "REACH_M",
    "CompactIr",
    "Encoded",
    "Levers",
    "compact_pack",
    "cut_hz",
    "decode",
    "degree_weights",
    "encode",
    "first_hz",
    "pass_hz",
    "share_within_reach",
    "write_compact",
]

#: What ``low/compact`` says it is.
FORMAT = "bins/1"
#: How far from a cell's centre its expansion is read: a head's 0.10 m at the
#: 0.20 m one cell is translated (``spatial.translate.TRANSLATE_WITHIN_M``).
REACH_M = 0.30
#: A degree's raised cosine is this wide, under the frequency the degree is whole
#: from. A ramp rings before an arrival as after it, for about one over its width,
#: and a response starts 11 ms into its window (``/mirror``'s ``lead_s``): 100 Hz
#: rings 10 ms. A ramp a quarter of the degree's frequency wide, 34 Hz for degree 3,
#: rang 30 ms, and what rang before the window's start came round to its end as
#: loud as -35 dB of the pressure's peak.
RAMP_HZ = 100.0
#: A degree that would enter under this is kept whole: there is nothing to save there.
MIN_CUT_HZ = 100.0
#: A degree's bins are kept from this far under where its raised cosine starts:
#: a response that ends in time reaches a little under its band, as over it.
GUARD_HZ = 15.0
#: What is left of a ramp's ring before the window's start comes round to the
#: window's end: the last of it falls to nothing over this long, for every degree
#: that has a ramp. What the response itself holds there is 70 dB under its peak.
WRAP_S = 0.050
#: The fade at the end of a degree that was cut, and the envelope the cut is read on.
FADE_S = 0.020
ENVELOPE_S = 0.010
#: A degree's length is a multiple of this many samples: 50 ms at 4 kHz.
LADDER = 200
#: With ``int16`` a channel has one scale for each block of bins this wide, in Hz:
#: one scale a channel leaves its noise as loud at 1 kHz, where the crossover has
#: taken the response, as at 200 Hz, where the response is.
BLOCK_HZ = 100.0
#: The values of a pair are appended to ``data`` in chunks of this many.
CHUNK = 1 << 16

_SAMPLES = {"float32": "<f4", "int16": "<i2"}
_INT16 = 32767.0


@dataclass(frozen=True)
class Levers:
    """Which levers a pack's low band is written with; none is today's ``low/ir``."""

    #: ``float32`` or ``int16``, which is of the bins and has their scales.
    sample: str = "float32"
    #: The transform's bins up to the ramp's top, not the samples.
    bins: bool = False
    #: Decibels under the whole at which a degree's share of the energy within reach
    #: lets it go (:func:`pass_hz`); 0 keeps every degree at every frequency.
    degree_db: float = 0.0
    #: Decibels under the pressure's loudest moment at which a degree, as it is heard
    #: within reach, ends; 0 keeps it whole.
    decay_db: float = 0.0

    def __post_init__(self) -> None:
        if self.sample not in _SAMPLES:
            raise ValueError(f"a sample is one of {sorted(_SAMPLES)}, not {self.sample!r}")
        if self.degree_db < 0.0 or self.decay_db < 0.0:
            raise ValueError("a lever is not negative")
        if not self.bins and (self.sample != "float32" or self.degree_db or self.decay_db):
            raise ValueError("int16, degree and decay are levers of the bins: say bins too")

    @property
    def off(self) -> bool:
        return self == Levers()

    def record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def parse(cls, text: str | None) -> Levers:
        """``bins,int16,degree=40,decay=60`` as levers; nothing, ``none`` or ``off`` is none."""
        found: dict[str, Any] = {}
        for word in (text or "").replace(" ", "").split(","):
            name, _, value = word.partition("=")
            if name in ("", "none", "off"):
                continue
            if name in _SAMPLES:
                found["sample"] = name
            elif name == "bins":
                found["bins"] = True
            elif name == "degree":
                found["degree_db"] = float(value)
            elif name == "decay":
                found["decay_db"] = float(value)
            else:
                raise ValueError(f"no lever is called {name!r}")
        return cls(**found)


def pass_hz(
    order: int, floor_db: float, sound_speed_m_s: float, *, reach_m: float = REACH_M
) -> np.ndarray:
    """From where each degree is kept whole, in Hz: ``[degree]``; 0 for every degree if off.

    A plane wave of unit pressure is, at a distance ``r`` from the centre
    of its expansion, the sum over the degrees of ``(2n + 1) j_n(k r)^2``,
    which is one; and ``j_n(x) <= x^n / (2n + 1)!!``. A degree is kept from
    the frequency where that bound on its share, at ``r = reach_m``, is
    ``floor_db`` under the whole:

    ``f_n = c / (2 pi R) * (10^(-floor_db / 20) (2n + 1)!! / sqrt(2n + 1))^(1 / n)``.

    A rule in ``k R`` over ``n`` alone (``k R >= alpha n``) lets the low
    degrees go too early: at ``k R = 0.5`` degree 1 still holds 0.08 of the
    energy. At 40 dB and 0.30 m the degrees 1 to 7 enter at 3, 47, 134,
    243, 362, 489 and 619 Hz.
    """
    found = np.zeros(order + 1)
    if floor_db <= 0.0:
        return found
    odd = 1.0
    for n in range(1, order + 1):
        odd *= 2 * n + 1
        found[n] = (10.0 ** (-floor_db / 20.0) * odd / np.sqrt(2 * n + 1)) ** (1.0 / n)
    found *= sound_speed_m_s / (2.0 * np.pi * reach_m)
    return found


def cut_hz(
    order: int, floor_db: float, sound_speed_m_s: float, *, reach_m: float = REACH_M
) -> np.ndarray:
    """:func:`pass_hz` as the lever applies it: a degree that enters under 100 Hz is whole."""
    whole = pass_hz(order, floor_db, sound_speed_m_s, reach_m=reach_m)
    whole[whole < MIN_CUT_HZ] = 0.0
    return whole


def first_hz(order: int, floor_db: float, sound_speed_m_s: float) -> np.ndarray:
    """From where each degree's bins are kept, in Hz, ``[degree]``: the file's ``first_hz``."""
    start = cut_hz(order, floor_db, sound_speed_m_s) - RAMP_HZ
    found: np.ndarray = np.maximum(start - GUARD_HZ, 0.0)
    return found


def degree_weights(freqs_hz: np.ndarray, passes_hz: np.ndarray) -> np.ndarray:
    """Each degree's weight at each frequency: ``[degree, frequency]``, 0 to 1.

    Nothing more than :data:`RAMP_HZ` under the degree's frequency, a raised
    cosine up to it and one above; one everywhere for a degree whose
    frequency is 0.
    """
    freqs = np.asarray(freqs_hz, dtype=float)[None, :]
    whole = np.asarray(passes_hz, dtype=float)[:, None]
    at = np.where(whole > 0.0, (freqs - (whole - RAMP_HZ)) / RAMP_HZ, 1.0)
    found: np.ndarray = 0.5 - 0.5 * np.cos(np.pi * np.clip(at, 0.0, 1.0))
    return found


@dataclass(frozen=True)
class Encoded:
    """One pair: each degree's length in samples, the scales, and the values.

    ``scale`` is ``[channel, block]``, what one step of an ``int16`` value
    is worth in each block of :data:`BLOCK_HZ`; zeros for another sample.
    """

    samples: np.ndarray
    scale: np.ndarray
    data: np.ndarray


def _bins(samples: int, from_hz: float, top_hz: float, rate_hz: float) -> tuple[int, int]:
    """The bins a degree of ``samples`` keeps: the first, and one past the last."""
    if samples == 0:
        return 0, 0
    first = int(np.floor(from_hz * samples / rate_hz + 1e-9))
    last = min(int(np.ceil(top_hz * samples / rate_hz - 1e-9)), samples // 2 - 1)
    return first, max(first, last + 1)


def _blocks(top_hz: float) -> int:
    """How many blocks of :data:`BLOCK_HZ` reach the last bin a degree may keep."""
    return int(top_hz // BLOCK_HZ) + 2


def _block_of(first: int, stop: int, samples: int, rate_hz: float, blocks: int) -> np.ndarray:
    """The block each of the bins ``first`` to ``stop - 1`` of a transform of ``samples`` is in."""
    at = np.arange(first, stop) * (rate_hz / (samples * BLOCK_HZ))
    found: np.ndarray = np.minimum((at + 1e-9).astype(np.int64), blocks - 1)
    return found


def share_within_reach(
    order: int, freqs_hz: np.ndarray, sound_speed_m_s: float, *, reach_m: float = REACH_M
) -> np.ndarray:
    """The bound on what a channel of each degree gives within reach: ``[degree, frequency]``.

    ``sqrt(2n + 1) (k R)^n / (2n + 1)!!``, and one where that is over one:
    the square is the bound of :func:`pass_hz` on the degree's share of a
    plane wave's energy, so a channel of any degree under this weight is on
    the pressure's own scale.
    """
    x = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) * reach_m / sound_speed_m_s
    found = np.ones((order + 1, x.size))
    odd = 1.0
    for n in range(1, order + 1):
        odd *= 2 * n + 1
        found[n] = np.minimum(np.sqrt(2 * n + 1) * x**n / odd, 1.0)
    return found


def _lengths(
    block: np.ndarray, degree: np.ndarray, decay_db: float, rate_hz: float, sound_speed_m_s: float
) -> np.ndarray:
    """How many samples of each degree are kept, a multiple of :data:`LADDER`: ``[degree]``.

    A degree is heard as :func:`share_within_reach` weighs it, and its
    level is the mean square of its channels so weighed, over
    :data:`ENVELOPE_S`. It ends :data:`FADE_S` after the last moment that
    level is within ``decay_db`` of the loudest moment of degree 0, the
    pressure at the cell. A degree that never is, is not kept at all.

    Unweighed, the upper degrees do not decay: under the frequency a sphere
    of the array's size can tell them at they hold the encoder's gain on
    the room's slowest modes, a floor 35 to 50 dB under the pressure's peak
    from 0.6 s to the window's end, which is nothing 0.30 m from the cell.
    """
    total = int(block.shape[-1])
    orders = int(degree.max()) + 1
    span = max(1, round(ENVELOPE_S * rate_hz))
    # The weight is a filter without delay: on a longer transform, so that what it rings
    # before the window's start does not come round and pass for the response's end.
    room = round(WRAP_S * rate_hz)
    padded = np.zeros((block.shape[0], total + 2 * room))
    padded[:, room : room + total] = block
    freqs = np.fft.rfftfreq(total + 2 * room, 1.0 / rate_hz)
    weight = share_within_reach(orders - 1, freqs, sound_speed_m_s)[degree]
    heard = np.fft.irfft(np.fft.rfft(padded, axis=-1) * weight, n=total + 2 * room, axis=-1)
    heard = heard[:, room : room + total]
    power = np.stack([np.mean(heard[degree == n] ** 2, axis=0) for n in range(orders)])
    summed = np.cumsum(np.pad(power, ((0, 0), (span, 0))), axis=1)
    level = (summed[:, span:] - summed[:, :-span]) / span
    floor = float(level[0].max()) * 10.0 ** (-decay_db / 10.0)
    found = np.zeros(orders, dtype=np.int32)
    for n in range(orders):
        over = np.nonzero(level[n] > floor)[0]
        if over.size:
            end = int(over[-1]) + 1 + round(FADE_S * rate_hz)
            found[n] = min(total, -(-end // LADDER) * LADDER)
    return found


def encode(
    ir: np.ndarray,
    levers: Levers,
    *,
    rate_hz: float,
    top_hz: float,
    sound_speed_m_s: float,
) -> Encoded:
    """One response, ``[channel, sample]`` at ``rate_hz``, as the bins ``levers`` keep.

    ``top_hz`` is where the response's spectrum ends, the top of the
    crossover's ramp. The values are each degree's channels in turn, each
    channel's bins from the degree's first, real part then imaginary.
    """
    if not levers.bins:
        raise ValueError("a response is encoded as bins or kept as it is")
    block = np.asarray(ir, dtype=np.float64)
    channels, total = block.shape
    degree = degrees_of(int(round(np.sqrt(channels))) - 1)
    orders = int(degree.max()) + 1
    stops = first_hz(orders - 1, levers.degree_db, sound_speed_m_s)
    if levers.degree_db > 0.0:
        whole = cut_hz(orders - 1, levers.degree_db, sound_speed_m_s)
        freqs = np.fft.rfftfreq(total, 1.0 / rate_hz)
        weights = degree_weights(freqs, whole)[degree]
        block = np.fft.irfft(np.fft.rfft(block, axis=-1) * weights, n=total, axis=-1)
        # What a raised cosine rang before the window's start has come round to its end.
        wrap = round(WRAP_S * rate_hz)
        block[whole[degree] > 0.0, total - wrap :] *= 0.5 + 0.5 * np.cos(
            np.pi * (np.arange(wrap) + 1.0) / wrap
        )
    lengths = np.full(orders, total, dtype=np.int32)
    if levers.decay_db > 0.0:
        lengths = _lengths(block, degree, levers.decay_db, rate_hz, sound_speed_m_s)
    fade = round(FADE_S * rate_hz)
    down = 0.5 + 0.5 * np.cos(np.pi * (np.arange(fade) + 1.0) / fade)
    blocks = _blocks(top_hz)
    scale = np.zeros((channels, blocks), dtype=np.float32)
    parts = []
    for n in range(orders):
        samples = int(lengths[n])
        first, stop = _bins(samples, float(stops[n]), top_hz, rate_hz)
        if stop == first:
            lengths[n] = 0
            continue
        piece = block[degree == n, :samples]
        if samples < total:
            piece = piece.copy()
            piece[:, samples - fade :] *= down
        spectrum = np.fft.rfft(piece, axis=-1)[:, first:stop]
        part = np.stack([spectrum.real, spectrum.imag], axis=-1)
        if levers.sample == "int16":
            of = _block_of(first, stop, samples, rate_hz, blocks)
            peak = np.abs(part).max(axis=2)
            steps = np.zeros((2 * n + 1, blocks), dtype=np.float32)
            for b in np.unique(of):
                steps[:, b] = peak[:, of == b].max(axis=1) / _INT16
            scale[degree == n] = steps
            unit = np.where(steps > 0.0, steps, 1.0).astype(np.float64)[:, of]
            part = np.clip(np.rint(part / unit[:, :, None]), -_INT16, _INT16)
        parts.append(part.ravel())
    data = np.concatenate(parts or [np.zeros(0)]).astype(_SAMPLES[levers.sample])
    return Encoded(samples=lengths, scale=scale, data=data)


def decode(
    encoded: Encoded,
    *,
    first_hz: np.ndarray,
    top_hz: float,
    rate_hz: float,
    samples: int,
) -> np.ndarray:
    """What :func:`encode` kept, as ``[channel, samples]`` float32: zeros where it kept nothing.

    ``first_hz`` is where each degree's bins start, ``[degree]``. A degree
    that was cut short ends in time, so it reaches a little over ``top_hz``
    (1e-4 of the spectrum's peak); it is brought back under it here, on the
    transform of the whole length, because a reader may assume a response is
    zero above the crossover's ramp.
    """
    orders = int(encoded.samples.shape[0])
    degree = degrees_of(orders - 1)
    out = np.zeros((degree.size, samples))
    cut = np.zeros(degree.size, dtype=bool)
    values = np.asarray(encoded.data)
    scaled = values.dtype.kind == "i"
    at = 0
    for n in range(orders):
        length = int(encoded.samples[n])
        first, stop = _bins(length, float(first_hz[n]), top_hz, rate_hz)
        if stop == first:
            continue
        count = (2 * n + 1) * (stop - first) * 2
        part = values[at : at + count].astype(np.float64).reshape(2 * n + 1, stop - first, 2)
        at += count
        if scaled:
            of = _block_of(first, stop, length, rate_hz, int(encoded.scale.shape[1]))
            part = part * encoded.scale[degree == n].astype(np.float64)[:, of, None]
        spectrum = np.zeros((2 * n + 1, length // 2 + 1), dtype=np.complex128)
        spectrum[:, first:stop] = part[..., 0] + 1j * part[..., 1]
        out[degree == n, :length] = np.fft.irfft(spectrum, n=length, axis=-1)
        cut[degree == n] = length < samples
    if at != values.size:
        raise ValueError(f"a pair holds {values.size} values and its lengths call for {at}")
    if cut.any():
        whole = np.fft.rfft(out[cut], axis=-1)
        whole[:, int(np.floor(top_hz * samples / rate_hz + 1e-9)) + 1 :] = 0.0
        out[cut] = np.fft.irfft(whole, n=samples, axis=-1)
    found: np.ndarray = out.astype(np.float32)
    return found


class CompactIr:
    """``low/compact`` read as ``low/ir`` is: ``ir[row]`` is ``[channel, sample]`` float32.

    A row is decoded when it is asked for and not kept: the engine keeps the
    spectra of the pairs it is using (:class:`reverberate.render.low.LowPart`).
    """

    def __init__(self, group: Any, channels: int, samples: int, rate_hz: float) -> None:
        self.group = group
        self.format = str(_text(group.attrs["format"]))
        if self.format != FORMAT:
            raise ValueError(f"low/compact is {self.format!r}; this reader knows {FORMAT!r}")
        self.levers = Levers(**json.loads(_text(group.attrs["levers_json"])))
        self.first_hz = np.asarray(group.attrs["first_hz"], dtype=float)
        self.top_hz = float(group.attrs["top_hz"])
        self.rate_hz = rate_hz
        self.offset = np.asarray(group["offset"][...], dtype=np.int64)
        self.lengths = np.asarray(group["samples"][...], dtype=np.int32)
        self.scale = np.asarray(group["scale"][...], dtype=np.float32) if "scale" in group else None
        self.data = group["data"]
        self.shape = (int(self.offset.shape[0]) - 1, channels, samples)
        self.dtype = np.dtype("<f4")
        self.ndim = 3

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        """What the responses take in the file."""
        values = int(self.data.size) * int(self.data.dtype.itemsize)
        scales = 0 if self.scale is None else self.scale.nbytes
        return values + self.offset.nbytes + self.lengths.nbytes + scales

    def __getitem__(self, row: Any) -> np.ndarray:
        if isinstance(row, tuple):
            return self[row[0]][row[1:]]
        if isinstance(row, slice):
            rows = range(*row.indices(self.shape[0]))
            if not rows:
                return np.zeros((0, *self.shape[1:]), dtype=self.dtype)
            return np.stack([self[r] for r in rows])
        row = int(row)
        if row < 0:
            row += self.shape[0]
        if not 0 <= row < self.shape[0]:
            raise IndexError(f"pair {row} of {self.shape[0]}")
        encoded = Encoded(
            samples=self.lengths[row],
            scale=np.zeros((0, 0), dtype=np.float32) if self.scale is None else self.scale[row],
            data=np.asarray(self.data[int(self.offset[row]) : int(self.offset[row + 1])]),
        )
        return decode(
            encoded,
            first_hz=self.first_hz,
            top_hz=self.top_hz,
            rate_hz=self.rate_hz,
            samples=self.shape[2],
        )


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def write_compact(
    low_group: Any,
    ir: Any,
    levers: Levers,
    *,
    rate_hz: float,
    top_hz: float,
    sound_speed_m_s: float,
) -> None:
    """A source's responses under ``low_group``, with ``levers``: what the trace calls.

    ``ir`` is ``[pair, channel, sample]``, an array or anything that gives a
    row when asked. What is written is ``low/compact``, a pair at a time,
    and no ``low/ir``.
    """
    pairs, channels, _ = (int(v) for v in ir.shape)
    if not levers.bins:
        raise ValueError("with no lever a pack keeps its low/ir, which its own writer writes")
    if isinstance(ir, CompactIr) and ir.levers == levers:
        low_group.copy(ir.group, "compact")
        return
    orders = int(round(np.sqrt(channels)))
    group = low_group.create_group("compact")
    group.attrs["format"] = FORMAT
    group.attrs["levers_json"] = json.dumps(levers.record(), sort_keys=True)
    group.attrs["top_hz"] = float(top_hz)
    group.attrs["reach_m"] = float(REACH_M)
    group.attrs["block_hz"] = float(BLOCK_HZ)
    group.attrs["first_hz"] = first_hz(orders - 1, levers.degree_db, sound_speed_m_s)
    data = group.create_dataset(
        "data", shape=(0,), maxshape=(None,), dtype=_SAMPLES[levers.sample], chunks=(CHUNK,)
    )
    offset = np.zeros(pairs + 1, dtype=np.int64)
    lengths = np.zeros((pairs, orders), dtype=np.int32)
    scale = np.zeros((pairs, channels, _blocks(top_hz)), dtype=np.float32)
    held: list[np.ndarray] = []
    waiting = written = 0

    def flush() -> None:
        nonlocal held, waiting, written
        if held:
            data.resize((written + waiting,))
            data[written:] = np.concatenate(held)
            written += waiting
            held, waiting = [], 0

    for row in range(pairs):
        made = encode(
            np.asarray(ir[row]),
            levers,
            rate_hz=rate_hz,
            top_hz=top_hz,
            sound_speed_m_s=sound_speed_m_s,
        )
        lengths[row], scale[row] = made.samples, made.scale
        offset[row + 1] = offset[row] + made.data.size
        held.append(made.data)
        waiting += made.data.size
        if waiting >= 64 * CHUNK:
            flush()
    flush()
    group.create_dataset("offset", data=offset)
    group.create_dataset("samples", data=lengths)
    if levers.sample == "int16":
        group.create_dataset("scale", data=scale)


def _held_bytes(ir: Any) -> int:
    """What a source's responses take where they are kept."""
    if isinstance(ir, CompactIr):
        return ir.nbytes
    return int(np.prod(ir.shape)) * int(np.dtype(ir.dtype).itemsize)


def compact_pack(
    source: Path,
    target: Path,
    levers: Levers,
    *,
    say: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """``source`` rewritten at ``target`` with its low band under ``levers``; the sizes.

    Everything but the responses is carried as it is, so the two packs are
    one trace heard two ways (``python -m reverberate.render check A --against B``).
    With no lever a pack that holds ``low/ir`` is copied, its own bytes, and
    one that holds ``low/compact`` is given its ``low/ir`` back.
    """
    from reverberate.render.pack import PackWriter, read_pack

    source, target = Path(source), Path(target)
    if target.resolve() == source.resolve():
        raise ValueError("a pack is not rewritten onto itself")
    with read_pack(source) as pack:
        before = sum(_held_bytes(item.low.ir) for item in pack.sources.values() if item.low)
        plain = not any(
            isinstance(item.low.ir, CompactIr) for item in pack.sources.values() if item.low
        )
        if levers.off and plain:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            size = source.stat().st_size
            return {
                "levers": levers.record(),
                "low_bytes": {"before": before, "after": before},
                "pack_bytes": {"before": size, "after": size},
            }
        with PackWriter(
            target,
            pack.header,
            pack.recipe,
            pack.listener,
            pack.cells,
            mirror=pack.mirror,
            crossover=pack.crossover,
            air=pack.air,
            directivity=pack.directivity,
            low_levers=levers,
        ) as writer:
            for item in pack.sources.values():
                if say is not None and item.low is not None:
                    say(f"{item.id}: {int(item.low.ir.shape[0])} pairs")
                writer.add_source(item)
    with read_pack(target) as made:
        after = sum(_held_bytes(item.low.ir) for item in made.sources.values() if item.low)
    return {
        "levers": levers.record(),
        "low_bytes": {"before": before, "after": after},
        "pack_bytes": {"before": source.stat().st_size, "after": target.stat().st_size},
    }
