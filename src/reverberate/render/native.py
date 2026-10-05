"""The signal engine's two inner loops in C, in single precision, with their ``numpy`` twins.

What the fast engine (:mod:`reverberate.render.fast`) does a sample at a
time is written here once in C and once in ``numpy``:

- :func:`early_interval`: one interval of the early part. Every path's two
  filtered signals are read at the path's moving delay through the
  interpolator of :mod:`reverberate.render.delay`, cross-faded, and laid on
  the listener's harmonics, which are linear between the direction nodes.
  The delay is computed in double precision, as the geometry is; the
  signals and the harmonics are in single precision.
- :func:`upsample`: the low band brought to the output rate, a polyphase
  product of a few taps an output sample.

The text is compiled once by the machine's own compiler and called through
``ctypes``, which lets go of the interpreter's lock, exactly as
:mod:`reverberate.mirror.native` builds its own (same cache, same switch:
``REVERBERATE_NO_NATIVE`` or no compiler gives the twin). **The twin is the
text, operation for operation**: every sum is taken in the text's order and
no product is fused with a sum (``-ffp-contract=off``), so a machine with
no compiler renders the same bits, and the tests hold one against the other
to the bit.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.native import _cache_root, _compiler
from reverberate.render import delay, noise

__all__ = [
    "FLAGS",
    "SOURCE",
    "available",
    "carrier",
    "disabled",
    "early_interval",
    "upsample",
    "why_not",
]

#: Flags of the build. Contraction off: a product and a sum are never one operation, so
#: the text and its twin round alike.
FLAGS = ("-O3", "-fPIC", "-shared", "-std=c99", "-ffp-contract=off", "-fno-fast-math")

SOURCE = r"""
#include <math.h>
#include <stddef.h>

#define RV_TAPS 12
#define RV_PHASES 1024

/* One end of a path over one interval: ``y`` read at the path's delay into ``got``.
   Returns 0, or 1 when a read falls outside the signal. */
static int rv_read(
    const float* y, long long length, double origin, const double* base, long long step,
    const float* table, float* got)
{
    for (long long n = 0; n < step; n++) {
        double position = 2.0 * (base[n] - origin);
        double whole = floor(position);
        double scaled = (position - whole) * RV_PHASES;
        long long phase = (long long)scaled;
        if (phase > RV_PHASES - 1) phase = RV_PHASES - 1;
        float blend = (float)(scaled - (double)phase);
        long long first = (long long)whole - (RV_TAPS / 2 - 1);
        if (first < 0 || first + RV_TAPS > length) return 1;
        const float* k0 = table + phase * RV_TAPS;
        const float* k1 = k0 + RV_TAPS;
        const float* s = y + first;
        float total = 0.0f;
        for (int j = 0; j < RV_TAPS; j++) total += (k0[j] + (k1[j] - k0[j]) * blend) * s[j];
        got[n] = total;
    }
    return 0;
}

/* ``early_interval`` of the module: see its twin for the arithmetic. */
int rv_early_interval(
    long long paths, long long step, long long nodes, long long channels,
    double start, double rate, double lead, double speed,
    const double* q0, const double* q1, const double* l0, const double* l1,
    const long long* ya, const long long* yb, const long long* la, const long long* lb,
    const double* wa, const double* wb,
    const float* harmonics, const float* table, float* out, double* base, float* work)
{
    long long within = step / nodes;
    float* a = work;
    float* b = work + step;
    float* signal = work + 2 * step;
    float* ramped = work + 3 * step;
    for (long long p = 0; p < paths; p++) {
        const double* s0 = q0 + 3 * p;
        const double* s1 = q1 + 3 * p;
        for (long long n = 0; n < step; n++) {
            double u = (double)n / (double)step;
            double x = s0[0] * (1.0 - u) + s1[0] * u - (l0[0] * (1.0 - u) + l1[0] * u);
            double y = s0[1] * (1.0 - u) + s1[1] * u - (l0[1] * (1.0 - u) + l1[1] * u);
            double z = s0[2] * (1.0 - u) + s1[2] * u - (l0[2] * (1.0 - u) + l1[2] * u);
            double tau = sqrt(x * x + y * y + z * z) / speed + lead;
            base[n] = start + (double)n - tau * rate;
        }
        const float* first = (const float*)(size_t)ya[p];
        const float* second = (const float*)(size_t)yb[p];
        if (first && rv_read(first, la[p], wa[p], base, step, table, a)) return 1;
        if (second && rv_read(second, lb[p], wb[p], base, step, table, b)) return 1;
        for (long long n = 0; n < step; n++) {
            double at = (double)n / (double)step;
            float u = (float)at;
            float less = (float)(1.0 - at);
            float v = (float)((double)(n % within) / (double)within);
            signal[n] = (first ? a[n] * less : 0.0f) + (second ? b[n] * u : 0.0f);
            ramped[n] = signal[n] * v;
        }
        const float* h = harmonics + p * (nodes + 1) * channels;
        for (long long node = 0; node < nodes; node++) {
            const float* h0 = h + node * channels;
            const float* h1 = h0 + channels;
            const float* s = signal + node * within;
            const float* r = ramped + node * within;
            for (long long c = 0; c < channels; c++) {
                float* restrict o = out + c * step + node * within;
                float from = h0[c];
                float change = h1[c] - h0[c];
                for (long long n = 0; n < within; n++) o[n] += s[n] * from + r[n] * change;
            }
        }
    }
    return 0;
}

/* ``noise.carrier``: Threefry 2x32 of (index, stream) under the key, twenty rounds, and
   the sum of the two words' eight bytes, centred and scaled. Whole numbers: exact. */
void rv_carrier(
    unsigned int key0, unsigned int key1, const unsigned int* streams, long long count_streams,
    unsigned int start, long long count, double mean, double scale, float* out)
{
    static const int turns[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
    unsigned int ks[3] = {key0, key1, key0 ^ key1 ^ 0x1BD11BDAu};
    for (long long s = 0; s < count_streams; s++) {
        float* o = out + s * count;
        for (long long i = 0; i < count; i++) {
            unsigned int x0 = start + (unsigned int)i + ks[0];
            unsigned int x1 = streams[s] + ks[1];
            for (int group = 0; group < 5; group++) {
                for (int r = 0; r < 4; r++) {
                    int bits = turns[group % 2][r];
                    x0 += x1;
                    x1 = ((x1 << bits) | (x1 >> (32 - bits))) ^ x0;
                }
                x0 += ks[(group + 1) % 3];
                x1 += ks[(group + 2) % 3] + (unsigned int)(group + 1);
            }
            unsigned int total = 0;
            for (int shift = 0; shift < 32; shift += 8)
                total += ((x0 >> shift) & 0xFFu) + ((x1 >> shift) & 0xFFu);
            o[i] = (float)(((double)total - mean) * scale);
        }
    }
}

/* out[c, factor q + r] = sum_j low[c, q + j] poly[r, j], for q < count. Each phase is
   summed tap by tap along the samples, which the compiler lays on vectors, and then
   placed: ``work`` holds ``count`` values. */
void rv_upsample(
    long long channels, long long count, long long width, long long factor,
    const float* restrict low, long long low_width, const float* restrict poly,
    float* restrict out, float* restrict work)
{
    for (long long c = 0; c < channels; c++) {
        const float* x = low + c * low_width;
        float* o = out + c * count * factor;
        for (long long r = 0; r < factor; r++) {
            const float* k = poly + r * width;
            for (long long q = 0; q < count; q++) work[q] = 0.0f;
            for (long long j = 0; j < width; j++) {
                float tap = k[j];
                const float* restrict s = x + j;
                for (long long q = 0; q < count; q++) work[q] += s[q] * tap;
            }
            for (long long q = 0; q < count; q++) o[q * factor + r] = work[q];
        }
    }
}
"""

_state: dict[str, Any] = {"off": 0, "library": None, "why": None, "tried": False}
_POINTER = ctypes.c_void_p


def _build() -> ctypes.CDLL:
    compiler = _compiler()
    if compiler is None:
        raise OSError("no C compiler on this machine (cc, gcc, clang, or CC)")
    version = subprocess.run(
        [*compiler.split(), "--version"], capture_output=True, text=True, check=False
    ).stdout
    name = hashlib.sha256(
        "\n".join([SOURCE, " ".join(FLAGS), compiler, version]).encode()
    ).hexdigest()[:24]
    root = _cache_root()
    target = root / f"render_{name}.so"
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            source = Path(scratch) / "render.c"
            source.write_text(SOURCE)
            built = Path(scratch) / "render.so"
            done = subprocess.run(
                [*compiler.split(), *FLAGS, str(source), "-o", str(built), "-lm"],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0 or not built.is_file():
                raise OSError(f"{compiler} did not build the engine's loops: {done.stderr[-800:]}")
            # Whole or not at all: another process of the machine may be building it too.
            built.replace(target)
    held = ctypes.CDLL(str(target))
    held.rv_early_interval.restype = ctypes.c_int
    held.rv_early_interval.argtypes = [
        *[ctypes.c_longlong] * 4,
        *[ctypes.c_double] * 4,
        *[_POINTER] * 15,
    ]
    held.rv_upsample.restype = None
    held.rv_upsample.argtypes = [
        *[ctypes.c_longlong] * 4,
        _POINTER,
        ctypes.c_longlong,
        _POINTER,
        _POINTER,
        _POINTER,
    ]
    held.rv_carrier.restype = None
    held.rv_carrier.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        _POINTER,
        ctypes.c_longlong,
        ctypes.c_uint,
        ctypes.c_longlong,
        ctypes.c_double,
        ctypes.c_double,
        _POINTER,
    ]
    return held


def _library() -> ctypes.CDLL | None:
    if _state["off"] or os.environ.get("REVERBERATE_NO_NATIVE"):
        return None
    if not _state["tried"]:
        _state["tried"] = True
        try:
            _state["library"] = _build()
        except OSError as error:
            _state["why"] = str(error)
    found: ctypes.CDLL | None = _state["library"]
    return found


def available() -> bool:
    """Whether the engine's loops run in C in this process."""
    return _library() is not None


def why_not() -> str | None:
    """Why the text is not in use here, for a log; ``None`` when it is."""
    if os.environ.get("REVERBERATE_NO_NATIVE"):
        return "REVERBERATE_NO_NATIVE is set"
    if _state["off"]:
        return "turned off by the caller"
    _library()
    why: str | None = _state["why"]
    return why


@contextlib.contextmanager
def disabled() -> Iterator[None]:
    """The ``numpy`` twins for the length of a block: what the tests hold the text against."""
    _state["off"] += 1
    try:
        yield
    finally:
        _state["off"] -= 1


def _table() -> np.ndarray:
    found = _state.get("table")
    if found is None:
        found = np.ascontiguousarray(delay.kernel_table(), dtype=np.float32)
        _state["table"] = found
    return np.asarray(found)


def _read_twin(y: np.ndarray, origin: float, base: np.ndarray) -> np.ndarray:
    position = 2.0 * (base - origin)
    whole = np.floor(position)
    scaled = (position - whole) * delay.PHASES
    phase = np.minimum(scaled.astype(np.int64), delay.PHASES - 1)
    blend = (scaled - phase).astype(np.float32)[:, None]
    first = whole.astype(np.int64) - (delay.TAPS // 2 - 1)
    if first.min() < 0 or first.max() + delay.TAPS > y.size:
        raise ValueError("a delay line was read outside the signal it holds")
    table = _table()
    kernel = table[phase] + (table[phase + 1] - table[phase]) * blend
    total = np.zeros(base.size, dtype=np.float32)
    for tap in range(delay.TAPS):  # in the text's order, a tap at a time
        total += kernel[:, tap] * y[first + tap]
    return total


def early_interval(
    out: np.ndarray,
    *,
    start: int,
    nodes: int,
    rate: float,
    lead: float,
    speed: float,
    q0: np.ndarray,
    q1: np.ndarray,
    l0: np.ndarray,
    l1: np.ndarray,
    first: list[tuple[np.ndarray, int] | None],
    second: list[tuple[np.ndarray, int] | None],
    harmonics: np.ndarray,
) -> None:
    """Add one interval's paths to ``out``, ``[channel, step]`` float32.

    The interval starts at the output sample ``start``. Path ``p`` goes
    from the apparent source ``q0[p]`` to ``q1[p]`` while the head goes
    from ``l0`` to ``l1``; at the sample ``n`` it is heard ``tau`` after
    it left, the distance over ``speed`` plus ``lead``, so it reads the dry
    sample ``start + n - tau rate``. ``first[p]`` and ``second[p]`` are the
    path's filtered signal at the interval's two ends, each ``(y, origin)``
    with ``y`` float32 at twice the rate, its sample 0 the dry sample
    ``origin``; ``None`` where the path has no gain at that end. The two
    reads are cross-faded over the interval, which is the format's linear
    gain. ``harmonics`` is ``[path, nodes + 1, channel]`` float32: the
    listener's basis at the nodes, linear between two.
    """
    channels, step = out.shape
    paths = int(q0.shape[0])
    if paths == 0:
        return
    q0 = np.ascontiguousarray(q0, dtype=np.float64)
    q1 = np.ascontiguousarray(q1, dtype=np.float64)
    l0 = np.ascontiguousarray(l0, dtype=np.float64)
    l1 = np.ascontiguousarray(l1, dtype=np.float64)
    harmonics = np.ascontiguousarray(harmonics, dtype=np.float32)
    library = _library()
    if library is not None:
        ends = []
        for side in (first, second):
            address = np.array([0 if e is None else e[0].ctypes.data for e in side], dtype=np.int64)
            length = np.array([0 if e is None else e[0].size for e in side], dtype=np.int64)
            origin = np.array([0.0 if e is None else e[1] for e in side], dtype=np.float64)
            ends.append((address, length, origin))
        base = np.empty(step, dtype=np.float64)
        work = np.empty(4 * step, dtype=np.float32)
        failed = library.rv_early_interval(
            paths,
            step,
            nodes,
            channels,
            float(start),
            float(rate),
            float(lead),
            float(speed),
            q0.ctypes.data,
            q1.ctypes.data,
            l0.ctypes.data,
            l1.ctypes.data,
            ends[0][0].ctypes.data,
            ends[1][0].ctypes.data,
            ends[0][1].ctypes.data,
            ends[1][1].ctypes.data,
            ends[0][2].ctypes.data,
            ends[1][2].ctypes.data,
            harmonics.ctypes.data,
            _table().ctypes.data,
            out.ctypes.data,
            base.ctypes.data,
            work.ctypes.data,
        )
        if failed:
            raise ValueError("a delay line was read outside the signal it holds")
        return
    within = step // nodes
    u = np.arange(step) / step
    v = ((np.arange(step) % within) / within).astype(np.float32)
    for p in range(paths):
        seen = (
            q0[p][None, :] * (1.0 - u)[:, None]
            + q1[p][None, :] * u[:, None]
            - (l0[None, :] * (1.0 - u)[:, None] + l1[None, :] * u[:, None])
        )
        tau = np.sqrt(np.sum(seen * seen, axis=1)) / speed + lead
        base = start + np.arange(step) - tau * rate
        signal = np.zeros(step, dtype=np.float32)
        for end, weight in ((first[p], 1.0 - u), (second[p], u)):
            if end is not None:
                signal += _read_twin(end[0], end[1], base) * weight.astype(np.float32)
        pieces = signal.reshape(nodes, within)
        ramped = (signal * v).reshape(nodes, within)
        h = harmonics[p]
        made = (
            h[:-1, :, None] * pieces[:, None, :] + (h[1:] - h[:-1])[:, :, None] * ramped[:, None, :]
        )
        out += made.transpose(1, 0, 2).reshape(channels, step)


def carrier(seed: int, streams: np.ndarray, start: int, count: int) -> np.ndarray:
    """:func:`reverberate.render.noise.carrier` in float32: the same values, which are exact."""
    library = _library()
    if library is None:
        return np.asarray(noise.carrier(seed, streams, start, count), dtype=np.float32)
    if start < 0 or start + count > 1 << 32:
        raise ValueError("a stream holds 2^32 values")
    chosen = np.ascontiguousarray(np.atleast_1d(streams), dtype=np.uint32)
    out = np.empty((chosen.size, count), dtype=np.float32)
    library.rv_carrier(
        int(seed) & 0xFFFFFFFF,
        (int(seed) >> 32) & 0xFFFFFFFF,
        chosen.ctypes.data,
        int(chosen.size),
        start,
        count,
        float(noise._BYTES_MEAN),
        float(noise._BYTES_SCALE),
        out.ctypes.data,
    )
    return out


def upsample(low: np.ndarray, poly: np.ndarray, count: int) -> np.ndarray:
    """The low band at the output rate, ``[channel, count factor]`` float32.

    ``out[c, factor q + r] = sum_j low[c, q + j] poly[j, r]``.

    ``low`` is ``[channel, count + width - 1]`` and ``poly`` ``[width, factor]``:
    each output sample is a few taps of the low band about it.
    """
    low = np.ascontiguousarray(low, dtype=np.float32)
    width, factor = (int(v) for v in poly.shape)
    channels = int(low.shape[0])
    if low.shape[1] < count + width - 1:
        raise ValueError("the low band is shorter than the taps reach")
    library = _library()
    if library is None:
        taps = np.asarray(poly, dtype=np.float32)
        made = np.zeros((channels, count, factor), dtype=np.float32)
        for j in range(width):  # in the text's order, a tap at a time
            made += low[:, j : j + count, None] * taps[j][None, None, :]
        return made.reshape(channels, count * factor)
    turned = np.ascontiguousarray(np.asarray(poly, dtype=np.float32).T)
    out = np.empty((channels, count * factor), dtype=np.float32)
    work = np.empty(count, dtype=np.float32)
    library.rv_upsample(
        channels,
        count,
        width,
        factor,
        low.ctypes.data,
        int(low.shape[1]),
        turned.ctypes.data,
        out.ctypes.data,
        work.ctypes.data,
    )
    return out
