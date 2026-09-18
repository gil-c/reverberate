"""The filters between the engine's pressure and the encoder, on the card.

:mod:`reverberate.audio` takes a receiver's pressure through four steps
before it is a response: the integration that undoes the differentiated
source together with a 40 Hz high pass, the low pass that removes the
dispersive top of the grid's band, the resampling to 48 kHz, and the air
absorption the solver has no term for. Each is a small, sequential piece of
arithmetic that runs a thousand times per listening point, and each is
reproduced here as a kernel that does the same arithmetic in the same order:

- ``sosfilt`` is scipy's direct form II transposed, one thread per row,
  sections in order, samples in order; the reversed pass of the low pass
  walks the same loop backwards rather than flipping the array twice.
- ``resample`` is resampy's ``kaiser_best`` loop, one thread per output
  sample: the left wing then the right, taps ascending, the filter table
  and its differences read from resampy itself so the numbers are its own.
- ``air_absorption`` is the square root Hann overlap-add of
  :func:`reverberate.audio.apply_air_absorption`, every frame transformed
  at once; the overlap-add is a gather that sums a sample's four frames in
  ascending order, as the CPU loop does.

With ``numpy`` every function here *is* the CPU path: it calls the module it
is a twin of, so a campaign run without a card computes what the old one
did. Bit-exactness between card and CPU holds where the arithmetic is
elementwise and sequential (the two IIR passes and the resampler, each
compiled without fused multiply-add) and is measured, not claimed, where a
transform is involved (the FFT under the air absorption).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import bilinear_zpk, butter, zpk2sos

from reverberate.audio import Atmosphere, frame_for
from reverberate.compute import raw_kernel

__all__ = [
    "air_absorption",
    "lowcut_sos",
    "lowpass_sos",
    "reduce_nodes",
    "resample",
    "resampler_table",
    "sosfilt",
    "sosfiltfilt",
]


# --------------------------------------------------------------------------
# receivers from nodes
# --------------------------------------------------------------------------


def reduce_nodes(u_out: Any, out_alpha: Any, xp: Any) -> Any:
    """:func:`reverberate.audio.reduce_nodes`: the weighted sum of each receiver's nodes.

    The sum runs node by node in order, which is what ``einsum`` does for one
    receiver, so a receiver read at one node (the campaign's ``nearest``
    interpolation, one weight of one) comes back as the pressure itself.
    """
    receivers, nodes = out_alpha.shape
    if u_out.shape[0] != receivers * nodes:
        raise ValueError(
            f"{u_out.shape[0]} node rows do not match {receivers} receivers of {nodes} nodes"
        )
    stacked = u_out.reshape(receivers, nodes, -1)
    total = out_alpha[:, 0:1] * stacked[:, 0, :]
    for node in range(1, nodes):
        total = total + out_alpha[:, node : node + 1] * stacked[:, node, :]
    return xp.asarray(total, dtype=xp.float64)


# --------------------------------------------------------------------------
# the IIR filters
# --------------------------------------------------------------------------


def lowcut_sos(
    sample_rate_hz: float, fcut: float, order: int, *, differentiated: bool
) -> np.ndarray:
    """The sections of :func:`reverberate.audio.integrate_and_lowcut`, as it designs them."""
    if fcut <= 0:
        raise ValueError("the card path needs a positive low cut, as every campaign has")
    zeros, poles, gain = butter(order, fcut * 2 * np.pi, btype="high", analog=True, output="zpk")
    if differentiated:
        zeros = zeros[1:]
    digital = bilinear_zpk(zeros, poles, gain, sample_rate_hz)
    return np.asarray(zpk2sos(*digital), dtype=np.float64)


def lowpass_sos(sample_rate_hz: float, fcut: float, order: int = 8) -> np.ndarray:
    """The sections of :func:`reverberate.audio.lowpass`: half the order, applied twice."""
    if order % 2:
        raise ValueError("order must be even so the two passes make the stated order")
    return np.asarray(
        butter(order // 2, 2.0 * fcut / sample_rate_hz, btype="low", output="sos"), dtype=np.float64
    )


SOSFILT_KERNEL = r"""
extern "C" __global__ void sosfilt(
    const double* __restrict__ sos, int sections,
    const double* __restrict__ x, double* __restrict__ y,
    int rows, int samples, int reverse)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double z0[16], z1[16];
    for (int s = 0; s < sections; ++s) { z0[s] = 0.0; z1[s] = 0.0; }
    const double* xr = x + (long long)row * samples;
    double* yr = y + (long long)row * samples;
    for (int step = 0; step < samples; ++step) {
        int n = reverse ? (samples - 1 - step) : step;
        double x_cur = xr[n];
        for (int s = 0; s < sections; ++s) {
            const double* c = sos + s * 6;
            double x_new = c[0] * x_cur + z0[s];
            z0[s] = c[1] * x_cur - c[4] * x_new + z1[s];
            z1[s] = c[2] * x_cur - c[5] * x_new;
            x_cur = x_new;
        }
        yr[n] = x_cur;
    }
}
"""


def sosfilt(sos: np.ndarray, signals: Any, xp: Any, *, reverse: bool = False) -> Any:
    """``scipy.signal.sosfilt`` along the last axis, zero initial state.

    ``reverse`` runs the recursion from the last sample to the first, which
    is ``sosfilt(sos, x[..., ::-1])[..., ::-1]`` without the two copies.
    """
    sos = np.asarray(sos, dtype=np.float64)
    if sos.shape[0] > 16:
        raise ValueError("at most 16 second order sections")
    if xp is np:
        from scipy.signal import sosfilt as scipy_sosfilt

        block = np.asarray(signals, dtype=np.float64)
        if reverse:
            return np.asarray(scipy_sosfilt(sos, block[..., ::-1], axis=-1)[..., ::-1], dtype=float)
        return np.asarray(scipy_sosfilt(sos, block, axis=-1), dtype=float)
    kernel = raw_kernel(SOSFILT_KERNEL, "sosfilt")
    block = xp.ascontiguousarray(xp.asarray(signals, dtype=xp.float64))
    rows, samples = block.shape
    out = xp.empty_like(block)
    threads = 128
    kernel(
        ((rows + threads - 1) // threads,),
        (threads,),
        (
            xp.asarray(sos.reshape(-1)),
            np.int32(sos.shape[0]),
            block,
            out,
            np.int32(rows),
            np.int32(samples),
            np.int32(1 if reverse else 0),
        ),
    )
    return out


def sosfiltfilt(sos: np.ndarray, signals: Any, xp: Any) -> Any:
    """Forward then backward, the zero phase pass of :func:`reverberate.audio.lowpass`."""
    return sosfilt(sos, sosfilt(sos, signals, xp), xp, reverse=True)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def resampler_table(sample_rate_hz: float, target_hz: float) -> dict[str, Any]:
    """resampy's ``kaiser_best`` table and the scalars its loop derives from the ratio."""
    from resampy.filters import get_filter

    sample_ratio = float(target_hz) / sample_rate_hz
    interp_win, precision, _ = get_filter("kaiser_best")
    interp_win = np.asarray(interp_win, dtype=np.float64)
    if sample_ratio < 1:
        interp_win = sample_ratio * interp_win
    interp_delta = np.diff(interp_win, append=interp_win[-1])
    scale = min(1.0, sample_ratio)
    num_table = int(precision)
    return {
        "interp_win": interp_win,
        "interp_delta": interp_delta,
        "num_table": num_table,
        "scale": scale,
        "index_step": int(scale * num_table),
        "time_increment": 1.0 / sample_ratio,
        "sample_ratio": sample_ratio,
    }


RESAMPLE_KERNEL = r"""
extern "C" __global__ void resample(
    const double* __restrict__ x, int rows, int n_orig,
    double* __restrict__ y, int n_out,
    const double* __restrict__ interp_win, const double* __restrict__ interp_delta, int nwin,
    int num_table, double scale, int index_step, double time_increment)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * n_out) return;
    int row = (int)(idx / n_out);
    int t = (int)(idx % n_out);
    const double* xr = x + (long long)row * n_orig;
    double time_register = (double)t * time_increment;
    int n = (int)time_register;
    double frac = scale * (time_register - (double)n);
    double index_frac = frac * (double)num_table;
    int offset = (int)index_frac;
    double eta = index_frac - (double)offset;
    double acc = 0.0;
    int i_max = min(n + 1, (nwin - offset) / index_step);
    for (int i = 0; i < i_max; ++i) {
        int tap = offset + i * index_step;
        double weight = interp_win[tap] + eta * interp_delta[tap];
        acc += weight * xr[n - i];
    }
    frac = scale - frac;
    index_frac = frac * (double)num_table;
    offset = (int)index_frac;
    eta = index_frac - (double)offset;
    int k_max = min(n_orig - n - 1, (nwin - offset) / index_step);
    for (int k = 0; k < k_max; ++k) {
        int tap = offset + k * index_step;
        double weight = interp_win[tap] + eta * interp_delta[tap];
        acc += weight * xr[n + k + 1];
    }
    y[idx] = acc;
}
"""


def resample(signals: Any, sample_rate_hz: float, target_hz: float, xp: Any) -> Any:
    """:func:`reverberate.audio.resample_to` on the card: resampy's loop, one thread a sample."""
    if sample_rate_hz == target_hz:
        return xp.array(signals, dtype=xp.float64, copy=True)
    if xp is np:
        from reverberate.audio import resample_to

        return resample_to(np.asarray(signals, dtype=float), sample_rate_hz, target_hz)
    table = resampler_table(sample_rate_hz, target_hz)
    block = xp.ascontiguousarray(xp.asarray(signals, dtype=xp.float64))
    rows, n_orig = block.shape
    n_out = int(n_orig * float(target_hz) / float(sample_rate_hz))
    if n_out < 1:
        raise ValueError("the signal is too short to resample")
    out = xp.zeros((rows, n_out), dtype=xp.float64)
    kernel = raw_kernel(RESAMPLE_KERNEL, "resample")
    total = rows * n_out
    threads = 256
    kernel(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            block,
            np.int32(rows),
            np.int32(n_orig),
            out,
            np.int32(n_out),
            xp.asarray(table["interp_win"]),
            xp.asarray(table["interp_delta"]),
            np.int32(table["interp_win"].size),
            np.int32(table["num_table"]),
            np.float64(table["scale"]),
            np.int32(table["index_step"]),
            np.float64(table["time_increment"]),
        ),
    )
    return out


# --------------------------------------------------------------------------
# air absorption
# --------------------------------------------------------------------------

OVERLAP_KERNEL = r"""
extern "C" __global__ void overlap_add(
    const double* __restrict__ frames, int rows, int nframes, int frame, int hop,
    const double* __restrict__ window,
    double* __restrict__ out, double* __restrict__ overlap, int length)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * length) return;
    int row = (int)(idx / length);
    int p = (int)(idx % length);
    int f_hi = p / hop;
    if (f_hi > nframes - 1) f_hi = nframes - 1;
    int f_lo = (p - frame) / hop + 1;
    if (p - frame < 0) f_lo = 0;
    double acc = 0.0, wsum = 0.0;
    for (int f = f_lo; f <= f_hi; ++f) {
        int i = p - f * hop;
        if (i < 0 || i >= frame) continue;
        double w = window[i];
        acc += frames[((long long)row * nframes + f) * frame + i] * w;
        if (row == 0) wsum += w * w;
    }
    out[idx] = acc;
    if (row == 0) overlap[p] = wsum;
}
"""


def air_absorption(
    signals: Any,
    sample_rate_hz: float,
    xp: Any,
    *,
    sound_speed_m_s: float,
    atmosphere: Atmosphere | None = None,
    start_time_s: float = 0.0,
    frame: int | None = None,
) -> Any:
    """:func:`reverberate.audio.apply_air_absorption`, every frame transformed at once."""
    if xp is np:
        from reverberate.audio import apply_air_absorption

        return apply_air_absorption(
            np.asarray(signals, dtype=float),
            sample_rate_hz,
            sound_speed_m_s=sound_speed_m_s,
            atmosphere=atmosphere,
            start_time_s=start_time_s,
            frame=frame,
        )
    block = xp.ascontiguousarray(xp.asarray(signals, dtype=xp.float64))
    rows, length = block.shape
    if frame is None:
        frame = frame_for(length / sample_rate_hz + start_time_s)
    if frame < 8 or frame % 4:
        raise ValueError("frame must be a multiple of 4 and at least 8 samples")
    hop = frame // 4
    window_np = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame) / frame))
    padded_length = length + 2 * frame
    padded = xp.zeros((rows, padded_length), dtype=xp.float64)
    padded[:, frame : frame + length] = block
    starts = np.arange(0, padded_length - frame + 1, hop)
    nframes = int(starts.size)
    # The gains, frame by frame, from the CPU's own arithmetic.
    frequency = np.fft.rfftfreq(frame, 1.0 / sample_rate_hz)
    attenuation = (atmosphere or Atmosphere()).attenuation_np_per_m(frequency)
    centres = (starts + frame / 2.0 - frame) / sample_rate_hz + start_time_s
    gains = np.exp(-attenuation[None, :] * sound_speed_m_s * np.maximum(centres, 0.0)[:, None])
    window = xp.asarray(window_np)
    index = xp.asarray(starts)[:, None] + xp.arange(frame)[None, :]
    frames = padded[:, index] * window[None, None, :]
    spectrum = xp.fft.rfft(frames, axis=-1)
    spectrum *= xp.asarray(gains)[None, :, :]
    frames = xp.ascontiguousarray(xp.fft.irfft(spectrum, n=frame, axis=-1))
    del spectrum
    out = xp.empty((rows, padded_length), dtype=xp.float64)
    overlap = xp.zeros(padded_length, dtype=xp.float64)
    kernel = raw_kernel(OVERLAP_KERNEL, "overlap_add")
    total = rows * padded_length
    threads = 256
    kernel(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            frames,
            np.int32(rows),
            np.int32(nframes),
            np.int32(frame),
            np.int32(hop),
            window,
            out,
            overlap,
            np.int32(padded_length),
        ),
    )
    interior = slice(frame, frame + length)
    if float(overlap[interior].min()) <= 0.0:
        raise ValueError("the window and hop do not cover every sample")
    return out[:, interior] / overlap[interior][None, :]
