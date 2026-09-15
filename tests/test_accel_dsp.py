"""The kernels' recursions, pinned to the CPU functions they stand in for.

A kernel cannot run here, so what is tested is the arithmetic each kernel
was written from: a Python loop with the same operations in the same order,
compared with scipy, resampy and :mod:`reverberate.audio` on random
signals. Equality is asked for where the operations are elementwise and
sequential; the card is then held to the same loop by the tests that run
on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate import audio
from reverberate.accel import dsp

RATE = 72818.9528707636


def sosfilt_reference(sos: np.ndarray, x: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    """The kernel's recursion, in Python."""
    sos = np.asarray(sos, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    for row in range(x.shape[0]):
        z0 = [0.0] * sos.shape[0]
        z1 = [0.0] * sos.shape[0]
        order = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])
        for n in order:
            x_cur = float(x[row, n])
            for s in range(sos.shape[0]):
                c = sos[s]
                x_new = c[0] * x_cur + z0[s]
                z0[s] = c[1] * x_cur - c[4] * x_new + z1[s]
                z1[s] = c[2] * x_cur - c[5] * x_new
                x_cur = x_new
            y[row, n] = x_cur
    return y


def resample_reference(signals: np.ndarray, sample_rate_hz: float, target_hz: float) -> np.ndarray:
    """The kernel's loop, in Python."""
    table = dsp.resampler_table(sample_rate_hz, target_hz)
    win, delta = table["interp_win"], table["interp_delta"]
    nwin = win.size
    num_table, scale, step, increment = (
        table["num_table"],
        table["scale"],
        table["index_step"],
        table["time_increment"],
    )
    x = np.asarray(signals, dtype=np.float64)
    n_orig = x.shape[1]
    n_out = int(n_orig * float(target_hz) / float(sample_rate_hz))
    y = np.zeros((x.shape[0], n_out))
    for row in range(x.shape[0]):
        for t in range(n_out):
            time_register = t * increment
            n = int(time_register)
            frac = scale * (time_register - n)
            index_frac = frac * num_table
            offset = int(index_frac)
            eta = index_frac - offset
            acc = 0.0
            for i in range(min(n + 1, (nwin - offset) // step)):
                weight = win[offset + i * step] + eta * delta[offset + i * step]
                acc += weight * x[row, n - i]
            frac = scale - frac
            index_frac = frac * num_table
            offset = int(index_frac)
            eta = index_frac - offset
            for k in range(min(n_orig - n - 1, (nwin - offset) // step)):
                weight = win[offset + k * step] + eta * delta[offset + k * step]
                acc += weight * x[row, n + k + 1]
            y[row, t] = acc
    return y


def overlap_add_reference(
    frames: np.ndarray, window: np.ndarray, hop: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """The gather kernel's sums in Python: each sample's frames ascending, window squared beside."""
    rows, nframes, frame = frames.shape
    out = np.zeros((rows, length))
    overlap = np.zeros(length)
    for p in range(length):
        f_hi = min(p // hop, nframes - 1)
        f_lo = 0 if p - frame < 0 else (p - frame) // hop + 1
        for f in range(f_lo, f_hi + 1):
            i = p - f * hop
            if i < 0 or i >= frame:
                continue
            out[:, p] += frames[:, f, i] * window[i]
            overlap[p] += window[i] * window[i]
    return out, overlap


def signals(rows: int = 3, samples: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, samples)) * np.exp(-np.arange(samples) / 150.0)


class TestIIR:
    def test_the_lowcut_sections_reproduce_integrate_and_lowcut(self) -> None:
        x = signals()
        sos = dsp.lowcut_sos(RATE, 40.0, 8, differentiated=True)
        expected = audio.integrate_and_lowcut(
            x, 1.0 / RATE, differentiated=True, fcut=40.0, order=8
        )
        assert np.array_equal(dsp.sosfilt(sos, x, np), expected)

    def test_the_lowpass_sections_reproduce_lowpass(self) -> None:
        x = signals()
        sos = dsp.lowpass_sos(RATE, 4000.0)
        assert np.array_equal(dsp.sosfiltfilt(sos, x, np), audio.lowpass(x, RATE, 4000.0))

    def test_the_kernel_s_recursion_is_scipy_s(self) -> None:
        from scipy.signal import sosfilt

        x = signals(rows=2, samples=300)
        sos = dsp.lowcut_sos(RATE, 40.0, 8, differentiated=True)
        forward = sosfilt_reference(sos, x)
        backward = sosfilt_reference(sos, x, reverse=True)
        expected_forward = sosfilt(sos, x, axis=-1)
        expected_backward = sosfilt(sos, x[..., ::-1], axis=-1)[..., ::-1]
        # Exact on x86 builds of scipy; a build that fuses multiply-adds can
        # differ in the last bit, which is what the tolerance below allows and
        # what the report on the card measures.
        assert np.allclose(forward, expected_forward, rtol=0, atol=1e-13 * np.abs(x).max())
        assert np.allclose(backward, expected_backward, rtol=0, atol=1e-13 * np.abs(x).max())

    def test_too_many_sections_are_refused(self) -> None:
        with pytest.raises(ValueError):
            dsp.sosfilt(np.zeros((17, 6)), signals(), np)


class TestResampling:
    @pytest.mark.parametrize("rate", [RATE, 18204.7382176909])
    def test_the_kernel_s_loop_is_resampy_s(self, rate: float) -> None:
        x = signals(rows=2, samples=250)
        expected = audio.resample_to(x, rate, 48000.0)
        got = resample_reference(x, rate, 48000.0)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)

    def test_the_numpy_path_is_the_cpu_path(self) -> None:
        x = signals()
        assert np.array_equal(
            dsp.resample(x, RATE, 48000.0, np), audio.resample_to(x, RATE, 48000.0)
        )

    def test_the_table_scales_with_the_ratio_only_when_decimating(self) -> None:
        down = dsp.resampler_table(RATE, 48000.0)
        up = dsp.resampler_table(18204.7, 48000.0)
        assert down["scale"] < 1.0 and up["scale"] == 1.0
        assert down["index_step"] == int(down["scale"] * down["num_table"])
        assert up["index_step"] == up["num_table"]


class TestAirAbsorption:
    def test_the_gather_sums_each_sample_s_frames_in_the_cpu_s_order(self) -> None:
        rng = np.random.default_rng(3)
        frame, hop, length = 16, 4, 64
        window = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame) / frame))
        starts = np.arange(0, length - frame + 1, hop)
        frames = rng.standard_normal((2, starts.size, frame))
        out = np.zeros((2, length))
        overlap = np.zeros(length)
        for f, start in enumerate(starts):
            out[:, start : start + frame] += frames[:, f] * window
            overlap[start : start + frame] += window * window
        got, got_overlap = overlap_add_reference(frames, window, hop, length)
        assert np.array_equal(got, out)
        assert np.array_equal(got_overlap, overlap)

    def test_the_numpy_path_is_the_cpu_path(self) -> None:
        x = signals(samples=2000)
        expected = audio.apply_air_absorption(x, 48000.0, sound_speed_m_s=343.2)
        assert np.array_equal(dsp.air_absorption(x, 48000.0, np, sound_speed_m_s=343.2), expected)


class TestReduce:
    def test_one_node_per_receiver_is_the_pressure_itself(self) -> None:
        u = signals(rows=4)
        assert np.array_equal(dsp.reduce_nodes(u, np.ones((4, 1)), np), u)

    def test_eight_nodes_match_audio(self) -> None:
        rng = np.random.default_rng(5)
        u = rng.standard_normal((16, 50))
        alpha = rng.random((2, 8))
        alpha /= alpha.sum(axis=1, keepdims=True)
        expected = audio.reduce_nodes(u, alpha)
        assert np.allclose(dsp.reduce_nodes(u, alpha, np), expected, rtol=0, atol=1e-15)

    def test_a_row_count_that_does_not_divide_is_refused(self) -> None:
        with pytest.raises(ValueError):
            dsp.reduce_nodes(signals(rows=3), np.ones((2, 2)), np)
