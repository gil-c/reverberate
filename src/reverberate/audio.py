"""From what the engine wrote to something a person can listen to.

The engine writes ``sim_outs.h5``: one row per **interpolation node**, at the
grid's own sample rate, in an order the comms file chose. That is four steps
away from a response, and every one of them has been got wrong somewhere in
this project's history, so each is named here.

**1. Eight nodes make one receiver.** The roadmap fixes receivers as not grid
constrained: ``interp_weights`` returns 8 nodes whose weights sum to 1.
``compare.response`` takes ``u_out[0]``, which is the first node of the first
receiver and not the receiver at all. It was right for W1's bit exactness
question, where any fixed node is as good as another, and it is wrong for
listening.

**2. A differentiated source must be integrated back.** ``write_comms`` applies
``diff_source`` for the single precision engines, which is PFFDTD's own trick
for injecting a wider band impulse. Reading the output without undoing it gives
the derivative of the response: audibly thin, and every decay metric measured on
the wrong signal. The ``diff`` flag travels in ``comms_out.h5`` and is read back
here rather than assumed.

**3. Above ``fmax`` the grid is lying.** The finite difference stencil is
dispersive near the top of its band, which is why the working point is quoted in
points per wavelength. Those frequencies are filtered out, with a symmetric
forward and reverse pass so the filter adds no group delay to a signal whose
timing is the thing being measured.

**4. Then, and only then, resample.** The grid rate is whatever the cell size
made it, 72.4 kHz at a 4 kHz working point. 48 kHz is a choice about delivery,
not about physics, and it happens last.

**What is deliberately not done here.** No per response normalisation: the
relative level between two receivers of one run is a measurement and dividing
each by its own peak would destroy it. A single gain is applied when writing a
WAV, is the same for every channel of that file, and is returned so it can be
recorded. And **air absorption is not modelled**: the solver has no viscosity
term and no Stokes filter is applied here, so the treble of a long tail is
overstated. Said out loud because the roadmap's own cost argument leans on air
absorbing treble.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import bilinear_zpk, butter, fftconvolve, sosfilt, zpk2sos

from reverberate.experiments.engine import sim_consts

__all__ = [
    "Reduced",
    "convolve",
    "integrate_and_lowcut",
    "lowpass",
    "read_engine_output",
    "reduce_nodes",
    "resample_to",
    "write_wav",
]


@dataclass(frozen=True)
class Reduced:
    """One row per receiver, at one stated sample rate."""

    signals: np.ndarray
    sample_rate_hz: float

    @property
    def receiver_count(self) -> int:
        return int(self.signals.shape[0])


def reduce_nodes(u_out: np.ndarray, out_alpha: np.ndarray) -> np.ndarray:
    """Weighted sum of each receiver's 8 interpolation nodes.

    ``u_out`` is ``[receiver * node, sample]`` already in the caller's receiver
    order, because the engine applies ``out_reorder`` on the way out.
    ``out_alpha`` is ``[receiver, node]`` and its rows sum to 1.
    """
    receivers, nodes = out_alpha.shape
    if u_out.shape[0] != receivers * nodes:
        raise ValueError(
            f"{u_out.shape[0]} node rows do not match {receivers} receivers of {nodes} nodes"
        )
    stacked = u_out.reshape(receivers, nodes, -1)
    return np.asarray(np.einsum("rn,rnt->rt", out_alpha, stacked), dtype=float)


def integrate_and_lowcut(
    signals: np.ndarray,
    ts: float,
    *,
    differentiated: bool,
    fcut: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Undo the source differentiation and remove the DC drift it leaves.

    When the source was differentiated, integrating alone would let numerical
    offset walk away, so the integrator and the high pass are designed as one
    analogue prototype and bilinear transformed together: a Butterworth high
    pass has ``order`` zeros at the origin, and dropping one of them is exactly
    an integration. That is PFFDTD's own construction, reproduced rather than
    imported because importing it would drag the numpy 1.26 pin into this
    interpreter.
    """
    if fcut <= 0:
        if not differentiated:
            return np.array(signals, dtype=float, copy=True)
        # Trapezoidal integrator. PFFDTD's own no-lowcut branch writes
        # ``a = [1, 1]``, which is a one pole low pass and not an integrator at
        # all; its own comment says the branch "shouldn't really use this". The
        # denominator here is ``[1, -1]``, which integrates. The branch is
        # reachable only when a caller asks for no high pass, and a caller who
        # does should know it drifts.
        taps = ts / 2 * np.array([1.0, 1.0])
        from scipy.signal import lfilter

        return np.asarray(lfilter(taps, np.array([1.0, -1.0]), signals, axis=-1), dtype=float)

    zeros, poles, gain = butter(order, fcut * 2 * np.pi, btype="high", analog=True, output="zpk")
    if differentiated:
        zeros = zeros[1:]
    digital = bilinear_zpk(zeros, poles, gain, 1.0 / ts)
    sos = zpk2sos(*digital)
    return np.asarray(sosfilt(sos, signals, axis=-1), dtype=float)


def lowpass(signals: np.ndarray, sample_rate_hz: float, fcut: float, order: int = 8) -> np.ndarray:
    """Remove the dispersive top of the band, without adding group delay.

    Run forwards then backwards, so the phase response cancels exactly. The
    order is halved first, because two passes of an order n filter make an
    order 2n magnitude response.
    """
    if order % 2:
        raise ValueError("order must be even so the two passes make the stated order")
    sos = butter(order // 2, 2.0 * fcut / sample_rate_hz, btype="low", output="sos")
    forward = sosfilt(sos, signals, axis=-1)
    return np.asarray(sosfilt(sos, forward[..., ::-1], axis=-1)[..., ::-1], dtype=float)


def resample_to(signals: np.ndarray, sample_rate_hz: float, target_hz: float) -> np.ndarray:
    """Resample to ``target_hz``. Delivery, not physics, so it happens last."""
    if sample_rate_hz == target_hz:
        return np.array(signals, dtype=float, copy=True)
    import resampy

    return np.asarray(
        resampy.resample(signals, sample_rate_hz, target_hz, filter="kaiser_best", axis=-1),
        dtype=float,
    )


def read_engine_output(run_dir: Path, comms_path: Path | None = None) -> tuple[Reduced, bool]:
    """Read ``sim_outs.h5`` and its comms file into one row per receiver.

    Returns the reduced signals at the grid rate and whether the source was
    differentiated, which the caller needs for :func:`integrate_and_lowcut` and
    must not guess.
    """
    run_dir = Path(run_dir)
    comms = Path(comms_path) if comms_path is not None else run_dir / "comms_out.h5"
    with h5py.File(run_dir / "sim_outs.h5", "r") as handle:
        u_out = np.asarray(handle["u_out"], dtype=np.float64)
    with h5py.File(comms, "r") as handle:
        out_alpha = np.asarray(handle["out_alpha"], dtype=np.float64)
        differentiated = bool(np.asarray(handle["diff"]).item())
    return Reduced(reduce_nodes(u_out, out_alpha), sim_consts(run_dir).sample_rate), differentiated


def convolve(dry: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve one dry signal with one impulse response, full length.

    Full rather than truncated to the dry length: the tail is the part the
    roadmap says makes separation hard, and cutting it here would hide exactly
    what the listen is for.
    """
    if dry.ndim != 1 or ir.ndim != 1:
        raise ValueError("convolve takes one dry signal and one response")
    return np.asarray(fftconvolve(dry, ir, mode="full"), dtype=float)


def write_wav(
    path: Path, signals: np.ndarray, sample_rate_hz: float, *, headroom_db: float = 1.0
) -> float:
    """Write a WAV and return the single gain applied to every channel.

    One gain for the whole file, returned rather than swallowed, so the
    relative level between channels survives and the absolute one is recorded
    instead of being quietly invented.
    """
    import soundfile

    block = np.atleast_2d(signals)
    peak = float(np.max(np.abs(block)))
    gain = 1.0 if peak == 0.0 else 10.0 ** (-headroom_db / 20.0) / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        str(path), (block * gain).T, int(round(sample_rate_hz)), subtype="FLOAT", format="WAV"
    )
    return gain
