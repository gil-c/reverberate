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
recorded. And air absorption is **not** applied by the four steps above: the
solver has no viscosity term. It is a fifth step, :func:`apply_air_absorption`,
which a caller applies explicitly and declares, because it carries two
parameters -- temperature and relative humidity -- that a response cannot be
read without. A response that has not been through it overstates the treble of
a long tail by 38 per cent of T60 at 16 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import bilinear_zpk, butter, fftconvolve, sosfilt, zpk2sos

from reverberate.experiments.engine import sim_consts

__all__ = [
    "Atmosphere",
    "DECIBELS_PER_NEPER",
    "Reduced",
    "air_absorption_np_per_m",
    "apply_air_absorption",
    "frame_for",
    "peak_gain",
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


def peak_gain(*blocks: np.ndarray, headroom_db: float = 1.0) -> float:
    """One gain for every block that has to stay comparable, from their joint peak.

    **The defect this exists to stop has already happened.** Roadmap W30
    records that ``render_audio`` wrote one WAV per receiver and scaled each by
    its own peak, so all six came back at 0.8913 and the distance between them
    was inaudible; the module's own rule, one gain because the level between two
    receivers is a measurement, was being applied per file in a run that writes
    one file per receiver. That fix is described in the roadmap as done, and it
    lived in a script that was never committed, so this function is the one the
    roadmap already refers to.

    It applies to more than receivers. Two ears of one head differ by their
    interaural level difference, which is the cue the dataset exists to carry;
    the channels of one ambisonic file differ by nothing but the direction. A
    per file normalisation destroys both.
    """
    peaks = [float(np.max(np.abs(np.asarray(block)))) for block in blocks if np.size(block)]
    peak = max(peaks) if peaks else 0.0
    return 1.0 if peak == 0.0 else 10.0 ** (-headroom_db / 20.0) / peak


def write_wav(
    path: Path,
    signals: np.ndarray,
    sample_rate_hz: float,
    *,
    headroom_db: float = 1.0,
    gain: float | None = None,
) -> float:
    """Write a WAV and return the single gain applied to every channel.

    One gain for the whole file, returned rather than swallowed, so the
    relative level between channels survives and the absolute one is recorded
    instead of being quietly invented.

    ``gain`` overrides that with one computed elsewhere, which is what a caller
    writing several files that must stay comparable passes: see
    :func:`peak_gain`. Left at ``None`` the file is scaled on its own peak,
    which is right for a file that stands alone and wrong for one of a set.
    """
    import soundfile

    block = np.atleast_2d(signals)
    gain = peak_gain(block, headroom_db=headroom_db) if gain is None else gain
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        str(path), (block * gain).T, int(round(sample_rate_hz)), subtype="FLOAT", format="WAV"
    )
    return gain


#: Decibels per neper, ``20 / ln 10``. Written out because rounding it to 8.686
#: is the whole of the fifth-digit disagreement between two correct
#: implementations of ISO 9613-1.
DECIBELS_PER_NEPER = 8.685889638065035


@dataclass(frozen=True)
class Atmosphere:
    """The air a response was computed in, as an object that can be written down.

    **The roadmap requires this of every run and a triple of loose floats does
    not satisfy it.** At 16 kHz the absorption coefficient runs from 0.252 dB/m
    at 80 per cent relative humidity to 0.466 at 30, a factor of 1.85, so a run
    that has not recorded its humidity has not recorded its own decay.
    """

    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    pressure_kpa: float = 101.325

    def __post_init__(self) -> None:
        if not 0.0 <= self.humidity_percent <= 100.0:
            raise ValueError(
                f"relative humidity must be a percentage in [0, 100], got {self.humidity_percent}"
            )
        if self.pressure_kpa <= 0.0:
            raise ValueError(f"pressure must be positive, got {self.pressure_kpa} kPa")
        if self.temperature_c <= -273.15:
            raise ValueError(f"temperature must be above absolute zero, got {self.temperature_c} C")

    def attenuation_np_per_m(self, frequency_hz: np.ndarray) -> np.ndarray:
        """This atmosphere's own absorption coefficient."""
        return air_absorption_np_per_m(
            frequency_hz,
            temperature_c=self.temperature_c,
            humidity_percent=self.humidity_percent,
            pressure_kpa=self.pressure_kpa,
        )

    def attenuation_db_per_m(self, frequency_hz: float | np.ndarray) -> np.ndarray:
        """The same coefficient in decibels per metre, the unit the standard tabulates.

        Scalar or array in, array out, so a caller can broadcast it against a
        band axis without special-casing one frequency.
        """
        freq = np.asarray(frequency_hz, dtype=float)
        return np.asarray(self.attenuation_np_per_m(freq) * DECIBELS_PER_NEPER, dtype=float)

    def record(self) -> dict[str, float | str]:
        """What a ``report.json`` has to carry for the response to be readable."""
        return {
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "pressure_kpa": self.pressure_kpa,
            "standard": "ISO 9613-1",
            "note": (
                "at 16 kHz the coefficient varies by a factor of 1.85 across "
                "ordinary indoor humidity, so these are part of the result"
            ),
        }


def air_absorption_np_per_m(
    frequency_hz: np.ndarray,
    *,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """Atmospheric absorption in nepers per metre, ISO 9613-1.

    The pure tone attenuation coefficient of still air, from the relaxation
    frequencies of oxygen and nitrogen plus the classical term. Returned in
    nepers rather than decibels because what uses it is an exponential gain,
    and converting in the caller is where a factor of 8.686 goes missing.

    **Humidity is not a detail.** At 16 kHz the coefficient runs from 0.25 dB/m
    at 80 per cent relative humidity to 0.47 dB/m at 30, a factor of 1.9, so
    every response that has been through this filter has to declare the two
    parameters beside the number.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    temperature = temperature_c + 273.15
    reference_temperature = 293.15
    triple_point = 273.16
    pressure = pressure_kpa / 101.325
    ratio = temperature / reference_temperature

    saturation = 10.0 ** (-6.8346 * (triple_point / temperature) ** 1.261 + 4.6151)
    molar_water = humidity_percent * saturation / pressure

    oxygen = pressure * (24.0 + 4.04e4 * molar_water * (0.02 + molar_water) / (0.391 + molar_water))
    nitrogen = (
        pressure
        * ratio**-0.5
        * (9.0 + 280.0 * molar_water * np.exp(-4.170 * (ratio ** (-1.0 / 3.0) - 1.0)))
    )

    classical = 1.84e-11 / pressure * np.sqrt(ratio)
    relaxation = ratio**-2.5 * (
        0.01275 * np.exp(-2239.1 / temperature) / (oxygen + frequency**2 / oxygen)
        + 0.1068 * np.exp(-3352.0 / temperature) / (nitrogen + frequency**2 / nitrogen)
    )
    # ISO 9613-1 writes the coefficient as ``8.686 f^2 (...)`` in dB/m, and that
    # 8.686 is the decibel conversion, so the neper form is the bracket itself.
    # Multiplying by it here and dividing again in the caller is where a factor
    # goes missing, and where two independent implementations disagree in the
    # fifth digit purely on which rounding of ``20 / ln 10`` each chose.
    return np.asarray(frequency**2 * (classical + relaxation), dtype=float)


def frame_for(duration_s: float) -> int:
    """The short time frame a response of this length needs, from measurement.

    The thresholds are the table in :func:`apply_air_absorption`, read at the
    0.05 dB line on its per bin probe: the shortest frame whose worst error
    stays under it. Longer responses need longer frames because the gain grows
    steeper in frequency with time, not because they hold more samples.

    Deliberately conservative. On a real decaying response even the shortest
    frame here moves octave band energy by 0.13 dB at most, so this is chosen
    for the probes and synthetic signals that do reach the leakage floor, and it
    costs a room response nothing.
    """
    if duration_s <= 0.1:
        return 256
    if duration_s <= 0.6:
        return 512
    if duration_s <= 1.2:
        return 1024
    return 2048


def apply_air_absorption(
    signals: np.ndarray,
    sample_rate_hz: float,
    *,
    sound_speed_m_s: float = 343.0,
    atmosphere: Atmosphere | None = None,
    start_time_s: float = 0.0,
    frame: int | None = None,
) -> np.ndarray:
    """Apply atmospheric absorption to an impulse response, in place of a solver term.

    ``atmosphere`` is the air the response is claimed to have crossed; ``None``
    is :class:`Atmosphere`'s own default, 20 C and 50 per cent, which a run
    still has to write down through :meth:`Atmosphere.record`.

    **This is exact rather than a concession, and the reason is the geometry of
    an impulse response.** Every sample arriving at time ``t`` has travelled
    exactly ``c t`` of path, whatever route it took around the room, so air
    absorption is precisely a per sample frequency dependent gain
    ``exp(-m(f) c t)``. It is a time varying filter, not a diffuse field
    approximation, it applies retroactively to responses already computed, and
    it needs no change to the pinned solver, whose kernel would otherwise carry
    a Stokes term in its inner loop.

    It matters most exactly where this project is most expensive. On W29's own
    measured decay the correction is -0.7 per cent of T60 at 1 kHz and
    **-37.7 per cent at 16 kHz**.

    Implemented as a short time Fourier transform with a square root Hann
    window at three quarters overlap. **The exact reconstruction comes from
    dividing by the summed window product, not from the choice of window**: any
    analysis and synthesis pair reconstructs under that normalisation, and a
    plain Hann does as well as a square root one. The square root pair is kept
    because it distributes the modification symmetrically between analysis and
    synthesis, which is the usual reason for it, and not because it is uniquely
    exact.

    **The frame length is chosen from the response, and it is not a detail.**
    Two errors pull against each other. A long frame freezes a gain that is
    changing with time. A short frame resolves frequency coarsely, and the gain
    ``exp(-m(f) c t)`` becomes a very steep function of frequency as ``t``
    grows: at 1 s it falls 125 dB between 50 Hz and 16 kHz, so a few bins of
    leakage from the loud bottom swamps the quiet top. Measured, worst error in
    dB over every bin whose own gain is above -120 dB:

    | frame | 5 ms | 50 ms | 0.5 s | 1 s | 2 s |
    | --- | --- | --- | --- | --- | --- |
    | 128 | 0.0006 | 0.008 | 0.79 | -- | -- |
    | 256 | 0.003 | 0.003 | 0.035 | 5.7 | 47 |
    | 512 | 0.003 | 0.004 | 0.011 | 0.30 | 1.1 |
    | 1024 | 0.015 | 0.041 | 0.003 | 0.095 | 0.018 |

    **What the frame buys is a dynamic range, not a length of time.** The error
    appears where the signal at a frequency has fallen below what leaks into
    that bin from the loud part of the spectrum, so what decides it is the ratio
    between the two. Duration only decides how long the signal takes to fall
    that far, which is why it works as a rule of thumb and fails as an
    explanation.

    **The quantity is where the curve leaves the analytic line, and not any
    floor it settles on**, because it settles on nothing: on a 16 kHz tone
    filtered here, once the tone is dead the block levels of the last half
    second span 43 dB at a 128 sample frame and 104 dB at 2048. That residue is
    numerical, any single number drawn from it depends on how it is averaged,
    and two implementations comparing tail averages will disagree by tens of dB
    while both filters are right. Two sessions of this project did exactly that.

    So: a 16 kHz tone falls 125 dB per second to air alone, and each frame
    follows that line until it departs from it by more than 6 dB, at

    | frame | departs below the tone's start at |
    | --- | --- |
    | 128 | -86 dB |
    | 256 | -100 dB |
    | 512 | -115 dB |
    | 1024 | -129 dB |
    | 2048 | -145 dB |

    A doubling buys of the order of 14 dB here.

    **Those levels belong to the square root Hann pair specifically, and the
    choice of pair is worth 50 to 70 dB.** Another implementation of this filter
    measured levels far lower on this same definition, and the whole difference
    is the window scheme rather than the statistic. Measured here, departure
    level with a Hann pair against the square root pair, same gain surface and
    same overlap:

    | frame | square root Hann | Hann | difference |
    | --- | --- | --- | --- |
    | 128 | -86 dB | -136 dB | 50 dB |
    | 256 | -100 dB | -151 dB | 51 dB |
    | 512 | -115 dB | -171 dB | 56 dB |
    | 1024 | -129 dB | -199 dB | 70 dB |

    A Hann pair is a squared window, whose spectrum falls away far faster, so it
    keeps leakage out of a weak bin much longer, and the gap widens with the
    frame. **Neither is wrong and both are exact wherever there is signal.** The
    square root pair is kept because it splits the modification evenly between
    analysis and synthesis, which the other does not, and because the margin
    below is already enormous. Anyone who ever needs another 50 dB of it should
    change the pair rather than the frame.

    **The margin a caller actually needs is enormous.** The default frame holds
    the line to about -100 dB, and the top band of a measured room response
    falls of the order of 30 dB. That is the mechanical reason an audit of real
    responses moves T30 by 0.37 per cent.

    A delta and a pure tone reach the floor quickly, because they leave nothing
    at the top of the band; a room response does not, because its tail is
    broadband. Measured on a decaying response with a direct sound, octave band
    energy at a 256 sample frame against a 4096 one:

    | duration | 4 kHz | 8 kHz | 16 kHz |
    | --- | --- | --- | --- |
    | 0.5 s | 0.019 | 0.064 | 0.128 |
    | 1.5 s | 0.017 | 0.066 | 0.107 |
    | 2.0 s | 0.020 | 0.059 | 0.096 |

    In dB, and it does not grow with duration. So the per bin table is the worst
    case of a probe rather than of a room, and the choice below is conservative
    on purpose: a longer frame costs a little arithmetic and protects the
    synthetic signals this project also filters, which are exactly the ones that
    empty their top of band and reach the floor.

    ``None``, the default, takes the frame from the length of the signal it is
    given. A caller who passes one explicitly is trusted and not corrected.

    ``start_time_s`` is the propagation time of the response's first sample.
    Zero for a response the solver started at the source, which is every
    response this project writes.
    """
    block = np.atleast_2d(np.asarray(signals, dtype=float))
    if frame is None:
        frame = frame_for(block.shape[1] / sample_rate_hz + start_time_s)
    if frame < 8 or frame % 4:
        raise ValueError("frame must be a multiple of 4 and at least 8 samples")
    hop = frame // 4
    window = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame) / frame))

    length = block.shape[1]
    padded = np.zeros((block.shape[0], length + 2 * frame), dtype=float)
    padded[:, frame : frame + length] = block
    out = np.zeros_like(padded)
    # The overlap sum of the analysis window times the synthesis window, summed
    # rather than quoted: the constant depends on the window and the hop, and a
    # remembered one is how a whole response acquires a quiet 2.5 dB of gain.
    overlap = np.zeros(padded.shape[1])

    frequency = np.fft.rfftfreq(frame, 1.0 / sample_rate_hz)
    attenuation = (atmosphere or Atmosphere()).attenuation_np_per_m(frequency)
    for start in range(0, padded.shape[1] - frame + 1, hop):
        centre = (start + frame / 2.0 - frame) / sample_rate_hz + start_time_s
        gain = np.exp(-attenuation * sound_speed_m_s * max(centre, 0.0))
        spectrum = np.fft.rfft(padded[:, start : start + frame] * window, axis=-1)
        out[:, start : start + frame] += np.fft.irfft(spectrum * gain, n=frame, axis=-1) * window
        overlap[start : start + frame] += window * window
    interior = slice(frame, frame + length)
    if np.min(overlap[interior]) <= 0.0:
        raise ValueError("the window and hop do not cover every sample")
    result = out[:, interior] / overlap[interior]
    return result if np.ndim(signals) > 1 else np.asarray(result[0])
