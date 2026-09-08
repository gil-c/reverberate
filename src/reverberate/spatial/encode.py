"""Ambisonic encoding: the array's pressures to spherical harmonic coefficients.

Inside a ball of free air the field is exactly

    p(r, omega) = sum_nm a_nm(omega) j_n(k r) Y_nm(direction)

so encoding is a linear least squares problem per frequency, with
``G[q, nm] = j_n(k r_q) Y_nm(d_q)`` over the array's own node positions. ``G``
is **real**: the geometry carries no phase, and all of it is in the pressures.
That halves the work and, more usefully, it means the fit cannot invent a phase.

**Three things make this fit an engineering problem rather than a solve.**

*The high orders are tiny at low frequency.* ``j_n(x)`` goes as ``x^n`` for
small ``x``, so at 1 kHz and 16 cm order 7 sits 63 dB below order 0. Inverting
that is amplifying the array's own noise by the same 63 dB. Tikhonov
regularisation caps that amplification at a stated figure:
``lambda = 1 / (4 g_max^2)`` makes ``sigma / (sigma^2 + lambda)`` at most
``g_max`` for every singular value, exactly and not approximately.

*The high orders are not tiny at high frequency, they are aliased.* Past
``k r = N`` the truncated expansion no longer describes the field on that
shell, and the residue folds into the orders that were kept. So each frequency
is fitted from the shells it can still describe, through a weight that tapers
to zero rather than cutting, since a cut moving between bins is a comb filter.

*The grid is not the continuum.* The solver's wave travels at a speed that
depends on frequency and on direction: at 10.5 points per wavelength the
7 point scheme is 1.0 per cent slow along the axes at 16 kHz and exact along
the diagonals. Over the 16 cm shell that is 0.5 rad of phase, far above the
solver's own noise floor, and the fit would read it as a field that is not
there. ``dispersion="numerical"`` fits with the scheme's own direction averaged
wavenumber instead of ``omega / c``. Whether it is worth it is measured on a
rehearsal box, not assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import roots_legendre, spherical_jn

from reverberate.spatial.array import ArrayDesign
from reverberate.spatial.field import monopole_coefficients
from reverberate.spatial.sh import channel_count, degrees_of, real_sh, scene_to_ambisonic

__all__ = [
    "Ambisonic",
    "ConditioningReport",
    "Dispersion",
    "EncoderSettings",
    "conditioning",
    "encode",
    "encode_spectrum",
    "numerical_wavenumber",
    "shell_weights",
    "well_posed",
]

#: Which wavenumber the radial terms are evaluated at. ``ideal`` is
#: ``omega / c``; ``numerical`` is the finite difference scheme's own, direction
#: averaged. See the module docstring.
Dispersion = Literal["ideal", "numerical"]


@dataclass(frozen=True)
class EncoderSettings:
    """Everything the fit does that is a choice rather than a consequence."""

    #: Order kept in the output.
    order: int = 7
    #: Order the least squares is solved at. Fitting above the order that is
    #: kept gives the residue of the shells' higher orders somewhere to go
    #: other than into the orders that are reported. Ten rather than nine is
    #: measured: it costs 1.46 times the solve and buys 8 dB at 8 kHz.
    fit_order: int = 10
    #: Largest amplification, in dB, from a receiver's pressure into a
    #: coefficient. Sets the Tikhonov weight exactly.
    gain_cap_db: float = 60.0
    #: How far below ``fit_order`` a shell's ``k r`` must stay to be used. The
    #: truncated fit has to *describe* the shells it is given, and a shell at
    #: ``k r = fit_order`` is exactly the one it cannot: measured on an analytic
    #: monopole, order 7 comes back at -20 dB with a margin of zero and at
    #: -40 dB or better across 1 to 16 kHz with a margin of four. It is the
    #: single most important number in this file.
    gate_margin: float = 4.0
    #: Width in ``k r`` of the raised cosine that closes the gate. A hard edge
    #: moving from bin to bin combs the spectrum.
    gate_taper: float = 2.0
    #: Bins above this are not fitted and are left at zero. They hold only the
    #: skirt of the band limiting low pass, and fitting them costs a third of
    #: the run for nothing. ``None`` fits every bin.
    max_frequency_hz: float | None = None
    dispersion: Dispersion = "numerical"

    def regularisation(self) -> float:
        return float(1.0 / (4.0 * (10.0 ** (self.gain_cap_db / 20.0)) ** 2))

    def __post_init__(self) -> None:
        # Reported here rather than deep in the solve, where it surfaces as a
        # broadcast error between 49 and 64 columns and says nothing about what
        # the caller did. Fitting below the order being reported asks for
        # coefficients the fit never solved for.
        if self.fit_order < self.order:
            raise ValueError(
                f"fit order {self.fit_order} is below the order being reported, "
                f"{self.order}; the fit would not solve for the channels asked for"
            )
        if self.order < 0:
            raise ValueError(f"order must be non negative, not {self.order}")

    @property
    def gate_kr(self) -> float:
        """The ``k r`` at which a shell starts being dropped.

        Deliberately allowed to fall below the order being reported. That is the
        whole premise of this design: an order is recoverable from a shell where
        ``k r`` is well under it, because the limit here is dynamic range and
        not a microphone's noise. Measured, order 7 comes back at -48 dB from a
        gate of 6, and a floor at the output order costs 22 dB at 8 kHz.

        What it must not do is close on everything, which is a condition on the
        number of receivers admitted rather than on the order.
        :func:`well_posed` is where that is checked.
        """
        return float(self.fit_order) - float(self.gate_margin)

    def record(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "fit_order": self.fit_order,
            "gain_cap_db": self.gain_cap_db,
            "gate_margin": self.gate_margin,
            "gate_kr": self.gate_kr,
            "gate_taper": self.gate_taper,
            "max_frequency_hz": self.max_frequency_hz,
            "dispersion": self.dispersion,
            "regularisation": self.regularisation(),
        }


@dataclass(frozen=True)
class Ambisonic:
    """One encoded response: ``[(order + 1)^2, sample]`` in ACN order and N3D.

    The signals are the *plane wave* coefficients ``b_nm``, which is what an
    ambisonic decoder expects: the first channel of a unit plane wave is 1 at
    every frequency. The interior coefficients ``a_nm`` the fit returns differ
    by ``i^n``, which is applied here so that nothing downstream has to
    remember it.
    """

    signals: np.ndarray
    sample_rate_hz: float
    order: int
    centre: np.ndarray
    normalisation: str = "N3D"
    ordering: str = "ACN"

    def __post_init__(self) -> None:
        if self.signals.ndim != 2:
            raise ValueError(f"signals must be [channel, sample], got {self.signals.shape}")
        if self.signals.shape[0] != channel_count(self.order):
            raise ValueError(f"{self.signals.shape[0]} channels do not match order {self.order}")

    @property
    def duration_s(self) -> float:
        return float(self.signals.shape[1] / self.sample_rate_hz)


def numerical_wavenumber(
    frequency_hz: np.ndarray,
    grid_step_m: float,
    sound_speed_m_s: float,
    *,
    courant: float = 1.0 / np.sqrt(3.0),
    directions: int = 24,
) -> np.ndarray:
    """The wavenumber the finite difference scheme actually propagates, averaged over direction.

    The 7 point scheme's dispersion relation is

        sin(omega Ts / 2)^2 = lambda^2 sum_axes sin(k_axis h / 2)^2

    with ``lambda`` the Courant number and ``Ts = lambda h / c``. Solved here
    for ``|k|`` along each of a set of directions and averaged, which is the
    isotropic wavenumber a spherical expansion can use. Above the scheme's
    cutoff, where the right hand side cannot reach the left, the ideal value is
    returned and the caller is expected to be band limiting well below it.
    """
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    ts = courant * grid_step_m / sound_speed_m_s
    target = np.sin(np.pi * frequency * ts) ** 2
    ideal = 2.0 * np.pi * frequency / sound_speed_m_s

    nodes, weights = roots_legendre(directions)
    cos_theta = np.asarray(nodes, dtype=float)
    phi = 2.0 * np.pi * (np.arange(directions) + 0.5) / directions
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    unit = np.concatenate(
        [
            np.stack(
                [
                    np.outer(sin_theta, np.cos(phi)).ravel(),
                    np.outer(sin_theta, np.sin(phi)).ravel(),
                    np.repeat(cos_theta, directions),
                ],
                axis=1,
            )
        ]
    )
    quadrature_weight = np.repeat(np.asarray(weights, dtype=float), directions) * (
        2.0 * np.pi / directions
    )
    quadrature_weight = quadrature_weight / quadrature_weight.sum()

    # Bisect on |k|: the right hand side is monotone in |k| up to the first
    # axis reaching pi / h, which is where the scheme stops propagating.
    low = np.zeros((frequency.size, unit.shape[0]))
    high = np.full_like(low, np.pi / grid_step_m)
    for _ in range(60):
        mid = 0.5 * (low + high)
        value = courant**2 * np.sum(
            np.sin(mid[:, :, None] * unit[None, :, :] * grid_step_m / 2.0) ** 2, axis=2
        )
        too_small = value < target[:, None]
        low = np.where(too_small, mid, low)
        high = np.where(too_small, high, mid)
    per_direction = 0.5 * (low + high)
    averaged = per_direction @ quadrature_weight
    unreachable = target >= courant**2 * 3.0
    return np.asarray(np.where(unreachable, ideal, averaged), dtype=float)


def shell_weights(radii: np.ndarray, k: np.ndarray, settings: EncoderSettings) -> np.ndarray:
    """Per frequency, per receiver weight in ``[0, 1]``, ``[frequency, receiver]``.

    One while ``k r`` is under :attr:`EncoderSettings.gate_kr`, zero once it is
    past that plus the taper, raised cosine between.
    """
    kr = (
        np.atleast_1d(np.asarray(k, dtype=float))[:, None] * np.asarray(radii, dtype=float)[None, :]
    )
    low = settings.gate_kr
    width = max(float(settings.gate_taper), 1e-9)
    ramp = np.clip((kr - low) / width, 0.0, 1.0)
    return np.asarray(0.5 * (1.0 + np.cos(np.pi * ramp)), dtype=float)


def well_posed(
    design: ArrayDesign, frequency_hz: np.ndarray, k: np.ndarray, settings: EncoderSettings
) -> dict[str, Any]:
    """Where the gate leaves fewer receivers than the fit has unknowns.

    The gate is what stops a shell the truncated fit cannot describe from
    folding into the orders that are kept, and it is stated as an offset below
    the fit order. Subtracting a margin tuned at a fit order of ten from a small
    one closes it on almost everything: at a fit order of five the gate sits at
    ``k r = 1``, which at 16 kHz is a radius of 1.1 mm, smaller than the grid
    cell, so the fit is determined by its regularisation and nothing else. That
    was measured as a direction of arrival error rising from 0.04 degrees to
    7.7 above 4 kHz, and it was silent.

    So the condition is counted rather than argued: the admitted weight has to
    exceed the number of unknowns. Reported rather than raised, because a
    frequency at the very bottom of the band legitimately has little to fit.
    """
    weights = shell_weights(design.radii, k, settings)
    admitted = weights.sum(axis=1)
    unknowns = channel_count(settings.fit_order)
    short = admitted < unknowns
    return {
        "unknowns": int(unknowns),
        "receivers": int(design.count),
        "under_determined_hz": [round(float(f), 1) for f in np.atleast_1d(frequency_hz)[short]],
        "least_admitted": round(float(admitted.min()), 1),
        "note": (
            "the gate admits fewer receivers than the fit has unknowns at these "
            "frequencies, so the answer there is set by the regularisation "
            "rather than by the field"
        ),
    }


def _basis(design: ArrayDesign, order: int) -> np.ndarray:
    """``Y_nm`` at every receiver, ``[receiver, channel]``, in the ambisonic frame.

    The centre node has no direction. Its radial term is ``j_0 = 1`` for the
    first channel and zero for every other, so any direction gives the right
    row; front is used and the radius does the rest.
    """
    offsets = scene_to_ambisonic(design.offsets)
    at_centre = np.linalg.norm(offsets, axis=1) == 0.0
    safe = np.where(at_centre[:, None], np.array([[1.0, 0.0, 0.0]]), offsets)
    return real_sh(order, safe)


def encode_spectrum(
    spectrum: np.ndarray,
    frequency_hz: np.ndarray,
    design: ArrayDesign,
    *,
    sound_speed_m_s: float,
    settings: EncoderSettings | None = None,
    chunk: int = 256,
) -> np.ndarray:
    """Plane wave coefficients ``b_nm`` per frequency, ``[frequency, channel]``.

    ``spectrum`` is ``[receiver, frequency]``, as ``numpy.fft.rfft`` returns it.
    """
    settings = settings or EncoderSettings()
    spectrum = np.asarray(spectrum)
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    if spectrum.shape != (design.count, frequency.size):
        raise ValueError(
            f"spectrum {spectrum.shape} does not match {design.count} receivers "
            f"and {frequency.size} frequencies"
        )

    if settings.dispersion == "numerical":
        k = numerical_wavenumber(frequency, design.grid_step_m, sound_speed_m_s)
    else:
        k = 2.0 * np.pi * frequency / sound_speed_m_s

    if settings.max_frequency_hz is not None:
        fitted = frequency <= float(settings.max_frequency_hz)
    else:
        fitted = np.ones(frequency.size, dtype=bool)

    fit_channels = channel_count(settings.fit_order)
    basis = _basis(design, settings.fit_order)
    degree = degrees_of(settings.fit_order)
    radii = design.radii
    lam = settings.regularisation()

    out = np.zeros((frequency.size, channel_count(settings.order)), dtype=complex)
    keep = channel_count(settings.order)
    for start in range(0, frequency.size, chunk):
        stop = min(start + chunk, frequency.size)
        if not fitted[start:stop].any():
            continue
        kk = k[start:stop]
        argument = kk[:, None, None] * radii[None, :, None]
        radial = np.asarray(
            spherical_jn(degree[None, None, :], argument), dtype=float
        )  # [bin, receiver, channel]
        matrix = radial * basis[None, :, :]
        weight = shell_weights(radii, kk, settings)
        weighted = matrix * weight[:, :, None]
        normal = np.einsum("bqc,bqd->bcd", weighted, matrix)
        normal[:, np.arange(fit_channels), np.arange(fit_channels)] += lam
        block = spectrum[:, start:stop].T  # [bin, receiver]
        rhs = np.einsum(
            "bqc,bqr->bcr",
            weighted,
            np.stack([block.real, block.imag], axis=2),
        )
        solved = np.linalg.solve(normal, rhs)
        block_out = solved[:, :keep, 0] + 1j * solved[:, :keep, 1]
        out[start:stop] = np.where(fitted[start:stop, None], block_out, 0.0)

    # a_nm to the plane wave convention b_nm: divide by i^n.
    return np.asarray(out / (1j ** degrees_of(settings.order))[None, :], dtype=complex)


def encode(
    pressure: np.ndarray,
    sample_rate_hz: float,
    design: ArrayDesign,
    *,
    sound_speed_m_s: float,
    settings: EncoderSettings | None = None,
    chunk: int = 256,
) -> Ambisonic:
    """Encode ``[receiver, sample]`` pressures into an :class:`Ambisonic` response."""
    settings = settings or EncoderSettings()
    pressure = np.asarray(pressure, dtype=float)
    if pressure.ndim != 2 or pressure.shape[0] != design.count:
        raise ValueError(
            f"pressure {pressure.shape} does not match the array's {design.count} receivers"
        )
    samples = pressure.shape[1]
    # **The record is padded before the transform, and cut back after it.**
    # The encoder is a filter: per bin it applies a gain that varies with
    # frequency, so it has an impulse response, and that response is close to
    # symmetric because the regularised inverse has little phase. Transforming
    # the whole record once and inverting it convolves that response
    # *circularly*, which sends everything the filter puts **before** an event
    # to the **end** of the record instead of before its start.
    #
    # Measured on a delta at the first sample, order 7 fitted at 10: the last
    # eighth of the record came back at -3.1 dB of the total, a mirror image of
    # the first eighth, and moving the delta moved the pair with it. In the
    # rendered bedroom that put the direct sound's pre-ring in the last 20 ms
    # at about -33 dB of the whole response, which is a burst of energy where a
    # room has none. It held the broadband Schroeder curve flat at -33 dB for
    # the whole tail, so no decay time could be read from it, and it is audible
    # as a faint copy of the whole signal half a second late.
    #
    # Padding to twice the length is the plain fix: the wrapped part lands in
    # the pad and the pad is discarded. The same measurement then reads -42 dB
    # in the last eighth and falls monotonically, which is the filter's own
    # decay rather than a reflection of its head.
    length = int(next_fast_len(2 * samples))
    spectrum = np.fft.rfft(pressure, n=length, axis=-1)
    frequency = np.fft.rfftfreq(length, 1.0 / sample_rate_hz)
    coefficients = encode_spectrum(
        spectrum,
        frequency,
        design,
        sound_speed_m_s=sound_speed_m_s,
        settings=settings,
        chunk=chunk,
    )
    signals = np.fft.irfft(coefficients.T, n=length, axis=-1)[..., :samples]
    return Ambisonic(
        signals=np.asarray(signals, dtype=float),
        sample_rate_hz=float(sample_rate_hz),
        order=settings.order,
        centre=design.centre,
    )


@dataclass(frozen=True)
class ConditioningReport:
    """What the array can actually recover, per frequency and per order.

    ``error_db`` is ``[frequency, order]``: the level of the encoding error of a
    known field, relative to that field's own energy in the same order. It is
    the only honest answer to "what order does this array support", and it is
    measured rather than read off the ``k r`` rule.
    """

    frequency_hz: np.ndarray
    error_db: np.ndarray
    effective_order: np.ndarray
    noise_floor_db: float
    threshold_db: float
    well_posed: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "frequency_hz": [round(float(f), 1) for f in self.frequency_hz],
            "well_posed": self.well_posed,
            "effective_order": [int(n) for n in self.effective_order],
            "error_db": [[round(float(v), 1) for v in row] for row in self.error_db],
            "noise_floor_db": self.noise_floor_db,
            "threshold_db": self.threshold_db,
            "note": (
                "error per order of a known monopole field encoded through the "
                "array, with numerical noise added at the stated floor; the "
                "effective order is the highest order still under the threshold"
            ),
        }


def conditioning(
    design: ArrayDesign,
    frequency_hz: np.ndarray,
    *,
    sound_speed_m_s: float,
    settings: EncoderSettings | None = None,
    source_distance_m: float = 1.5,
    noise_floor_db: float = -100.0,
    threshold_db: float = -20.0,
    seed: int = 0,
) -> ConditioningReport:
    """Encode a known field through the array and measure what came back.

    The field is a monopole at ``source_distance_m``, which is a real room
    distance rather than a plane wave, so the near field terms the encoder will
    meet are present. Noise at ``noise_floor_db`` relative to the array's own
    pressure is added, standing for the solver's float32 floor: without it the
    answer is limited only by double precision and every order looks perfect.
    """
    settings = settings or EncoderSettings()
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    k = 2.0 * np.pi * frequency / sound_speed_m_s
    rng = np.random.default_rng(seed)

    direction = np.array([0.37, 0.51, 0.77])
    direction = direction / np.linalg.norm(direction)
    source = source_distance_m * direction

    positions = scene_to_ambisonic(design.offsets)
    distance = np.linalg.norm(positions - source, axis=1)
    field = np.exp(-1j * k[:, None] * distance[None, :]) / (4.0 * np.pi * distance[None, :])
    scale = np.sqrt(np.mean(np.abs(field) ** 2, axis=1))[:, None]
    noisy = field + scale * 10.0 ** (noise_floor_db / 20.0) * (
        rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
    ) / np.sqrt(2.0)

    truth = monopole_coefficients(source, k, settings.order) / (1j ** degrees_of(settings.order))
    estimate = encode_spectrum(
        noisy.T,
        frequency,
        design,
        sound_speed_m_s=sound_speed_m_s,
        settings=replace(settings, dispersion="ideal", max_frequency_hz=None),
    )

    degree = degrees_of(settings.order)
    error = np.full((frequency.size, settings.order + 1), np.inf)
    for n in range(settings.order + 1):
        columns = degree == n
        reference = np.sum(np.abs(truth[:, columns]) ** 2, axis=1)
        residual = np.sum(np.abs(estimate[:, columns] - truth[:, columns]) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            error[:, n] = 10.0 * np.log10(np.where(reference > 0, residual / reference, np.inf))

    under = error <= threshold_db
    effective = np.full(frequency.size, -1, dtype=int)
    for row in range(frequency.size):
        highest = -1
        for n in range(settings.order + 1):
            if not under[row, n]:
                break
            highest = n
        effective[row] = highest
    posedness = well_posed(design, frequency, k, settings)
    return ConditioningReport(
        frequency_hz=frequency,
        well_posed=posedness,
        error_db=error,
        effective_order=effective,
        noise_floor_db=noise_floor_db,
        threshold_db=threshold_db,
    )
