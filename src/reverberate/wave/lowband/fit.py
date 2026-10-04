"""From the records on the card to a cell's response in the cache form, the fit prepared once.

The filters are those of :class:`reverberate.accel.pairs.PairEncoder`, in
its order and with its kernels: the integration with the 40 Hz low cut, the
zero phase low pass at ``fmax``, the resampling to 4 kHz, the window of the
cache form. They run where the records are, at the grid's rate, and nothing
is kept at any other.

**The fit is an operator, not a solve.** :func:`reverberate.accel.encode.
encode_point` solves the regularised normal equations of every bin for every
cell: a batched factorisation of 3600 matrices of 121 by 121, the same
matrices each time, because every cell's array is the same shells about a
node. Here they are solved once per geometry against the weighted array
matrix, which leaves ``E[bin] = (G'WG + lambda I)^-1 G'W`` cut to the 64
channels kept, 1.8 GB on the card for the grid to 1500 Hz; a cell is then
two transforms and one batched product. The two agree to the conditioning
of the normal matrices times the rounding of float64, and the tests measure
it.

The wavenumber the fit reads the array with is the scheme's own
(:func:`reverberate.wave.lowband.scheme.wavenumber`): on the Cartesian grid
it is the encoder's, to the bit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reverberate.accel import dsp
from reverberate.accel.encode import BandEncoder, geometry_key, prepare_band
from reverberate.accel.lattice import sim_constants
from reverberate.accel.pairs import LOWCUT_HZ, LOWCUT_ORDER, REFERENCE_FMAX_HZ
from reverberate.compute import to_numpy
from reverberate.spatial.encode import EncoderSettings
from reverberate.spatial.lowband import LOW_RATE_HZ
from reverberate.wave.lowband.scheme import CARTESIAN, Scheme, wavenumber

__all__ = ["CellEncoder", "FitOperator", "level_scale"]


def level_scale(scheme: Scheme, fmax_hz: float, ts: float, ppw: float) -> float:
    """What puts a solve on the scale of a field's 8 kHz solve, whatever its grid.

    A source is one unit of volume velocity over one step, on either grid
    (``write_comms`` scales it by the node's volume), so a response's level
    goes as the time step. On the Cartesian grid at 10.5 points per
    wavelength that is ``fmax / 8000``, :mod:`reverberate.accel.pairs`'s own
    figure, and is given as such; any other grid is given the ratio of the
    steps, of which that figure is the special case.
    """
    if not scheme.fcc and ppw == CARTESIAN.ppw:
        return float(fmax_hz / REFERENCE_FMAX_HZ)
    reference = sim_constants(20.0, 50.0, REFERENCE_FMAX_HZ, CARTESIAN.ppw).ts
    return float(reference / ts)


@dataclass
class FitOperator:
    """One geometry's fit as a product: ``[chunk][bin, channel, receiver]`` on the device."""

    encoder: BandEncoder
    parts: list[Any]
    bytes_on_device: int

    @classmethod
    def prepare(cls, encoder: BandEncoder, xp: Any) -> FitOperator:
        """Solve the band's normal equations against the weighted matrix, once; free both."""
        keep = encoder.keep
        parts = []
        for weighted, normal in zip(encoder.weighted, encoder.normal, strict=True):
            solved = xp.linalg.solve(normal, xp.transpose(weighted, (0, 2, 1)))
            parts.append(xp.ascontiguousarray(solved[:, :keep, :]))
        encoder.weighted.clear()
        encoder.normal.clear()
        return cls(encoder=encoder, parts=parts, bytes_on_device=int(sum(p.nbytes for p in parts)))

    def apply(self, signals: Any, xp: Any) -> Any:
        """:func:`reverberate.accel.encode.encode_point` by the operator: ``[channel, sample]``."""
        encoder = self.encoder
        spectrum = xp.fft.rfft(xp.asarray(signals, dtype=xp.float64), n=encoder.length, axis=-1)
        fitted_bins = xp.asarray(np.flatnonzero(encoder.fitted))
        block = spectrum[:, fitted_bins].T
        solved_parts = []
        start = 0
        for part in self.parts:
            count = int(part.shape[0])
            rows = block[start : start + count]
            solved = xp.matmul(part, xp.stack([rows.real, rows.imag], axis=2))
            solved_parts.append(solved[:, :, 0] + 1j * solved[:, :, 1])
            start += count
        coefficients = xp.zeros((spectrum.shape[1], encoder.keep), dtype=xp.complex128)
        coefficients[fitted_bins] = xp.concatenate(solved_parts, axis=0)
        coefficients = coefficients / xp.asarray(1j**encoder.degree_keep)[None, :]
        return xp.fft.irfft(coefficients.T, n=encoder.length, axis=-1)[..., : encoder.samples]


@dataclass
class CellEncoder:
    """A cell's node records to its order 7 response at 4 kHz, on one device."""

    scheme: Scheme
    grid_rate_hz: float
    grid_step_m: float
    sound_speed_m_s: float
    fmax_hz: float
    settings: EncoderSettings
    xp: Any
    samples: int
    scale: float
    operators: dict[str, FitOperator] = field(default_factory=dict)
    prepare_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.lowcut = dsp.lowcut_sos(
            self.grid_rate_hz, LOWCUT_HZ, LOWCUT_ORDER, differentiated=True
        )
        self.lowpass = dsp.lowpass_sos(self.grid_rate_hz, self.fmax_hz)

    def operator_for(self, offsets: np.ndarray) -> FitOperator:
        """The fit of this geometry: prepared once for the campaign."""
        key = geometry_key(offsets)
        with self._lock:
            if key not in self.operators:
                scheme = self.scheme
                encoder = prepare_band(
                    offsets,
                    grid_step_m=self.grid_step_m,
                    sound_speed_m_s=self.sound_speed_m_s,
                    settings=self.settings,
                    samples=self.samples,
                    sample_rate_hz=LOW_RATE_HZ,
                    xp=self.xp,
                    wavenumber_of=(
                        (lambda f: wavenumber(scheme, f, self.grid_step_m, self.sound_speed_m_s))
                        if scheme.fcc
                        else None
                    ),
                )
                self.operators[key] = FitOperator.prepare(encoder, self.xp)
                self.prepare_s += encoder.prepare_s
            return self.operators[key]

    def signals(self, u_out: Any) -> Any:
        """Node records at the grid's rate as responses at 4 kHz over the window, float64."""
        xp = self.xp
        signals = xp.asarray(u_out, dtype=xp.float64)
        signals = dsp.sosfilt(self.lowcut, signals, xp)
        signals = dsp.sosfiltfilt(self.lowpass, signals, xp)
        signals = dsp.resample(signals, self.grid_rate_hz, LOW_RATE_HZ, xp)
        block = xp.zeros((signals.shape[0], self.samples), dtype=xp.float64)
        kept = min(self.samples, int(signals.shape[1]))
        block[:, :kept] = signals[:, :kept]
        return block

    def cell(self, u_out: Any, offsets: np.ndarray) -> np.ndarray:
        """One cell's response in the cache form, ``[channel, sample]`` float32 on the host."""
        operator = self.operator_for(offsets)
        encoded = operator.apply(self.signals(u_out), self.xp)
        return np.asarray(to_numpy(encoded * self.scale), dtype=np.float32)
