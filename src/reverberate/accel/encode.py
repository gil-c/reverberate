"""The order 7 fit of every listening point, on the card, per bin in one batch.

:mod:`reverberate.spatial.encode` fits, for every frequency bin, the spherical
harmonic coefficients of a ball of about a thousand grid nodes: a regularised
least squares whose matrix ``G[q, nm] = j_n(k r_q) Y_nm(d_q)`` depends on the
array's geometry and on the bin, and not on the pressure. A campaign's
arrays are the same geometry at every point -- the same shells snapped onto
the same lattice about a node -- so the matrix, its Gram ``G'WG + lambda I``
and the gate weights are the same for every point of a band, and were being
rebuilt a thousand times from the same spherical Bessel functions: that is
most of the 16 to 125 seconds a point cost on the encode boxes.

Here a band is prepared once. The radial terms, the harmonics and the gate
are computed on the host by the CPU module's own functions, bit for bit, and
the weighted matrix of every fitted bin is kept on the card. A point then
costs two transforms, a batched matrix-vector product for its right hand
sides, and one batched solve of the same Gram matrices. Where two points do
not share a geometry (a shell that snapped differently), each geometry gets
its own preparation, and the campaign reports how many there were.

What the card does not reproduce bit for bit, and is measured instead: the
Gram product (BLAS accumulates in its own order), the transforms (cuFFT
against pocketfft) and the solve (cuBLAS's batched LU against LAPACK's).
Everything upstream of the fit is in :mod:`reverberate.accel.dsp`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.fft import next_fast_len
from scipy.special import spherical_jn

from reverberate.accel import dsp
from reverberate.compute import to_numpy
from reverberate.spatial.array import ArrayDesign
from reverberate.spatial.encode import (
    EncoderSettings,
    _basis,
    numerical_wavenumber,
    shell_weights,
)
from reverberate.spatial.sh import channel_count, degrees_of

__all__ = [
    "BandEncoder",
    "BandPipeline",
    "encode_band",
    "encode_point",
    "geometry_key",
    "merge_encoded",
    "prepare_band",
]

#: Bins per chunk when the weighted matrix is built and uploaded.
CHUNK = 256


#: Offsets are compared at this resolution, a picometre: two arrays whose
#: nodes sit at the same places relative to their centres differ only by the
#: rounding of ``node - centre`` at different places on the lattice, which is
#: 1e-17 m, and would otherwise be prepared twice for nothing.
GEOMETRY_RESOLUTION_M = 1e-12


def geometry_key(offsets: np.ndarray) -> str:
    """One string per array geometry: arrays with the same offsets share a preparation."""
    rounded = np.rint(np.asarray(offsets, dtype=np.float64) / GEOMETRY_RESOLUTION_M).astype(
        np.int64
    )
    return hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()[:16]


@dataclass
class BandEncoder:
    """One band's fit, prepared for one array geometry."""

    settings: EncoderSettings
    sample_rate_hz: float
    samples: int
    length: int
    fitted: np.ndarray
    degree_keep: np.ndarray
    #: ``[chunk][bin, receiver, channel]`` on the device: ``G`` times the gate.
    weighted: list[Any] = field(default_factory=list)
    #: ``[chunk][bin, channel, channel]`` on the device: ``G'WG + lambda I``.
    normal: list[Any] = field(default_factory=list)
    chunk_bins: list[int] = field(default_factory=list)
    bytes_on_device: int = 0
    prepare_s: float = 0.0

    @property
    def keep(self) -> int:
        return channel_count(self.settings.order)


def prepare_band(
    offsets: np.ndarray,
    *,
    grid_step_m: float,
    sound_speed_m_s: float,
    settings: EncoderSettings,
    samples: int,
    sample_rate_hz: float,
    xp: Any,
    chunk: int = CHUNK,
) -> BandEncoder:
    """Everything of the fit that does not depend on the pressure, for one geometry."""
    started = time.time()
    offsets = np.asarray(offsets, dtype=np.float64)
    radii = np.linalg.norm(offsets, axis=1)
    design = ArrayDesign(
        centre=np.zeros(3),
        positions=offsets,
        offsets=offsets,
        radii=radii,
        shell=np.zeros(offsets.shape[0], dtype=int),
        nominal_radii=(float(radii.max()),),
        grid_step_m=float(grid_step_m),
        requested_centre=np.zeros(3),
    )
    length = int(next_fast_len(2 * samples))
    frequency = np.fft.rfftfreq(length, 1.0 / sample_rate_hz)
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
    lam = settings.regularisation()
    encoder = BandEncoder(
        settings=settings,
        sample_rate_hz=float(sample_rate_hz),
        samples=int(samples),
        length=length,
        fitted=fitted,
        degree_keep=degrees_of(settings.order),
    )
    fitted_bins = np.flatnonzero(fitted)
    eye = xp.eye(fit_channels, dtype=xp.float64)
    for start in range(0, fitted_bins.size, chunk):
        bins = fitted_bins[start : start + chunk]
        kk = k[bins]
        argument = kk[:, None, None] * radii[None, :, None]
        # Eleven degrees, not 121 channels: ``j_n`` depends on the degree
        # alone, and the CPU module evaluates it once per channel, which is
        # where most of its seconds a point went.
        per_degree = np.asarray(
            spherical_jn(np.arange(settings.fit_order + 1)[None, None, :], argument),
            dtype=np.float64,
        )
        radial = per_degree[:, :, degree]
        matrix = radial * basis[None, :, :]
        weight = shell_weights(radii, kk, settings)
        weighted_np = matrix * weight[:, :, None]
        matrix_d = xp.asarray(matrix)
        weighted_d = xp.asarray(weighted_np)
        normal = xp.matmul(xp.transpose(weighted_d, (0, 2, 1)), matrix_d)
        normal = normal + lam * eye[None, :, :]
        del matrix_d
        encoder.weighted.append(weighted_d)
        encoder.normal.append(normal)
        encoder.chunk_bins.append(int(bins.size))
        encoder.bytes_on_device += int(weighted_d.nbytes + normal.nbytes)
    encoder.prepare_s = time.time() - started
    return encoder


def encode_point(encoder: BandEncoder, signals: Any, xp: Any) -> Any:
    """``encode(signals, ...)`` for one point: ``[channel, sample]`` on the device."""
    spectrum = xp.fft.rfft(xp.asarray(signals, dtype=xp.float64), n=encoder.length, axis=-1)
    fitted_bins = xp.asarray(np.flatnonzero(encoder.fitted))
    block = spectrum[:, fitted_bins].T  # [bin, receiver]
    keep = encoder.keep
    solved_parts = []
    start = 0
    for weighted, normal, count in zip(
        encoder.weighted, encoder.normal, encoder.chunk_bins, strict=True
    ):
        part = block[start : start + count]
        rhs = xp.matmul(xp.transpose(weighted, (0, 2, 1)), xp.stack([part.real, part.imag], axis=2))
        solved = xp.linalg.solve(normal, rhs)
        solved_parts.append(solved[:, :keep, 0] + 1j * solved[:, :keep, 1])
        start += count
    coefficients = xp.zeros((spectrum.shape[1], keep), dtype=xp.complex128)
    coefficients[fitted_bins] = xp.concatenate(solved_parts, axis=0)
    coefficients = coefficients / xp.asarray(1j**encoder.degree_keep)[None, :]
    out = xp.fft.irfft(coefficients.T, n=encoder.length, axis=-1)[..., : encoder.samples]
    return out


@dataclass
class BandPipeline:
    """The filters and the fits of one band, ready for every point of a solve."""

    band: str
    source: str
    grid_rate_hz: float
    delivery_rate_hz: float
    fmax_hz: float
    grid_step_m: float
    sound_speed_m_s: float
    settings: EncoderSettings
    lowcut: np.ndarray
    lowpass: np.ndarray
    differentiated: bool
    delivery_samples: int
    encoders: dict[str, BandEncoder] = field(default_factory=dict)

    @classmethod
    def from_plan(
        cls,
        plan: dict[str, Any],
        band: str,
        source: str,
        *,
        differentiated: bool,
        delivery_rate_hz: float = 48000.0,
    ) -> BandPipeline:
        spec = plan["bands"][band]
        rate = float(spec["sample_rate_hz"])
        samples = int(spec["samples"])
        settings = EncoderSettings(
            order=int(plan["encoder"]["order"]),
            fit_order=int(plan["encoder"]["fit_order"]),
            max_frequency_hz=float(spec["fmax_hz"]),
        )
        return cls(
            band=band,
            source=source,
            grid_rate_hz=rate,
            delivery_rate_hz=float(delivery_rate_hz),
            fmax_hz=float(spec["fmax_hz"]),
            grid_step_m=float(spec["grid_step_m"]),
            sound_speed_m_s=float(plan["sound_speed_m_s"]),
            settings=settings,
            lowcut=dsp.lowcut_sos(
                rate,
                float(plan["lowcut_hz"]),
                int(plan["lowcut_order"]),
                differentiated=differentiated,
            ),
            lowpass=dsp.lowpass_sos(rate, float(spec["fmax_hz"])),
            differentiated=differentiated,
            delivery_samples=int(samples * float(delivery_rate_hz) / rate),
        )

    def encoder_for(self, offsets: np.ndarray, samples: int, xp: Any) -> BandEncoder:
        """The preparation for this geometry and this record length.

        The length is the resampled record's own, not the plan's: the engine
        runs one step more than the plan's ``samples`` and the CPU child
        keeps every sample it wrote, so the low band of hssd_0076 is 57 603
        samples at 48 kHz and not 57 600. The assembly reads the band's
        length as the response's duration and synthesises the tails to it;
        three samples less shifted every synthesised tail of the field.
        """
        key = f"{geometry_key(offsets)}:{samples}"
        if key not in self.encoders:
            self.encoders[key] = prepare_band(
                offsets,
                grid_step_m=self.grid_step_m,
                sound_speed_m_s=self.sound_speed_m_s,
                settings=self.settings,
                samples=samples,
                sample_rate_hz=self.delivery_rate_hz,
                xp=xp,
            )
        return self.encoders[key]

    def signals_of(self, u_out: Any, out_alpha: Any, xp: Any) -> Any:
        """From node rows to delivery rate responses: the four steps of :mod:`audio`."""
        signals = dsp.reduce_nodes(xp.asarray(u_out, dtype=xp.float64), xp.asarray(out_alpha), xp)
        signals = dsp.sosfilt(self.lowcut, signals, xp)
        signals = dsp.sosfiltfilt(self.lowpass, signals, xp)
        signals = dsp.resample(signals, self.grid_rate_hz, self.delivery_rate_hz, xp)
        return dsp.air_absorption(
            signals, self.delivery_rate_hz, xp, sound_speed_m_s=self.sound_speed_m_s
        )

    def point(
        self, u_out: Any, out_alpha: Any, positions: np.ndarray, centre: np.ndarray, xp: Any
    ) -> np.ndarray:
        """One point's order 7 response, ``[channel, sample]`` float32 on the host."""
        offsets = np.asarray(positions, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
        signals = self.signals_of(u_out, out_alpha, xp)
        encoder = self.encoder_for(offsets, int(signals.shape[1]), xp)
        return to_numpy(encode_point(encoder, signals, xp)).astype(np.float32)


def encode_band(
    plan: dict[str, Any],
    band: str,
    source: dict[str, Any],
    *,
    pressure: Path,
    comms: Path,
    positions: np.ndarray,
    output: Path,
    xp: Any,
    row_offset: int = 0,
    rows: list[list[int] | None] | None = None,
    say: Any = None,
    pressure_dtype: str = "float32",
) -> dict[str, Any]:
    """Every point of one solve, written as the first campaigns' encode child wrote ``encoded.h5``.

    ``pressure`` is the engine's ``sim_outs.h5`` (or a float32 shrink of it):
    the rows are rounded to float32 before the fit, as the shrink did, so
    an unshrunk output encodes to the same numbers. ``rows`` and
    ``row_offset`` describe a slice of the plan, as the encode job did.
    """
    say = say or (lambda message: None)
    spec = plan["bands"][band]
    plan_rows: list[list[int] | None] = spec["rows"] if rows is None else rows
    with h5py.File(comms, "r") as handle:
        out_alpha = np.asarray(handle["out_alpha"][...], dtype=np.float64)
        differentiated = bool(np.asarray(handle["diff"]).item())
    # The comms file is either the plan's (one row per receiver of the whole
    # band, the encode boxes' case) or the slice's own (the campaign's case,
    # one row per receiver of this pressure file); its rows are read in the
    # space it was written in.
    alpha_is_plan = out_alpha.shape[0] == positions.shape[0]
    pipeline = BandPipeline.from_plan(plan, band, source["name"], differentiated=differentiated)
    indices = [i for i, r in enumerate(plan_rows) if r is not None]
    started = time.time()
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    read_s = 0.0
    with h5py.File(pressure, "r") as handle:
        u_all = handle["u_out"]
        for done, index in enumerate(indices, start=1):
            span = plan_rows[index]
            assert span is not None
            start, stop = span
            t0 = time.time()
            block = np.asarray(u_all[start:stop])
            read_s += time.time() - t0
            if pressure_dtype == "float32":
                block = block.astype(np.float32).astype(np.float64)
            plan_start, plan_stop = start + row_offset, stop + row_offset
            centre = np.asarray(spec["centres"][index], dtype=float)
            alpha = out_alpha[plan_start:plan_stop] if alpha_is_plan else out_alpha[start:stop]
            signals = pipeline.point(block, alpha, positions[plan_start:plan_stop], centre, xp)
            results[index] = (signals, centre)
            if done % 20 == 0 or done == len(indices):
                elapsed = time.time() - started
                say(
                    f"encoded {done}/{len(indices)} in {elapsed:.0f} s"
                    f" ({elapsed / done:.2f} s a point, {read_s / done:.2f} s of it reading)"
                )
    order_indices = sorted(results)
    channels, samples = results[order_indices[0]][0].shape
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        data = handle.create_dataset(
            "signals", shape=(len(order_indices), channels, samples), dtype=np.float32
        )
        for row, index in enumerate(order_indices):
            data[row] = results[index][0]
        handle.create_dataset("point_index", data=np.asarray(order_indices))
        handle.create_dataset("centres", data=np.asarray([results[i][1] for i in order_indices]))
        handle.attrs["sample_rate_hz"] = float(pipeline.delivery_rate_hz)
        handle.attrs["order"] = int(pipeline.settings.order)
        handle.attrs["band"] = band
        handle.attrs["source"] = source["name"]
    geometries = len(pipeline.encoders)
    report = {
        "band": band,
        "source": source["name"],
        "points": len(order_indices),
        "geometries": geometries,
        "prepare_s": round(sum(e.prepare_s for e in pipeline.encoders.values()), 1),
        "device_bytes": int(sum(e.bytes_on_device for e in pipeline.encoders.values())),
        "encode_s": round(time.time() - started, 1),
        "read_s": round(read_s, 1),
        "output": str(output),
    }
    say(json.dumps(report))
    return report


def merge_encoded(parts: list[Path], merged: Path) -> None:
    """One encoded file from the parts of a band, points in plan order."""
    signals, index, centres = [], [], []
    rate = order = None
    attrs: dict[str, Any] = {}
    for part in parts:
        with h5py.File(part, "r") as handle:
            if handle["signals"].shape[0] == 0:
                continue
            signals.append(np.asarray(handle["signals"][...]))
            index.append(np.asarray(handle["point_index"][...]))
            centres.append(np.asarray(handle["centres"][...]))
            rate, order = float(handle.attrs["sample_rate_hz"]), int(handle.attrs["order"])
            attrs = {k: handle.attrs[k] for k in ("band", "source")}
    stacked = np.concatenate(signals)
    points = np.concatenate(index)
    order_of = np.argsort(points)
    with h5py.File(merged, "w") as handle:
        handle.create_dataset("signals", data=stacked[order_of])
        handle.create_dataset("point_index", data=points[order_of])
        handle.create_dataset("centres", data=np.concatenate(centres)[order_of])
        handle.attrs["sample_rate_hz"] = rate
        handle.attrs["order"] = order
        for k, v in attrs.items():
            handle.attrs[k] = v
