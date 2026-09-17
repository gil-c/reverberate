"""One response from two solvers: the wave field low, the geometric mirror high.

The wave solver answers where a room is modal and a geometric model cannot
be right: below the Schroeder frequency the response is a handful of
resonances, not a sum of rays. The mirror answers where the wave solver is
expensive: the cost of a finite difference grid goes as the fourth power of
the frequency it resolves. This module joins the two into one response per
point, in the reference's own format, so the walk-through app plays it
beside the two it is made of.

**The join.** How two estimates of the same field add through the
crossover depends on whether they share a phase there, and on 0076 they do
for a while and then stop: over the octave around 1 kHz the wave field and
the mirror correlate 0.83 to 0.98 over the direct sound, 0.24 to 0.49 over
the first 40 ms, and under 0.25 after 200 ms. So the response is joined
twice. The part up to a few milliseconds after the onset, where the two
carry the same arrival, uses masks that add to one in **pressure**
(``cos^2 + sin^2``), which two coherent signals rebuild exactly. What
follows uses masks that add to one in **power** (``cos + sin`` of the same
angle, squared), which is right for two independent tails and would
otherwise leave a 3 dB hole at the cutoff. Both pairs are applied in the
frequency domain: no group delay, so neither solver's arrival times move.

**The level.** The mirror is calibrated against the wave field band by
band, so the two already agree at the join to about a decibel. What is
left is measured per point in the crossover octave and reported as
``seam_db``; ``match`` scales the mirror by it, one scalar per point, so
the join has no step in it. Nothing else of either solver is touched.

**The clock.** The mirror field is written on the reference's own clock and
scale (:mod:`reverberate.mirror.field`), so the two are already aligned
when they arrive here. A hybrid of an unaligned pair would still be flat in
power, but its direct sound would arrive twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "Crossover",
    "blend",
    "seam_db",
    "write_hybrid_field",
]


@dataclass(frozen=True)
class Crossover:
    """Where the two solvers meet, and over how wide a band."""

    #: The frequency the two masks cross, each at half the power.
    cutoff_hz: float = 1000.0
    #: The ramp's width in octaves, centred on the cutoff. One octave takes
    #: the wave field from full at 707 Hz to nothing at 1414 Hz.
    width_octaves: float = 1.0
    #: How long after the onset the two are still taken as one arrival and
    #: joined in pressure, and how long that hand-over takes. Zero joins the
    #: whole response in power.
    coherent_s: float = 0.005
    coherent_fade_s: float = 0.005

    def record(self) -> dict[str, Any]:
        return {
            "cutoff_hz": self.cutoff_hz,
            "width_octaves": self.width_octaves,
            "coherent_s": self.coherent_s,
            "coherent_fade_s": self.coherent_fade_s,
        }

    def masks(self, samples: int, rate: float, *, power: bool = True) -> tuple[Any, Any]:
        """The low and high masks over ``rfftfreq(samples, 1 / rate)``.

        With ``power`` the two squares add to one, which is right where the
        two responses are independent; without it the two themselves add to
        one, which is right where they carry the same arrival.
        """
        freqs = np.fft.rfftfreq(samples, 1.0 / rate)
        ramp = np.zeros_like(freqs)
        above = freqs > 0.0
        ramp[above] = np.log2(freqs[above] / self.cutoff_hz) / self.width_octaves + 0.5
        angle = np.clip(ramp, 0.0, 1.0) * (0.5 * np.pi)
        low, high = np.cos(angle), np.sin(angle)
        return (low, high) if power else (low**2, high**2)

    def onset_window(self, signal: np.ndarray, rate: float) -> np.ndarray:
        """One where the two responses carry the same arrival, zero after it.

        The onset is the loudest sample of ``signal``, which for a point
        with a direct path is the direct sound itself.
        """
        samples = int(np.asarray(signal).shape[-1])
        window = np.zeros(samples)
        if self.coherent_s <= 0.0:
            return window
        onset = int(np.argmax(np.abs(np.asarray(signal, dtype=float))))
        held = onset + int(round(self.coherent_s * rate))
        fade = max(int(round(self.coherent_fade_s * rate)), 1)
        window[: min(held, samples)] = 1.0
        stop = min(held + fade, samples)
        if stop > held:
            ramp = np.arange(stop - held) / fade
            window[held:stop] = 0.5 * (1.0 + np.cos(np.pi * ramp))
        return window

    def band_hz(self) -> tuple[float, float]:
        """The band the ramp covers, where both solvers are heard."""
        half = 0.5 * self.width_octaves
        return self.cutoff_hz * 2.0**-half, self.cutoff_hz * 2.0**half


def seam_db(low: np.ndarray, high: np.ndarray, rate: float, crossover: Crossover) -> float:
    """How far apart the two are, in decibels, over the band they share.

    Positive means the low side holds more energy than the high side there.
    It is the step a listener would hear at the join if nothing levelled the
    two, and the honest measure of how well the mirror's calibration met the
    wave field at the cutoff.
    """
    lo_hz, hi_hz = crossover.band_hz()
    samples = int(np.asarray(low).shape[-1])
    freqs = np.fft.rfftfreq(samples, 1.0 / rate)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not band.any():
        return 0.0
    a = float(np.sum(np.abs(np.fft.rfft(np.asarray(low, dtype=float), axis=-1)[..., band]) ** 2))
    b = float(np.sum(np.abs(np.fft.rfft(np.asarray(high, dtype=float), axis=-1)[..., band]) ** 2))
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float(10.0 * np.log10(a / b))


def blend(
    low: np.ndarray,
    high: np.ndarray,
    rate: float,
    crossover: Crossover,
    *,
    match: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``low`` under the cutoff and ``high`` over it, joined without a step.

    Both are ``[channel, sample]`` at ``rate`` and must be the same length.
    ``match`` scales ``high`` by the seam so the two carry the same energy
    over the band they share; the record says by how much.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.shape != high.shape:
        raise ValueError(f"the two responses differ in shape: {low.shape} and {high.shape}")
    step = seam_db(low[0], high[0], rate, crossover)
    gain = 10.0 ** (step / 20.0) if match else 1.0
    high = high * gain
    samples = low.shape[-1]
    together = crossover.onset_window(low[0], rate)
    apart = 1.0 - together
    lo_power, hi_power = crossover.masks(samples, rate, power=True)
    spectrum = (
        np.fft.rfft(low * apart, axis=-1) * lo_power + np.fft.rfft(high * apart, axis=-1) * hi_power
    )
    if together.any():
        lo_press, hi_press = crossover.masks(samples, rate, power=False)
        spectrum = spectrum + (
            np.fft.rfft(low * together, axis=-1) * lo_press
            + np.fft.rfft(high * together, axis=-1) * hi_press
        )
    joined = np.fft.irfft(spectrum, n=samples, axis=-1)
    return joined, {
        "crossover": crossover.record(),
        "seam_db": round(step, 3),
        "applied_gain_db": round(20.0 * np.log10(gain), 3),
        "matched": bool(match),
        "coherent_samples": int(together.sum()),
    }


def write_hybrid_field(
    target: Path,
    low: Path,
    high: Path,
    *,
    crossover: Crossover,
    match: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A field whose responses are ``low`` under the cutoff and ``high`` over it.

    Both files are fields of :doc:`the reference's format </formats/ambisonic-field>`
    over the same lattice, in the same point order. Everything that is not
    ``/ir`` is copied from ``low``, which is the reference itself in the use
    this was written for.
    """
    target, low, high = Path(target), Path(low), Path(high)
    target.parent.mkdir(parents=True, exist_ok=True)
    seams: list[float] = []
    with h5py.File(low, "r") as a, h5py.File(high, "r") as b:
        rate = float(a.attrs["sample_rate_hz"])
        if float(b.attrs["sample_rate_hz"]) != rate:
            raise ValueError("the two fields do not share a sample rate")
        shape = tuple(int(v) for v in a["ir"].shape)
        if tuple(int(v) for v in b["ir"].shape) != shape:
            raise ValueError(f"the two fields differ in shape: {shape} and {b['ir'].shape}")
        points, channels, samples = shape
        with h5py.File(target, "w") as out:
            ir = out.create_dataset(
                "ir",
                shape=shape,
                dtype=np.float32,
                chunks=(1, channels, samples),
            )
            for point in range(points):
                joined, record = blend(
                    np.asarray(a["ir"][point], dtype=float),
                    np.asarray(b["ir"][point], dtype=float),
                    rate,
                    crossover,
                    match=match,
                )
                ir[point] = joined.astype(np.float32)
                seams.append(float(record["seam_db"]))
            for name in a:
                if name != "ir" and isinstance(a[name], h5py.Dataset):
                    out.create_dataset(name, data=np.asarray(a[name][...]))
            for name in a.attrs:
                if name != "provenance_json":
                    out.attrs[name] = a.attrs[name]
            summary = {
                "kind": "hybrid: the wave field under the cutoff, the mirror over it",
                "low": str(low),
                "high": str(high),
                "crossover": crossover.record(),
                "matched": bool(match),
                "seam_db": {
                    "median": round(float(np.median(seams)), 3),
                    "p10": round(float(np.percentile(seams, 10)), 3),
                    "p90": round(float(np.percentile(seams, 90)), 3),
                },
                **(provenance or {}),
            }
            out.attrs["hybrid"] = "two solvers joined; see provenance_json"
            out.attrs["provenance_json"] = json.dumps(summary, sort_keys=True)
    return summary
