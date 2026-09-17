"""Two solvers joined into one response."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.mirror.hybrid import Crossover, blend, seam_db, write_hybrid_field


def test_the_masks_are_power_complementary() -> None:
    """Two responses that do not share a phase still sum to the right power."""
    crossover = Crossover(cutoff_hz=1000.0, width_octaves=1.0)
    low, high = crossover.masks(8192, 48000.0)
    np.testing.assert_allclose(low**2 + high**2, 1.0, rtol=0, atol=1e-12)
    # ... and the pair for the part where the two carry the same arrival adds
    # to one in pressure instead.
    near, far = crossover.masks(8192, 48000.0, power=False)
    np.testing.assert_allclose(near + far, 1.0, rtol=0, atol=1e-12)
    freqs = np.fft.rfftfreq(8192, 1.0 / 48000.0)
    # Everything under the ramp is the low side's, everything over it the high side's.
    assert low[np.searchsorted(freqs, 400.0)] == pytest.approx(1.0)
    assert high[np.searchsorted(freqs, 400.0)] == pytest.approx(0.0)
    assert low[np.searchsorted(freqs, 3000.0)] == pytest.approx(0.0, abs=1e-12)
    assert high[np.searchsorted(freqs, 3000.0)] == pytest.approx(1.0)
    # They cross at the cutoff, each at half the power.
    at = np.searchsorted(freqs, 1000.0)
    assert low[at] ** 2 == pytest.approx(0.5, abs=0.01)


def test_the_join_keeps_each_solver_in_its_own_band() -> None:
    """Under the cutoff the low side comes back untouched, over it the high side."""
    rate = 48000.0
    rng = np.random.default_rng(0)
    samples = 24000
    low = rng.standard_normal((4, samples))
    high = rng.standard_normal((4, samples))
    joined, record = blend(low, high, rate, Crossover(coherent_s=0.0), match=False)
    assert record["applied_gain_db"] == 0.0
    freqs = np.fft.rfftfreq(samples, 1.0 / rate)
    spectra = {
        name: np.fft.rfft(value, axis=-1)
        for name, value in (("j", joined), ("l", low), ("h", high))
    }
    under = freqs < 500.0
    over = freqs > 2500.0
    np.testing.assert_allclose(spectra["j"][:, under], spectra["l"][:, under], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(spectra["j"][:, over], spectra["h"][:, over], rtol=1e-9, atol=1e-9)


def test_the_seam_is_measured_and_levelled() -> None:
    """A high side 6 dB down is measured as such and brought back up."""
    rate = 48000.0
    rng = np.random.default_rng(1)
    low = rng.standard_normal((2, 12000))
    high = low * 10.0 ** (-6.0 / 20.0)
    crossover = Crossover(coherent_s=0.0)
    assert seam_db(low[0], high[0], rate, crossover) == pytest.approx(6.0, abs=1e-6)
    joined, record = blend(low, high, rate, crossover, match=True)
    assert record["seam_db"] == pytest.approx(6.0, abs=1e-3)
    assert record["applied_gain_db"] == pytest.approx(6.0, abs=1e-3)
    # Levelled, the high side carries the low side's own energy, so the join of
    # two copies of one signal holds between 1.5 and 3 dB more over the band.
    assert -3.1 < seam_db(low[0], joined[0], rate, crossover) < -1.5


def _field(path: Path, ir: np.ndarray, rate: float = 48000.0) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ir", data=ir.astype(np.float32))
        handle.create_dataset("point_index", data=np.arange(ir.shape[0]))
        handle.attrs["sample_rate_hz"] = rate
        handle.attrs["order"] = 1
        handle.attrs["provenance_json"] = json.dumps({"who": "the test"})
    return path


def test_the_hybrid_field_keeps_the_lattice(tmp_path: Path) -> None:
    """Everything that is not the responses is the low field's own."""
    rng = np.random.default_rng(2)
    low = rng.standard_normal((3, 4, 6000))
    high = rng.standard_normal((3, 4, 6000))
    summary = write_hybrid_field(
        tmp_path / "out.h5",
        _field(tmp_path / "low.h5", low),
        _field(tmp_path / "high.h5", high),
        crossover=Crossover(coherent_s=0.0),
    )
    assert summary["crossover"]["cutoff_hz"] == 1000.0
    with h5py.File(tmp_path / "out.h5", "r") as handle:
        assert handle["ir"].shape == (3, 4, 6000)
        np.testing.assert_array_equal(handle["point_index"][...], np.arange(3))
        assert float(handle.attrs["sample_rate_hz"]) == 48000.0
        written = json.loads(handle.attrs["provenance_json"])
        joined = np.asarray(handle["ir"][0], dtype=float)
    assert written["kind"].startswith("hybrid")
    freqs = np.fft.rfftfreq(6000, 1.0 / 48000.0)
    under = freqs < 400.0
    got = np.fft.rfft(joined, axis=-1)[:, under]
    want = np.fft.rfft(low[0], axis=-1)[:, under]
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def test_a_shared_arrival_is_rebuilt_and_a_shared_tail_is_not() -> None:
    """The onset joins in pressure, so a direct sound the two agree on comes back whole."""
    rate = 48000.0
    rng = np.random.default_rng(3)
    low = np.zeros((2, 12000))
    low[:, 100] = 1.0
    low[:, 400:] = rng.standard_normal((2, 11600)) * 0.01
    joined, record = blend(low, low, rate, Crossover(), match=False)
    assert record["coherent_samples"] > 0
    assert joined[0, 100] == pytest.approx(1.0, abs=1e-3)
    # Past the hand-over two copies of one tail add in power, which is 3 dB more
    # than one of them over the crossover band and nothing outside it.
    late = slice(2000, 12000)
    freqs = np.fft.rfftfreq(10000, 1.0 / rate)
    ratio = 20 * np.log10(
        np.abs(np.fft.rfft(joined[0, late])) / np.abs(np.fft.rfft(low[0, late]) + 1e-30)
    )
    assert abs(float(np.median(ratio[freqs < 300.0]))) < 0.5
    assert float(np.median(ratio[(freqs > 800.0) & (freqs < 1200.0)])) > 1.5
