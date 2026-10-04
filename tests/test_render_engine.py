"""The signal engine against renders known in closed form, and against itself.

The synthetic free field gives the analytic checks: the delay, the ``1 / d``
gain and the direction of a source at rest, the Doppler shift of a listener
who walks, the monopole's own expansion under the crossover. The engine
against itself gives the rest: blocks of any size, two processes, two
devices, stems against the mix.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import fftconvolve

from reverberate.compute import cuda_available
from reverberate.metrics import octave_bank, octave_filter_rows
from reverberate.mirror.ism import Paths
from reverberate.mirror.render import RenderSettings as MirrorSettings
from reverberate.mirror.render import early_signals
from reverberate.render import delay
from reverberate.render.__main__ import main
from reverberate.render.benchmark import density_pack
from reverberate.render.dry import DryTrack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.output import open_signal, write_signal
from reverberate.render.pack import (
    ScenePack,
    monopole_low_response,
    read_pack,
    synthetic_free_field,
    write_pack,
)
from reverberate.render.translate import OperatorTranslation, SpatialTranslation
from reverberate.spatial.lowband import from_stored, to_stored
from reverberate.spatial.sh import degrees_of, real_sh, scene_to_ambisonic
from reverberate.spatial.translate import fusion_operator, translation_operator

C = 343.2
FS = 48000


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).standard_normal(int(round(seconds * FS))))


def band_noise(seconds: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Noise between two frequencies, faded at both ends."""
    n = int(round(seconds * FS))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    shape = np.clip((freqs - low_hz) / 50.0, 0, 1) * np.clip((high_hz - freqs) / 50.0, 0, 1)
    return np.asarray(np.fft.irfft(np.fft.rfft(noise(seconds)) * shape, n) * np.hanning(n))


def through_bank(dry: np.ndarray) -> np.ndarray:
    """The dry signal through every band of the octave bank at unit gain, summed."""
    bands = octave_bank(FS).filters.shape[1]
    return np.asarray(
        octave_filter_rows(np.repeat(dry[None], bands, 0), FS, np.arange(bands)).sum(0)
    )


def small_dense(moving: bool = True, **changes: Any) -> ScenePack:
    return density_pack(duration_s=0.3, moving=moving, bins=30, **changes)


# --------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------


def test_a_source_at_rest_gives_the_delay_the_gain_and_the_direction() -> None:
    distance = 480 * C / FS  # ten milliseconds, a whole number of samples
    direction = np.array([0.6, 0.0, -0.8])
    pack = synthetic_free_field(
        source=distance * direction, listener_start=(0, 0, 0), duration_s=0.5
    )
    dry = band_noise(0.4, 0.0, 22000.0)  # the engine fades the band out from 22 kHz
    out = Engine(pack, {"s1": dry}).render()
    assert out.shape == (64, 24000)
    expected = np.zeros(24000)
    expected[480 : 480 + dry.size] = through_bank(dry) / distance
    harmonics = real_sh(7, scene_to_ambisonic(direction[None]))[0]
    middle = slice(1000, 19000)  # clear of the bank's ringing round the signal's two ends
    peak = np.abs(expected).max()
    # The pack holds its gains and directions in float32: 1e-7 of them.
    np.testing.assert_allclose(
        out[:, middle], harmonics[:, None] * expected[None, middle], atol=1e-6 * peak
    )
    # The delay, read off the signal itself.
    lag = np.argmax(fftconvolve(out[0], dry[::-1])) - (dry.size - 1)
    assert lag == 480


def test_at_rest_the_early_part_is_the_mirrors_own_render() -> None:
    """One path through ``mirror.render.early_signals``, convolved with the dry signal."""
    distance = 480 * C / FS
    direction = np.array([0.0, 0.6, 0.8])
    pack = synthetic_free_field(
        source=distance * direction, listener_start=(0, 0, 0), duration_s=0.3
    )
    gains = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
    source = pack.sources["s1"]
    shaped = dataclasses.replace(
        source.early, gain=np.repeat(gains[None], source.early.gain.shape[0], 0)
    )
    pack = dataclasses.replace(pack, sources={"s1": dataclasses.replace(source, early=shaped)})
    dry = band_noise(0.2, 0.0, 22000.0)
    out = Engine(pack, {"s1": dry}).render()
    paths = Paths(
        receiver=np.zeros(3),
        image=np.array([0], dtype=np.int32),
        order=np.array([0], dtype=np.int32),
        length_m=np.array([distance]),
        direction=direction[None],
        gain=gains[None].astype(float),
        points=np.zeros((1, 5, 3)),
        sequence=np.full((1, 3), -1, dtype=np.int32),
    )
    pulse = early_signals(paths, MirrorSettings(order=7, duration_s=0.1), C)
    expected = fftconvolve(pulse, np.append(dry, 0.0)[None, :], axes=1)[:, : out.shape[1]]
    assert expected.shape == out.shape
    # The pack's gains and directions are float32: 1e-7 of them, and nothing else.
    np.testing.assert_allclose(out, expected, atol=3e-7 * np.abs(expected).max())


def test_the_interpolator_holds_the_band_to_its_stated_error() -> None:
    upper = delay.worst_error(1000.0, 20000.0)
    lower = delay.worst_error(20.0, 1000.0)
    # Measured: -94.0 dB (2.0e-5) from 1 to 20 kHz, -115.6 dB under 1 kHz.
    assert upper["error_db"] < -93.0 and upper["level_db"] < 2e-4
    assert lower["error_db"] < -114.0
    # A whole delay is one tap of one.
    table = delay.kernel_table()
    np.testing.assert_allclose(table[0], np.eye(delay.TAPS)[delay.TAPS // 2 - 1], atol=1e-15)
    with pytest.raises(ValueError, match="outside"):
        delay.read(np.zeros((1, 16, 1)), np.array([[2.0]]))


def test_a_delay_between_two_samples_is_the_delayed_signal() -> None:
    distance = 480.37 * C / FS
    pack = synthetic_free_field(source=(distance, 0, 0), listener_start=(0, 0, 0), duration_s=0.3)
    dry = band_noise(0.2, 1000.0, 20000.0)
    out = Engine(pack, {"s1": dry}).render()[0]
    n = out.size
    spectrum = np.fft.rfft(through_bank(dry), n)
    shift = np.exp(-2j * np.pi * np.fft.rfftfreq(n) * 480.37)
    expected = np.fft.irfft(spectrum * shift, n) / distance
    assert np.abs(out - expected).max() < 1e-4 * np.abs(expected).max()


def test_a_listener_walking_towards_the_source_hears_the_doppler_shift() -> None:
    speed, tone_hz = 1.5, 8000.0
    pack = synthetic_free_field(
        source=(5.0, 1.5, 0.0),
        listener_start=(0.0, 1.5, 0.0),
        listener_end=(speed * 0.8, 1.5, 0.0),
        duration_s=0.8,
    )
    tone = np.sin(2.0 * np.pi * tone_hz * np.arange(int(0.8 * FS)) / FS)
    out = Engine(pack, {"s1": tone}).render()[0]
    cut = slice(9600, 28800)
    window = out[cut] * np.hanning(cut.stop - cut.start)
    padded = 64 * window.size
    heard = np.argmax(np.abs(np.fft.rfft(window, padded))) * FS / padded
    # The analytic shift is 34.97 Hz; the transform's own grid is 0.04 Hz.
    assert heard == pytest.approx(tone_hz * (1.0 + speed / C), abs=0.1)
    # And the waveform: the tone at the moving delay, over the moving distance.
    t = np.arange(cut.start, cut.stop) / FS
    far = 5.0 - speed * t
    expected = np.sin(2.0 * np.pi * tone_hz * (t - far / C)) / far
    assert np.abs(out[cut] - expected).max() < 5e-4 * np.abs(expected).max()


def test_a_birth_and_a_death_make_no_click() -> None:
    steps = 17
    heard = np.zeros(steps, dtype=bool)
    heard[5:12] = True
    pack = synthetic_free_field(
        source=(2.0, 0, 0), listener_start=(0, 0, 0), duration_s=0.8, audible=heard
    )
    hz = 200.0
    dry = np.sin(2.0 * np.pi * hz * np.arange(int(0.8 * FS)) / FS)
    out = Engine(pack, {"s1": dry}).render()[0]
    step = pack.header.step_samples
    # Silent until the step before its first, then a fade one step long.
    assert np.all(out[: 4 * step] == 0.0) and np.all(out[12 * step :] == 0.0)
    level = np.abs(through_bank(dry)[step:-step]).max() / 2.0  # the bank's gain at the tone
    assert np.abs(out[6 * step : 10 * step]).max() == pytest.approx(level, rel=1e-3)
    # No sample to sample jump beyond the tone's own slope and the fade's, at gain 1 / d.
    bound = (2.0 * np.pi * hz / FS + 1.0 / step) * level
    assert np.abs(np.diff(out)).max() <= 1.001 * bound
    assert abs(out[4 * step]) < 1e-3 and abs(out[12 * step - 1]) < 1e-3


def test_a_path_whose_image_jumps_is_one_dying_and_one_born() -> None:
    near, far = 480 * C / FS, 960 * C / FS
    pack = synthetic_free_field(source=(near, 0, 0), listener_start=(0, 0, 0), duration_s=0.3)
    source = pack.sources["s1"]
    delays = np.array(source.early.delay_s)
    gains = np.array(source.early.gain)
    delays[3:] = far / C  # the same identity, 3.4 m further from one step to the next
    gains[3:] = 1.0 / far
    moved = dataclasses.replace(source.early, delay_s=delays, gain=gains)
    pack = dataclasses.replace(pack, sources={"s1": dataclasses.replace(source, early=moved)})
    dry = band_noise(0.3, 0.0, 22000.0)
    out = Engine(pack, {"s1": dry}).render()[0]
    # Not one delay swept over 3.4 m in a step: the near one fading out under the far one.
    banked = np.concatenate([np.zeros(960), through_bank(dry)])
    u = np.arange(2400) / 2400.0
    t = 960 + 2 * 2400 + np.arange(2400)
    expected = (1.0 - u) * banked[t - 480] / near + u * banked[t - 960] / far
    np.testing.assert_allclose(out[2 * 2400 : 3 * 2400], expected, atol=1e-6 * np.abs(out).max())


def reference_low(pack: ScenePack, dry: np.ndarray, source: np.ndarray) -> np.ndarray:
    """The monopole at the head, convolved with the dry signal at 48 kHz, by its transform."""
    head = np.asarray(pack.listener.position[0])
    response = monopole_low_response(source - head, pack.header, pack.crossover).astype(float)
    n = dry.size + 2 * FS  # the response is 1.2 s long
    # The response's transform on the 48 kHz grid: the low rate's, padded to the same
    # bins, times the ratio of the rates, the stored samples being the 48 kHz response's.
    low = 12.0 * np.fft.rfft(response, n // 12, axis=1)
    transfer = np.zeros((64, n // 2 + 1), dtype=complex)
    transfer[:, : low.shape[1]] = low
    return np.asarray(np.fft.irfft(np.fft.rfft(dry, n)[None] * transfer, n)[:, : dry.size])


def degree_error_db(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """The error's energy over the reference's, per ambisonic degree, in dB."""
    degrees = degrees_of(7)
    error = ((got - want) ** 2).sum(axis=1)
    level = (want**2).sum(axis=1)
    return np.array(
        [10 * np.log10(error[degrees == n].sum() / level[degrees == n].sum()) for n in range(8)]
    )


def test_level_b_under_the_crossover_is_the_analytic_monopole() -> None:
    source = np.array([0.5, 1.5, 3.0])
    dry = band_noise(0.5, 200.0, 1300.0)
    middle = slice(2400, 21000)
    # On a cell: the solved response itself, to the ripple of the filter between the rates.
    pack = synthetic_free_field(
        level="B", source=source, listener_start=(0, 1.5, 0), duration_s=0.5
    )
    low = Engine(pack, {"s1": dry}).stem("s1", parts=("low",))
    want = reference_low(pack, dry, source)
    assert np.abs(low - want)[:, middle].max() < 3e-6 * np.abs(want[:, middle]).max()
    # Translated 0.10 m from one cell, and fused from two 0.20 m either side. An order 7
    # expansion moved keeps its low degrees and loses its top ones, which is the
    # estimator's doing and is measured here, degree by degree.
    moved = synthetic_free_field(
        level="B",
        source=source,
        listener_start=(0.1, 1.5, 0),
        cell_origin=(0, 1.5, 0),
        duration_s=0.5,
    )
    fused = synthetic_free_field(
        level="B",
        source=source,
        listener_start=(0.2, 1.5, 0),
        cell_origin=(0, 1.5, 0),
        duration_s=0.5,
        fuse=True,
    )
    errors = {}
    for name, made in (("moved", moved), ("fused", fused)):
        got = Engine(made, {"s1": dry}).stem("s1", parts=("low",))[:, middle]
        errors[name] = degree_error_db(got, reference_low(made, dry, source)[:, middle])
    # Measured: moved -59 -56 -54 -53 -44 -31 -19 -9; fused -52 -54 -58 -51 -45 -44 -35 -23.
    assert np.all(errors["moved"][:4] < -45.0) and errors["moved"][7] > -15.0
    assert np.all(errors["fused"][:6] < -40.0) and errors["fused"][7] < -18.0


def test_level_b_joins_the_two_sides_into_the_whole_band() -> None:
    """Pressure masks over the onset add to one: low and high give the delayed signal."""
    source = np.array([0.5, 1.5, 3.0])
    pack = synthetic_free_field(
        level="B", source=source, listener_start=(0, 1.5, 0), duration_s=0.5
    )
    dry = band_noise(0.4, 400.0, 3000.0)
    out = Engine(pack, {"s1": dry}).render()[0]
    far = float(np.linalg.norm(source - [0, 1.5, 0]))
    n = out.size
    shift = np.exp(-2j * np.pi * np.fft.rfftfreq(n) * far / C * FS)
    expected = np.fft.irfft(np.fft.rfft(through_bank(dry), n) * shift, n) / far
    # The mask is a filter of 85 ms here and a spectrum of 1.2 s in the pack: 0.5 per cent.
    assert np.abs(out - expected).max() < 5e-3 * np.abs(expected).max()


def test_the_engine_s_translation_is_the_library_s_operators_applied() -> None:
    """``SpatialTranslation`` never forms ``T`` or ``G``; it is them all the same."""
    applied = SpatialTranslation(2, C, quadrature_degree=8)
    freqs = np.array([200.0, 700.0])
    rng = np.random.default_rng(1)

    def operator(offsets: np.ndarray, at: np.ndarray) -> np.ndarray:
        if offsets.shape[0] == 1:
            made = translation_operator(offsets[0], at, 2, sound_speed_m_s=C, quadrature_degree=8)
        else:
            made = fusion_operator(offsets, at, 2, sound_speed_m_s=C, quadrature_degree=8)
        return np.asarray(made)

    for cells in (1, 2):
        fields = rng.standard_normal((cells, 9, 2)) + 1j * rng.standard_normal((cells, 9, 2))
        offsets = np.array([[0.1, 0.0, 0.05], [-0.3, 0.0, 0.05]])[:cells]
        direct = applied.to_head(fields, offsets, freqs, np)
        via = OperatorTranslation(operator).to_head(fields, offsets, freqs, np)
        # The operators are kept in single precision.
        np.testing.assert_allclose(via, direct, atol=2e-6)
    # A head on its cell is the cell: one cell is translated, not regularised.
    still = applied.to_head(fields[:1], np.zeros((1, 3)), freqs, np)
    np.testing.assert_allclose(still, fields[0], atol=1e-12)
    with pytest.raises(ValueError, match="one cell or from two"):
        applied.to_head(np.zeros((3, 9, 2), dtype=complex), np.zeros((3, 3)), freqs, np)


def test_a_response_stored_by_the_low_band_library_is_rendered_at_its_own_level() -> None:
    """``spatial.lowband.to_stored`` in, the engine out: the 48 kHz response's low side."""
    pack = synthetic_free_field(
        level="B", source=(0.5, 1.5, 3.0), listener_start=(0, 1.5, 0), duration_s=0.5
    )
    source = pack.sources["s1"]
    assert source.low is not None
    # A wave response at 48 kHz: a few arrivals on 64 channels, band limited as a solve is.
    rng = np.random.default_rng(3)
    response = np.zeros((64, 57600))
    for at in (2400, 2700, 3500, 6000):
        response[:, at] = rng.standard_normal(64) / at
    spectrum = np.fft.rfft(response, axis=1)
    freqs = np.fft.rfftfreq(57600, 1.0 / FS)
    edges = np.clip((1700.0 - freqs) / 200.0, 0, 1)
    response = np.fft.irfft(spectrum * edges, 57600, axis=1)
    stored = to_stored(response, FS, pack.crossover)
    assert stored.shape == (64, 4800) and stored.dtype == np.float32
    # What the pack holds is what the library gives back at 48 kHz, to float32.
    low_side = from_stored(stored, FS)
    held = dataclasses.replace(source.low, ir=np.stack([stored, stored]))
    pack = dataclasses.replace(pack, sources={"s1": dataclasses.replace(source, low=held)})
    dry = band_noise(0.5, 100.0, 1400.0)
    got = Engine(pack, {"s1": dry}).stem("s1", parts=("low",))
    n = dry.size + 2 * FS
    want = np.fft.irfft(np.fft.rfft(dry, n)[None] * np.fft.rfft(low_side, n, axis=1), n)[
        :, : dry.size
    ]
    middle = slice(2400, 21000)
    assert np.abs(got - want)[:, middle].max() < 1e-5 * np.abs(want[:, middle]).max()
    # And that is the response's own level: the masks pass what lies under the ramp.
    under = np.fft.irfft(spectrum * edges * (freqs < 600.0), 57600, axis=1)
    quiet = band_noise(0.5, 150.0, 500.0)
    got = Engine(pack, {"s1": quiet}).stem("s1", parts=("low",))
    want = np.fft.irfft(np.fft.rfft(quiet, n)[None] * np.fft.rfft(under, n, axis=1), n)[
        :, : quiet.size
    ]
    level = 10 * np.log10((got[:, middle] ** 2).sum() / (want[:, middle] ** 2).sum())
    assert abs(level) < 0.05


def test_the_tail_reads_through_the_bank_at_the_energy_the_histogram_holds() -> None:
    pack = density_pack(duration_s=0.7, moving=False, bins=300)
    source = pack.sources["s1"]
    assert source.tail is not None
    energy = np.zeros_like(source.tail.energy)
    energy[:, 50:250] = 1e-4
    moments = np.zeros_like(source.tail.moments)
    moments[..., 0] = energy
    flat = dataclasses.replace(source.tail, energy=energy, moments=moments)
    pack = dataclasses.replace(
        pack,
        header=dataclasses.replace(pack.header, has_low=False),
        sources={"s1": dataclasses.replace(source, tail=flat, low=None)},
        air=dataclasses.replace(pack.air, enabled=False),
        mirror=dataclasses.replace(pack.mirror, signature=np.ones(1), lowcut_hz=0.0, lead_s=0.0),
    )
    click = np.zeros(int(0.1 * FS))
    click[1200] = 1.0
    engine = Engine(pack, {"s1": click})
    tail = engine.stem("s1", parts=("tail",))
    bands = octave_bank(FS).filters.shape[1]
    read = octave_filter_rows(np.repeat(tail[0:1], bands, 0), FS, np.arange(bands))
    got = np.sum(read**2, axis=1)
    # Every band from 250 Hz up reads what was asked within half a decibel, as the mirror's.
    np.testing.assert_allclose(10 * np.log10(got[1:7] / (200 * 1e-4)), 0.0, atol=0.5)
    # Nothing before the histogram's first bin, and channels as loud as each other.
    assert np.abs(tail[:, : 1200 + 50 * 96 - 300]).max() < 1e-12 * np.abs(tail).max()
    level = np.sum(tail**2, axis=1)
    assert np.all(np.abs(10 * np.log10(level[1:16] / level[0])) < 1.5)
    # One carrier a source: the same seed the same tail, another seed another.
    np.testing.assert_array_equal(Engine(pack, {"s1": click}).stem("s1", parts=("tail",)), tail)
    other = dataclasses.replace(pack.sources["s1"], tail_seed=source.tail_seed + 1)
    again = Engine(dataclasses.replace(pack, sources={"s1": other}), {"s1": click})
    assert not np.allclose(again.stem("s1", parts=("tail",)), tail)


# --------------------------------------------------------------------------
# the engine against itself
# --------------------------------------------------------------------------


def test_blocks_of_any_size_give_the_same_samples() -> None:
    pack = small_dense()
    dry = {"s1": noise(0.3)}
    runs = RenderSettings(chunk_steps=2)  # three runs in the scene: blocks cross their joins
    exact = Engine(pack, dry, settings=runs).render()
    whole = exact.astype(np.float32)
    assert np.abs(whole).max() > 0.0
    for size in (1000, 7777):
        blocks = list(Engine(pack, dry, settings=runs).blocks(size))
        assert all(block.shape[1] == size for block in blocks[:-1])
        np.testing.assert_array_equal(np.concatenate(blocks, axis=1), whole)
    # A seek is a slice: no state is carried from what precedes it.
    np.testing.assert_array_equal(
        Engine(pack, dry, settings=runs).render(5000, 9000), exact[:, 5000:9000]
    )
    # The run's own length is a setting, not a block size: the band's taper and the air
    # are filters on the run's transform, and another length moves the result by 2e-8.
    other = Engine(pack, dry).render()
    assert np.abs(other - exact).max() < 1e-7 * np.abs(exact).max()
    with pytest.raises(ValueError, match="inside the scene"):
        Engine(pack, dry).render(0, whole.shape[1] + 1)


def test_the_stems_add_up_to_the_mix_and_the_parts_to_the_stem() -> None:
    pack = small_dense(sources=2)
    dry = {"s1": noise(0.3, 1), "s2": noise(0.3, 2)}
    engine = Engine(pack, dry)
    mix = engine.render()
    stems = [engine.stem(name) for name in ("s1", "s2")]
    np.testing.assert_array_equal(stems[0] + stems[1], mix)
    assert not np.allclose(stems[0], stems[1])
    parts = sum(engine.stem("s1", parts=(part,)) for part in ("early", "low", "tail"))
    np.testing.assert_allclose(parts, stems[0], atol=1e-12 * np.abs(stems[0]).max())
    streamed = list(engine.stems(4800, sources=["s2"]))
    np.testing.assert_array_equal(
        np.concatenate([block["s2"] for block in streamed], axis=1), stems[1].astype(np.float32)
    )
    with pytest.raises(KeyError, match="holds no source"):
        engine.render(sources=["s9"])


_CHILD = """
import hashlib, sys
import numpy as np
from reverberate.render.benchmark import density_pack
from reverberate.render.engine import Engine
pack = density_pack(duration_s=0.3, moving=True, bins=30)
dry = np.random.default_rng(0).standard_normal(14400)
print(hashlib.sha256(Engine(pack, {"s1": dry}).render().tobytes()).hexdigest())
"""


def test_two_processes_give_one_output() -> None:
    here = hashlib.sha256(Engine(small_dense(), {"s1": noise(0.3)}).render().tobytes()).hexdigest()
    child = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert child.stdout.strip() == here


gpu = pytest.mark.skipif(not cuda_available(), reason="needs a CUDA device and cupy")


@gpu
def test_the_card_renders_what_the_host_renders_to_a_millionth_of_the_peak() -> None:
    """V4. The tail's noise is the same bits on both; the transforms round apart."""
    import cupy

    from reverberate.render import noise as generator

    np.testing.assert_array_equal(
        cupy.asnumpy(generator.carrier(7, np.arange(3), 5, 1000, cupy)),
        generator.carrier(7, np.arange(3), 5, 1000),
    )
    for moving in (False, True):
        pack = small_dense(moving)
        dry = {"s1": noise(0.3)}
        host = Engine(pack, dry, gpu=False).render()
        card = Engine(pack, dry, gpu=True).render()
        assert np.abs(card - host).max() <= 1e-6 * np.abs(host).max()
    free = synthetic_free_field(
        level="B", listener_start=(0, 1.5, 0), listener_end=(0.3, 1.5, 0), duration_s=0.4, fuse=True
    )
    dry = {"s1": noise(0.4)}
    host = Engine(free, dry, gpu=False).render()
    card = Engine(free, dry, gpu=True).render()
    assert np.abs(card - host).max() <= 1e-6 * np.abs(host).max()


# --------------------------------------------------------------------------
# dry audio, switches, files
# --------------------------------------------------------------------------


def test_clips_are_placed_at_their_times_with_their_gains() -> None:
    pack = synthetic_free_field(source=(2.0, 0, 0), listener_start=(0, 0, 0), duration_s=0.6)
    clip = noise(0.5, 3)
    recipe = json.loads(pack.recipe)
    recipe["sources"][0]["activity"] = [
        {
            "start_s": 0.1,
            "end_s": 0.3,
            "clip": {"library": "test", "name": "a", "sha256": "0" * 64},
            "clip_offset_s": 0.05,
            "gain_db": -6.0,
        }
    ]
    raw = (json.dumps(recipe, sort_keys=True, separators=(",", ":")) + "\n").encode()
    header = dataclasses.replace(pack.header, recipe_sha256=hashlib.sha256(raw).hexdigest())
    louder = dataclasses.replace(pack.sources["s1"], gain_db=6.0)
    pack = dataclasses.replace(pack, header=header, recipe=raw, sources={"s1": louder})
    asked: list[str] = []

    def load(named: Any) -> tuple[np.ndarray, float]:
        asked.append(named["name"])
        return clip, float(FS)

    out = Engine(pack, clips=load).render()
    placed = np.zeros(int(0.6 * FS))
    placed[4800:14400] = clip[2400:12000] * 10.0 ** (-6.0 / 20.0)  # the engine adds the source's
    np.testing.assert_allclose(out, Engine(pack, {"s1": placed}).render(), atol=1e-12)
    assert asked == ["a"] and np.abs(out).max() > 0.0
    assert np.abs(out[:, :4000]).max() < 1e-6 * np.abs(out).max()
    with pytest.raises(KeyError, match="no dry signal"):
        Engine(pack).render()
    short = DryTrack.from_recipe(recipe["sources"][0], lambda named: (clip[:1000], float(FS)))
    with pytest.raises(ValueError, match="shorter than its interval"):
        short.read(0, 24000)
    with pytest.raises(ValueError, match="mono"):
        DryTrack.from_array(np.zeros((2, 10)))


def test_the_directivity_switch_renders_the_same_pack_both_ways() -> None:
    pack = small_dense(False)
    dry = {"s1": noise(0.3)}
    as_packed = Engine(pack, dry).stem("s1", parts=("early",))
    omni = Engine(pack, dry, settings=RenderSettings(directivity=False)).stem(
        "s1", parts=("early",)
    )
    assert not np.allclose(as_packed, omni)
    off = dataclasses.replace(pack.sources["s1"], directivity_enabled=False)
    packed_off = dataclasses.replace(pack, sources={"s1": off})
    np.testing.assert_array_equal(Engine(packed_off, dry).stem("s1", parts=("early",)), omni)
    forced = Engine(packed_off, dry, settings=RenderSettings(directivity=True))
    np.testing.assert_array_equal(forced.stem("s1", parts=("early",)), as_packed)


def test_the_signal_is_written_in_blocks_and_read_back_by_range(tmp_path: Path) -> None:
    pack = small_dense(False)
    path = write_pack(tmp_path / "pack.h5", pack)
    dry = {"s1": noise(0.3)}

    with read_pack(path) as held:
        engine = Engine(held, dry)
        header = write_signal(
            tmp_path / "scene",
            engine.blocks(5000),
            sample_rate_hz=engine.sample_rate_hz,
            order=held.header.order,
            recipe_sha256=held.header.recipe_sha256,
            sources=list(held.sources),
        )
        whole = engine.render().astype(np.float32)
    # From the file's own tables the render is the one from memory's.
    np.testing.assert_array_equal(whole, Engine(pack, dry).render().astype(np.float32))
    signal = open_signal(tmp_path / "scene")
    assert signal.header == header
    assert (header["frames"], header["channels"], header["complete"]) == (14400, 64, True)
    assert header["peak"] == pytest.approx(float(np.abs(whole).max()))
    np.testing.assert_array_equal(signal.read(3000, 9000), whole[:, 3000:9000])
    data = (tmp_path / "scene.f32").read_bytes()
    assert hashlib.sha256(data).hexdigest() == header["sha256"]
    assert len(data) == 14400 * 64 * 4
    (tmp_path / "scene.f32").write_bytes(data[:-4])
    with pytest.raises(ValueError, match="not the size"):
        open_signal(tmp_path / "scene")


def test_the_command_line_states_the_interpolators_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["interpolator"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["1 kHz to 20 kHz"]["error_db"] < -93.0
