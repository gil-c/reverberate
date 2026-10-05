"""The audio chain, link by link, each against a truth that does not come from the chain.

``docs/open-questions/chain-audit.md`` is the audit these tests pin. A unit
source in free air reads ``1 / d`` at every frequency: that is the scale of a
pack, and every link between a path and a sample is held to it here on a
small exact case. Where the chain is wrong today and the remedy belongs to
another change, the test is marked ``xfail`` and its reason carries the error
measured, so the defect cannot hide behind a levelling scalar again.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.metrics import band_centres, octave_bank
from reverberate.mirror.direct import apply_signature, direct_energy, measure_signature
from reverberate.mirror.directivity import voice_v1
from reverberate.mirror.files import align_to_reference
from reverberate.mirror.hybrid import Crossover, blend, seam_db
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, paths_for
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import Histogram
from reverberate.mirror.render import (
    RenderSettings,
    band_pulse_energy,
    early_signals,
    tail_from_histogram,
)
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
from reverberate.spatial.sh import real_sh
from test_mirror_ism import RECEIVER, SOURCE, box_scene

RATE = 48000.0
C = 343.2


def db(ratio: float | np.ndarray) -> np.ndarray:
    return np.asarray(20.0 * np.log10(np.maximum(np.abs(ratio), 1e-30)))


def at(freqs: np.ndarray, hz: float) -> int:
    return int(np.argmin(np.abs(freqs - hz)))


def one_path(length_m: float, gain: float | None = None) -> Paths:
    """The direct sound of a unit source ``length_m`` away, from the front."""
    level = 1.0 / length_m if gain is None else gain
    return Paths(
        receiver=np.zeros(3),
        image=np.zeros(1, dtype=np.int32),
        order=np.zeros(1, dtype=int),
        length_m=np.array([length_m]),
        direction=np.array([[1.0, 0.0, 0.0]]),
        gain=np.full((1, len(OCTAVE_BANDS)), level),
        points=np.zeros((1, 5, 3)),
        sequence=np.full((1, 3), -1, dtype=int),
    )


# --------------------------------------------------------------------------
# 1. free field, the band above the crossover
# --------------------------------------------------------------------------


def test_the_bank_s_bands_add_to_one_in_pressure() -> None:
    """A pulse laid in every band at one gain is the pulse: the bands' sum is flat.

    Measured: within 0.02 dB from 354 Hz to 23.5 kHz, -0.19 dB at 177 Hz and
    -2.7 dB at 125 Hz, where the lowest band's filter is one bin of 512 wide;
    all of that is under the crossover. In power the same bands add to
    -2.7 dB at every band edge, which is the tail's ripple below.
    """
    kernels = np.asarray(octave_bank(int(RATE)).filters, dtype=float).T
    spectra = np.fft.rfft(kernels, 1 << 15, axis=1)
    freqs = np.fft.rfftfreq(1 << 15, 1.0 / RATE)
    heard = (freqs >= 354.0) & (freqs <= 23500.0)
    assert np.abs(db(spectra.sum(axis=0)[heard])).max() < 0.05
    edge = at(freqs, 1000.0 * np.sqrt(2.0))
    assert 10.0 * np.log10(np.sum(np.abs(spectra[:, edge]) ** 2)) == pytest.approx(-2.7, abs=0.1)


def test_a_unit_source_s_direct_sound_is_one_over_its_distance_at_every_frequency() -> None:
    """``early_signals`` of one path: the spectrum times the distance, 500 Hz to 20 kHz.

    Measured -0.06 dB at every frequency: the windowed sinc that lays a pulse
    between two samples sums to a little under one.
    """
    distance = 2.0
    settings = RenderSettings(order=0, duration_s=0.1)
    signals = early_signals(one_path(distance), settings, C)
    freqs = np.fft.rfftfreq(signals.shape[1], 1.0 / RATE)
    level = db(np.fft.rfft(signals[0]) * distance)
    heard = (freqs >= 500.0) & (freqs <= 20000.0)
    assert np.abs(level[heard]).max() < 0.1
    # And it stands where the path says: its largest sample on the path's own.
    assert int(np.argmax(np.abs(signals[0]))) == int(round(distance / C * RATE))


def _reference_field(path: Path, distances: np.ndarray, pulse: np.ndarray, lead_s: float) -> Path:
    """A field of direct sounds alone: ``FIELD_UNIT_AT_1M / d`` times ``pulse``, at ``d / c``."""
    samples = 4800
    centre = int(np.argmax(np.abs(pulse)))
    with h5py.File(path, "w") as handle:
        ir = handle.create_dataset("ir", shape=(distances.size, 1, samples), dtype=np.float32)
        for point, distance in enumerate(distances):
            first = int(round((distance / C + lead_s) * RATE)) - centre
            block = np.zeros(samples)
            block[first : first + pulse.size] = FIELD_UNIT_AT_1M / distance * pulse
            ir[point, 0] = block
        handle.create_dataset("direct_path_m", data=distances)
        handle.attrs["sample_rate_hz"] = RATE
    return path


def _dispersed_pulse(spread_s: float, size: int = 512) -> np.ndarray:
    """A pulse of flat spectrum whose high frequencies come late, as a grid's dispersion makes it.

    Unit magnitude at every frequency, so its level is that of a unit pulse;
    its group delay grows as the square of the frequency to ``spread_s`` at
    the Nyquist frequency.
    """
    freqs = np.fft.rfftfreq(size, 1.0 / RATE)
    nyquist = 0.5 * RATE
    phase = -2.0 * np.pi * spread_s * freqs**3 / (3.0 * nyquist**2)
    pulse = np.fft.irfft(np.exp(1j * phase), size)
    return np.asarray(np.roll(pulse, size // 4))


def _alignment_error_db(tmp_path: Path, pulse: np.ndarray) -> float:
    """The mirror's level at 1 kHz over the reference's, once aligned on it, in dB."""
    distances = np.linspace(0.8, 3.0, 6)
    reference = _reference_field(tmp_path / "S1.h5", distances, pulse, lead_s=0.01)
    with h5py.File(reference, "r") as handle:
        omni = [np.asarray(handle["ir"][p, 0], dtype=float) for p in range(distances.size)]
    taps, _ = measure_signature(omni, RATE)
    settings = RenderSettings(order=0, duration_s=0.1)
    energies = {}
    for point, distance in enumerate(distances):
        rendered = apply_signature(early_signals(one_path(float(distance)), settings, C), taps)
        energies[point] = direct_energy(rendered, RATE)
    alignment = align_to_reference(reference, energies, sound_speed_m_s=C)
    freqs = np.fft.rfftfreq(4096, 1.0 / RATE)
    signature = np.abs(np.fft.rfft(taps, 4096))[at(freqs, 1000.0)]
    return float(db(alignment.gain * signature / FIELD_UNIT_AT_1M))


def test_the_alignment_puts_the_mirror_on_a_reference_whose_pulse_is_a_pulse(
    tmp_path: Path,
) -> None:
    """A reference whose direct sound is one sample: gain times signature is its level."""
    pulse = np.zeros(64)
    pulse[16] = 1.0
    assert abs(_alignment_error_db(tmp_path, pulse)) < 0.3


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.files.align_to_reference reads a level as the energy of 0.5 ms round the "
        "direct sound, and mirror.direct.measure_signature gives the mirror a minimum phase "
        "pulse of unit energy: a reference whose pulse is spread in time holds less of its "
        "energy in that window than the mirror's, and the gain comes out low at every "
        "frequency. Measured here -2.7 dB at 1 kHz; on the validated field of hssd_0076 "
        "-2.35 dB at 1 kHz, -2.68 at 2 kHz, -2.61 at 4 kHz (126 points), which is the "
        "seam's median of +2.3 dB"
    ),
)
def test_the_alignment_puts_the_mirror_on_a_reference_whose_pulse_is_dispersed(
    tmp_path: Path,
) -> None:
    """The same level, the high frequencies up to 0.5 ms late: the mirror lands on it still."""
    assert abs(_alignment_error_db(tmp_path, _dispersed_pulse(0.0005))) < 0.3


# --------------------------------------------------------------------------
# the tail: what a histogram's energy becomes
# --------------------------------------------------------------------------


def _flat_tail() -> tuple[np.ndarray, np.ndarray, float]:
    """The tail of a histogram that holds one unit of energy a bin in every band.

    Returns the tail's power spectrum from 40 ms on, its frequencies, and
    what a white noise of one unit of energy a bin holds a frequency: the
    truth, whose reading through the bank is ``band_pulse_energy`` a bin,
    which is the scale the histogram is given.
    """
    bins, bin_s = 300, 0.002
    settings = RenderSettings(order=0, duration_s=bins * bin_s, tail_from_s=0.01)
    energy = np.ones((bins, len(OCTAVE_BANDS)))
    moments = np.zeros((bins, len(OCTAVE_BANDS), 16))
    moments[:, :, 0] = energy
    histogram = Histogram(
        energy=energy[None],
        moments=moments[None],
        hits=np.ones((1, bins), dtype=np.int64),
        bin_s=bin_s,
        bands_hz=tuple(OCTAVE_BANDS),
        order=3,
        rays=1,
    )
    tail, _ = tail_from_histogram(
        histogram,
        0,
        settings,
        sound_speed_m_s=C,
        start_s=0.0,
        seed=7,
        bursts=settings.tail_bursts,
        scale_per_band=np.asarray(band_pulse_energy(RATE)),
    )
    first = int(0.04 * RATE)
    held = np.asarray(tail[0, first:], dtype=float)
    freqs = np.fft.rfftfreq(held.size, 1.0 / RATE)
    per_sample = 1.0 / (bin_s * RATE)
    return np.abs(np.fft.rfft(held)) ** 2, freqs, held.size * per_sample


def test_the_tail_holds_the_energy_its_histogram_holds() -> None:
    """Over the bands above the crossover together, the tail is the truth's energy to 0.3 dB.

    Measured -0.19 dB on this draw, -0.14 dB in expectation.
    """
    power, freqs, truth = _flat_tail()
    heard = (freqs >= 707.0) & (freqs <= 11314.0)
    assert 10.0 * np.log10(power[heard].mean() / truth) == pytest.approx(0.0, abs=0.3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the tail is a noise a band, each through its band's filter, and the bands add in "
        "power to -2.7 dB at every edge; mirror.render.bank_reading's inverse then raises "
        "each band so that the bank reads what was asked. Of a flat histogram the tail is "
        "+1.0 dB at 1, 2, 4 and 8 kHz and -1.7 dB at 1.4, 2.8 and 5.7 kHz in expectation "
        "(+1.2, -2.1, +1.0, -2.3, +1.1, -2.0 dB on this draw): a comb of 2.7 dB an octave "
        "in every tail above the crossover"
    ),
)
def test_the_tail_of_a_flat_histogram_is_flat() -> None:
    """A sixth of an octave at the bands' centres and at their edges: the same level."""
    power, freqs, truth = _flat_tail()
    levels = []
    for centre in (1000.0, 1414.2, 2000.0, 2828.4, 4000.0, 5656.9):
        band = (freqs >= centre * 2.0 ** (-1 / 12)) & (freqs <= centre * 2.0 ** (1 / 12))
        levels.append(10.0 * np.log10(power[band].mean() / truth))
    assert float(np.max(levels) - np.min(levels)) < 1.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.render.tail_from_histogram and render.tail.TailPart start the tail "
        "tail_from_s = 10 ms after the first arrival and drop what the rays hold before: "
        "everything scattered, everything off a surface the tree does not mirror, and every "
        "order above 3 in those 10 ms. The first bounce's scattered part alone, integrated "
        "over the walls of a box of absorption 0.2: the reflections of the first 10 ms are "
        "0.3 to 1.5 dB short at scattering 0.2 and 0.9 to 3.1 dB short at 0.4, the shell's "
        "under the calibration in use (boxes of 4 x 3 x 2.5 m and 9 x 6 x 2.6 m, 1 to 6 m "
        "from the source; the tracer's own rays give -0.72 dB where the integral gives "
        "-0.74). On the validated field the mirror is 0.5 dB short of the wave field from "
        "1.5 to 10 ms in the median, over its 2.4 dB, and 4 to 5 dB at the ninth decile"
    ),
)
def test_what_the_rays_hold_before_the_tail_starts_is_rendered() -> None:
    """A histogram whose energy lies 2 to 8 ms after the direct sound: it must be heard."""
    bins, bin_s = 100, 0.002
    distance = 2.0
    settings = RenderSettings(order=0, duration_s=bins * bin_s)
    first = int(np.ceil(distance / C / bin_s)) + 1
    energy = np.zeros((bins, len(OCTAVE_BANDS)))
    energy[first : first + 3] = 1.0
    moments = np.zeros((bins, len(OCTAVE_BANDS), 16))
    moments[:, :, 0] = energy
    histogram = Histogram(
        energy=energy[None],
        moments=moments[None],
        hits=np.ones((1, bins), dtype=np.int64),
        bin_s=bin_s,
        bands_hz=tuple(OCTAVE_BANDS),
        order=3,
        rays=1,
    )
    tail, _ = tail_from_histogram(
        histogram,
        0,
        settings,
        sound_speed_m_s=C,
        start_s=distance / C,
        seed=1,
        bursts=settings.tail_bursts,
        scale_per_band=np.asarray(band_pulse_energy(RATE)),
    )
    assert float(np.sum(np.asarray(tail) ** 2)) > 0.0


# --------------------------------------------------------------------------
# 3. the split between the images and the rays
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.ism.IsmSettings.flutter_order = 6 renders the images of orders 4 to 6 "
        "between two facing walls, and MirrorSettings.traced_rays tells the rays to leave "
        "out orders 1 to max_order = 3 only: those reflections are in the images and in "
        "the histogram. In a 4 x 3 x 2.5 m box, 18 of 75 paths. By the box's own lattice of "
        "images they hold 0.5 to 3.8 per cent of the images' energy (two boxes, 1 to 6 m, "
        "scattering 0 to 0.4), counted twice: 0.02 to 0.16 dB of the early reflections"
    ),
)
def test_no_image_is_also_a_ray() -> None:
    """Every path the tree renders is of an order the rays were told to leave to it."""
    settings = MirrorSettings()
    scene = box_scene(alpha=0.2, scattering=0.0)
    ism = IsmSettings()
    paths = paths_for(scene, grow_tree(scene, SOURCE, ism), RECEIVER, ism)
    assert int(paths.order.max()) <= settings.traced_rays().skip_specular_order


# --------------------------------------------------------------------------
# 4. the join
# --------------------------------------------------------------------------


def _pulse_at(sample: int, samples: int = 24000, cut_hz: float | None = None) -> np.ndarray:
    """A unit pulse, or one band limited as a solve to ``cut_hz`` band limits its own."""
    freqs = np.fft.rfftfreq(samples, 1.0 / RATE)
    spectrum = np.exp(-2j * np.pi * freqs * sample / RATE)
    if cut_hz is not None:
        # ``accel.dsp.lowpass_sos`` forward and backward: Butterworth of order 4, squared.
        spectrum = spectrum / (1.0 + (freqs / cut_hz) ** 8)
    return np.asarray(np.fft.irfft(spectrum, samples))


def _level_round(signal: np.ndarray, sample: int, hz: float) -> float:
    """The spectrum of 10 ms round ``sample`` at ``hz``, under a raised cosine."""
    half = int(0.005 * RATE)
    piece = signal[sample - half : sample + half + 1] * np.hanning(2 * half + 1)
    freqs = np.fft.rfftfreq(4800, 1.0 / RATE)
    return float(np.abs(np.fft.rfft(piece, 4800))[at(freqs, hz)])


def test_the_two_bands_direct_sounds_add_to_the_direct_sound() -> None:
    """In the window the masks add to one in pressure: a shared pulse comes back to 0.01 dB."""
    pulse = _pulse_at(1000)[None, :]
    joined, record = blend(pulse, pulse, RATE, Crossover())
    assert record["seam_db"] == pytest.approx(0.0, abs=1e-6)
    for hz in (500.0, 850.0, 1000.0, 1189.0, 2000.0):
        ratio = _level_round(joined[0], 1000, hz) / _level_round(pulse[0], 1000, hz)
        assert float(db(ratio)) == pytest.approx(0.0, abs=0.01)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.hybrid.Crossover joins in pressure for coherent_s = 5 ms after the onset, "
        "fades over 5 ms, and joins in power after: an arrival both bands carry after "
        "10 ms is +3.0 dB at 1 kHz (+2.1 dB over the octave). On the validated field the "
        "two bands' correlation over 707 to 1414 Hz is 0.96 at the direct sound, 0.77 from "
        "5 to 10 ms, 0.41 from 10 to 20 ms, 0.19 from 20 to 50 ms and 0 after: +1.5 dB at "
        "1 kHz from 10 to 20 ms, +0.75 dB from 20 to 50 ms"
    ),
)
def test_a_reflection_both_bands_carry_comes_back_at_its_level() -> None:
    """The direct sound and one reflection 20 ms later, the same in both bands."""
    both = (_pulse_at(1000) + 0.5 * _pulse_at(1000 + 960))[None, :]
    joined, _ = blend(both, both, RATE, Crossover())
    ratio = _level_round(joined[0], 1960, 1000.0) / _level_round(both[0], 1960, 1000.0)
    assert abs(float(db(ratio))) < 0.5


def test_what_a_solve_to_1500_hz_loses_under_the_low_mask_is_a_quarter_of_a_decibel() -> None:
    """Heard through the low side's power mask, the solve's own band limit costs 0.2 dB."""
    samples = 24000
    freqs = np.fft.rfftfreq(samples, 1.0 / RATE)
    low_mask, _ = Crossover().masks(samples, RATE, power=True)
    band = (freqs >= 707.0) & (freqs <= 1415.0)
    kept = 1.0 / (1.0 + (freqs / 1500.0) ** 8) ** 2
    lost = 10.0 * np.log10(np.sum((low_mask**2 * kept)[band]) / np.sum((low_mask**2)[band]))
    assert lost == pytest.approx(-0.21, abs=0.05)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.hybrid.seam_db (trace.level.pair_seam_db) reads the wave side before its "
        "mask, over 707 to 1414 Hz with one weight for every frequency, and a pair is solved "
        "to 1500 Hz and band limited there: the band limit is read as a level. Measured "
        "here -0.9 dB for two bands on one scale; on 92 pairs of the first scene under "
        "0.9 m, -1.07 dB (-0.45 dB at 1 kHz, -1.4 at 1189 Hz, -6.0 at 1414 Hz), where the "
        "mask leaves 0.2 dB of it to be heard. The scene's seams are 0.9 dB low by it"
    ),
)
def test_the_seam_of_two_bands_on_one_scale_is_zero() -> None:
    """A pair solved to 1500 Hz and the mirror, both a unit source's direct sound."""
    low = _pulse_at(1000, cut_hz=1500.0)
    high = _pulse_at(1000)
    assert abs(seam_db(low, high, RATE, Crossover())) < 0.2


# --------------------------------------------------------------------------
# 2 and 6. what a level means at the ear, and a voice's axis
# --------------------------------------------------------------------------


def test_a_plane_wave_of_unit_pressure_is_one_on_the_first_channel() -> None:
    """N3D: channel 0 is the pressure, and the 64 channels hold 64 times its energy."""
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    harmonics = real_sh(7, axes)
    np.testing.assert_allclose(harmonics[:, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(harmonics**2, axis=1), 64.0, atol=1e-9)


def test_the_decoder_gives_the_ear_the_pressure_it_is_given() -> None:
    """Under 300 Hz a head is no obstacle: a unit plane wave is 0 dB at both ears.

    So full scale at the ear is full scale on channel 0, 86 dB SPL at 1 m by
    ``docs/formats/clip-library.md``, and the decode adds no gain of its own.
    Measured: -0.18 dB from the front, +0.22 and +0.06 dB from the left.
    """
    from reverberate.spatial.binaural import design_decoder
    from reverberate.spatial.hrtf import sphere_head

    order = 3
    decoder = design_decoder(
        sphere_head(RATE, 512, quadrature_degree=20),
        order=order,
        sample_rate_hz=RATE,
        filter_length=512,
    )
    spectra = np.fft.rfft(decoder.filters, 4096, axis=-1)
    freqs = np.fft.rfftfreq(4096, 1.0 / RATE)
    low = (freqs >= 100.0) & (freqs <= 300.0)
    for axis in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]):
        ears = np.einsum("c,ecf->ef", real_sh(order, np.array([axis]))[0], spectra)
        level = 10.0 * np.log10(np.mean(np.abs(ears[:, low]) ** 2, axis=1))
        assert np.abs(level).max() < 0.5
    # And a source on the left is louder at the left ear, ear 0, above 1 kHz.
    ears = np.einsum("c,ecf->ef", real_sh(order, np.array([[0, 1.0, 0]]))[0], spectra)
    high = (freqs >= 1000.0) & (freqs <= 4000.0)
    power = np.mean(np.abs(ears[:, high]) ** 2, axis=1)
    assert power[0] > 2.0 * power[1]


def test_a_voice_on_its_axis_is_above_the_clip_s_level_by_its_directivity() -> None:
    """Unit mean power, so the axis is over ``1 / d``: what the audit hands the owner.

    ``clip-library.md`` stores a voice at 60 dB SPL at 1 m, and the engine
    multiplies an arrival by the pattern, which is normalised to unit mean
    power: ahead of a talker the direct sound is 3.0 dB over that at 1 kHz
    and 6.3 dB at 8 kHz, while the band under the crossover is solved for an
    omnidirectional source and stays at 0 dB. A change of the normalisation
    moves these numbers and must say so.
    """
    axis = voice_v1().gain_db[:, 0]
    np.testing.assert_allclose(axis, [0.96, 1.41, 2.26, 3.04, 4.08, 5.26, 6.25], atol=0.01)
    assert band_centres(int(RATE))[3] == 1000


# --------------------------------------------------------------------------
# the mirror's colour above the crossover, and what a bounce returns
# --------------------------------------------------------------------------


def _band_limited_pulse(cut_hz: float, size: int = 256) -> np.ndarray:
    """A pulse flat to ``cut_hz`` and gone above it, as a field solved to there holds its own."""
    freqs = np.fft.rfftfreq(size, 1.0 / RATE)
    pulse = np.fft.irfft(1.0 / (1.0 + (freqs / cut_hz) ** 16), size)
    return np.asarray(np.roll(pulse, size // 4))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.direct.measure_signature gives the mirror the reference's direct spectrum, "
        "the band limit of its grid with it, and trace.assets.MirrorAssets carries that "
        "filter into a pack whose scale is physical and whose wave band ends at 1414 Hz: "
        "render.engine.SourceRenderer puts every dry signal through it (DryTrack.high). In "
        "the first scene's pack the mirror's direct sound is -2.4 dB re 1 / d at 1 kHz, "
        "-2.6 at 2 kHz, -2.5 at 4 kHz, -4.2 at 8 kHz, -5.7 at 12 kHz and -7.4 at 16 kHz: "
        "1.8 to 5 dB of treble that no physics took, over what the gain takes"
    ),
)
def test_the_mirror_s_direct_sound_is_as_flat_as_a_unit_source_s(tmp_path: Path) -> None:
    """A pack is physical: ``1 / d`` at 12 kHz as at 2 kHz, whatever grid the reference was on."""
    distances = np.linspace(0.8, 3.0, 6)
    reference = _reference_field(
        tmp_path / "S1.h5", distances, _band_limited_pulse(8000.0), lead_s=0.01
    )
    with h5py.File(reference, "r") as handle:
        omni = [np.asarray(handle["ir"][p, 0], dtype=float) for p in range(distances.size)]
    taps, _ = measure_signature(omni, RATE)
    freqs = np.fft.rfftfreq(4096, 1.0 / RATE)
    colour = db(np.abs(np.fft.rfft(taps, 4096)))
    assert abs(float(colour[at(freqs, 12000.0)] - colour[at(freqs, 2000.0)])) < 0.5


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.parameters.image_scene gives the images the class's scattering and "
        "image_absorption_scale, and apply_parameters gives the rays shell_scattering and "
        "absorption_scale: the rays leave the images (1 - alpha)(1 - 0.4) of a bounce on "
        "the shell and scatter the rest, and the images render (1 - alpha')(1 - 0.095) of "
        "it. Under the calibration in use (c3cec6aab28bb582, hssd_0076, the shell 975 of "
        "1070 m2) a bounce of orders 1 to 3 returns 1.17 of what it received at 1 kHz "
        "where the wall keeps 0.82: the images are 2.4 dB a bounce over what the rays left "
        "them (1.0 dB at 125 Hz, 2.7 dB at 8 kHz), and the fit's tail_gain_db and the tail's "
        "first 10 ms, which are dropped, take some of it back"
    ),
)
def test_a_bounce_returns_no_more_than_the_wall_keeps() -> None:
    """Images and rays share a bounce: what one renders is what the other left out."""
    from dataclasses import replace

    from reverberate.mirror.geometry import MaterialTable
    from reverberate.mirror.parameters import Parameters, apply_parameters, image_scene

    scene = box_scene(alpha=0.18, scattering=0.05)
    shell = MaterialTable(("shell",), scene.materials.absorption, scene.materials.scattering)
    scene = replace(scene, labels=("shell",), materials=shell)
    # The calibration in use, at 1 kHz, on every band.
    fitted = Parameters(
        absorption_scale=tuple(1.039 for _ in OCTAVE_BANDS),
        scattering_scale=1.906,
        shell_scattering=0.4,
        image_absorption_scale=tuple(0.362 for _ in OCTAVE_BANDS),
    )
    rays = apply_parameters(scene, fitted).materials
    images = image_scene(scene, fitted).materials
    kept = 1.0 - rays.absorption[0]
    scattered = kept * rays.scattering[0]
    mirrored = (1.0 - images.absorption[0]) * (1.0 - images.scattering[0])
    assert float(np.max((scattered + mirrored) / kept)) < 1.02


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mirror.render.render_point, trace.level.mirror_omni and the engine's tail take "
        "what a sphere of radius r catches of the direct rays as r^2 / (4 d^2), the small "
        "angle's value; a sphere catches (1 - sqrt(1 - r^2 / d^2)) / 2, which the tracer "
        "gives to 0.05 dB (40 000 rays). The tail of a source 0.25 m away is 0.97 dB too "
        "loud, 0.18 dB at 0.5 m, 0.04 dB at 1 m, with the 0.2 m sphere of every field"
    ),
)
def test_the_tail_s_scale_is_what_the_sphere_catches() -> None:
    """The rays' own direct energy at 0.25 m against what the renderer expects of it."""
    from reverberate.mirror.rays import RaySettings, trace

    scene = box_scene(alpha=0.2, scattering=0.2, size=np.array([6.0, 5.0, 4.0]))
    source = np.array([2.0, 2.5, 2.0])
    radius, distance = 0.2, 0.25
    histogram = trace(
        scene,
        source,
        source[None, :] + np.array([[distance, 0.0, 0.0]]),
        RaySettings(rays=3000, duration_s=0.002, bin_s=0.0001, receiver_radius_m=radius, order=0),
    )
    caught = float(histogram.energy[0, :, 3].sum())
    expected = radius**2 / (4.0 * max(distance, 1.05 * radius) ** 2)
    assert abs(10.0 * np.log10(caught / expected)) < 0.5


# --------------------------------------------------------------------------
# 6. the bottom of the spectrum
# --------------------------------------------------------------------------


def test_the_wave_band_is_whole_from_50_hz_and_gone_at_20() -> None:
    """The fit's integrator and low cut against a plain integrator: where the band starts.

    -3 dB at 40 Hz, -1 dB at 43.7 Hz, -0.14 dB at 50 Hz, nothing lost from
    63 Hz up; -17 dB at 31.5 Hz, -49 dB at 20 Hz. The pair cache of the first
    scene holds the same: its third octaves from 40 Hz up are level with the
    rest, 31.5 Hz 8 to 10 dB under and 20 Hz 40 dB under. Nothing in the
    chain cuts at 80 Hz but the stand-in of the next test.
    """
    from scipy.signal import sosfreqz

    from reverberate.accel import dsp
    from reverberate.accel.pairs import LOWCUT_HZ, LOWCUT_ORDER

    grid_rate = np.sqrt(3.0) * 10.5 * 1500.0
    sections = dsp.lowcut_sos(grid_rate, LOWCUT_HZ, LOWCUT_ORDER, differentiated=True)
    freqs = np.array([20.0, 31.5, 40.0, 50.0, 63.0, 80.0, 125.0, 250.0])
    _, response = sosfreqz(sections, worN=freqs, fs=grid_rate)
    level = db(np.abs(response) * 2.0 * np.pi * freqs)
    level -= level[-1]
    np.testing.assert_allclose(level[3:], 0.0, atol=0.2)
    assert level[2] == pytest.approx(-3.0, abs=0.2)
    assert level[1] == pytest.approx(-16.6, abs=1.0)
    assert level[0] < -45.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "trace.engines.FreeFieldPairs.response keeps nothing under 80 Hz and raises a "
        "cosine from 80 to 160 Hz (-6 dB at 120 Hz), and render.pack.synthetic_free_field "
        "does the same (SYNTHETIC_HIGHPASS_HZ): the stand-in for a solve starts an octave "
        "over the solve, whose band is whole from 50 Hz. A level read on a free field pack "
        "is short of what a solved pack holds from 40 to 160 Hz, and 'neither band renders "
        "under 80 Hz' (first-scene-defects.md) is true of the stand-in alone"
    ),
)
def test_the_free_field_stand_in_holds_the_band_a_solve_holds(tmp_path: Path) -> None:
    """A monopole 2 m away at 63 Hz: the level a solve's own low cut leaves it, -0.0 dB."""
    from reverberate.spatial.lowband import LOW_RATE_HZ, LOW_SAMPLES
    from reverberate.trace.engines import FreeFieldPairs

    pairs = FreeFieldPairs(np.array([[2.0, 0.0, 0.0]]), np.zeros((1, 3)), tmp_path, gain=1.0)
    omni = np.asarray(pairs.response(0, 0)[0], dtype=float)
    freqs = np.fft.rfftfreq(LOW_SAMPLES, 1.0 / LOW_RATE_HZ)
    spectrum = np.abs(np.fft.rfft(omni))
    assert float(db(spectrum[at(freqs, 63.0)] / spectrum[at(freqs, 400.0)])) > -1.0


def test_a_sealed_room_keeps_the_volume_its_source_gave_it() -> None:
    """Why the wave band has a low cut at all: the pressure zone of a room with no leak.

    The solver's unit source is a step of volume velocity: in free air its
    pressure is a pulse, ``1 / (4 pi d)``, and in a closed room the volume it
    goes on giving has nowhere to go, so the mean pressure climbs, by
    ``c^2 / V`` a second for a unit source, under the room's first
    resonance. That is the model's own answer and a sealed room's (cabin
    gain), not an error of the scheme; a dwelling leaks and its walls give,
    which the model does not hold, so what it computes down there is not a
    dwelling's. The plain integral of the solver's record shows it and the
    fit's low cut removes it. Measured in this box of 25 litres: the plain
    integral stands at 8 to 11 after 55 to 110 ms, where the direct pulse
    is under 1 for one step, and the low cut leaves a mean of 5e-5.
    """
    from scipy.signal import sosfilt

    import test_wave_lowband as wave
    from reverberate.accel import dsp
    from reverberate.wave.lowband.box import box_arrays
    from reverberate.wave.lowband.problem import build_problem
    from reverberate.wave.lowband.scheme import CARTESIAN
    from reverberate.wave.lowband.solver import drive_for, solve

    arrays, grid = box_arrays(CARTESIAN, (26, 22, 18), room=((4, 4, 4), (20, 17, 13)))
    source = np.array([[8.0, 9.0, 8.0]]) * grid.h
    nodes = np.array([wave.node(grid, s) for s in ((12, 11, 9), (18, 6, 6), (6, 15, 11))])
    problem = build_problem(arrays, wave.seeds_of(grid, source))
    steps = 2000
    record = solve(problem, drive_for(problem, grid, source, [nodes], steps * grid.Ts), np)
    record = np.asarray(record, dtype=float)
    plain = np.cumsum(record, axis=-1) * grid.Ts
    cut = sosfilt(dsp.lowcut_sos(1.0 / grid.Ts, 40.0, 8, differentiated=True), record)
    late = slice(steps // 2, steps)
    # The same at the three nodes, as a pressure with no wavelength is, and far over the pulse.
    held = plain[:, late].mean(axis=1)
    assert held.min() > 3.0 and held.max() / held.min() < 1.2
    assert abs(float(cut[:, late].mean())) < 0.02 * float(held.mean())
    # And it climbs as the volume says: c^2 / V a second for the unit source, within a third.
    volume = float(np.prod(np.array([17, 14, 10]) * grid.h))
    early = slice(steps // 8, steps // 2)
    slope = np.polyfit(np.arange(steps)[early] * grid.Ts, plain[:, early].mean(axis=0), 1)[0]
    assert slope == pytest.approx(C**2 / volume * grid.Ts, rel=0.35)
