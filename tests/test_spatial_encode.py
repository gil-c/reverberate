"""What the encoder recovers, measured against fields whose answer is known.

The expensive failure is a fit that returns a plausible field which is not the
one in the room, so every test here starts from an analytic field and asks for
the coefficients back. Synthetic, offline, seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.spatial.array import ArrayDesign, design_array, fibonacci_directions
from reverberate.spatial.encode import (
    EncoderSettings,
    conditioning,
    encode,
    encode_spectrum,
    numerical_wavenumber,
    shell_weights,
    well_posed,
)
from reverberate.spatial.field import plane_wave_coefficients
from reverberate.spatial.sh import acn, degrees_of, scene_to_ambisonic
from reverberate.wave.comms import Grid

SOUND_SPEED = 343.2
STEP = 0.0020428571428571427


def a_grid(nodes: int = 320, h: float = STEP) -> Grid:
    axis = np.arange(nodes) * h
    return Grid(
        h=h, Ts=h / SOUND_SPEED / np.sqrt(3.0), l2=1 / 3, fcc_flag=0, xv=axis, yv=axis, zv=axis
    )


def an_array(grid: Grid | None = None) -> ArrayDesign:
    grid = grid or a_grid()
    centre = np.array([grid.xv[len(grid.xv) // 2]] * 3)
    return design_array(centre, grid)


def test_the_array_is_made_of_grid_nodes_at_the_radii_it_claims() -> None:
    grid = a_grid()
    array = an_array(grid)
    for position in array.positions:
        for axis, value in zip((grid.xv, grid.yv, grid.zv), position, strict=True):
            assert np.min(np.abs(axis - value)) < 1e-12
    # Every node sits within one cell diagonal of its nominal shell.
    for shell, nominal in enumerate(array.nominal_radii):
        on_shell = array.radii[array.shell == shell]
        if shell == 0:
            continue
        assert np.all(np.abs(on_shell - nominal) < np.sqrt(3.0) * grid.h)


def test_fibonacci_directions_are_unit_vectors_that_cover_the_sphere() -> None:
    d = fibonacci_directions(200)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d.mean(axis=0), 0.0, atol=0.02)
    assert np.allclose(fibonacci_directions(1), [[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        fibonacci_directions(0)


def test_a_plane_wave_is_encoded_to_its_own_coefficients() -> None:
    """The plane wave convention: the first channel of a unit plane wave is one."""
    array = an_array()
    frequency = np.array([1000.0, 4000.0])
    k = 2.0 * np.pi * frequency / SOUND_SPEED
    direction = np.array([0.48, -0.6, 0.64])
    direction = direction / np.linalg.norm(direction)
    projection = scene_to_ambisonic(array.offsets) @ direction
    field = np.exp(1j * k[:, None] * projection[None, :])
    estimate = encode_spectrum(
        field.T,
        frequency,
        array,
        sound_speed_m_s=SOUND_SPEED,
        settings=EncoderSettings(dispersion="ideal"),
    )
    truth = plane_wave_coefficients(direction, 7) / (1j ** degrees_of(7))
    assert np.allclose(estimate[0, 0], 1.0, atol=1e-3)
    error = np.linalg.norm(estimate - truth[None, :], axis=1) / np.linalg.norm(truth)
    assert np.all(20.0 * np.log10(error) < -30.0)


def test_a_plane_wave_from_the_left_lands_in_the_left_channel() -> None:
    """A silent sign error mirrors the room; only a direction test catches it."""
    array = an_array()
    frequency = np.array([2000.0])
    k = 2.0 * np.pi * frequency / SOUND_SPEED
    # Scene coordinates: the listener faces +x, so scene -z is the ambisonic +y.
    from_left = scene_to_ambisonic(np.array([[0.0, 0.0, -1.0]]))[0]
    projection = scene_to_ambisonic(array.offsets) @ from_left
    field = np.exp(1j * k[:, None] * projection[None, :])
    estimate = encode_spectrum(
        field.T,
        frequency,
        array,
        sound_speed_m_s=SOUND_SPEED,
        settings=EncoderSettings(dispersion="ideal"),
    )[0]
    assert estimate[acn(1, -1)].real > 1.6
    assert abs(estimate[acn(1, 1)]) < 0.05
    assert abs(estimate[acn(1, 0)]) < 0.05


def test_a_near_field_monopole_is_recovered_to_order_seven_across_the_band() -> None:
    """The acceptance figure of the design: order 7, better than -30 dB, 1 to 16 kHz."""
    array = an_array()
    frequency = np.array([1000.0, 4000.0, 16000.0])
    report = conditioning(array, frequency, sound_speed_m_s=SOUND_SPEED)
    assert np.all(report.error_db[:, 7] < -30.0)
    assert np.all(report.effective_order == 7)


def test_below_a_few_hundred_hertz_the_high_orders_are_gone_and_the_report_says_so() -> None:
    """Physics, not a defect: at 125 Hz and 16 cm, ``k r`` is 0.37."""
    array = an_array()
    report = conditioning(array, np.array([125.0]), sound_speed_m_s=SOUND_SPEED)
    assert report.effective_order[0] <= 4
    assert report.error_db[0, 0] < -60.0


def test_widening_the_gate_to_the_fit_order_destroys_the_top_of_the_band() -> None:
    """The gate margin is the single most important number in the encoder."""
    array = an_array()
    frequency = np.array([16000.0])
    good = conditioning(array, frequency, sound_speed_m_s=SOUND_SPEED)
    wide = conditioning(
        array,
        frequency,
        sound_speed_m_s=SOUND_SPEED,
        settings=EncoderSettings(gate_margin=0.0),
    )
    assert wide.error_db[0, 7] > good.error_db[0, 7] + 15.0


def test_the_gate_drops_a_shell_the_fit_cannot_describe() -> None:
    settings = EncoderSettings(fit_order=10, gate_margin=4.0, gate_taper=2.0)
    radii = np.array([0.01, 0.05, 0.16])
    weights = shell_weights(radii, np.array([2.0 * np.pi * 16000.0 / SOUND_SPEED]), settings)[0]
    assert weights[0] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)
    assert np.all(np.diff(weights) <= 0.0)


def test_the_noise_gain_never_exceeds_the_cap() -> None:
    """Tikhonov at ``1 / (4 g^2)`` caps ``sigma / (sigma^2 + lambda)`` at ``g`` exactly."""
    settings = EncoderSettings(gain_cap_db=40.0)
    sigma = np.logspace(-8, 2, 2001)
    gain = sigma / (sigma**2 + settings.regularisation())
    assert gain.max() <= 10.0 ** (40.0 / 20.0) * (1.0 + 1e-9)


def test_the_numerical_wavenumber_is_slower_than_the_ideal_one_and_tends_to_it() -> None:
    """The scheme is 0.4 per cent slow at 16 kHz here, and exact as the step shrinks."""
    frequency = np.array([1000.0, 16000.0])
    ideal = 2.0 * np.pi * frequency / SOUND_SPEED
    coarse = numerical_wavenumber(frequency, STEP, SOUND_SPEED)
    fine = numerical_wavenumber(frequency, STEP / 8.0, SOUND_SPEED)
    assert np.all(coarse > ideal)
    assert coarse[1] / ideal[1] > 1.001
    assert np.all(np.abs(fine / ideal - 1.0) < np.abs(coarse / ideal - 1.0))


def test_the_filter_does_not_wrap_its_own_pre_ring_onto_the_end_of_the_record() -> None:
    """One transform of a whole record convolves circularly, and that lands badly.

    The encoder is a filter with an almost symmetric impulse response, so half
    of what it puts around an event comes *before* it. Taken over the whole
    record at once, that half comes out at the **end** instead: the rendered
    bedroom carried the direct sound's pre-ring in its last 20 ms at about
    -33 dB of the whole response, which held the Schroeder curve flat for the
    entire tail and sounded like a faint copy half a second late.
    """
    array = an_array()
    samples = 2048
    pressure = np.zeros((array.count, samples))
    pressure[:, 0] = 1.0

    signals = encode(
        pressure,
        48000.0,
        array,
        sound_speed_m_s=SOUND_SPEED,
        settings=EncoderSettings(order=3, fit_order=5),
    ).signals[0]

    eighth = samples // 8
    head = float(signals[:eighth] @ signals[:eighth])
    tail = float(signals[-eighth:] @ signals[-eighth:])
    # Circularly, the last eighth mirrors the first and reads within a few dB
    # of it. It has to be far below instead.
    assert 10.0 * np.log10(tail / head) < -30.0


def test_a_gate_that_closes_on_everything_is_reported_rather_than_silent() -> None:
    """The trap: a margin tuned at fit order ten, subtracted from a small one.

    At a fit order of five the gate sits at ``k r = 1``, which at 16 kHz is a
    radius of 1.1 mm, under one grid cell. The fit is then determined by its
    regularisation and not by the field, and the only symptom is a direction of
    arrival that quietly drifts by degrees.
    """
    array = an_array()
    frequency = np.array([1000.0, 16000.0])
    k = 2.0 * np.pi * frequency / SOUND_SPEED
    healthy = well_posed(array, frequency, k, EncoderSettings(order=7, fit_order=10))
    assert healthy["under_determined_hz"] == []
    assert healthy["least_admitted"] > healthy["unknowns"]

    starved = well_posed(array, frequency, k, EncoderSettings(order=3, fit_order=5))
    assert starved["under_determined_hz"] == [16000.0]
    assert starved["least_admitted"] < starved["unknowns"]
