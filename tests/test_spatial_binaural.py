"""The head, and the decode through it, against the answers that are known.

The rigid sphere is analytic, so almost everything here is checked against a
closed form or against a published approximation rather than against itself.
The two failures this guards are both silent: swapped ears, which mirrors the
room and changes no level, and a truncated decode, which loses the interaural
level difference the consumer of this dataset separates voices with.

Synthetic, offline, a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.spatial.binaural import (
    covariance_correction,
    design_decoder,
    ild_db,
    interaural_coherence,
    itd_s,
    render,
)
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.hrtf import (
    EAR_AZIMUTH_DEG,
    HEAD_RADIUS_M,
    HrtfSet,
    ear_directions,
    project,
    sphere_hrtf,
    sphere_hrtf_sh,
    woodworth_itd_s,
)
from reverberate.spatial.sh import channel_count, directions, quadrature, real_sh

RATE = 48000.0
TAPS = 512


def a_head(quadrature_degree: int = 60) -> HrtfSet:
    grid, weights = quadrature(quadrature_degree)
    frequency = np.fft.rfftfreq(TAPS, 1.0 / RATE)
    sampled = sphere_hrtf(grid, frequency)
    return HrtfSet(sampled.responses, grid, frequency, sampled.description, weights=weights)


def a_plane_wave(order: int, azimuth_deg: float, samples: int = 4096) -> Ambisonic:
    direction = directions(np.array(np.radians(azimuth_deg)), np.array(0.0))
    signals = np.zeros((channel_count(order), samples))
    signals[:, 200] = real_sh(order, direction[None, :])[0]
    return Ambisonic(signals, RATE, order, np.zeros(3))


def test_the_sphere_is_transparent_at_zero_frequency() -> None:
    """A head small against the wavelength is not there, and the gain is one."""
    front = directions(np.array(0.0), np.array(0.0))[None, :]
    assert np.allclose(np.abs(sphere_hrtf(front, np.array([0.0])).responses[:, 0, 0]), 1.0)


def test_the_projected_sphere_matches_its_closed_form_wherever_it_is_audible() -> None:
    """The addition theorem gives ``H_nm`` exactly; the projection must agree."""
    order = 20
    head = a_head(quadrature_degree=2 * order + 40)
    numeric = project(head, order)
    closed = sphere_hrtf_sh(order, head.frequency_hz)
    audible = np.abs(closed) > 0.01 * np.abs(closed).max()
    assert np.max(np.abs(numeric - closed)[audible] / np.abs(closed)[audible]) < 1e-6


def test_the_left_ear_is_on_the_left() -> None:
    ears = ear_directions()
    assert ears[0, 1] > 0.9
    assert ears[1, 1] < -0.9
    assert np.allclose(ears[:, 2], 0.0)
    # Ears sit behind the coronal plane, which is what tells front from back.
    assert EAR_AZIMUTH_DEG > 90.0
    assert np.all(ears[:, 0] < 0.0)


def test_a_source_on_the_left_is_louder_and_earlier_at_the_left_ear() -> None:
    """Two silent flips at once: the Hankel kind and the ear order."""
    left = directions(np.array(np.pi / 2), np.array(0.0))[None, :]
    response = sphere_hrtf(left, np.array([4000.0])).responses[:, 0, 0]
    assert 20.0 * np.log10(abs(response[0]) / abs(response[1])) > 5.0


def test_the_shadow_deepens_with_frequency_and_vanishes_in_the_median_plane() -> None:
    left = directions(np.array(np.pi / 2), np.array(0.0))[None, :]
    frequency = np.array([250.0, 1000.0, 4000.0, 8000.0])
    response = sphere_hrtf(left, frequency).responses
    ild = 20.0 * np.log10(np.abs(response[0, 0]) / np.abs(response[1, 0]))
    assert np.all(np.diff(ild) > 0.0)
    assert ild[0] < 1.0
    assert ild[-1] > 15.0
    overhead = directions(np.array(0.0), np.array(0.6))[None, :]
    median = sphere_hrtf(overhead, np.array([4000.0])).responses[:, 0, 0]
    assert abs(median[0]) == pytest.approx(abs(median[1]), rel=1e-9)


def test_the_decoded_delay_has_the_sign_and_the_size_woodworth_predicts() -> None:
    """Positive is the left ear leading, and a source on the left must give that."""
    decoder = design_decoder(a_head(), order=7, sample_rate_hz=RATE, filter_length=TAPS)
    for azimuth in (90.0, 45.0, -90.0):
        measured = itd_s(render(a_plane_wave(7, azimuth), decoder), RATE)
        predicted = float(woodworth_itd_s(np.array(np.radians(azimuth))))
        assert np.sign(measured) == np.sign(predicted)
        assert abs(measured - predicted) < 150e-6
    ahead = itd_s(render(a_plane_wave(7, 0.0), decoder), RATE)
    assert abs(ahead) < 1.0 / RATE


def test_the_decode_is_exact_below_the_cut_on() -> None:
    """Order 7 is far more than a sphere needs under 1 kHz, so nothing may be lost."""
    head = a_head()
    decoder = design_decoder(head, order=7, sample_rate_hz=RATE, filter_length=TAPS)
    direction = directions(np.array(np.pi / 2), np.array(0.0))
    rendered = render(a_plane_wave(7, 90.0), decoder)
    spectrum = np.fft.rfft(rendered[:, : TAPS + 256], n=TAPS)
    reference = sphere_hrtf(direction[None, :], head.frequency_hz).responses[:, 0, :]
    band = (head.frequency_hz > 200.0) & (head.frequency_hz < 1000.0)
    error = 20.0 * np.log10(np.abs(spectrum[:, band]) / np.abs(reference[:, band]))
    # Not zero: the filter's ends are tapered so its impulse response stays
    # compact, and that taper costs a fraction of a decibel.
    assert np.max(np.abs(error)) < 0.3


def test_magnitude_least_squares_recovers_the_level_a_plain_fit_loses() -> None:
    """Above the cut-on the plain fit is several dB out; matching magnitude fixes it."""
    head = a_head()
    direction = directions(np.array(np.pi / 2), np.array(0.0))
    reference = sphere_hrtf(direction[None, :], head.frequency_hz).responses[:, 0, :]
    band = (head.frequency_hz > 4000.0) & (head.frequency_hz < 16000.0)
    errors = {}
    for name, cut_on in (("plain", 1e9), ("magls", 2000.0)):
        decoder = design_decoder(
            head, order=7, sample_rate_hz=RATE, filter_length=TAPS, magls_cut_on_hz=cut_on
        )
        spectrum = np.fft.rfft(render(a_plane_wave(7, 90.0), decoder)[:, : TAPS + 256], n=TAPS)
        errors[name] = float(
            np.mean(np.abs(20.0 * np.log10(np.abs(spectrum[:, band]) / np.abs(reference[:, band]))))
        )
    assert errors["magls"] < 0.5 * errors["plain"]


def test_a_truncated_decode_without_magnitude_matching_goes_quiet_in_a_diffuse_field() -> None:
    """The measured reason MagLS is on by default: 4.7 dB at order 7, 15.9 at order 1."""
    head = a_head()
    rng = np.random.default_rng(0)
    diffuse = rng.standard_normal((channel_count(7), 1 << 15))
    weights = head.weights
    assert weights is not None
    reference = (np.abs(head.responses[0]) ** 2 * weights[:, None]).sum(axis=0) / (4.0 * np.pi)
    band = (head.frequency_hz > 2000.0) & (head.frequency_hz < 16000.0)
    truth_db = 10.0 * np.log10(np.mean(reference[band]))

    levels = {}
    for name, cut_on in (("plain", 1e9), ("magls", 2000.0)):
        decoder = design_decoder(
            head, order=7, sample_rate_hz=RATE, filter_length=TAPS, magls_cut_on_hz=cut_on
        )
        rendered = render(Ambisonic(diffuse, RATE, 7, np.zeros(3)), decoder)
        spectrum = np.fft.rfft(rendered, axis=-1)
        grid = np.fft.rfftfreq(rendered.shape[1], 1.0 / RATE)
        inside = (grid > 2000.0) & (grid < 16000.0)
        levels[name] = 10.0 * np.log10(
            np.mean(np.abs(spectrum[0, inside]) ** 2) / rendered.shape[1]
        )
    assert levels["plain"] - truth_db < -3.0
    assert abs(levels["magls"] - truth_db) < 0.5


def test_the_covariance_correction_is_the_identity_when_there_is_nothing_to_correct() -> None:
    reference = np.array([[2.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.5]])
    assert np.allclose(covariance_correction(reference, reference), np.eye(2), atol=1e-6)


def test_the_covariance_correction_restores_the_covariance_it_is_given() -> None:
    reference = np.array([[2.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.5]])
    truncated = np.array([[0.9, 0.6 + 0.0j], [0.6 - 0.0j, 0.8]])
    matrix = covariance_correction(reference, truncated)
    assert np.allclose(matrix @ truncated @ matrix.conj().T, reference, atol=1e-6)


def test_turning_the_field_swaps_the_ears() -> None:
    """A source on the left, with the field turned half a turn, is a source on the right."""
    decoder = design_decoder(a_head(), order=7, sample_rate_hz=RATE, filter_length=TAPS)
    ambisonic = a_plane_wave(7, 90.0)
    straight = render(ambisonic, decoder)
    turned = render(ambisonic, decoder, field_yaw_rad=np.pi)
    assert ild_db(straight) == pytest.approx(-ild_db(turned), abs=0.05)
    assert itd_s(straight, RATE) == pytest.approx(-itd_s(turned, RATE), abs=1.0 / RATE)


def test_a_rotation_of_the_field_moves_the_image_by_the_same_angle() -> None:
    decoder = design_decoder(a_head(), order=7, sample_rate_hz=RATE, filter_length=TAPS)
    turned = render(a_plane_wave(7, 30.0), decoder, field_yaw_rad=np.radians(60.0))
    direct = render(a_plane_wave(7, 90.0), decoder)
    assert ild_db(turned) == pytest.approx(ild_db(direct), abs=0.05)


def test_a_decoder_and_a_response_must_agree_on_order_and_rate() -> None:
    decoder = design_decoder(a_head(), order=7, sample_rate_hz=RATE, filter_length=TAPS)
    with pytest.raises(ValueError, match="order"):
        render(a_plane_wave(3, 0.0), decoder)
    with pytest.raises(ValueError, match="Hz"):
        render(Ambisonic(np.zeros((64, 128)), 44100.0, 7, np.zeros(3)), decoder)


def test_a_head_sampled_off_the_decoder_s_grid_is_refused() -> None:
    head = a_head()
    with pytest.raises(ValueError, match="frequency grid"):
        design_decoder(head, order=3, sample_rate_hz=RATE, filter_length=256)


def test_uncorrelated_ears_read_as_incoherent_and_a_shared_signal_as_coherent() -> None:
    rng = np.random.default_rng(1)
    shared = rng.standard_normal(4800)
    _, coherent = interaural_coherence(np.stack([shared, shared]), RATE)
    _, incoherent = interaural_coherence(rng.standard_normal((2, 4800)), RATE)
    assert np.all(coherent > 0.99)
    assert np.mean(incoherent) < 0.35


def test_the_head_radius_and_ear_angle_are_the_documented_ones() -> None:
    assert pytest.approx(0.0875) == HEAD_RADIUS_M
    assert pytest.approx(100.0) == EAR_AZIMUTH_DEG
