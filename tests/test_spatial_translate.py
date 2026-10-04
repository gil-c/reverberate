"""The 64 channel translation and fusion against fields known in closed form, and the cell rule.

A field of plane waves has exact order 7 coefficients about any point: a
wave from ``s`` of amplitude ``a`` is ``a Y_c(s) exp(i k s . x)`` in channel
``c`` of an expansion centred on ``x``. Nothing is solved and nothing is read.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.spatial.sh import channel_count, degrees_of, real_sh, scene_to_ambisonic
from reverberate.spatial.translate import (
    FUSE_WITHIN_M,
    MODE_EXACT,
    MODE_FUSED,
    MODE_TRANSLATED,
    REGULARISATION,
    SOUND_SPEED_M_S,
    cell_stride,
    choose_cells,
    clearance_m,
    fusion_inverse,
    fusion_operator,
    fusion_weights,
    kernels,
    namespace_of,
    serving_radius_m,
    translation_operator,
    translation_weights,
)

ORDER = 7


class PlaneWaves:
    """A few plane waves in scene coordinates, and their coefficients about any point."""

    def __init__(self, count: int = 5, seed: int = 44) -> None:
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(count, 3))
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.amplitudes = rng.uniform(0.3, 1.0, size=count)
        self.basis = real_sh(ORDER, scene_to_ambisonic(self.directions))

    def coefficients(self, point: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """``[frequency, channel]`` of the order 7 expansion centred on ``point``."""
        k = 2 * np.pi * freqs_hz / SOUND_SPEED_M_S
        waves = self.amplitudes[None, :] * np.exp(
            1j * k[:, None] * (self.directions @ point)[None, :]
        )
        return np.asarray(waves @ self.basis)


def frequency_of(kd: float, distance_m: float) -> np.ndarray:
    return np.array([kd * SOUND_SPEED_M_S / (2 * np.pi * distance_m)])


def error_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(estimate - truth) ** 2) / np.sum(np.abs(truth) ** 2)))


class TestTranslationOperator:
    field = PlaneWaves()
    centre = np.array([0.3, 1.7, -0.2])
    offset = np.array([0.12, -0.05, 0.38])  # about 0.40 m, off every axis

    def test_it_is_a_matrix_per_frequency_from_64_channels_to_64(self) -> None:
        operator = translation_operator(self.offset, np.array([100.0, 200.0, 400.0]), ORDER)
        assert operator.shape == (3, channel_count(ORDER), channel_count(ORDER))
        assert operator.dtype == np.complex64

    @pytest.mark.parametrize("kd", [0.5, 3.6, 7.3])
    def test_its_first_row_is_the_pressure_s_closed_form(self, kd: float) -> None:
        freqs = frequency_of(kd, float(np.linalg.norm(self.offset)))
        row = translation_operator(self.offset, freqs, ORDER)[0, 0]
        assert np.abs(row - translation_weights(self.offset, freqs, ORDER)[0]).max() < 1e-6

    def test_no_offset_is_the_identity(self) -> None:
        operator = translation_operator(np.zeros(3), np.array([500.0]), ORDER)[0]
        assert np.abs(operator - np.eye(channel_count(ORDER))).max() < 1e-6

    def test_the_lower_degrees_of_a_field_are_reproduced_at_the_new_centre(self) -> None:
        """Output degree ``n`` needs the input up to ``n + k d``: at ``k d = 1``, degree 3 holds."""
        freqs = frequency_of(1.0, float(np.linalg.norm(self.offset)))
        moved = np.einsum(
            "fce,fe->fc",
            translation_operator(self.offset, freqs, ORDER),
            self.field.coefficients(self.centre, freqs),
        )
        truth = self.field.coefficients(self.centre + self.offset, freqs)
        low = degrees_of(ORDER) <= 3
        assert error_db(moved[:, low], truth[:, low]) < -50.0
        assert error_db(moved, truth) < -10.0, "the top degrees are cut, not lost"

    def test_the_offset_is_from_the_centre_to_the_target(self) -> None:
        freqs = frequency_of(1.0, float(np.linalg.norm(self.offset)))
        there = self.field.coefficients(self.centre, freqs)
        truth = self.field.coefficients(self.centre + self.offset, freqs)[:, :16]
        backwards = np.einsum(
            "fce,fe->fc", translation_operator(-self.offset, freqs, ORDER), there
        )[:, :16]
        assert error_db(backwards, truth) > -10.0

    def test_a_batch_of_offsets_is_each_offset_s_operator(self) -> None:
        offsets = np.stack([self.offset, -0.5 * self.offset, np.array([0.0, 0.1, 0.0])])
        freqs = np.array([250.0, 900.0])
        batch = translation_operator(offsets, freqs, ORDER)
        assert batch.shape[:2] == (3, 2)
        for row, offset in zip(batch, offsets, strict=True):
            # To rounding: a batch and a single offset do not take the same path through BLAS.
            assert np.abs(row - translation_operator(offset, freqs, ORDER)).max() < 1e-6

    def test_the_namespace_is_the_arguments_own_unless_one_is_given(self) -> None:
        assert namespace_of(np.zeros(3)) is np
        assert namespace_of(np.zeros(3), xp="given") == "given"


class TestFusionOperator:
    field = PlaneWaves()
    target = np.array([0.3, 1.7, -0.2])
    step = np.array([0.0, 0.0, 0.30])

    def cells(self, freqs: np.ndarray) -> np.ndarray:
        below = self.field.coefficients(self.target - self.step, freqs)
        above = self.field.coefficients(self.target + self.step, freqs)
        return np.concatenate([below, above], axis=1)

    def test_one_cell_fused_is_its_translation_over_one_plus_lambda(self) -> None:
        freqs = np.array([300.0, 1000.0])
        one = fusion_operator(self.step[None, :], freqs, ORDER)
        moved = translation_operator(self.step, freqs, ORDER)
        assert one.shape == (2, 64, 64)
        assert np.abs(one * (1 + REGULARISATION) - moved).max() < 1e-5

    def test_two_cells_either_side_hold_where_one_has_failed(self) -> None:
        """At ``k d = 6`` the pair gives the low degrees and one translation does not."""
        freqs = frequency_of(6.0, 0.30)
        offsets = np.stack([self.step, -self.step])  # the target seen from each cell
        fused = np.einsum("fce,fe->fc", fusion_operator(offsets, freqs, ORDER), self.cells(freqs))
        alone = np.einsum(
            "fce,fe->fc",
            translation_operator(self.step, freqs, ORDER),
            self.field.coefficients(self.target - self.step, freqs),
        )
        truth = self.field.coefficients(self.target, freqs)
        low = degrees_of(ORDER) <= 2
        assert error_db(fused[:, low], truth[:, low]) < -20.0
        assert error_db(alone[:, low], truth[:, low]) > error_db(fused[:, low], truth[:, low]) + 10

    def test_the_inverse_is_factored_once_and_serves_every_target(self) -> None:
        freqs = np.array([400.0, 800.0])
        cells = np.stack([-self.step, self.step])  # about the first target
        inverse = fusion_inverse(cells, freqs, ORDER)
        targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.05, 0.0, -0.1]])
        offsets = targets[:, None, :] - cells[None, :, :]
        batch = fusion_operator(offsets, freqs, ORDER, inverse=inverse)
        assert batch.shape == (3, 2, 64, 128)
        for row, offset in zip(batch, offsets, strict=True):
            assert np.abs(row - fusion_operator(offset, freqs, ORDER)).max() < 1e-5

    def test_a_batch_without_the_inverse_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fusion_inverse"):
            fusion_operator(np.zeros((2, 2, 3)), np.array([400.0]), ORDER)

    def test_the_pressure_weights_are_the_operator_s_first_row(self) -> None:
        freqs = np.array([300.0, 900.0])
        offsets = np.stack([self.step, -self.step])
        weights = fusion_weights(offsets, freqs, ORDER)
        operator = fusion_operator(offsets, freqs, ORDER)
        assert weights.shape == (2, 2, 64) and weights.dtype == np.complex64
        assert np.abs(weights.reshape(2, 128) - operator[:, 0, :]).max() < 1e-6


class TestKernels:
    def test_a_kernel_s_transform_is_the_operator_and_its_centre_is_time_zero(self) -> None:
        taps, rate = 32, 4000.0
        freqs = np.fft.rfftfreq(taps, 1 / rate)
        operator = translation_operator(np.array([0.0, 0.0, 0.1]), freqs, ORDER)
        kernel = kernels(operator, taps)
        assert kernel.shape == (64, 64, taps) and kernel.dtype == np.float32
        back = np.fft.rfft(np.roll(kernel, -taps // 2, axis=-1), axis=-1)
        # The bin at Nyquist of a real kernel is real; the band under it is the operator.
        assert np.abs(np.moveaxis(back, -1, 0)[:-1] - operator[:-1]).max() < 1e-5

    def test_no_offset_is_one_tap_at_the_centre(self) -> None:
        taps = 16
        operator = translation_operator(np.zeros(3), np.fft.rfftfreq(taps, 1 / 4000.0), ORDER)
        kernel = kernels(operator, taps)
        assert kernel[5, 5, taps // 2] == pytest.approx(1.0, abs=1e-5)
        assert np.abs(np.delete(kernel[5, 5], taps // 2)).max() < 1e-5

    def test_frequencies_that_are_not_the_taps_transform_are_refused(self) -> None:
        with pytest.raises(ValueError, match="taps"):
            kernels(np.zeros((9, 4, 4), dtype=complex), 32)


class TestClearance:
    square = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )

    def test_over_a_face_beside_an_edge_and_past_a_corner(self) -> None:
        points = np.array([[0.5, 0.7, 0.5], [0.5, 0.0, -0.3], [-0.3, 0.4, -0.4]])
        expected = [0.7, 0.3, np.sqrt(0.09 + 0.16 + 0.16)]
        assert clearance_m(points, self.square) == pytest.approx(expected, abs=1e-12)

    def test_the_cull_changes_nothing(self) -> None:
        rng = np.random.default_rng(3)
        triangles = rng.uniform(-2, 2, size=(400, 3, 3))
        triangles[::7] *= 3.0  # some wide ones, which are never culled
        points = rng.uniform(-2, 2, size=(20, 3))
        everything = clearance_m(points, triangles, large_m=0.0)
        assert np.abs(clearance_m(points, triangles, large_m=0.25) - everything).max() < 1e-12

    def test_triangles_of_another_shape_are_refused(self) -> None:
        with pytest.raises(ValueError, match="triangle"):
            clearance_m(np.zeros((1, 3)), np.zeros((4, 3)))


class TestChoosingCells:
    cells = np.array([[0.0, 1.7, 0.0], [0.4, 1.7, 0.0], [0.8, 1.7, 0.0]])
    free = np.full(3, 1.0)

    def test_a_head_on_a_cell_is_exact(self) -> None:
        assert choose_cells(np.array([0.4, 1.7, 0.0004]), self.cells, self.free) == (
            MODE_EXACT,
            (1, -1),
        )

    def test_a_head_between_two_cells_is_fused_from_the_nearer_first(self) -> None:
        assert choose_cells(np.array([0.25, 1.7, 0.0]), self.cells, self.free) == (
            MODE_FUSED,
            (1, 0),
        )

    def test_a_second_cell_on_the_same_side_is_not_taken(self) -> None:
        """Past the last cell there is one cell to read, however near the next one is."""
        mode, slots = choose_cells(np.array([-0.1, 1.7, 0.0]), self.cells, self.free)
        assert (mode, slots) == (MODE_TRANSLATED, (0, -1))

    def test_a_cell_is_not_read_past_its_serving_radius(self) -> None:
        """The far cell may not serve: the head is read from the near one alone."""
        radius = np.array([1.0, 0.1, 1.0])
        assert choose_cells(np.array([0.2, 1.7, 0.0]), self.cells, radius) == (
            MODE_TRANSLATED,
            (0, -1),
        )

    def test_a_head_no_cell_may_serve_is_an_error_that_says_how_far(self) -> None:
        with pytest.raises(LookupError, match=r"0\.2\d\d m away"):
            choose_cells(np.array([0.2, 1.7, 0.2]), self.cells, np.full(3, 0.1))
        with pytest.raises(LookupError):
            choose_cells(np.array([-FUSE_WITHIN_M - 0.01, 1.7, 0.0]), self.cells, self.free)

    def test_the_serving_radius_is_a_share_of_the_clearance_and_of_the_source_s_distance(
        self,
    ) -> None:
        clearance = np.array([0.6, 0.6, 0.2])
        source = np.array([0.4, 1.7, 1.0])
        radius = serving_radius_m(self.cells, clearance, source)
        assert radius == pytest.approx([0.15 * np.sqrt(1.16), 0.15, 0.1])
        assert serving_radius_m(self.cells, clearance) == pytest.approx([0.3, 0.3, 0.1])
        both = serving_radius_m(self.cells, clearance, np.stack([source, self.cells[0] + 0.1]))
        assert both[0] == pytest.approx(0.15 * np.sqrt(0.03))

    def test_a_far_source_reads_every_fourth_cell_of_a_line_and_a_near_one_all(self) -> None:
        assert cell_stride(2.0, 0.15) == 4
        assert cell_stride(1.0, 0.15) == 2
        assert cell_stride(0.5, 0.15) == 1
        assert cell_stride(0.3, 0.15) == 1, "never less than every cell"
        assert cell_stride(10.0, 0.15) == 4, "two cells are fused 0.30 m away at most"
