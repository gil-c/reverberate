"""Tests for the outer envelope, and for the limits of absorption compensation.

The envelope exists because HSSD colliders are convex decompositions whose
triangle area counts faces buried between adjacent pieces, inflating one
apartment's furniture surface from 698 m² to 1316 m². The tests below check
that it removes that inflation, that it refuses to approximate objects it
would misrepresent, and that flat surfaces cost the two triangles they should.

The last group is the important one. Simulation showed that rescaling
absorption does **not** make a bad geometric approximation acoustically
equivalent, so these encode the boundary rather than the hope.
"""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra
import pytest
import trimesh

from reverberate.acoustics import MIN_WAVELENGTH
from reverberate.geometry.absorption import absorbing_power, compensate
from reverberate.geometry.decimation import decimate_within, deviation
from reverberate.geometry.envelope import (
    MAX_ENVELOPE_DEVIATION,
    acoustic_envelope,
)


def convex_decomposition_of_a_box(pieces: int = 8) -> trimesh.Trimesh:
    """A box built from overlapping convex slabs, as a collider file would be.

    The interior faces where the slabs meet are exactly the ones that inflate
    the raw area while being unreachable by sound.
    """
    slabs = []
    width = 2.0 / pieces
    for index in range(pieces):
        slab = trimesh.creation.box(extents=(width, 1.0, 1.0))
        slab.apply_translation([-1.0 + width * (index + 0.5), 0.0, 0.0])
        slabs.append(slab)
    combined = trimesh.util.concatenate(slabs)
    assert isinstance(combined, trimesh.Trimesh)
    return combined


def test_a_flat_panel_keeps_the_two_triangles_it_needs() -> None:
    """Section 5.3 bounds the smallest *feature* worth keeping. A plane has no
    features, so its cost must not grow with its size."""
    small = trimesh.creation.box(extents=(1.0, 0.05, 1.0))
    large = trimesh.creation.box(extents=(8.0, 0.05, 6.0))

    small_reduced, _ = decimate_within(small)
    large_reduced, _ = decimate_within(large)

    assert len(large_reduced.faces) == len(small_reduced.faces) == 12


def test_deviation_driven_reduction_stays_inside_the_physical_limit() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)

    reduced, error = decimate_within(sphere, max_deviation=MIN_WAVELENGTH / 2.0)

    assert len(reduced.faces) < len(sphere.faces)
    assert error <= MIN_WAVELENGTH / 2.0


def test_a_curved_object_costs_more_triangles_than_a_flat_one() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    panel = trimesh.creation.box(extents=(1.0, 0.05, 1.0))

    curved, _ = decimate_within(sphere)
    flat, _ = decimate_within(panel)

    assert len(curved.faces) > len(flat.faces)


def test_envelope_removes_the_buried_interior_area() -> None:
    """The measured defect: a decomposition claims area it does not expose."""
    decomposed = convex_decomposition_of_a_box(pieces=8)
    solid_area = trimesh.creation.box(extents=(2.0, 1.0, 1.0)).area

    envelope = acoustic_envelope(decomposed)

    assert decomposed.area > solid_area * 1.5
    assert envelope.area == pytest.approx(solid_area, rel=0.05)
    assert envelope.area_ratio < 0.8


def test_envelope_of_a_convex_object_is_faithful_and_cheap() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)

    envelope = acoustic_envelope(sphere)

    assert envelope.deviation <= MAX_ENVELOPE_DEVIATION
    assert len(envelope.mesh.faces) < len(sphere.faces)


def test_a_concave_object_is_not_silently_flattened() -> None:
    """A shelf's cavity is acoustically real. The envelope must either keep it
    or decline to approximate, never quietly fill it in."""
    panels = []
    for translation, extents in (
        ((0.0, 0.9, 0.0), (1.6, 0.05, 0.6)),
        ((0.0, 0.05, 0.0), (1.6, 0.05, 0.6)),
        ((-0.8, 0.5, 0.0), (0.05, 0.9, 0.6)),
        ((0.8, 0.5, 0.0), (0.05, 0.9, 0.6)),
    ):
        panel = trimesh.creation.box(extents=list(extents))
        panel.apply_translation(list(translation))
        panels.append(panel)
    shelf = trimesh.util.concatenate(panels)
    assert isinstance(shelf, trimesh.Trimesh)

    envelope = acoustic_envelope(shelf)

    filled_hull_area = shelf.convex_hull.area
    assert envelope.deviation <= MAX_ENVELOPE_DEVIATION
    assert not np.isclose(envelope.area, filled_hull_area, rtol=0.02)


def test_envelope_reports_what_it_did_rather_than_asserting_it() -> None:
    envelope = acoustic_envelope(convex_decomposition_of_a_box())

    text = envelope.summary()

    assert "deviation" in text
    assert envelope.original_faces > 0
    assert envelope.bodies >= 1


# --- the limits of compensation -------------------------------------------
#
# Measured by simulation on real HSSD colliders, max_order=1, 3000 rays, in a
# 4 x 2.6 x 3 m room. Compensation helps only while the geometry is already
# close; past that it saturates at alpha = 1 and overshoots badly:
#
#   deviation 7 cm,  x2.98 : RT60  +0.4 % uncompensated,  -1.2 % compensated
#   deviation 32 cm, x2.67 : RT60 -12.8 % uncompensated,  -7.2 % compensated
#   deviation 77 cm, x3.77 : RT60 -59.4 % uncompensated, -86.6 % compensated
#
# The third row is the one that matters: compensation made it *worse*, so it
# cannot be treated as a licence to approximate more aggressively.


def test_compensation_conserves_absorbing_power_only_until_it_caps() -> None:
    """The saturation point is where equivalence stops, so it must be visible."""
    material = pra.Material(0.4)
    gentle = compensate(material, original_area=10.0, reduced_area=9.0)
    severe = compensate(material, original_area=10.0, reduced_area=2.0)

    assert absorbing_power(gentle.material, 9.0) == pytest.approx(absorbing_power(material, 10.0))
    assert severe.capped
    assert absorbing_power(severe.material, 2.0) < absorbing_power(material, 10.0)


def test_a_faithful_envelope_needs_only_a_gentle_compensation() -> None:
    """The property the whole design leans on: geometric fidelity and a small
    compensation factor come together, so a large factor is a warning sign."""
    decomposed = convex_decomposition_of_a_box(pieces=8)
    envelope = acoustic_envelope(decomposed)

    entry = compensate(pra.Material(0.3), envelope.area, float(envelope.mesh.area))

    assert envelope.deviation <= MAX_ENVELOPE_DEVIATION
    assert entry.applied_factor < 1.5
    assert not entry.capped


def test_deviation_is_measured_against_the_original_surface() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    crude = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    assert deviation(sphere, sphere) == pytest.approx(0.0, abs=1e-6)
    assert deviation(sphere, crude) > MAX_ENVELOPE_DEVIATION
