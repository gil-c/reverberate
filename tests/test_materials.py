"""Tests for semantic label to pyroomacoustics material mapping (roadmap
section 5.5)."""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra

from reverberate.geometry.materials import material_for_label


def test_known_label_resolves_to_the_expected_material() -> None:
    rng = np.random.default_rng(0)
    material = material_for_label("wall", rng)
    assert isinstance(material, pra.Material)
    assert "plasterboard" in material.energy_absorption["description"].lower()


def test_substring_match_resolves_specific_labels_to_a_coarser_category() -> None:
    rng = np.random.default_rng(0)
    material = material_for_label("wall_clock", rng)
    assert (
        material.energy_absorption["coeffs"]
        == pra.Material("hard_surface").energy_absorption["coeffs"]
    )


def test_unknown_label_falls_back_deterministically_with_a_seeded_rng() -> None:
    material_a = material_for_label("some_never_seen_label", np.random.default_rng(42))
    material_b = material_for_label("some_never_seen_label", np.random.default_rng(42))
    assert material_a.energy_absorption["coeffs"] == material_b.energy_absorption["coeffs"]


def test_none_label_also_falls_back_without_raising() -> None:
    material = material_for_label(None, np.random.default_rng(0))
    assert isinstance(material, pra.Material)


def test_different_labels_can_resolve_to_different_materials() -> None:
    rng = np.random.default_rng(0)
    wall = material_for_label("wall", rng)
    window = material_for_label("window", rng)
    assert wall.energy_absorption["coeffs"] != window.energy_absorption["coeffs"]
