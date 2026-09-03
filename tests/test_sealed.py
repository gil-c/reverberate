"""Tests for the census of air the solver seals off.

Sealing is correct and it is also destructive: the simulation stops carrying
sound through a region. The census exists so that never happens silently, so
what it must get right is the distinction between air that is sealed because
an object is closed and air that is sealed because something went wrong.
"""

from __future__ import annotations

import pyroomacoustics as pra
import pytest
import trimesh

from reverberate.geometry.pra_room import MeshMaterialAssignment
from reverberate.geometry.sealed import sealed_regions


def obstacle(mesh: trimesh.Trimesh, name: str = "cabinet_0") -> MeshMaterialAssignment:
    return MeshMaterialAssignment(mesh=mesh, material=pra.Material(0.1), name=name)


class TestSealedRegions:
    def test_a_closed_body_seals_its_own_volume(self) -> None:
        box = trimesh.creation.box(extents=(0.5, 0.4, 0.2))
        report = sealed_regions([obstacle(box)])

        assert len(report.interiors) == 1
        assert report.sealed_volume_m3 == pytest.approx(0.5 * 0.4 * 0.2)
        assert report.interiors[0].owner == "cabinet_0"
        assert report.unclosed == []

    def test_each_body_of_one_obstacle_is_counted_apart(self) -> None:
        """A chair is many closed bodies, and each is its own cavity."""
        left = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        right = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        right.apply_translation([1.0, 0.0, 0.0])
        both = trimesh.util.concatenate([left, right])
        assert isinstance(both, trimesh.Trimesh)

        report = sealed_regions([obstacle(both, "seat_3")])

        assert len(report.interiors) == 2
        assert {r.owner for r in report.interiors} == {"seat_3"}

    def test_the_shell_is_never_sealed(self) -> None:
        """Its interior is the room; sealing it would seal the simulation."""
        box = trimesh.creation.box(extents=(4.0, 2.5, 3.0))
        report = sealed_regions([obstacle(box, "shell_wall")])

        assert report.interiors == []
        assert report.unclosed == []

    def test_an_open_body_is_reported_not_assumed(self) -> None:
        """Its inside cannot be told from its outside, so nothing is claimed."""
        open_sheet = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
        report = sealed_regions([obstacle(open_sheet, "curtain_2")])

        assert report.interiors == []
        assert report.unclosed == ["curtain_2"]

    def test_the_first_mode_says_whether_it_would_have_been_audible(self) -> None:
        """A cavity of side L rings at c/2L; that is why size is worth quoting."""
        box = trimesh.creation.box(extents=(1.286, 0.5, 0.5))
        region = sealed_regions([obstacle(box)]).interiors[0]

        assert region.extent_m == pytest.approx(1.286)
        assert region.first_mode_hz == pytest.approx(133.0, abs=1.0)

    def test_the_record_is_ordered_largest_first(self) -> None:
        """The viewer and a reader both want the ones that would have boomed."""
        small = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        big = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        big.apply_translation([5.0, 0.0, 0.0])
        both = trimesh.util.concatenate([small, big])
        assert isinstance(both, trimesh.Trimesh)

        report = sealed_regions([obstacle(both)])
        record = report.record()
        volumes = [region.volume_m3 for region in report.interiors]
        listed = record["interiors"]
        assert isinstance(listed, list)
        volumes = [entry["volume_m3"] for entry in listed]

        assert volumes == sorted(volumes, reverse=True)
        assert record["sealed_volume_m3"] == pytest.approx(1.001, abs=1e-6)
