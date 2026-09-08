"""Tests for the census of air the solver seals off.

Sealing is correct and it is also destructive: the simulation stops carrying
sound through a region. The census exists so that never happens silently, so
what it must get right is the distinction between air that is sealed because
an object is closed and air that is sealed because something went wrong.
"""

from __future__ import annotations

from typing import Any, cast

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


def test_one_object_of_many_slivers_is_reported_as_one_object() -> None:
    """The page prints a count of this list in a sentence about objects.

    A carve on a 2 mm cell splits a christmas tree into thousands of closed
    twigs and a few thousand slivers under four faces, and one name per
    unjudged body made the viewer say "32 300 bodies are not closed" where the
    honest figure was 195 objects.
    """
    from reverberate.geometry.sealed import SealedReport

    report = SealedReport(unclosed=["christmas_tree_38"] * 2808 + ["decoration_3"] * 1749)
    record = report.record()
    assert record["unclosed_bodies"] == ["christmas_tree_38", "decoration_3"]
    assert record["unclosed_body_counts"] == {"christmas_tree_38": 2808, "decoration_3": 1749}
    assert "4557 bodies not closed over 2 objects" in report.summary()


def test_the_record_lists_the_cavities_that_matter_and_counts_the_rest() -> None:
    """The record is embedded in the run page and the page shows eight rows.

    A carve on a 2 mm cell splits a plant into thousands of closed leaves and
    every one is an interior: measured on this flat, 11 278 regions of which
    10 899 are under a tenth of a litre and hold 24.6 litres between them. The
    list took the report from 0.18 MB to 2.54 MB, so what is dropped has to be
    counted rather than merely absent.
    """
    from reverberate.geometry.sealed import REPORT_MIN_VOLUME_M3, SealedRegion, SealedReport

    big = SealedRegion(owner="wardrobe_1", volume_m3=0.9, extent_m=2.0, centroid=(0, 0, 0))
    crumbs = [
        SealedRegion(owner=f"plant_{n}", volume_m3=1e-5, extent_m=0.02, centroid=(0, 0, 0))
        for n in range(500)
    ]
    record = SealedReport(interiors=[*crumbs, big]).record()
    listed = cast(list[dict[str, Any]], record["interiors"])
    omitted = cast(dict[str, Any], record["interiors_omitted"])

    assert [row["owner"] for row in listed] == ["wardrobe_1"]
    assert record["interior_count"] == 501
    assert omitted["count"] == 500
    assert omitted["volume_m3"] == round(500 * 1e-5, 6)
    assert omitted["below_m3"] == REPORT_MIN_VOLUME_M3
    # The total is the total, whatever the list holds.
    assert record["sealed_volume_m3"] == round(0.9 + 500 * 1e-5, 6)
