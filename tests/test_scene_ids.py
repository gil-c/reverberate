"""Tests for this project's names for the HSSD scenes.

The table is a record, so what matters is that it stays one, and that a storey
suffix cannot be confused with the second half of a compound HSSD id -- which is
the whole reason the suffix lives on our names and not on theirs.
"""

from __future__ import annotations

import pytest

from reverberate.geometry.scene_ids import local_name, scene_names, scene_of


def test_the_table_gives_one_name_to_each_scene() -> None:
    """`scene_names` raises on a duplicate, so loading it is the assertion."""
    table = scene_names()

    assert len({entry.local for entry in table.values()}) == len(table)
    assert all(e.local.startswith("hssd_") and e.local[5:].isdigit() for e in table.values())


def test_a_storey_of_a_compound_id_survives_the_round_trip() -> None:
    """The trap: `105515175_173104107` already ends in a number, so a name built
    by appending to it could not be taken apart again."""
    name = local_name("105515175_173104107", 1)

    assert name == f"{local_name('105515175_173104107')}_1"
    assert scene_of(name) == ("105515175_173104107", 1)


def test_a_storey_a_scene_does_not_have_is_refused() -> None:
    """Answering with the base name would turn an off-by-one into a plausible
    name for the wrong volume."""
    with pytest.raises(ValueError, match="storeys"):
        local_name("102344022", 2)
