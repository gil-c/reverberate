"""category_from_filename is a pure function: no glb loading, no data/ needed."""

from __future__ import annotations

from reverberate.viz.mesh_viewer import SEMANTIC_UNLABELLED, category_from_filename


def test_multi_word_category_before_uuid() -> None:
    name = "Cabinet_Shelf_Desk_e314cd3c-e309-4614-98d2-13f99208ced8_4.glb"
    assert category_from_filename(name) == "Cabinet_Shelf_Desk"


def test_ceil_file_has_no_embedded_uuid_and_is_unlabelled() -> None:
    assert category_from_filename("ceil.glb") == SEMANTIC_UNLABELLED
