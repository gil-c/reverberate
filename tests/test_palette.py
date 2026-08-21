"""Unit tests for the shared category colour palette."""

from __future__ import annotations

from reverberate.viz.palette import color_for_category, muted_color_for_category


def test_color_for_category_is_deterministic() -> None:
    categories = ["chair", "sofa", "wall"]
    assert color_for_category("sofa", categories) == color_for_category("sofa", categories)


def test_color_for_category_differs_between_categories() -> None:
    categories = ["chair", "sofa", "wall"]
    colors = {color_for_category(c, categories) for c in categories}
    assert len(colors) == len(categories)


def test_muted_color_for_category_is_lighter_than_the_vivid_one() -> None:
    categories = ["chair", "sofa", "wall"]
    vivid = color_for_category("chair", categories)
    muted = muted_color_for_category("chair", categories)
    vivid_sum = sum(int(vivid[i : i + 2], 16) for i in (1, 3, 5))
    muted_sum = sum(int(muted[i : i + 2], 16) for i in (1, 3, 5))
    assert muted_sum > vivid_sum
