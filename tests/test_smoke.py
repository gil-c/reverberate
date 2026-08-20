"""Smoke test that the package imports and the test suite itself runs."""

from __future__ import annotations

import reverberate


def test_package_imports() -> None:
    assert reverberate is not None
