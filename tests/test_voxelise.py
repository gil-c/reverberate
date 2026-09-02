"""Tests for the voxelisation cache and for the files it replaces upstream.

The replacement is the delicate part. PFFDTD is a 2021 checkout this project
pins and edits in one place, and an edit installed over the wrong upstream, or
missed entirely, changes the geometry that reaches the solver without changing
anything a reader would look at.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from reverberate.wave import vendored
from reverberate.wave.voxelise import SceneSpec, ensure_patched


def make_spec(tmp_path: Path) -> SceneSpec:
    """A spec whose inputs exist, so ``key`` can actually be computed."""
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "mats_hash": {
                    "wall": {
                        "tris": [[0, 1, 2]],
                        "pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "sides": [2],
                        "color": [128, 128, 128],
                    }
                },
                "sources": [],
                "receivers": [],
            }
        )
    )
    mats = tmp_path / "mats"
    mats.mkdir()
    (mats / "wall.h5").write_bytes(b"not really an h5, but bytes are bytes")
    return SceneSpec(
        model_json=model,
        mat_folder=mats,
        mat_files={"wall": "wall.h5"},
        fmax=4000.0,
        ppw=10.5,
    )


def test_the_key_covers_the_grid_and_the_geometry(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    assert spec.key != SceneSpec(**{**spec.__dict__, "fmax": 8000.0}).key
    assert len(spec.key) == 32


class TestEnsurePatched:
    """The replacement is only safe while the pin it was derived from holds."""

    @staticmethod
    def _checkout(root: Path, contents: bytes) -> Path:
        target = root / "python" / "voxelizer" / "vox_scene.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(contents)
        return target

    def test_it_installs_our_copy_over_the_upstream_it_was_derived_from(
        self, tmp_path: Path
    ) -> None:
        upstream = tmp_path / "up.py"
        ours = vendored.patched_path("python/voxelizer/vox_scene.py").read_bytes()
        target = self._checkout(tmp_path / "pffdtd", b"whatever")
        # Stand in for the real upstream by declaring its digest.
        digest = hashlib.sha256(b"whatever").hexdigest()
        with mock.patch.dict(vendored.UPSTREAM_SHA256, {"python/voxelizer/vox_scene.py": digest}):
            written = ensure_patched(tmp_path / "pffdtd")
        assert written == ["python/voxelizer/vox_scene.py"]
        assert target.read_bytes() == ours
        assert not upstream.exists()

    def test_installing_twice_writes_once(self, tmp_path: Path) -> None:
        """Idempotent, so calling it before every voxelisation costs nothing."""
        ours = vendored.patched_path("python/voxelizer/vox_scene.py").read_bytes()
        self._checkout(tmp_path / "pffdtd", ours)
        assert ensure_patched(tmp_path / "pffdtd") == []

    def test_it_refuses_a_file_it_did_not_derive_from(self, tmp_path: Path) -> None:
        """A changed upstream is a merge nobody did, not a file to overwrite."""
        self._checkout(tmp_path / "pffdtd", b"upstream moved on")
        with pytest.raises(RuntimeError, match="neither the upstream"):
            ensure_patched(tmp_path / "pffdtd")

    def test_it_refuses_a_checkout_at_another_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "pffdtd"
        self._checkout(root, b"anything")
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--quiet",
                "-m",
                "not the pinned commit",
            ],
            check=True,
        )
        with pytest.raises(RuntimeError, match="derived from"):
            ensure_patched(root)


class TestPatchedVoxeliser:
    def test_the_replacement_seals_the_non_air_side(self) -> None:
        """The whole point of patch 5, asserted on the source we ship.

        Upstream takes the material away from the back side and leaves it
        adjacent to its neighbours, which is what makes the inside of a closed
        object a rigid cavity the air kernel keeps driving.
        """
        source = vendored.patched_path("python/voxelizer/vox_scene.py").read_text()
        assert "adj_bn[back_side,:] = False" in source
        assert "REVERBERATE PATCH 5" in source

    def test_the_voxeliser_is_part_of_the_cache_key(self, tmp_path: Path) -> None:
        """Otherwise an entry from before the patch answers for one after it."""
        spec = make_spec(tmp_path)
        before = spec.key
        with mock.patch.object(vendored, "UPSTREAM_COMMIT", "0" * 40):
            assert spec.key != before
