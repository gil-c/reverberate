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
from dataclasses import replace
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
    assert spec.key != replace(spec, fmax=8000.0).key
    assert len(spec.key) == 32


class TestEnsurePatched:
    """The replacement is only safe while the pin it was derived from holds."""

    @staticmethod
    def _checkout(root: Path, contents: bytes) -> Path:
        """A checkout holding every file the pin covers, all with ``contents``.

        All of them, not just the one a test is about: ``ensure_patched`` walks
        the whole of ``PATCHED_FILES`` and a checkout missing one of them is
        not the situation any of these tests means to describe.
        """
        for relative in vendored.PATCHED_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        return root / "python" / "voxelizer" / "vox_scene.py"

    @staticmethod
    def _all_declared(digest: str) -> dict[str, str]:
        """``UPSTREAM_SHA256`` with every entry claiming ``digest``."""
        return dict.fromkeys(vendored.UPSTREAM_SHA256, digest)

    def test_it_installs_our_copy_over_the_upstream_it_was_derived_from(
        self, tmp_path: Path
    ) -> None:
        upstream = tmp_path / "up.py"
        ours = vendored.patched_path("python/voxelizer/vox_scene.py").read_bytes()
        target = self._checkout(tmp_path / "pffdtd", b"whatever")
        # Stand in for the real upstream by declaring its digest.
        digest = hashlib.sha256(b"whatever").hexdigest()
        with mock.patch.dict(vendored.UPSTREAM_SHA256, self._all_declared(digest)):
            written = ensure_patched(tmp_path / "pffdtd")
        assert written == list(vendored.PATCHED_FILES)
        assert target.read_bytes() == ours
        assert not upstream.exists()

    def test_installing_twice_writes_once(self, tmp_path: Path) -> None:
        """Idempotent, so calling it before every voxelisation costs nothing."""
        root = tmp_path / "pffdtd"
        for relative in vendored.PATCHED_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(vendored.patched_path(relative).read_bytes())
        assert ensure_patched(root) == []

    def test_it_refuses_a_file_it_did_not_derive_from(self, tmp_path: Path) -> None:
        """A changed upstream is a merge nobody did, not a file to overwrite."""
        self._checkout(tmp_path / "pffdtd", b"upstream moved on")
        with pytest.raises(RuntimeError, match="neither the upstream"):
            ensure_patched(tmp_path / "pffdtd")

    def test_a_second_call_for_the_same_checkout_is_not_reverified(self, tmp_path: Path) -> None:
        """Voxelising many scenes against one checkout should not re-run the
        git subprocess and file digests before every one of them.

        Corrupting the checkout after the first call and asserting the second
        call does not raise proves the second call skipped verification
        entirely, rather than merely being fast.
        """
        root = tmp_path / "pffdtd"
        for relative in vendored.PATCHED_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(vendored.patched_path(relative).read_bytes())
        target = root / "python" / "voxelizer" / "vox_scene.py"
        assert ensure_patched(root) == []

        target.write_bytes(b"neither upstream nor ours")
        assert ensure_patched(root) == []

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

    def test_the_index_never_changes_what_the_solver_is_given(self) -> None:
        """Patch 6 is a speed change, so the acceptance is identity.

        Not asserted here -- a real voxelisation needs the PFFDTD checkout and
        several minutes -- but the source has to keep the two properties the
        identity rests on, and both are one edit away from being lost:

        - the candidate lists stay ascending by triangle index, which is the
          order ``np.nonzero`` produced, so ``candidates[hits]`` is unchanged;
        - a lattice the binning does not recognise falls back to the scan
          rather than to a guess.

        The measured run is in ``scripts/check_vox_index.sh``: one bedroom at
        4 kHz, with and without the patch, ``vox_out.h5`` byte for byte.
        """
        source = vendored.patched_path("python/voxelizer/vox_grid_base.py").read_text()
        assert "REVERBERATE PATCH 6" in source
        assert "kind='stable'" in source
        assert "return None" in source

    def test_the_numpy_repair_is_declared_rather_than_applied_by_hand(self) -> None:
        """``np.float`` went in numpy 1.20 and upstream still reads it.

        The fix was in the checkout, made by hand and recorded in no file, so
        ``ensure_patched`` neither knew about it nor would restore it: a fresh
        clone at the pinned commit could not voxelise at all.
        """
        assert "python/common/myfuncs.py" in vendored.PATCHED_FILES
        source = vendored.patched_path("python/common/myfuncs.py").read_text()
        assert "np.finfo(float).eps" in source
        assert "np.float)" not in source

    def test_every_replacement_declares_the_upstream_it_came_from(self) -> None:
        """A file in ``PATCHED_FILES`` with no digest would install over
        anything, which is the check the whole module exists for."""
        assert set(vendored.PATCHED_FILES) == set(vendored.UPSTREAM_SHA256)
        for relative in vendored.PATCHED_FILES:
            assert vendored.patched_path(relative).is_file()

    def test_the_voxeliser_is_part_of_the_cache_key(self, tmp_path: Path) -> None:
        """Otherwise an entry from before the patch answers for one after it.

        Two specs, not two reads of one: ``key`` is cached per instance (see
        ``test_key_is_computed_once_per_spec`` below), so re-reading the same
        spec's ``key`` after mutating global state would just answer from its
        own cache, which is not what this is checking.
        """
        before_dir, after_dir = tmp_path / "before", tmp_path / "after"
        before_dir.mkdir()
        after_dir.mkdir()
        before = make_spec(before_dir).key
        with mock.patch.object(vendored, "UPSTREAM_COMMIT", "0" * 40):
            assert make_spec(after_dir).key != before

    def test_key_is_computed_once_per_spec(self, tmp_path: Path) -> None:
        """A single ``voxelise()`` call reads ``spec.key`` several times.

        Recomputing it from disk on every access would mean re-reading the
        mesh and every material file that many times over; caching it means a
        spec whose backing files have since disappeared still answers from
        what it already computed.
        """
        spec = make_spec(tmp_path)
        first = spec.key
        Path(spec.model_json).unlink()
        assert spec.key == first


class TestScratchDirectory:
    """Two voxelisations must not share a scratch directory."""

    def test_the_child_runs_in_its_own_staging_directory(self, tmp_path: Path) -> None:
        """PFFDTD spills per-voxel results through a *relative* ``mmap_dat/``.

        ``vox_scene.py:61`` sets ``DAT_FOLDER = 'mmap_dat'`` and clears it at the
        start of every run, and ``adj_check.dat`` is memory-mapped beside it. A
        child inheriting the caller's working directory therefore shares that
        scratch with every other voxelisation started from the same place, and
        the loser dies on an assertion about triangle counts that reads like a
        defect in the scene rather than a collision.

        Asserted on the ``cwd`` the subprocess is given, because that is the
        whole of the fix and the alternative -- running two real voxelisations
        and checking neither corrupts the other -- costs minutes and a
        gigabyte.
        """
        import importlib
        from unittest import mock

        # By path, not ``from reverberate.wave import voxelise``: the package
        # re-exports the *function* of that name, which would shadow the module.
        module = importlib.import_module("reverberate.wave.voxelise")

        spec = make_spec(tmp_path)
        recorded: dict[str, object] = {}

        def fake_run(argv: list[str], **kwargs: object) -> object:
            recorded.update(kwargs)
            raise RuntimeError("stop here: the call is what is under test")

        with (
            mock.patch.object(module, "ensure_patched", lambda: []),
            mock.patch.object(module, "pffdtd_dir", lambda: tmp_path),
            mock.patch.object(module, "pffdtd_python", lambda: "python"),
            mock.patch.object(module.subprocess, "run", fake_run),
            mock.patch.object(module, "cache_root", lambda: tmp_path / "cache"),
            pytest.raises(RuntimeError, match="stop here"),
        ):
            module.voxelise(spec)

        cwd = recorded.get("cwd")
        assert cwd is not None
        assert Path(str(cwd)).name.startswith(f".{spec.key}.partial.")
