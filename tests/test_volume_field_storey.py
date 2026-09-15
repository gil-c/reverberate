"""The storey's audit meshes as ``walk.json`` names them."""

from __future__ import annotations

from pathlib import Path

from reverberate.experiments.w40_volume_field.storey import audit_meshes


class TestAuditMeshes:
    def test_only_payloads_that_exist_are_named_relative_to_the_run(self, tmp_path: Path) -> None:
        """The app refuses a mesh of another scene, and the first campaign's
        hard-coded paths drew hssd_0002's walls around hssd_0076: a mesh is
        named only when this campaign built it."""
        audit = tmp_path / "audit"
        (audit / "1000" / "voxels").mkdir(parents=True)
        (audit / "1000" / "voxels" / "rooms.json").write_text("{}")
        (audit / "4000" / "voxels").mkdir(parents=True)  # built nothing
        assert audit_meshes(tmp_path, audit) == {"1000": "audit/1000/voxels"}
        assert audit_meshes(tmp_path, tmp_path / "elsewhere") == {}
