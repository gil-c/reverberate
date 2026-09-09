"""Tests for sizing a rented machine before renting it.

Every rental this project has lost was lost to a number nobody worked out
beforehand. These check the arithmetic against the runs that produced the
measurements, so a requirement that drifts fails here rather than on a machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverberate.wave.remote_voxelise import (
    MachineNeed,
    grid_shape_of,
    install_entry,
    nodes_from_shape,
    payload_need_for,
    voxelise_need,
)
from reverberate.wave.voxelise import SceneSpec


class _Offer:
    def __init__(self, cores: float, ram: float, disk: float, vram: float = 0.0) -> None:
        self.cpu_cores = cores
        self.ram_gb = ram
        self.disk_gb = disk
        self.gpu_ram_gb = vram


@pytest.fixture
def model(tmp_path: Path) -> Path:
    """A model whose points span the flat's own bounding box.

    Measured: 23.58 x 2.89 x 18.44 m gives a 2894 x 360 x 2265 grid at 4 kHz.
    """
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "mats_hash": {
                    "shell": {
                        "pts": [[-21.07, -0.07, -13.11], [2.51, 2.82, 5.33]],
                        "tris": [[0, 1, 0]],
                        "sides": [2],
                    }
                }
            }
        )
    )
    return path


class TestGridShape:
    def test_it_reproduces_the_flat_at_4_kHz(self, model: Path) -> None:
        """Against the grid the flat actually built: 2894 x 360 x 2265.

        Worked out from the model's own points rather than by voxelising, which
        is what lets a machine be sized before one is rented.
        """
        shape = grid_shape_of(model, 4000.0, 10.5)
        for got, want in zip(shape, (2894, 360, 2265), strict=True):
            assert abs(got - want) <= 2, f"{shape} against (2894, 360, 2265)"

    def test_four_times_the_frequency_is_sixty_four_times_the_points(self, model: Path) -> None:
        """Points, not axes. The ``+1`` and the 3.5-cell pad do not scale, so a
        2.89 m axis grows by 3.93 rather than 4 -- and it is the product that
        sizes the disk, because PFFDTD's guard is over the whole grid."""

        def points(shape: tuple[int, int, int]) -> float:
            return float(shape[0]) * shape[1] * shape[2]

        ratio = points(grid_shape_of(model, 16000.0, 10.5)) / points(
            grid_shape_of(model, 4000.0, 10.5)
        )
        assert abs(ratio / 64.0 - 1.0) < 0.05


class TestNodeCount:
    def test_it_reproduces_the_flat_at_16_kHz_from_the_4_kHz_fit(self) -> None:
        """The constant is read off 4 kHz; 16 kHz is the check that it travels.

        Measured 1 089 464 499 nodes on 11549 x 1415 x 9035. Boundary nodes
        cover a surface, so they go as the grid to the two thirds.
        """
        predicted = nodes_from_shape((11549, 1415, 9035))
        assert abs(predicted / 1_089_464_499 - 1.0) < 0.10


class TestVoxeliseNeed:
    def test_the_disk_it_asks_for_is_what_the_flat_needed(self, model: Path) -> None:
        """The 16 kHz run needed 330 GB free and was given 400.

        PFFDTD's guard compares the whole grid against *half* the free space and
        asks a question on a stdin the child has consumed, so too little disk
        presents as a hang. The requirement is twice the grid plus the entry.
        """
        need = voxelise_need(model, 16000.0, slabs=16)
        assert 320 <= need.disk_gb <= 420

    def test_a_room_sized_job_does_not_ask_for_a_terabyte(self, model: Path) -> None:
        """A requirement that over-asks refuses machines that would have done."""
        assert voxelise_need(model, 1000.0).disk_gb < 60


class TestPayloadNeed:
    def test_it_covers_what_the_lossless_flat_actually_used(self) -> None:
        """Measured 16.9 GB for 66 159 665 nodes at one block each.

        It must err *high*: a requirement that under-asks is a machine rented to
        swap for hours, which is how this number came to be measured.
        """
        need = payload_need_for(66_159_665, (2894, 360, 2265), 100_000_000)
        assert 16.9 <= need.ram_gb <= 30.0

    def test_a_coarser_budget_asks_for_much_less(self) -> None:
        """The block budget, not the grid, is what decides the lattice."""
        fine = payload_need_for(66_159_665, (2894, 360, 2265), 100_000_000)
        coarse = payload_need_for(66_159_665, (2894, 360, 2265), 20_000_000)
        assert coarse.ram_gb < fine.ram_gb / 2


class TestUnmet:
    def test_it_names_every_shortfall_rather_than_the_first(self) -> None:
        """A caller printing one reason sends the operator round the loop twice."""
        need = MachineNeed(cores=32, ram_gb=64, disk_gb=400, why="test")
        problems = need.unmet(_Offer(cores=8, ram=16, disk=100))
        assert len(problems) == 3

    def test_a_machine_that_fits_reports_nothing(self) -> None:
        need = MachineNeed(cores=16, ram_gb=32, disk_gb=200, why="test")
        assert need.unmet(_Offer(cores=32, ram=64, disk=500)) == []

    def test_vram_is_only_asked_of_a_stage_that_wants_a_gpu(self) -> None:
        """Voxelising and payload-building never touch the card; the solve does."""
        cpu = MachineNeed(cores=4, ram_gb=8, disk_gb=50, why="test")
        assert cpu.unmet(_Offer(cores=8, ram=16, disk=100, vram=0)) == []
        gpu = MachineNeed(cores=4, ram_gb=8, disk_gb=50, why="t", needs_gpu=True, vram_gb=24)
        assert gpu.unmet(_Offer(cores=8, ram=16, disk=100, vram=8))


def test_merge_satisfies_both_stages() -> None:
    """One rental voxelises and then builds the payload, and they want
    different things: cores and disk against memory."""
    voxelise = MachineNeed(cores=32, ram_gb=8, disk_gb=400, why="voxelise")
    payload = MachineNeed(cores=4, ram_gb=64, disk_gb=40, why="payload")
    both = voxelise.merge(payload)
    assert (both.cores, both.ram_gb, both.disk_gb) == (32, 64, 400)
    assert "voxelise" in both.why and "payload" in both.why


class TestInstallEntry:
    """A fetched grid is not a cache entry until it is installed as one.

    Everything downstream addresses a grid by its key and reads
    ``manifest.json`` for the material labels, so a rented voxelisation that
    stopped at the transfer had to be recomputed locally to be usable -- the
    opposite of what the split exists for.
    """

    @pytest.fixture
    def spec(self, tmp_path: Path, model: Path) -> SceneSpec:
        from reverberate.wave.voxelise import SceneSpec

        materials = tmp_path / "materials"
        materials.mkdir()
        (materials / "shell.h5").write_bytes(b"not really an h5, and never read here")
        return SceneSpec(
            model_json=model,
            mat_folder=materials,
            mat_files={"shell": "shell.h5"},
            fmax=1000.0,
            ppw=10.5,
            slabs=4,
        )

    def _fetched(self, root: Path) -> Path:
        from reverberate.wave.voxelise import CACHE_FILES

        root.mkdir(parents=True, exist_ok=True)
        for name in CACHE_FILES:
            (root / name).write_bytes(b"x" * 16)
        return root

    def test_it_lands_in_the_cache_under_the_key(
        self, spec: SceneSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
        fetched = self._fetched(tmp_path / "landing")
        entry = install_entry(spec, fetched, {"Nbt": 7}, wall_s=12.5)
        assert entry.complete
        assert entry.path.name == spec.key
        assert not fetched.exists()

    def test_the_manifest_carries_what_the_audit_view_reads(
        self, spec: SceneSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``audit_view`` takes its material labels from here, and
        ``grid_page`` its fmax and its model path. A manifest missing either
        fails deep inside a build rather than at the transfer."""
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
        entry = install_entry(spec, self._fetched(tmp_path / "landing"), {"Nbt": 7}, wall_s=1.0)
        assert entry.manifest["materials"] == {"shell": "shell.h5"}
        assert entry.manifest["fmax"] == 1000.0
        assert entry.manifest["slabs"] == 4
        assert entry.manifest["computed_on"] == "rented"
        assert entry.manifest["Nbt"] == 7
        assert entry.manifest["geometry_sha256"]

    def test_it_refuses_a_transfer_that_is_missing_a_file(
        self, spec: SceneSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Half a grid installed under a key is a cache poisoned forever: the
        key is content addressed, so nothing later would ever recompute it."""
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
        fetched = self._fetched(tmp_path / "landing")
        (fetched / "vox_out.h5").unlink()
        with pytest.raises(RuntimeError, match="vox_out.h5"):
            install_entry(spec, fetched, {}, wall_s=1.0)


class TestLaunchCommand:
    """The shell quoting that costs a rental when it is wrong.

    ``cmd1 && cmd2 & cmd3`` backgrounds the whole ``&&`` chain. The first
    detached launcher did exactly that: it backgrounded the creation of the job
    file, took the chain's pid for the child's, and left ``ssh`` waiting out its
    five minute timeout on a machine where the child was already running.
    """

    DIR = "/root/vox/16000"

    def _command(self) -> str:
        from reverberate.wave.remote_voxelise import launch_command

        return launch_command({"fmax": 16000.0}, self.DIR)

    def test_only_the_launcher_is_backgrounded(self) -> None:
        """Everything before the ``&`` must be a completed statement, so the
        setup has run by the time the child starts."""
        command = self._command()
        head, _, tail = command.partition(" & ")
        assert tail.strip() == "echo $!"
        # The chain that prepares the job ends in ';', not in '&&', so the '&'
        # applies to the setsid alone.
        assert "; setsid sh " in head
        assert "&& setsid" not in head

    def test_the_job_reaches_the_child_as_a_file(self) -> None:
        """Piping it in was what tied the job to the connection."""
        command = self._command()
        assert f"> {self.DIR}/job.json" in command
        assert f"< {self.DIR}/job.json" in command

    def test_every_descriptor_is_off_the_ssh_channel(self) -> None:
        """``ssh`` returns when nothing still holds the channel, and a
        background process holding stdout holds it."""
        command = self._command()
        assert "< /dev/null > /dev/null 2>&1 &" in command
        assert f"> {self.DIR}/voxelise.out 2>&1" in command

    def test_the_exit_status_is_kept_beside_the_log(self) -> None:
        """A child that dies without printing its report is diagnosed by it --
        137 is the kernel, and that is a different fix from a bad scene."""
        assert f"echo $? > {self.DIR}/voxelise.rc" in self._command()

    def test_the_stale_files_of_a_previous_attempt_are_removed(self) -> None:
        """A retry in the same directory must not read the last run's log."""
        command = self._command()
        assert f"rm -f {self.DIR}/voxelise.out {self.DIR}/voxelise.rc" in command
