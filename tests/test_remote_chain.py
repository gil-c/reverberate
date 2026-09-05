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
    nodes_from_shape,
    payload_need_for,
    voxelise_need,
)


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
