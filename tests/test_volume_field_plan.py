"""The arithmetic a campaign is sized and cut by, pinned against the runs that measured it.

Every rental this project lost was lost to a number nobody had worked out
beforehand: a mid band that needed twice its output in RAM, a shard cut through
a point's rows, a grid that did not fit the card. These fail here first.
"""

from __future__ import annotations

from reverberate.experiments.w40_volume_field.encode import shard_bounds
from reverberate.experiments.w40_volume_field.plan import BYTES_PER_SAMPLE, pitch_for
from reverberate.experiments.w40_volume_field.solve import (
    RAM_PER_OUTPUT,
    VRAM_FIXED_GB,
    VRAM_PER_NODE_B,
    sizing,
    slices_for,
)


class TestShardBounds:
    def test_every_point_lands_in_exactly_one_shard_with_rows_at_its_own_origin(self) -> None:
        """A shard takes points while it stays under its share of the rows;
        the last shard takes the rest. Here the share is 25 rows: the first
        point (10) fits, the second would end at 25 and goes to the last."""
        rows = [[0, 10], None, [10, 25], [25, 30], [30, 50]]
        bounds, per_shard = shard_bounds(rows, 2)
        assert bounds == [(0, 10), (10, 50)]
        assert per_shard[0] == [[0, 10], None, None, None, None]
        assert per_shard[1] == [None, None, [0, 15], [15, 20], [20, 40]]

    def test_a_shard_never_cuts_through_a_point(self) -> None:
        """The pressure is row-contiguous per point; a cut inside one would
        hand half a point to each box."""
        rows = [[i * 7, (i + 1) * 7] for i in range(11)]
        bounds, per_shard = shard_bounds(rows, 4)
        starts = {b[0] for b in bounds} | {b[1] for b in bounds}
        assert starts <= {r[0] for r in rows} | {rows[-1][1]}
        placed = [sum(r is not None for r in shard) for shard in per_shard]
        assert sum(placed) == 11 and all(n >= 2 for n in placed)


class TestSlices:
    def test_the_mid_band_of_hssd_0076_needs_two_slices_on_a_188_gb_host(self) -> None:
        """437 points x 1021 nodes x 29128 samples x 8 B = 104 GB of output;
        at 2.2 x that plus 8 GB it passes 188 GB once and fits in two."""
        output = 437 * 1021 * 29128 * BYTES_PER_SAMPLE
        assert slices_for(output, 180.0) == 2
        assert slices_for(output, 250.0) == 1
        assert slices_for(output, 90.0) == 3

    def test_sizing_reads_the_plan_and_multiplies_card_time_by_the_slices(self) -> None:
        plan = {
            "bands": {
                "low": {
                    "cost": {"sim_outs_bytes": 72e9, "grid_points": 1.5e7, "estimated_gpu_s": 60}
                },
                "mid": {
                    "cost": {"sim_outs_bytes": 104e9, "grid_points": 9.6e8, "estimated_gpu_s": 420}
                },
                "high": {
                    "cost": {"sim_outs_bytes": 78e9, "grid_points": 9.0e9, "estimated_gpu_s": 2600}
                },
            }
        }
        jobs = [({"name": "S1"}, b) for b in ("low", "mid", "high")]
        size = sizing(plan, jobs, slice_ram_gb=180.0)
        assert size["parts"] == {"low": 1, "mid": 2, "high": 1}
        assert abs(size["vram_gb"] - (9.0e9 * VRAM_PER_NODE_B / 1e9 + VRAM_FIXED_GB)) < 1e-6
        assert size["vram_gb"] > 80.0, "the storey of hssd_0076 does not fit one 80 GB card"
        assert abs(size["ram_gb"] - (RAM_PER_OUTPUT * 78.0 + 8.0)) < 1e-6
        assert abs(size["gpu_hours"] - (60 + 2 * 420 + 2600) / 3600) < 1e-9


class TestGridBudget:
    def test_the_pitch_that_fills_a_cap_puts_the_cap_on_the_floor(self) -> None:
        pitch = pitch_for(85.6, 560)
        assert 0.35 <= pitch <= 0.45
        assert 85.6 / pitch**2 <= 560 * 1.05
