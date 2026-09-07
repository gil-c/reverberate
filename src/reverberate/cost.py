"""What a solve will cost, in hours and in money, before it is bought.

Roadmap W12: do not launch a long job without showing that number first. This
module is that number. It exists because the project has twice concluded
something about cost from a figure that was not a measurement -- B0's first
report called the cost model an order of magnitude wrong when the whole
discrepancy was a 3.7x tariff difference, and ``w20_first_listen``'s local
throughput constant was once read off a column labelled ``engine_s`` that did
not hold local timings, and was wrong by a factor of 37.

Two rules follow, and both are enforced by the signatures here rather than by
a convention someone has to remember.

**Every cost figure carries the hourly rate it assumes.** ``usd_per_hour`` is a
required keyword argument with no default. A cost without a rate is not a
measurement, so this module makes one impossible to produce by accident.

**The card count is part of the price.** Memory scales with the bounding box,
not with the air inside it, and a room that needs three cards costs three times
the hourly rate for the same wall clock. :class:`SolveCost` therefore reports
cards alongside hours, because a figure that omits them understates a large
room by exactly that factor.

The throughput and footprint constants are measured, from this project's own
runs, and reproduce every cost figure in roadmap W30's tables. Their derivation
is in the constants' own documentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BYTES_PER_POINT",
    "CUDA_CONTEXT_BYTES",
    "SECONDS_PER_POINT_STEP",
    "SolveCost",
    "cards_for",
    "estimate",
    "steps_for",
]

#: Seconds of solver per grid point per time step, on one A100 80 GB.
#:
#: **Measured**, from B0's anchor re-used by W29: ``bedroom_only`` at 16 kHz,
#: 4 607 993 520 grid points, a 60 ms window at the grid's own 291 275.8 Hz,
#: which is 17 477 steps, took 1211 s of engine. That is
#: ``1211 / (4.607993520e9 * 17477) = 1.5038e-11``, or 6.650e10 point updates
#: per second. The roadmap quotes 1.503e-11 and 6.65e10 for the same run.
#:
#: It is a hardware roofline rather than a code limit: 6.65e10 updates at
#: roughly 30 bytes of traffic each is about 2.0 TB/s, which is the A100's
#: HBM2e bandwidth. Below about 1e9 points the figure falls to 50e9 and below
#: 1e8 to 30e9, because the grid stops filling the machine. Every run this
#: module is used for is far above that, so one constant is honest here; a
#: smaller grid will be over-priced, which is the safe direction.
SECONDS_PER_POINT_STEP = 1.5038e-11

#: Bytes of device memory per grid point of volume.
#:
#: **Measured twice, on the same room.** B0 reported 41.2 GB for 4.610e9
#: points and W29 reported 40 533 MiB for 4.607993520e9, giving 8.94 and 9.22
#: bytes per point. The lower is used, with the headroom argument of
#: :func:`cards_for` covering the difference. It decomposes into ``u0`` and
#: ``u1`` at float32 plus one mask bit, which is 8.125 bytes, and a surface
#: term that varies with how furnished the room is. **Refit per scene**: a
#: bare room has far fewer boundary nodes than a furnished one.
BYTES_PER_POINT = 8.94

#: Fixed device memory a CUDA context costs before any grid is allocated.
#: **Measured**, 2.13 GB, and it comes off the card's capacity once per card.
CUDA_CONTEXT_BYTES = 2.13e9


def steps_for(duration_s: float, sample_rate_hz: float) -> int:
    """Time steps a window of ``duration_s`` costs, at the grid's own rate.

    The rate is the *solver's*, not the delivery rate: at 16 kHz and 10.5
    points per wavelength the grid runs at about 291 kHz, so one second of
    response is 291 276 steps and not 48 000. Read it from the voxelisation
    manifest's ``sample_rate_hz`` rather than deriving it, because PFFDTD's
    own value differs from ``c sqrt(3) / h`` by about 0.1 per cent and the
    manifest is what the engine will actually use.

    Mirrors ``reverberate.wave.comms.source_signal``, which is where the
    number really lands as ``Nt``.
    """
    if duration_s < 0.0:
        raise ValueError(f"duration must not be negative, got {duration_s} s")
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz} Hz")
    return int(math.ceil(duration_s * sample_rate_hz))


def cards_for(
    grid_points: int,
    *,
    card_ram_gb: float = 80.0,
    bytes_per_point: float = BYTES_PER_POINT,
    headroom: float = 1.0,
) -> int:
    """How many cards of ``card_ram_gb`` the grid needs, at least one.

    Multi-GPU subdomain splitting is implemented in the pinned engine and on by
    default -- slabs along ``x`` with one halo layer, selected by
    ``CUDA_VISIBLE_DEVICES`` rather than by a flag -- so the memory ceiling is
    a cost decision and not a capability gap.

    ``headroom`` above 1.0 buys margin against the per-scene variation in the
    surface term. It is not applied by default, because the default
    ``bytes_per_point`` is already the lower of the two measurements and
    inflating both would double-count.
    """
    if grid_points <= 0:
        raise ValueError(f"grid points must be positive, got {grid_points}")
    if card_ram_gb <= 0.0:
        raise ValueError(f"card memory must be positive, got {card_ram_gb} GB")
    usable = card_ram_gb * 1e9 - CUDA_CONTEXT_BYTES
    if usable <= 0.0:
        raise ValueError(f"a {card_ram_gb} GB card cannot hold the CUDA context alone")
    return max(1, int(math.ceil(grid_points * bytes_per_point * headroom / usable)))


@dataclass(frozen=True)
class SolveCost:
    """The price of one solve, with the rate and the card count it assumes."""

    grid_points: int
    steps: int
    sources: int
    duration_s: float
    sample_rate_hz: float
    cards: int
    card_ram_gb: float
    usd_per_hour_per_card: float

    @property
    def point_steps(self) -> float:
        """Grid points times time steps times sources. The unit of work."""
        return float(self.grid_points) * self.steps * self.sources

    @property
    def gpu_seconds(self) -> float:
        """Engine seconds on one card. Excludes provisioning and transfer."""
        return self.point_steps * SECONDS_PER_POINT_STEP

    @property
    def wall_clock_hours(self) -> float:
        """Hours of wall clock, assuming the split scales perfectly across cards.

        An optimistic bound. The halo exchange is overlapped with the air
        kernel, so the loss is small, but it is not zero and nobody has
        measured it here. Named as a bound rather than folded into a fudge
        factor.
        """
        return self.gpu_seconds / (3600.0 * self.cards)

    @property
    def gpu_hours(self) -> float:
        """Card-hours billed: wall clock times the number of cards."""
        return self.gpu_seconds / 3600.0

    @property
    def usd(self) -> float:
        """Money, at the rate this estimate was given. Engine time only."""
        return self.gpu_hours * self.usd_per_hour_per_card

    @property
    def device_bytes(self) -> float:
        """Device memory the grid needs, before the CUDA context."""
        return self.grid_points * BYTES_PER_POINT

    def summary(self) -> str:
        """One line, and it always names the rate."""
        return (
            f"{self.grid_points:.3e} points x {self.steps} steps: "
            f"{self.wall_clock_hours:.2f} h on {self.cards} x {self.card_ram_gb:.0f} GB, "
            f"{self.usd:.2f} USD at {self.usd_per_hour_per_card:.3f} USD/h per card"
        )

    def record(self) -> dict[str, Any]:
        """What a ``report.json`` or a ``plan.json`` carries."""
        return {
            "grid_points": self.grid_points,
            "steps": self.steps,
            "sources": self.sources,
            "duration_s": self.duration_s,
            "sample_rate_hz": self.sample_rate_hz,
            "point_steps": self.point_steps,
            "gpu_seconds": self.gpu_seconds,
            "gpu_hours": self.gpu_hours,
            "wall_clock_hours": self.wall_clock_hours,
            "cards": self.cards,
            "card_ram_gb": self.card_ram_gb,
            "device_bytes": self.device_bytes,
            "usd_per_hour_per_card": self.usd_per_hour_per_card,
            "usd": self.usd,
            "seconds_per_point_step": SECONDS_PER_POINT_STEP,
            "bytes_per_point": BYTES_PER_POINT,
        }


def estimate(
    grid_points: int,
    duration_s: float,
    sample_rate_hz: float,
    *,
    usd_per_hour_per_card: float,
    sources: int = 1,
    card_ram_gb: float = 80.0,
    headroom: float = 1.0,
) -> SolveCost:
    """Price one solve.

    ``usd_per_hour_per_card`` is required and has no default. Use the *billed*
    rate rather than the quoted one: Vast's running ``dph_total`` includes
    storage and bandwidth and has run 20 to 40 per cent above the advertised
    figure on every rental this project has made.

    Receivers do not appear because they are free: 64 bytes per receiver per
    step, about 1.9 MB per receiver per 100 ms, and no effect on the stencil.
    Sources are not free -- each needs its own solve -- which is why they
    multiply.
    """
    if sources < 1:
        raise ValueError(f"a solve needs at least one source, got {sources}")
    if usd_per_hour_per_card < 0.0:
        raise ValueError(f"rate must not be negative, got {usd_per_hour_per_card}")
    return SolveCost(
        grid_points=grid_points,
        steps=steps_for(duration_s, sample_rate_hz),
        sources=sources,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        cards=cards_for(grid_points, card_ram_gb=card_ram_gb, headroom=headroom),
        card_ram_gb=card_ram_gb,
        usd_per_hour_per_card=usd_per_hour_per_card,
    )
