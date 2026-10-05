"""What a trace is predicted to take on an offered machine: wall hours and USD, before renting.

A rental taken by its hourly price is the slowest way through 1763 wave
solves. The low band divides over a machine's cards, one batch a card
(:class:`reverberate.wave.lowband.pairs.LowbandPairs`), and so do the rays
(:func:`reverberate.mirror.engine.histogram_on_devices`); the early trace,
the levelling and the pack's write are one process on one card and the
host, and the way home is the laptop's line. :func:`predict` adds those up
for one offer, and :mod:`reverberate.gpu.onebox` rents the lowest total
within the wall time allowed.

**What is measured and what is not.** The low band's constants are one RTX
3080 20 GB's (``wave.lowband.pairs``), every other stage's one RTX 3090's
(``trace.plan``, ``docs/adr/0016-appendix-trace-cost.md``). A card of
another kind is priced through :data:`CARDS`, its throughput on a stencil
bound by memory traffic over the RTX 3080's. **Three entries rest on a
measurement of the stencil**, and two of those through the present engine
and not the batched solver; the others are the ratio of the cards' memory
bandwidths, which the two measured ratios show to be a poor guide (see the
table). A prediction on an estimated card says so, and its watchdog is
given a larger margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reverberate.spatial.lowband import solve_fmax_hz
from reverberate.trace import plan as plans
from reverberate.wave.lowband import pairs as batched

__all__ = ["CARDS", "Card", "card_of", "predict", "predictor", "throughput_table"]


@dataclass(frozen=True)
class Card:
    """A kind of card: its stencil throughput over one RTX 3080 20 GB's, and where that is from."""

    #: What an offer's ``gpu_name`` contains, compared without case or spaces.
    name: str
    throughput: float
    measured: bool
    source: str


_BANDWIDTH = "estimate: memory bandwidth over the RTX 3080's 760 GB/s, not measured"

#: Longest name first where one contains another. The measured rows:
#:
#: - RTX 3080 20 GB, the batched solver itself: 110 s a source position, 1.4e10 node
#:   updates a second (instance 54201838, 2026-10-05). The unit.
#: - A100: the present engine (PFFDTD) solved a position in 15.9 s on 2 x A100 80 GB, the
#:   grid split over both (1.354e11 updates a second, 2026-10-03), against 135 s on the RTX
#:   3080 for the same engine and grid: 135 / (2 x 15.9) = 4.2 a card. Assumed to carry
#:   over to the batched solver.
#: - Tesla P100: the present engine took 183 s a position on 2 x P100 (the first smoke,
#:   instance 54194831, 2026-10-04), one engine process a card as ``accel.pairs`` runs
#:   them, against 135 s: 0.74. If those 183 s were of both cards on one solve it is
#:   half that.
#: - RTX 3090: **measured for every stage but the solve**, of which it is the reference
#:   card; its stencil throughput is the bandwidth estimate like the others'.
#:
#: The bandwidth ratio would give the A100 2.0 to 2.5 and the P100 0.96: it is 70 per cent
#: short on one and 30 per cent long on the other.
CARDS: tuple[Card, ...] = (
    Card("RTX 3080 Ti", 1.2, False, _BANDWIDTH),
    Card("RTX 3080", 1.0, True, "the batched solver, 110 s a position, 2026-10-05"),
    Card("RTX 3090 Ti", 1.3, False, _BANDWIDTH),
    Card(
        "RTX 3090",
        1.2,
        False,
        _BANDWIDTH + "; the card every stage but the solve was measured on",
    ),
    Card("RTX 4090", 1.3, False, _BANDWIDTH),
    Card("RTX 4080", 0.95, False, _BANDWIDTH),
    Card("RTX 4070 Ti", 0.65, False, _BANDWIDTH),
    Card("RTX 5090", 2.3, False, _BANDWIDTH),
    Card("RTX 5080", 1.25, False, _BANDWIDTH),
    Card("RTX A6000", 1.0, False, _BANDWIDTH),
    Card("RTX A5000", 1.0, False, _BANDWIDTH),
    Card("RTX A4000", 0.6, False, _BANDWIDTH),
    Card("RTX 6000Ada", 1.25, False, _BANDWIDTH),
    Card("A100", 4.2, True, "PFFDTD, 15.9 s a position on 2 x A100 against 135 s on the RTX 3080"),
    Card("H100", 4.2, False, "estimate: taken as the A100's, not measured"),
    Card("H200", 4.2, False, "estimate: taken as the A100's, not measured"),
    Card("L40S", 1.1, False, _BANDWIDTH),
    Card("L40", 1.1, False, _BANDWIDTH),
    Card("A40", 0.9, False, _BANDWIDTH),
    Card("Tesla V100", 1.2, False, _BANDWIDTH),
    Card("Tesla P100", 0.74, True, "PFFDTD, 183 s a position on 2 x P100 against 135 s"),
    Card("RTX 2080 Ti", 0.8, False, _BANDWIDTH),
    Card("Titan RTX", 0.9, False, _BANDWIDTH),
)


def _plain(name: str) -> str:
    return "".join(str(name).lower().split()).replace("_", "")


def card_of(gpu_name: str) -> Card | None:
    """The table's row for an offer's card; ``None`` for a card it has no figure for."""
    asked = _plain(gpu_name)
    for card in sorted(CARDS, key=lambda c: -len(c.name)):
        if _plain(card.name) in asked or _plain(card.name).removeprefix("tesla") == asked:
            return card
    return None


def throughput_table() -> list[str]:
    """The table, a line a card, as a log shows it."""
    return [
        f"{card.name:<12} {card.throughput:>4.2f}  {'measured ' if card.measured else 'ESTIMATED'}"
        f"  {card.source}"
        for card in CARDS
    ]


def predict(
    record: dict[str, Any],
    *,
    gpu_name: str,
    num_gpus: int,
    gpu_ram_gb: float,
    dph_total: float,
    fetch_pairs: bool = True,
    synced: bool = True,
    check: str | None = None,
    low_engine: str = "lowband",
    low_ppw: float | None = None,
    low_seconds: float | None = None,
    rays: int | None = None,
) -> dict[str, Any] | None:
    """Wall hours and USD of a plan's trace on one offer; ``None`` where it cannot be said.

    ``record`` is a plan's. The low band is what the machine will do: the
    solves, counted with the cells a card of this memory holds records for
    (:func:`reverberate.wave.lowband.pairs.solves_needed`), and the pairs,
    over the cards and the card's throughput. The rays divide over the
    cards and are otherwise the measured card's; the stages bound by the
    host are the measured machine's as they are. The pack goes home after
    the run at the laptop's line; the pair cache goes home while the run
    lasts when it is ``synced``, and only what the run did not leave time
    for is waited for.

    ``None`` for a card with no row in :data:`CARDS`, for a card too small
    to hold one cell's records beside the grid, and for the present engine,
    whose price was fitted on one machine only.
    """
    card = card_of(gpu_name)
    if card is None or low_engine != "lowband":
        return None
    counts = [int(c) for c in record.get("cells_a_position", [])]
    positions, pairs = int(record["source_positions"]), int(record["pairs"])
    window = batched.MEASURED_DURATION_S if low_seconds is None else float(low_seconds)
    if counts:
        solves = batched.solves_needed(counts, float(gpu_ram_gb), ppw=low_ppw, duration_s=window)
        if solves is None:
            return None
    else:
        if batched.cells_a_solve(float(gpu_ram_gb), ppw=low_ppw, duration_s=window) < 1:
            return None
        solves = positions
    cards = max(1, int(num_gpus))
    low = batched.estimate(
        solves,
        pairs,
        fmax_hz=solve_fmax_hz(),
        duration_s=window,
        rate_usd_per_hour=dph_total,
        cards=cards,
        ppw=low_ppw,
        speed=card.throughput,
    )
    base = plans.estimate(
        record,
        rate_usd_per_hour=dph_total,
        fetch_pairs=fetch_pairs,
        check=check,
        low_engine=low_engine,
        low_ppw=low_ppw,
        low_seconds=low_seconds,
        rays=rays,
    )
    seconds = dict(base["seconds"])
    seconds["low"] = float(low["seconds"])
    seconds["rays"] = seconds["rays"] / cards
    if synced and fetch_pairs:
        # Brought home while the machine works: only what the run leaves no time for.
        before = sum(v for k, v in seconds.items() if k not in ("transfer_pack", "transfer_pairs"))
        seconds["transfer_pairs"] = max(0.0, seconds["transfer_pairs"] - before)
    total = float(sum(seconds.values()))
    return {
        "hours": total / 3600.0,
        "usd": total / 3600.0 * float(dph_total),
        "measured": card.measured,
        "note": f"{solves} solves ({solves - positions} for the records), {card.name} x{cards}"
        f" at {card.throughput:g}",
        "seconds": {name: round(value, 1) for name, value in seconds.items()},
        "solves": solves,
        "extra_solves": solves - positions,
        "card": card.name,
        "cards": cards,
        "throughput": card.throughput,
    }


def predictor(record: dict[str, Any], **options: Any) -> Any:
    """:func:`predict` of ``record`` as :func:`reverberate.gpu.onebox.price_offers` calls it."""

    def priced(offer: Any) -> dict[str, Any] | None:
        return predict(
            record,
            gpu_name=str(offer.gpu_name),
            num_gpus=int(offer.num_gpus),
            gpu_ram_gb=float(offer.gpu_ram_gb),
            dph_total=float(offer.dph_total),
            **options,
        )

    return priced
