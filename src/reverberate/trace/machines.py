"""What a trace is predicted to take on an offered machine: wall hours and USD, before renting.

A rental taken by its hourly price is the slowest way through 1529 wave
solves, and a rental is billed from the moment it exists to the moment its
pack is home. :func:`predict` prices that whole span on one offer, as the
trace runs since it is **one queue over every card and every core**
(``docs/adr/0016-appendix-every-card-every-core.md``):

- **the start**: the instance answering, the engine built, the bundle
  pushed, then the grid voxelised and the launches planned. No card works;
- **the solves**, a launch a card: a source position's seconds on this kind
  of card (:data:`SOURCE_S`), a launch's own seconds, a pair's fit, over
  the cards; and **what the cards idle at the end**, half a launch each,
  since the last launches do not end together;
- **the rays**, a site a card, on the same cards;
- **the host's stages** (the early trace, the levelling, the pack's rows)
  on the host's cores *while the cards solve*: they add to the wall only
  where a host of few cores takes longer over them than its cards over
  theirs;
- **the write and the check**, one process;
- **the way home**, at the machine's rate: the pack, in the form it is
  written in, and what is left of the pair cache once the run has ended.
  :data:`LINE` is what the proxy carries; a direct line is priced by where
  the host is, once one has been seen to work.

**Checked against the two whole scenes of 2026-10-05**, the first that
ran (:func:`against`, ``python -m reverberate.trace ledger``): their solves
are predicted at 331 and 255 minutes and took 330 and 246. Those two ran
the code of before the queue, their other stages one after the other;
``queue=False`` prices a run that way, which is what makes the comparison
one of like with like.

**What is measured and what is not.** A source position's seconds are
measured on the RTX 3090 (88 s on the validated grid, 34.6 s at 7.2 points
per wavelength, eight and four cards) and on the RTX 3080 (110 and 36 s).
A card of another kind is priced through :data:`CARDS`, its throughput on
a stencil bound by memory traffic over the RTX 3080's; a prediction on an
estimated card says so, and its watchdog is given a larger margin.

**The table chooses an offer and nothing else.** Once on the machine the
trace reads its cards and measures them (:mod:`reverberate.trace.resources`),
says its own prediction before its long work and is told the hours the
watchdog leaves it (``--max-hours``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate.trace import plan as plans
from reverberate.wave.lowband import pairs as batched

__all__ = [
    "CARDS",
    "LINE",
    "SOURCE_S",
    "Card",
    "actual",
    "against",
    "card_of",
    "line_bytes_per_s",
    "predict",
    "predictor",
    "throughput_table",
]


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
#: - RTX 3090, the batched solver on the first whole scene: 88.0 s a source position in
#:   the mean of 1505 on 8 cards (1.76e10 node updates a second a card, instance 54262814,
#:   2026-10-05), and 90 s on 4 cards in launches of 8 (instance 54299322): 110 / 88.
#: - A100: the present engine (PFFDTD) solved a position in 15.9 s on 2 x A100 80 GB, the
#:   grid split over both (1.354e11 updates a second, 2026-10-03), against 135 s on the RTX
#:   3080 for the same engine and grid: 135 / (2 x 15.9) = 4.2 a card. Assumed to carry
#:   over to the batched solver.
#: - Tesla P100: the present engine took 183 s a position on 2 x P100 (the first smoke,
#:   instance 54194831, 2026-10-04), one engine process a card as ``accel.pairs`` runs
#:   them, against 135 s: 0.74. If those 183 s were of both cards on one solve it is
#:   half that.
#:
#: The bandwidth ratio would give the A100 2.0 to 2.5 and the P100 0.96: it is 70 per cent
#: short on one and 30 per cent long on the other.
CARDS: tuple[Card, ...] = (
    Card("RTX 3080 Ti", 1.2, False, _BANDWIDTH),
    Card("RTX 3080", 1.0, True, "the batched solver, 110 s a position, 2026-10-05"),
    Card("RTX 3090 Ti", 1.3, False, _BANDWIDTH),
    Card(
        "RTX 3090",
        1.25,
        True,
        "the batched solver, 88 s a position on 8 cards, the first whole scene, 2026-10-05",
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

#: Card seconds of one source position of 1.2 s on the grid to 1500 Hz, by the card and
#: the grid's points per wavelength, where a run measured them. The RTX 3090's are the two
#: whole scenes of 2026-10-05: 132 393 card seconds for 1505 positions on 8 cards, 51 824
#: for 1496 on 4 at 7.2 points. Between the two grids the RTX 3080 goes as the points to
#: the power 2.96 and the RTX 3090 as 2.47: another card or grid is priced from the RTX
#: 3080's 110 s, its throughput, and :data:`reverberate.wave.lowband.pairs.PPW_EXPONENT`.
SOURCE_S: dict[tuple[str, float], float] = {
    ("RTX 3080", 10.5): 110.0,
    ("RTX 3080", 7.2): 36.0,
    ("RTX 3090", 10.5): 88.0,
    ("RTX 3090", 7.2): 34.6,
}
#: Card seconds a pair's fit takes beside its launch's steps, where a run measured them:
#: 10 065 s for the 18 219 pairs of the first scene on the validated grid (0.74 s a pair
#: where a launch held one position heard at 84 cells, 0.27 s where it held twelve heard
#: at two), 3 167 s for 15 187 pairs at 7.2 points.
FIT_S: dict[tuple[str, float], float] = {("RTX 3090", 10.5): 0.55, ("RTX 3090", 7.2): 0.21}
#: From the rental to the trace's first line: the instance answering, its cards asked,
#: the engine built, the interpreter made, the bundle pushed. 8.1 and 13.4 minutes on the
#: two hosts of 2026-10-05, of which the build and the interpreter were 3.6 and 8.3.
START_S = 630.0
#: The grid voxelised, the arrays placed and the pairs assigned, before any launch: 108 s
#: on the validated grid, 240 s at 7.2 points, whose arrays are half as large again.
PREPARE_S = {10.5: 110.0, 7.2: 240.0}
#: Of the cards' time, what each idles at the end of the solves, in launches: the last
#: launches do not end together and nothing is left to give the card that ends first.
#: 555 s a card after launches of 20.7 minutes, 255 s after launches of 19.5.
TAIL_LAUNCHES = 0.5
#: The host's stages take this many times their work on the laptop's one core, on a
#: rented host's workers: the early trace's work is 1.6 times one worker's at eight and
#: 2.5 at sixteen, and a rented core is slower. One figure, not measured on a host of the
#: queue's own run; the stages hide under the solves on any host of eight cores or more.
HOST_OVERHEAD = 2.0
#: Host workers the stages are divided over, at most: beyond it the early trace loses.
HOST_WORKERS_AT_MOST = 16

#: Bytes a second that come home, by the way they come. ``proxy`` is measured: four
#: streams through Vast's ssh proxy from two hosts in California, 2026-10-05 (one stream
#: 2.0 to 2.3 MB/s, eight 5.9). The others are the laptop's own line to a host there, as
#: a speed test reads it, and what a direct connection is bounded by: **not measured on a
#: transfer**, and used only when a caller says the direct way works (``line="direct"``).
LINE: dict[str, float] = {"proxy": 4.4e6, "europe": 70e6, "united states": 17e6, "elsewhere": 8e6}
#: The country codes an offer's location ends with, of hosts the laptop's line reaches as
#: it reaches Europe.
_EUROPE = frozenset(
    ("AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR")
    + ("HU", "IE", "IS", "IT", "LT", "LU", "LV", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE")
    + ("SI", "SK", "UA")
)


def line_bytes_per_s(
    location: str = "", inet_up_mbps: float = 0.0, *, line: str = "proxy"
) -> tuple[float, bool]:
    """Bytes a second a pack comes home at from a host, and whether a transfer measured it.

    ``line`` is ``proxy`` (what every run so far used) or ``direct``: the
    laptop's line to the host's part of the world, by the country code its
    offer ends with, and no more than four fifths of what the host says
    its own line sends.
    """
    if line != "direct":
        return LINE["proxy"], True
    code = str(location).rsplit(",", 1)[-1].strip().upper()
    where = "europe" if code in _EUROPE else "united states" if code == "US" else "elsewhere"
    rate = LINE[where]
    if inet_up_mbps > 0.0:
        rate = min(rate, 0.8 * float(inet_up_mbps) * 1e6 / 8.0)
    return rate, False


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


def _a_source_s(card: Card, ppw: float, window_s: float) -> float:
    """Card seconds of one source position on this card and grid."""
    held = SOURCE_S.get((card.name, round(ppw, 2)))
    if held is None:
        grid = (ppw / batched.MEASURED_PPW) ** batched.PPW_EXPONENT
        held = batched.SOLVE_S_AT_1500 * grid / card.throughput
    return float(held) * window_s / batched.MEASURED_DURATION_S


def _a_fit_s(card: Card, ppw: float) -> float:
    held = FIT_S.get((card.name, round(ppw, 2)))
    return float(held) if held is not None else batched.PAIR_S / card.throughput


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
    low_levers: str | None = None,
    cpu_cores: float = 0.0,
    location: str = "",
    inet_up_mbps: float = 0.0,
    line: str = "proxy",
    queue: bool = True,
    launches: int | None = None,
    launch_sources: int | None = None,
    billed_dph: float | None = None,
) -> dict[str, Any] | None:
    """Wall hours and USD of a plan's trace on one offer; ``None`` where it cannot be said.

    ``billed_dph`` is what the rental is billed an hour where that is not
    the offer's ``dph_total``: the offer's hour with the disk the run asks
    for (:meth:`reverberate.gpu.vast.Offer.billed_dph`). The USD are of it.

    ``record`` is a plan's. ``seconds`` is the wall on the machine, stage
    after stage as the module's words give them, and sums to the hours;
    ``work`` is what the cards and the cores do in each, in their own
    seconds. The fetch is in the total: the machine bills while the pack
    comes home, so a host whose line is faster or whose hour is cheaper
    wins by it. ``low_levers`` is the form the pack is written in
    (:func:`reverberate.trace.plan.pack_pair_bytes`); ``line`` how it comes
    home (:func:`line_bytes_per_s`); ``cpu_cores`` the offer's, for the
    host's stages (unknown: enough of them).

    ``queue`` false prices the trace of before the queue, as the two whole
    scenes of 2026-10-05 ran: a launch's records on its card, so that a
    position heard at more cells than the card holds records for is solved
    more than once, and the early trace, the levelling and the write one
    after the other in one process. ``launches`` and ``launch_sources`` are
    then what that run's log says it planned.

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
    ppw = batched.MEASURED_PPW if low_ppw is None else float(low_ppw)
    if batched.cells_a_solve(float(gpu_ram_gb), ppw=low_ppw, duration_s=window) < 1:
        return None
    solves = positions
    if not queue and counts:
        # Records on the card: a position heard at more cells than it holds is solved again.
        counted = batched.solves_needed(counts, float(gpu_ram_gb), ppw=low_ppw, duration_s=window)
        if counted is None:
            return None
        solves = counted
    cards = max(1, int(num_gpus))
    base = plans.estimate(
        record,
        rate_usd_per_hour=dph_total,
        fetch_pairs=fetch_pairs,
        check=check,
        low_engine=low_engine,
        low_ppw=low_ppw,
        low_seconds=low_seconds,
        rays=rays,
        low_levers=low_levers,
    )
    planned = dict(base["seconds"])
    source_s, fit_s = _a_source_s(card, ppw, window), _a_fit_s(card, ppw)
    a_launch = int(launch_sources or batched.LAUNCH_SOURCES)
    made = int(launches) if launches else -(-solves // a_launch)
    from reverberate.trace.resources import REFERENCE

    work = {
        "solve_card_s": solves * source_s + made * float(REFERENCE["launch_s"]),
        "fit_card_s": pairs * fit_s,
        "rays_card_s": planned["rays"],
        "host_core_s": HOST_OVERHEAD
        * (
            (int(record["step_pairs"]) + pairs) * float(REFERENCE["paths_job_s"])
            + pairs * (float(REFERENCE["level_pair_s"]) + float(REFERENCE["row_pair_s"]))
        ),
    }
    tail = TAIL_LAUNCHES * min(a_launch, max(solves, 1)) * source_s if cards > 1 and solves else 0.0
    low = 0.0
    if solves:
        low = (work["solve_card_s"] + work["fit_card_s"]) / cards + tail
        low += float(REFERENCE["start_s"])
    prepare = PREPARE_S.get(round(ppw, 2), PREPARE_S[10.5])
    seconds: dict[str, float] = {"start": START_S, "prepare": prepare, "low": low}
    seconds["rays"] = work["rays_card_s"] / cards
    if queue:
        workers = max(1.0, min(float(cpu_cores) / 2.0, HOST_WORKERS_AT_MOST)) if cpu_cores else 16.0
        host = work["host_core_s"] / workers
        # Under the solves and the rays: only what a host of few cores takes beyond them.
        seconds["host_beyond_the_cards"] = max(0.0, host - seconds["low"] - seconds["rays"])
    else:
        seconds["paths"] = planned["paths"]
        seconds["level"] = planned["level"]
    seconds["write"] = planned["write"]
    seconds["check"] = planned["check"]
    rate, line_measured = line_bytes_per_s(location, inet_up_mbps, line=line)
    on_machine = float(sum(seconds.values()))
    seconds["transfer_pack"] = float(base["pack_gb"]) * 1e9 / rate
    home = float(base["pair_cache_gb"]) * 1e9 / rate if fetch_pairs else 0.0
    if synced and fetch_pairs:
        # Brought home while the machine works: only what the run leaves no time for.
        home = max(0.0, home - (on_machine - START_S))
    seconds["transfer_pairs"] = home
    total = float(sum(seconds.values()))
    billed = float(dph_total if billed_dph is None else billed_dph)
    return {
        "hours": total / 3600.0,
        "usd": total / 3600.0 * billed,
        "rate_usd_per_hour": round(billed, 4),
        "measured": card.measured,
        "note": f"{solves} solves, {card.name} x{cards} at {source_s:.0f} s a position;"
        f" the fetch {(seconds['transfer_pack'] + home) / 60.0:.0f} min of it;"
        f" billed {billed:.3f} USD/h with its disk",
        "seconds": {name: round(value, 1) for name, value in seconds.items()},
        "work": {name: round(value, 1) for name, value in work.items()},
        "solves": solves,
        "extra_solves": solves - positions,
        "launches": made,
        "card": card.name,
        "cards": cards,
        "throughput": card.throughput,
        "source_s": round(source_s, 1),
        "pack_gb": base["pack_gb"],
        "pair_cache_gb": base["pair_cache_gb"],
        "line_bytes_per_s": rate,
        "line_measured": line_measured,
        "fetch_usd": round((seconds["transfer_pack"] + home) / 3600.0 * billed, 3),
        "queue": bool(queue),
    }


def predictor(record: dict[str, Any], *, disk_gb: float = 0.0, **options: Any) -> Any:
    """:func:`predict` of ``record`` as :func:`reverberate.gpu.onebox.price_offers` calls it.

    ``disk_gb`` is the disk the rental will ask for: an offer that says
    what its disk costs is priced with it.
    """

    def priced(offer: Any) -> dict[str, Any] | None:
        billed = getattr(offer, "billed_dph", None)
        return predict(
            record,
            gpu_name=str(offer.gpu_name),
            num_gpus=int(offer.num_gpus),
            gpu_ram_gb=float(offer.gpu_ram_gb),
            dph_total=float(offer.dph_total),
            cpu_cores=float(getattr(offer, "cpu_cores", 0.0) or 0.0),
            location=str(getattr(offer, "location", "") or ""),
            inet_up_mbps=float(getattr(offer, "inet_up_mbps", 0.0) or 0.0),
            billed_dph=float(billed(disk_gb)) if billed is not None and disk_gb else None,
            **options,
        )

    return priced


# --------------------------------------------------------------------------
# what a run really took, from its own logs, against the prediction
# --------------------------------------------------------------------------

_STAMP = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \+\s*[0-9.]+ h \| "
_STAGE = re.compile(_STAMP + r"(\w+): (start|done in [0-9.]+ min)")
_BATCH = re.compile(
    _STAMP + r"batch of (\d+): (\d+) pair\(s\), solve ([0-9.]+) s at \S+ updates/s,"
    r" encode ([0-9.]+) s"
)
_PLANNED = re.compile(
    r"solve: (\d+) solve\(s\) in (\d+) batch\(es\) on (\d+) card\(s\), up to (\d+)"
)
_WATCH = re.compile(r"up\s+([0-9.]+) h cost\s+([0-9.]+) USD")
#: The stages of a trace as its log names them, in the order they run.
STAGES = ("voxelise", "plan", "assign", "solve", "paths", "rays", "level", "write", "check")


def _at(stamp: str) -> float:
    import calendar

    return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")))


def actual(home: Path) -> dict[str, Any]:
    """What a run took, read in what it brought home: ``pulled/campaign.log`` and the watches.

    A run that failed and was resumed is several attempts in one log.
    ``attempts`` has each one's stages in wall seconds (a stage that did
    not end is counted to the attempt's last line), its launches and their
    card seconds; ``stages`` sums them, ``idle_s`` is the time between an
    attempt's last line and the next one's first, and ``billed`` the last
    ``up H h cost X USD`` a watcher's line says, in ``onebox.json`` or in a
    driver's log kept beside it. Nothing is written.
    """
    home = Path(home)
    log = home / "pulled" / "campaign.log"
    attempts: list[dict[str, Any]] = []
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            stamp = re.match(_STAMP, line)
            if stamp is None:
                continue
            now = _at(stamp.group(1))
            if "| trace " in line and "steps" in line and "source(s)" in line:
                attempts.append(
                    {
                        "first": now,
                        "last": now,
                        "stages": {},
                        "open": {},
                        "launches": 0,
                        "sources": 0,
                        "pairs": 0,
                        "solve_card_s": 0.0,
                        "fit_card_s": 0.0,
                        "longest_launch_s": 0.0,
                        "failed": "",
                    }
                )
            if not attempts:
                continue
            held = attempts[-1]
            held["last"] = now
            stage = _STAGE.match(line)
            if stage is not None:
                name = stage.group(2)
                if stage.group(3) == "start":
                    held["open"][name] = now
                elif name in held["open"]:
                    began = held["open"].pop(name)
                    held["stages"][name] = held["stages"].get(name, 0.0) + now - began
            batch = _BATCH.match(line)
            if batch is not None:
                solve, fit = float(batch.group(4)), float(batch.group(5))
                held["launches"] += 1
                held["sources"] += int(batch.group(2))
                held["pairs"] += int(batch.group(3))
                held["solve_card_s"] += solve
                held["fit_card_s"] += fit
                held["longest_launch_s"] = max(held["longest_launch_s"], solve + fit)
            planned = _PLANNED.search(line)
            if planned is not None:
                held["planned"] = {
                    "solves": int(planned.group(1)),
                    "launches": int(planned.group(2)),
                    "cards": int(planned.group(3)),
                    "launch_sources": int(planned.group(4)),
                }
            if "trace FAILED" in line:
                held["failed"] = line.split("trace FAILED:", 1)[-1].strip()[:200]
    stages: dict[str, float] = {}
    for held in attempts:
        # A stage that did not end, ended with its attempt.
        for name, began in held.pop("open").items():
            held["stages"][name] = held["stages"].get(name, 0.0) + held["last"] - began
        for name, value in held["stages"].items():
            stages[name] = stages.get(name, 0.0) + value
    idle = sum(
        max(0.0, later["first"] - earlier["last"])
        for earlier, later in zip(attempts, attempts[1:], strict=False)
    )
    billed: dict[str, float] = {}
    texts = []
    record = home / "onebox.json"
    if record.is_file():
        texts += [str(line) for line in json.loads(record.read_text()).get("watches", [])]
    for name in sorted(home.glob("driver*.log")):
        texts += name.read_text(errors="replace").splitlines()
    for line in texts:
        watch = _WATCH.search(line)
        if watch is not None and float(watch.group(1)) >= billed.get("hours", 0.0):
            billed = {"hours": float(watch.group(1)), "usd": float(watch.group(2))}
    if billed.get("hours"):
        billed["rate_usd_per_hour"] = round(billed["usd"] / billed["hours"], 3)
    return {
        "attempts": attempts,
        "stages": {name: round(stages[name], 1) for name in STAGES if name in stages},
        "idle_s": round(idle, 1),
        "launches": sum(a["launches"] for a in attempts),
        "sources": sum(a["sources"] for a in attempts),
        "pairs": sum(a["pairs"] for a in attempts),
        "solve_card_s": round(sum(a["solve_card_s"] for a in attempts), 1),
        "fit_card_s": round(sum(a["fit_card_s"] for a in attempts), 1),
        "longest_launch_s": max((a["longest_launch_s"] for a in attempts), default=0.0),
        "on_machine_s": round(sum(a["last"] - a["first"] for a in attempts), 1),
        "billed": billed,
    }


def against(
    home: Path,
    *,
    gpu_name: str,
    num_gpus: int,
    gpu_ram_gb: float,
    dph_total: float | None = None,
    cpu_cores: float = 0.0,
    queue: bool | None = None,
    fetch_s: float | None = None,
) -> dict[str, Any]:
    """A run's prediction beside what it took, stage by stage: ``lines`` to print, and both.

    The plan and the grid are the bundle's the run left in ``home``; the
    machine is given, with the rate it was billed at (left out: the one
    its watches say). ``queue`` says which trace ran: left out, the log
    decides, a run whose ``paths`` stage has a line of its own being of
    before the queue. The prediction is then made as that run planned its
    launches, so that what is compared is the model and not the plan.
    """
    home = Path(home)
    took = actual(home)
    spec = json.loads((home / "bundle" / "campaign.json").read_text())
    record = json.loads((home / "bundle" / "trace" / "plan.json").read_text())
    told = dict(spec.get("trace") or {})
    low = dict(told.get("low") or {})
    rate = dph_total if dph_total is not None else took["billed"].get("rate_usd_per_hour", 0.0)
    if queue is None:
        queue = "paths" not in took["stages"]
    first = next((a.get("planned") for a in took["attempts"] if a.get("planned")), None) or {}
    longest = max(
        (a.get("planned", {}).get("launch_sources", 0) for a in took["attempts"]), default=0
    )
    said = predict(
        record,
        gpu_name=gpu_name,
        num_gpus=num_gpus,
        gpu_ram_gb=gpu_ram_gb,
        dph_total=float(rate),
        fetch_pairs=False,
        check=told.get("check"),
        low_ppw=low.get("ppw"),
        low_seconds=low.get("seconds"),
        low_levers=told.get("low_levers"),
        cpu_cores=cpu_cores,
        queue=bool(queue),
        launches=None if queue else first.get("launches"),
        launch_sources=None if queue else (longest or None),
    )
    if said is None:
        raise ValueError(f"no prediction for {num_gpus} x {gpu_name} of {gpu_ram_gb:g} GB")
    predicted = dict(said["seconds"])
    stages = dict(took["stages"])
    rows: list[tuple[str, float | None, float | None, str]] = [
        ("start: rental to first line", predicted["start"], None, "the driver's, not in the log"),
        (
            "prepare: voxelise, plan, assign",
            predicted["prepare"],
            sum(stages.get(n, 0.0) for n in ("voxelise", "plan", "assign")),
            f"{len(took['attempts'])} attempt(s)",
        ),
        (
            "solve, wall",
            predicted["low"],
            stages.get("solve"),
            f"{took['launches']} launches, the longest {took['longest_launch_s'] / 60.0:.1f} min",
        ),
        (
            "  solves, card seconds",
            said["work"]["solve_card_s"],
            took["solve_card_s"] or None,
            f"{said['solves']} predicted, {took['sources']} made;"
            f" {took['solve_card_s'] / max(took['sources'], 1):.1f} s a position",
        ),
        (
            "  fits, card seconds",
            said["work"]["fit_card_s"],
            took["fit_card_s"] or None,
            f"{record['pairs']} pairs planned, {took['pairs']} made",
        ),
        ("rays", predicted["rays"], stages.get("rays"), "a site a card when queued"),
    ]
    if queue:
        rows.append(
            (
                "host stages beyond the cards",
                predicted["host_beyond_the_cards"],
                sum(stages.get(n, 0.0) for n in ("paths", "level")) or None,
                "under the solves",
            )
        )
    else:
        rows += [
            ("paths (one process)", predicted["paths"], stages.get("paths"), ""),
            ("level (one process)", predicted["level"], stages.get("level"), ""),
        ]
    rows += [
        ("write", predicted["write"], stages.get("write"), ""),
        ("check", predicted["check"], stages.get("check"), ""),
        ("between attempts", 0.0, took["idle_s"], "a failure, then the relaunch"),
        (
            "fetch of the pack",
            predicted["transfer_pack"],
            fetch_s,
            f"{said['pack_gb']} GB at {said['line_bytes_per_s'] / 1e6:g} MB/s",
        ),
    ]
    lines = [
        f"{home.name}: {num_gpus} x {said['card']} at {float(rate):.3f} USD/h,"
        f" {'the queue' if queue else 'the trace of before the queue'};"
        f" {record['source_positions']} positions, {record['pairs']} pairs",
        f"{'stage':<34} {'predicted s':>12} {'actual s':>10} {'USD':>7} {'actual USD':>10}  note",
    ]
    total_p = total_a = 0.0
    for name, want, got, note in rows:
        inner = name.startswith("  ")
        if not inner:
            total_p += float(want or 0.0)
            total_a += float(got or 0.0)
        per = float(rate) / 3600.0
        lines.append(
            f"{name:<34} {'' if want is None else f'{want:12.0f}':>12}"
            f" {'' if got is None else f'{got:10.0f}':>10}"
            f" {'' if want is None or inner else f'{want * per:7.2f}':>7}"
            f" {'' if got is None or inner else f'{got * per:10.2f}':>10}  {note}"
        )
    billed = took["billed"]
    lines.append(
        f"{'sum of the stages':<34} {total_p:12.0f} {total_a:10.0f}"
        f" {total_p * float(rate) / 3600.0:7.2f} {total_a * float(rate) / 3600.0:10.2f}"
        "  the actual start is not in the log"
    )
    if billed:
        lines.append(
            f"billed at the last watch: {billed['hours']:.2f} h, {billed['usd']:.2f} USD;"
            f" predicted for the run as it was planned: {said['hours']:.2f} h,"
            f" {said['usd']:.2f} USD"
        )
    return {"lines": lines, "predicted": said, "actual": took, "rate_usd_per_hour": float(rate)}
