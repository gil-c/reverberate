"""Where a trace gets its low band pairs: a campaign on a card, or a monopole in free air.

A trace asks three things of the low band: where the arrays really stood,
the pairs it names solved into the dwelling's cache, and each pair's key.
:class:`CardPairs` answers with :class:`reverberate.accel.pairs.PairsCampaign`
on the rented machine, one engine process a source position.
:class:`BatchedPairs` answers with the same campaign on the batched solver
(:mod:`reverberate.wave.lowband`): many source positions a launch, the
receivers and the fit on the card, on the bundle's grid or on a cheaper one
under its own key. :class:`FreeFieldPairs` answers in closed form with
no room at all, so that the whole chain runs on a laptop and in the tests:
a pack made with it replays, and says in its provenance that its low band
is not a solve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from reverberate.accel.pairs import PairCache
from reverberate.spatial.field import monopole_coefficients
from reverberate.spatial.lowband import LOW_RATE_HZ, LOW_SAMPLES, pair_key
from reverberate.spatial.sh import degrees_of, scene_to_ambisonic

__all__ = [
    "BatchedPairs",
    "CardPairs",
    "FreeFieldPairs",
    "PairsEngine",
    "build_engine",
    "cache_levers",
]


def build_engine(
    told: dict[str, Any],
    bundle: Path,
    out: Path,
    *,
    gpu: bool | None = None,
    pffdtd_dir: Path | str = Path("/root/pffdtd"),
) -> Any:
    """The engine a command line asked for, from what it was told: in any process of the run.

    ``told`` is a few words (``kind`` and the grid's options), so that a
    worker's process builds the engine the run's own process built, from
    the same files, and nothing of it crosses a pipe.
    """
    engine = _engine(told, bundle, out, gpu=gpu, pffdtd_dir=pffdtd_dir)
    # The form the run's pairs are written in is the bundle's, in every process of the run.
    engine.cache.levers = cache_levers(bundle)
    return engine


def cache_levers(bundle: Path) -> str | None:
    """The form a bundle's run keeps its pair cache in (``trace.pair_cache``); ``None``: samples.

    A bundle that does not say, one made before the cache had a compact
    form, keeps the samples.
    """
    import json

    from reverberate.accel.pairs import _cache_levers

    spec = Path(bundle) / "campaign.json"
    if not spec.is_file():
        return None
    told = dict(json.loads(spec.read_text()).get("trace") or {}).get("pair_cache")
    return _cache_levers(str(told)) if told else None


def _engine(
    told: dict[str, Any],
    bundle: Path,
    out: Path,
    *,
    gpu: bool | None = None,
    pffdtd_dir: Path | str = Path("/root/pffdtd"),
) -> Any:
    kind = str(told.get("kind", "pffdtd"))
    if kind == "free-field":
        from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
        from reverberate.trace.assets import MirrorAssets

        held = Path(bundle) / "trace"
        with np.load(held / "plan.npz") as plan:
            cells = np.concatenate([plan["cells"], plan["patch_cells"].reshape(-1, 3)])
        mirror = MirrorAssets.load(held / "mirror")
        # In the cache form: on the geometric clock and the field's scale.
        return FreeFieldPairs(
            np.load(held / "positions.npy"),
            cells,
            out,
            sound_speed_m_s=mirror.settings.sound_speed_m_s,
            gain=FIELD_UNIT_AT_1M,
        )
    if kind == "lowband":
        return BatchedPairs(
            Path(bundle) / "pairs",
            out,
            pffdtd_dir=Path(pffdtd_dir),
            devices=told.get("devices"),
            gpu=gpu,
            scheme=str(told.get("scheme") or "cartesian"),
            ppw=told.get("ppw"),
            batch=told.get("batch"),
        )
    return CardPairs(
        Path(bundle) / "pairs",
        out,
        pffdtd_dir=Path(pffdtd_dir),
        devices=told.get("devices"),
        gpu=gpu,
        solvers=told.get("solvers"),
    )


class PairsEngine(Protocol):
    """What :class:`reverberate.trace.run.Trace` asks of the low band."""

    cache: PairCache
    voxel_low_key: str
    solver: str

    def key_of(self, position: int, cell: int) -> str: ...

    def place(self) -> list[np.ndarray | None]:
        """The centre the array of each cell stands on; ``None`` where none could stand."""
        ...

    def solve(self, heard_at: list[list[int]]) -> dict[str, Any]:
        """The pairs not yet in the cache, solved into it; how many were, and how many were not."""
        ...


class CardPairs:
    """The pairs campaign of a bundle, stage by stage, on the machine that holds the card."""

    def __init__(
        self,
        bundle: Path,
        out: Path,
        *,
        pffdtd_dir: Path,
        devices: str | None = None,
        gpu: bool | None = None,
        solvers: int | None = None,
    ) -> None:
        from reverberate.accel.pairs import PairsCampaign

        self.campaign = PairsCampaign(
            bundle=bundle,
            out=out,
            pffdtd_dir=pffdtd_dir,
            devices=devices,
            gpu=gpu,
            solvers=solvers,
        )
        self.cache = self.campaign.cache
        self.voxel_low_key = self.campaign.keys["low"]
        self.solver = str(self.campaign.spec["solver"])
        self.report: dict[str, Any] = {}

    def key_of(self, position: int, cell: int) -> str:
        return self.campaign.key_of(position, cell)

    def place(self) -> list[np.ndarray | None]:
        self.report["voxelise"] = self.campaign.stage("voxelise", self.campaign.voxelise)
        placed = self.campaign.stage("plan", self.campaign.place)
        self.report["placed"] = {k: v for k, v in placed.items() if k != "centres"}
        return [None if c is None else np.asarray(c, dtype=float) for c in placed["centres"]]

    def solve(self, heard_at: list[list[int]]) -> dict[str, Any]:
        self.campaign.heard_at = [sorted(int(c) for c in cells) for cells in heard_at]
        wanted = sum(len(cells) for cells in heard_at)
        solves = self.campaign.stage("solve", self.campaign.solve)
        solved = sum(int(r["cells"]) for r in solves)
        self.report["solves"] = solves
        return {
            "solved": solved,
            "cached": wanted - solved,
            "source_positions_solved": sum(1 for r in solves if not r.get("skipped")),
        }


class BatchedPairs(CardPairs):
    """The pairs campaign on the batched low band solver; what a trace asks is unchanged.

    ``scheme`` and ``ppw`` name the grid: left alone, the bundle's own. The
    report carries the grid as the solver cut it, a record a launch, and
    what a source position and a pair cost in seconds.
    """

    def __init__(
        self,
        bundle: Path,
        out: Path,
        *,
        pffdtd_dir: Path = Path("/root/pffdtd"),
        devices: str | None = None,
        gpu: bool | None = None,
        scheme: str = "cartesian",
        ppw: float | None = None,
        batch: int | None = None,
    ) -> None:
        from reverberate.wave.lowband.pairs import LowbandPairs

        self.lowband = LowbandPairs(
            bundle=bundle,
            out=out,
            pffdtd_dir=pffdtd_dir,
            devices=devices,
            gpu=gpu,
            scheme=scheme,
            ppw=ppw,
            batch=batch,
        )
        self.campaign = self.lowband
        self.cache = self.campaign.cache
        self.voxel_low_key = self.campaign.keys["low"]
        self.solver = str(self.campaign.spec["solver"])
        self.report = {}

    def solve(self, heard_at: list[list[int]]) -> dict[str, Any]:
        found = super().solve(heard_at)
        self.report["grid"] = getattr(self.lowband, "problem_record", {})
        self.report["batches"] = self.lowband.batches
        return found

    # ---- the solve as jobs of the trace's queue (:mod:`reverberate.trace.pool`) -------------

    #: Where a launch runs: on a worker that holds a card.
    launches_on = "card"

    def launches(
        self, heard_at: list[list[int]], *, free_bytes: list[float], host_bytes: float
    ) -> list[dict[str, Any]]:
        """Every launch still to make: ``name``, ``payload``, ``bytes`` on a card, ``pairs``."""
        self.campaign.heard_at = [sorted(int(c) for c in cells) for cells in heard_at]
        return [
            {
                "name": held["name"],
                "payload": {"items": held["items"]},
                "bytes": held["bytes"],
                "updates": held["updates"],
                "pairs": held["pairs"],
            }
            for held in self.lowband.launches(free_bytes, host_bytes)
        ]

    def kept_bytes(self) -> float:
        """What this process keeps on its card for the run, once it has made a launch: the fit."""
        if getattr(self.lowband, "_ready", None) is None:
            return 0.0
        return float(self.lowband.fit_bytes())

    def run_launch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One launch on this process's device; in ``parted`` where the device does not hold it."""
        record = self.lowband.run_launch(payload["items"])
        if "parted" in record:
            return {
                "parted": [
                    {"name": p["name"], "payload": {"items": p["items"]}, "bytes": p["bytes"]}
                    for p in record["parted"]
                ]
            }
        return record

    def solved_report(
        self, records: list[dict[str, Any]], heard_at: list[list[int]]
    ) -> dict[str, Any]:
        """What the launches made, as :meth:`solve` says it."""
        solves = self.lowband.gathered([r for r in records if r.get("batch")])
        wanted = sum(len(cells) for cells in heard_at)
        solved = sum(int(r["cells"]) for r in solves)
        self.report["solves"] = solves
        self.report["grid"] = getattr(self.lowband, "problem_record", {})
        self.report["batches"] = self.lowband.batches
        return {
            "solved": solved,
            "cached": wanted - solved,
            "source_positions_solved": len(solves),
        }


class FreeFieldPairs:
    """A monopole in free air at every pair, in the cache form: no room, no card, no solve.

    The response is the interior expansion of a point source about the cell
    (:func:`reverberate.spatial.field.monopole_coefficients`), times
    ``gain`` and band limited as a low only solve is. The cache form is on
    the geometric clock and on the field's scale: ``lead_s`` zero and
    ``gain`` :data:`reverberate.spatial.lowband.FIELD_UNIT_AT_1M`. A
    ``lead_s`` delays the response, which is how a field's low band comes
    and what a trace refuses. ``centres`` moves the arrays off the cells
    asked for, as a grid does.
    """

    def __init__(
        self,
        sources: np.ndarray,
        cells: np.ndarray,
        out: Path,
        *,
        sound_speed_m_s: float = 343.2,
        lead_s: float = 0.0,
        gain: float = 1.0,
        centres: list[np.ndarray | None] | None = None,
    ) -> None:
        self.sources = np.asarray(sources, dtype=float).reshape(-1, 3)
        self.cells = np.asarray(cells, dtype=float).reshape(-1, 3)
        self.voxel_low_key = "free-field"
        self.solver = "free-field monopole, no room: reverberate.trace.engines/1"
        self.cache = PairCache(Path(out) / "pairs", self.voxel_low_key)
        self.sound_speed_m_s = float(sound_speed_m_s)
        self.lead_s, self.gain = float(lead_s), float(gain)
        self.centres = centres if centres is not None else list(self.cells)
        #: The pairs solved, in order: what a resumed run must leave empty.
        self.solved: list[tuple[int, int]] = []

    def key_of(self, position: int, cell: int) -> str:
        return pair_key(
            self.voxel_low_key,
            self.sources[position],
            self.cells[cell],
            encoder={"order": 7},
            solver=self.solver,
        )

    def place(self) -> list[np.ndarray | None]:
        return [None if c is None else np.asarray(c, dtype=float) for c in self.centres]

    def response(self, position: int, cell: int) -> np.ndarray:
        """``[64, 4800]`` float32: the monopole at ``position`` seen from the cell's centre."""
        centre = self.centres[cell]
        if centre is None:
            raise ValueError(f"no array stands on cell {cell}")
        freqs = np.fft.rfftfreq(LOW_SAMPLES, 1.0 / LOW_RATE_HZ)
        keep = freqs > 80.0
        k = 2.0 * np.pi * freqs[keep] / self.sound_speed_m_s
        seen = scene_to_ambisonic((self.sources[position] - np.asarray(centre))[None, :])[0]
        spectrum = np.zeros((freqs.size, 64), dtype=complex)
        spectrum[keep] = (
            4.0 * np.pi * monopole_coefficients(seen, k, 7) / (1j ** degrees_of(7))[None]
        )
        edges = np.clip((freqs - 80.0) / 80.0, 0, 1) * np.clip((1800.0 - freqs) / 200.0, 0, 1)
        spectrum *= (0.5 - 0.5 * np.cos(np.pi * edges))[:, None]
        spectrum *= np.exp(-2j * np.pi * freqs * self.lead_s)[:, None]
        scale = self.gain * LOW_RATE_HZ / 48000.0
        response = np.fft.irfft(spectrum.T, LOW_SAMPLES, axis=-1) * scale
        # A solve starts in silence and does not wrap; a band limited expansion rings
        # both ways. It is faded in over the lead, or without one over the time the sound
        # takes to arrive, and out over the last 50 ms.
        time_s = np.arange(LOW_SAMPLES) / LOW_RATE_HZ
        before = (
            self.lead_s
            if self.lead_s > 0.0
            else float(np.linalg.norm(seen)) / (self.sound_speed_m_s)
        )
        rise = np.clip(time_s / before, 0.0, 1.0) if before > 0.0 else np.ones_like(time_s)
        fall = np.clip((time_s[-1] - time_s) / 0.05, 0.0, 1.0)
        window = (0.5 - 0.5 * np.cos(np.pi * rise)) * (0.5 - 0.5 * np.cos(np.pi * fall))
        return np.asarray(response * window[None, :], dtype=np.float32)

    #: Where a launch runs: on any worker, with ``numpy``.
    launches_on = "host"
    #: Source positions a launch holds.
    LAUNCH_POSITIONS = 8

    def launches(self, heard_at: list[list[int]], **_: Any) -> list[dict[str, Any]]:
        """The pairs not yet in the cache, a few source positions a launch."""
        items: list[tuple[int, list[int]]] = []
        for position, cells in enumerate(heard_at):
            todo = [
                int(cell)
                for cell in cells
                if self.centres[cell] is not None
                and not self.cache.has(self.key_of(position, cell))
            ]
            if todo:
                items.append((position, todo))
        return [
            {
                "name": f"p{block[0][0]:06d}",
                "payload": {"items": block},
                "bytes": 0.0,
                "pairs": [(position, cell) for position, cells in block for cell in cells],
            }
            for block in (
                items[first : first + self.LAUNCH_POSITIONS]
                for first in range(0, len(items), self.LAUNCH_POSITIONS)
            )
        ]

    def run_launch(self, payload: dict[str, Any]) -> dict[str, Any]:
        made = 0
        for position, cells in payload["items"]:
            for cell in cells:
                key = self.key_of(int(position), int(cell))
                if self.cache.has(key):
                    continue
                self.cache.write(
                    key,
                    self.response(int(position), int(cell)),
                    {
                        "source_m": [float(v) for v in self.sources[position]],
                        "cell_m": [float(v) for v in self.cells[cell]],
                        "centre_m": [float(v) for v in np.asarray(self.centres[cell])],
                        "solver": self.solver,
                    },
                )
                self.solved.append((int(position), int(cell)))
                made += 1
        return {"pairs": made}

    def solved_report(
        self, records: list[dict[str, Any]], heard_at: list[list[int]]
    ) -> dict[str, Any]:
        solved = sum(int(r.get("pairs", 0)) for r in records)
        wanted = sum(len(cells) for cells in heard_at)
        return {"solved": solved, "cached": wanted - solved, "source_positions_solved": 0}

    def solve(self, heard_at: list[list[int]]) -> dict[str, Any]:
        records = [self.run_launch(launch["payload"]) for launch in self.launches(heard_at)]
        return self.solved_report(records, heard_at)
