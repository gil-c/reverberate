"""Where a trace gets its low band pairs: the campaign on a card, or a monopole in free air.

A trace asks three things of the low band: where the arrays really stood,
the pairs it names solved into the dwelling's cache, and each pair's key.
:class:`CardPairs` answers with :class:`reverberate.accel.pairs.PairsCampaign`
on the rented machine. :class:`FreeFieldPairs` answers in closed form with
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

__all__ = ["CardPairs", "FreeFieldPairs", "PairsEngine"]


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

    def solve(self, heard_at: list[list[int]]) -> dict[str, Any]:
        solved = cached = 0
        for position, cells in enumerate(heard_at):
            for cell in cells:
                key = self.key_of(position, cell)
                if self.cache.has(key) or self.centres[cell] is None:
                    cached += 1
                    continue
                self.cache.write(
                    key,
                    self.response(position, cell),
                    {
                        "source_m": [float(v) for v in self.sources[position]],
                        "cell_m": [float(v) for v in self.cells[cell]],
                        "centre_m": [float(v) for v in np.asarray(self.centres[cell])],
                        "solver": self.solver,
                    },
                )
                self.solved.append((position, cell))
                solved += 1
        return {"solved": solved, "cached": cached, "source_positions_solved": 0}
