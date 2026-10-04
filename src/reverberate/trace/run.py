"""The trace on the machine: a bundle in, ``pack.h5`` out, resumed from what is on disk.

One process, the stages in order, each leaving what the next and a rerun
look for:

1. ``voxelise``, ``plan``: the low grid and an array at every cell
   (:class:`reverberate.accel.pairs.PairsCampaign`); the centres the arrays
   really got are the pack's ``/cells``;
2. ``assign``: the cell rule on those centres (:func:`reverberate.trace.plan.assign`),
   which names the pairs;
3. ``solve``: the pairs not in the cache, a source position at a time;
4. ``paths``: the batched early trace of every source over its audible
   steps, and of every pair at rest for the levelling, with the diffracted
   onsets (``early/<source>.npz``);
5. ``rays``: the histograms at the tail's sites over the tail's cells, in
   their cache (``tails/``), and each source's table;
6. ``level``: a pair's seam and onset (``level.jsonl``, a line a pair);
7. ``write``: the pack, a source at a time, through
   :class:`reverberate.render.pack.PackWriter`;
8. ``check``: the pack read back whole, and rendered by the signal engine on
   the host and on the card, which must agree to 1e-6 of the peak.

``status.json``, ``campaign.log`` and the two markers are a campaign's, so
:mod:`reverberate.gpu.onebox` watches a trace as it watches a field.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.accel.pairs import PairCache
from reverberate.audio import Atmosphere
from reverberate.compute import Devices, device_report, to_numpy, xp_for
from reverberate.metrics import band_centres
from reverberate.mirror.hybrid import Crossover
from reverberate.mirror.moving import (
    KIND_DIRECT,
    EarlyTable,
    MovingSettings,
    prepare,
    trace_early,
)
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.tails import TailCache, TailTable, tail_table
from reverberate.render.pack import (
    Air,
    Cells,
    Early,
    Header,
    Level,
    Listener,
    Low,
    Mirror,
    PackWriter,
    Source,
    Tail,
    default_fusion,
    read_pack,
    tail_seed,
)
from reverberate.scenes import canonical_bytes, load_recipe
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M, LOW_RATE_HZ
from reverberate.spatial.translate import clearance_m
from reverberate.trace.assets import MirrorAssets, directivity_models, found_assets, mismatched
from reverberate.trace.engines import CardPairs, PairsEngine
from reverberate.trace.level import (
    first_arrival_s,
    mirror_omni,
    pair_low,
    pair_seam_db,
    step_levels,
)
from reverberate.trace.plan import (
    Profile,
    Tracks,
    assign,
    pairs_of,
    tail_cells,
    tail_sites_of,
    tracks_of,
)

__all__ = ["KIND", "Journal", "Trace", "render_check", "run_trace"]

#: What a trace's ``campaign.json`` says it is, and the command line reads.
KIND = "scene-trace"
#: (step, image) pairs validated at once on a card: the appendix's figure, 5000 steps and 1 GB.
CARD_PAIRS_PER_VALIDATION = 3_000_000
#: The engine's two array modules must agree to this share of the peak (V4).
RENDER_TOLERANCE = 1e-6
#: The pairs' onsets must trail their direct sound by the pack's lead to this, s: half a
#: period of the crossover's 1 kHz, beyond which the two bands cancel at the join.
CLOCK_S = 0.5e-3
#: Seconds of the pack the check renders, and the block it renders them in.
CHECK_SECONDS = 60.0
CHECK_BLOCK_S = 5.0

_EARLY_FIELDS = (
    "offsets",
    "path_id",
    "delay_s",
    "arrival",
    "departure",
    "gain",
    "order",
    "kind",
    "length_m",
    "sequence",
    "rank",
    "source",
    "listener",
)


@dataclass
class Journal:
    """The log, the status and the timings of a run that has no campaign to keep them."""

    out: Path
    started: float = field(default_factory=time.time)
    status: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    quiet: bool = False

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.status = {
            "stage": "start",
            "started": self.started,
            "device": device_report(),
            "cpus": os.cpu_count(),
        }

    def say(self, message: str) -> None:
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} +{(time.time() - self.started) / 3600:.2f} h"
            f" | {message}"
        )
        if not self.quiet:
            print(line, flush=True)
        with (self.out / "campaign.log").open("a") as handle:
            handle.write(line + "\n")

    def set_status(self, **fields: Any) -> None:
        self.status.update(fields)
        self.status["updated"] = time.time()
        self.status["elapsed_s"] = round(time.time() - self.started, 1)
        path = self.out / "status.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status, indent=1, default=str))
        tmp.replace(path)

    def stage(self, name: str, work: Any, **status: Any) -> Any:
        self.set_status(stage=name, stage_started=time.time(), **status)
        self.say(f"{name}: start")
        t0 = time.time()
        result = work()
        self.timings[name] = round(self.timings.get(name, 0.0) + time.time() - t0, 1)
        self.say(f"{name}: done in {(time.time() - t0) / 60:.1f} min")
        return result


def _digest(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _save_early(path: Path, table: EarlyTable, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        digest=np.asarray(digest),
        sound_speed_m_s=np.asarray(table.sound_speed_m_s),
        record=np.asarray(json.dumps(table.record, default=str)),
        **{name: getattr(table, name) for name in _EARLY_FIELDS},
    )
    partial.replace(path)


def _load_early(path: Path, digest: str) -> EarlyTable | None:
    if not path.is_file():
        return None
    with np.load(path) as held:
        if str(held["digest"]) != digest:
            return None
        return EarlyTable(
            **{name: np.asarray(held[name]) for name in _EARLY_FIELDS},
            sound_speed_m_s=float(held["sound_speed_m_s"]),
            record=json.loads(str(held["record"])),
        )


@dataclass
class Trace:
    """One recipe's pack, made on this machine from a bundle."""

    bundle: Path
    out: Path
    #: Where the pairs come from; the bundle's campaign on a card unless given.
    engine: PairsEngine | None = None
    gpu: bool | None = None
    devices: Devices | None = None
    pffdtd_dir: Path = Path("/root/pffdtd")
    card_devices: str | None = None
    solvers: int | None = None
    quiet: bool = False

    def __post_init__(self) -> None:
        self.bundle, self.out = Path(self.bundle), Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.spec = json.loads((self.bundle / "campaign.json").read_text())
        if self.spec.get("kind") != KIND:
            raise ValueError(f"{self.bundle} is not the bundle of a scene trace")
        told = dict(self.spec["trace"])
        held = self.bundle / "trace"
        self.recipe = load_recipe(held / "recipe.json")
        self.recipe_bytes = canonical_bytes(self.recipe)
        self.profile = Profile.from_record(told["profile"])
        self.crossover = Crossover(**told.get("crossover", {}))
        self.told = told
        self.assets = MirrorAssets.load(held / "mirror")
        with np.load(held / "plan.npz") as plan:
            self.asked = np.asarray(plan["cells"], dtype=float)
            self.kind = np.asarray(plan["kind"], dtype=np.uint8)
            self.patch_cells = np.asarray(plan["patch_cells"], dtype=float).reshape(-1, 3)
            self.patch_source = int(plan["patch_source"])
        self.tracks: Tracks = tracks_of(self.recipe, self.profile)
        self.xp = xp_for(self.gpu)
        if self.devices is None:
            self.devices = Devices.detect() if self.xp is not np else Devices.host(1)
        if self.engine is None:
            self.engine = CardPairs(
                self.bundle / "pairs",
                self.out,
                pffdtd_dir=self.pffdtd_dir,
                devices=self.card_devices,
                gpu=self.gpu,
                solvers=self.solvers,
            )
        campaign = getattr(self.engine, "campaign", None)
        self.journal: Any = (
            campaign if campaign is not None else Journal(self.out, quiet=self.quiet)
        )
        self.report: dict[str, Any] = {"kind": KIND, "profile": self.profile.record()}
        sources = self.bundle / "trace" / "positions.npy"
        if sources.is_file() and not np.array_equal(np.load(sources), self.tracks.positions):
            raise RuntimeError(
                "the source positions of the bundle are not those this code reads in the recipe"
            )

    # ---- what the bundle is checked against ------------------------------------------------

    def check_assets(self) -> list[str]:
        """The recipe's asset keys that are not this trace's; refused unless the bundle allows."""
        assert self.engine is not None
        found = found_assets(
            self.recipe,
            self.assets,
            voxel_low_key=self.engine.voxel_low_key,
            export_sha256=str(self.told.get("export_sha256", "")),
        )
        wrong = mismatched(self.recipe, found)
        self.report["assets"] = found
        self.report["assets_mismatched"] = wrong
        if wrong and not self.told.get("allow_asset_mismatch", False):
            raise RuntimeError(
                "the recipe was generated against other assets than this trace finds: "
                + ", ".join(wrong)
            )
        if wrong:
            self.journal.say(f"assets: the recipe disagrees on {', '.join(wrong)}; allowed")
        return wrong

    # ---- stages ----------------------------------------------------------------------------

    def assign(self, centres: list[np.ndarray | None]) -> None:
        """The cell rule on the centres the arrays got, and the pairs it names."""
        scene = self.asked.shape[0]
        if len(centres) != scene + self.patch_cells.shape[0]:
            raise RuntimeError(f"{len(centres)} arrays for {scene} cells and the patch")
        self.placed = np.array([c is not None for c in centres[:scene]], dtype=bool)
        self.cells = np.array(
            [self.asked[i] if c is None else c for i, c in enumerate(centres[:scene])], dtype=float
        ).reshape(-1, 3)
        self.clearance = clearance_m(self.cells, self.assets.triangles)
        moved = np.linalg.norm(self.cells - self.asked, axis=1)
        self.low, refused = assign(self.tracks, self.cells, self.clearance, usable=self.placed)
        self.heard_at = pairs_of(self.tracks, self.low)
        if self.patch_cells.shape[0] and self.patch_source >= 0:
            stood = [scene + i for i, c in enumerate(centres[scene:]) if c is not None]
            self.heard_at[self.patch_source] = sorted(
                set(self.heard_at[self.patch_source]) | set(stood)
            )
        self.pairs = sorted(
            (position, cell)
            for position, cells in enumerate(self.heard_at)
            for cell in cells
            if cell < scene
        )
        self.report["cells"] = {
            "asked": scene,
            "without_an_array": int((~self.placed).sum()),
            "moved_m": {
                "median": round(float(np.median(moved)), 4) if scene else 0.0,
                "worst": round(float(moved.max()), 4) if scene else 0.0,
            },
            "patch": int(self.patch_cells.shape[0]),
            "patch_without_an_array": sum(1 for c in centres[scene:] if c is None),
        }
        self.report["fallback_steps"] = [[name, step] for name, step in refused]
        self.report["pairs"] = {
            "scene": len(self.pairs),
            "all": sum(len(cells) for cells in self.heard_at),
        }
        self.journal.say(
            f"assign: {scene} cells ({int((~self.placed).sum())} without an array),"
            f" {len(self.pairs)} pairs, {len(refused)} step(s) the rule refused"
        )

    def solve(self) -> None:
        """The pairs the bundle carried into the cache, then those still to be solved."""
        assert self.engine is not None
        carried_root = self.bundle / "pairs_cache"
        carried = 0
        if (carried_root / self.engine.voxel_low_key).is_dir():
            held = PairCache(carried_root, self.engine.voxel_low_key)
            for key, record in held.records().items():
                if held.has(key) and not self.engine.cache.has(key):
                    self.engine.cache.write(
                        key, held.read(key), {k: v for k, v in record.items() if k != "key"}
                    )
                    carried += 1
        self.report["low_pairs"] = {**self.engine.solve(self.heard_at), "carried": carried}

    def paths(self) -> None:
        """Every source's early table, and the table of the pairs at rest."""
        settings = self.assets.settings
        c = settings.sound_speed_m_s
        moving = MovingSettings()
        if self.xp is not np:
            moving = replace(moving, pairs_per_validation=CARD_PAIRS_PER_VALIDATION)
        tracks = self.tracks
        self.ms = prepare(self.assets.catalogue, settings, moving)
        every = [tracks.listener, self.cells, tracks.positions]
        every += [track.position for track in tracks.sources.values()]
        heads = np.concatenate([tracks.listener, self.cells])
        region = (heads.min(axis=0) - 0.5, heads.max(axis=0) + 0.5)
        onsets = onset_field(self.assets.catalogue, np.concatenate(every), sound_speed_m_s=c)
        identity = [self.assets.catalogue.key, settings.record(), [list(r) for r in region]]
        self.early: dict[str, EarlyTable] = {}
        records: dict[str, Any] = {}

        def table(name: str, source: np.ndarray, head: np.ndarray, audible: Any) -> EarlyTable:
            heard = np.ones(source.shape[0], dtype=bool) if audible is None else audible
            digest = _digest(source, head, heard, identity)
            path = self.out / "early" / f"{name}.npz"
            found = _load_early(path, digest)
            if found is None:
                found = trace_early(
                    self.ms, source, head, audible=heard, region=region, onsets=onsets, xp=self.xp
                )
                _save_early(path, found, digest)
                records[name] = found.record
            else:
                records[name] = {"cached": True}
            return found

        for number, (name, track) in enumerate(tracks.sources.items(), start=1):
            self.journal.set_status(source=name, job=f"{number}/{len(tracks.sources)}")
            self.early[name] = table(name, track.position, tracks.listener, track.audible)
        rows = np.asarray(self.pairs, dtype=np.int64).reshape(-1, 2)
        self.journal.set_status(source="pairs at rest", job=f"{rows.shape[0]} pairs")
        self.pair_early = table(
            "_pairs", tracks.positions[rows[:, 0]], self.cells[rows[:, 1]], None
        )
        self.report["paths"] = records
        self.report["distance_fields"] = int(onsets.solved)

    def rays(self) -> None:
        """Each source's tail table, in the pack's types; the histograms in their cache."""
        settings = self.assets.settings
        tracks = self.tracks
        self.tail_rows = tail_cells(self.cells, self.kind)
        self.tail_cache = TailCache(self.out / "tails")
        self.tail: dict[str, dict[str, np.ndarray]] = {}
        # The source a pair is levelled with: the first whose steps read it.
        self.owner: dict[tuple[int, int], str] = {}
        for number, (name, track) in enumerate(tracks.sources.items(), start=1):
            self.journal.set_status(source=name, job=f"{number}/{len(tracks.sources)}")
            held = tail_table(
                self.ms,
                settings,
                track.position,
                tracks.listener,
                tail_sites_of(self.recipe, track),
                self.cells[self.tail_rows],
                audible=track.audible,
                devices=self.devices,
                cache=self.tail_cache,
                xp=self.xp,
            ).pack()
            held["hist_cell"] = self.tail_rows[held["hist_cell"]].astype(np.int32)
            self.tail[name] = held
            chosen = self.low[name]
            for step in np.flatnonzero(track.audible):
                for position in track.slot[step]:
                    for cell in chosen.cell[step]:
                        if position >= 0 and cell >= 0:
                            self.owner.setdefault((int(position), int(cell)), name)
        self.report["rays"] = {
            "tail_cells": int(self.tail_rows.size),
            "histograms_traced": int(self.tail_cache.misses),
            "histograms_read": int(self.tail_cache.hits),
        }

    def pair_tails(self, name: str, mine: list[int]) -> TailTable:
        """The tails of the pairs in ``mine`` at rest, as the pack gives them to ``name``'s steps.

        A step with the source on the pair's solved position and the head on
        its cell: the sites and the cells either side, and their weights.
        """
        rows = np.asarray([self.pairs[j] for j in mine], dtype=np.int64)
        return tail_table(
            self.ms,
            self.assets.settings,
            self.tracks.positions[rows[:, 0]],
            self.cells[rows[:, 1]],
            tail_sites_of(self.recipe, self.tracks.sources[name]),
            self.cells[self.tail_rows],
            devices=self.devices,
            cache=self.tail_cache,
            xp=self.xp,
        )

    def level_pair(
        self, j: int, tails: TailTable, local: int, atmosphere: Atmosphere
    ) -> dict[str, Any]:
        """Pair ``j``: its seam, its onset, the mirror's first arrival, how its response ends."""
        assert self.engine is not None
        settings = self.assets.settings
        _, onset, aired = pair_low(
            self.engine.cache.read(self.pair_key[j]),
            self.crossover,
            atmosphere,
            sound_speed_m_s=settings.sound_speed_m_s,
            lead_s=self.assets.pack_lead_s,
        )
        mirror = mirror_omni(
            self.pair_early,
            j,
            tails,
            local,
            self.assets,
            seed=settings.seed + self.pairs[j][1],
            xp=self.xp,
        )
        return {
            "seam_db": pair_seam_db(aired, mirror, self.crossover),
            "onset_s": onset,
            "first_s": first_arrival_s(self.pair_early, j),
            # A diffracted onset has no reflection either: the direct path is a kind, not an order.
            "direct": bool(np.any(self.pair_early.kind[self.pair_early.rows(j)] == KIND_DIRECT)),
            # The engine convolves a response as it stands: one that rings up to its
            # last 50 ms was cut, or wrapped, by whatever made it.
            "end_db": _end_db(aired),
        }

    def level(self) -> None:
        """Every pair's seam and onset, a line each in ``level.jsonl``."""
        assert self.engine is not None
        atmosphere = Atmosphere(**self.recipe.atmosphere.to_dict())
        settings = self.assets.settings
        ledger = self.out / "level.jsonl"
        known: dict[str, dict[str, Any]] = {}
        identity = _digest(
            self.crossover.record(),
            self.recipe.atmosphere.to_dict(),
            settings.record(),
            {"lead_s": self.assets.pack_lead_s},
        )
        if ledger.is_file():
            for line in ledger.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("identity") == identity:
                        known[str(record["key"])] = record
        self.pair_key = [self.engine.key_of(position, cell) for position, cell in self.pairs]
        missing = [key for key in self.pair_key if not self.engine.cache.has(key)]
        if missing:
            raise RuntimeError(f"{len(missing)} pair(s) are not in the cache after the solve")
        made = 0
        started = time.time()
        # A source at a time: the tails of its pairs are a table that is not kept.
        todo: dict[str, list[int]] = {}
        for j, key in enumerate(self.pair_key):
            if key not in known:
                todo.setdefault(self.owner[self.pairs[j]], []).append(j)
        with ledger.open("a") as handle:
            for owner, mine in todo.items():
                tails = self.pair_tails(owner, mine)
                for local, j in enumerate(mine):
                    record = {
                        "key": self.pair_key[j],
                        "identity": identity,
                        **self.level_pair(j, tails, local, atmosphere),
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    known[self.pair_key[j]] = record
                    made += 1
                    if made % 50 == 0:
                        elapsed = time.time() - started
                        left = len(self.pair_key) - len(known)
                        self.journal.set_status(
                            job=f"{len(known)}/{len(self.pair_key)}",
                            remaining_s=round(elapsed / made * left, 1),
                        )
        self.levels = [known[key] for key in self.pair_key]
        seams = np.array([r["seam_db"] for r in self.levels], dtype=float)
        trails = np.array([r["onset_s"] - r["first_s"] for r in self.levels if r["direct"]])
        self.report["level"] = {
            "pairs": len(self.levels),
            "made": made,
            "read": len(self.levels) - made,
            "seam_db": _spread(seams),
            "end_db": _spread(np.array([r["end_db"] for r in self.levels], dtype=float), 1),
            # What a pair's loudest sample trails its direct sound by: the pack's lead and
            # the solver's pulse. Away from the lead, the two bands are not on one clock.
            "trail_s_of_pairs_with_a_direct_path": _spread(trails, 6),
            "lead_s": self.assets.pack_lead_s,
        }
        if trails.size and abs(float(np.median(trails)) - self.assets.pack_lead_s) > CLOCK_S:
            raise RuntimeError(
                "the low band and the mirror are not on one clock: the pairs' onsets trail their"
                f" direct sound by {np.median(trails) * 1e3:.2f} ms in the median, lead included,"
                f" and the pack's lead is {self.assets.pack_lead_s * 1e3:.2f} ms. The pair cache"
                " is on the geometric clock: a response in it starts when its source does"
            )

    def write(self) -> Path:
        """``pack.h5``, a source at a time."""
        assert self.engine is not None
        recipe, tracks, settings = self.recipe, self.tracks, self.assets.settings
        atmosphere = Atmosphere(**recipe.atmosphere.to_dict())
        floor = recipe.dwelling.floor_y_m
        stations = np.asarray([s.position for s in recipe.stations], dtype=float).reshape(-1, 3)
        rooms = [s.room for s in recipe.stations]
        if stations.shape[0]:
            near = np.linalg.norm(self.cells[:, None, :] - stations[None, :, :], axis=2).argmin(
                axis=1
            )
            room = tuple(rooms[int(i)] for i in near)
        else:
            room = ("",) * self.cells.shape[0]
        rays = settings.traced_rays()
        models = directivity_models()
        named = sorted(
            {recipe.source(name).directivity.model for name in tracks.sources} | {"omni"}
        )
        provenance = {
            "recipe_sha256": hashlib.sha256(self.recipe_bytes).hexdigest(),
            "assets": self.report.get("assets", {}),
            "assets_mismatched": self.report.get("assets_mismatched", []),
            "code_version": str(self.told.get("code_version", "unknown")),
            "solver": self.engine.solver,
            "low_pairs": self.report.get("low_pairs", {}),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": self.profile.record(),
            "fallback_steps": len(self.report.get("fallback_steps", [])),
            "device": self.journal.status.get("device", {}),
            "timings_s": dict(self.journal.timings),
            # A cost without its rate is not written: the laptop, which rented, adds them.
            "cost": [],
        }
        header = Header(
            profile="trace",
            recipe_sha256=provenance["recipe_sha256"],
            dwelling=recipe.dwelling.name,
            scene_id=recipe.dwelling.scene_id,
            duration_s=tracks.duration_s,
            steps=tracks.steps,
            sound_speed_m_s=settings.sound_speed_m_s,
            bank_bands_hz=band_centres(48000),
            has_low=True,
            has_tail=True,
            fusion=default_fusion(),
            provenance=provenance,
        )
        target = self.out / "pack.h5"
        partial = self.out / "pack.partial.h5"
        seam = np.array([r["seam_db"] for r in self.levels], dtype=float)
        onset = np.array([r["onset_s"] for r in self.levels], dtype=float)
        trail = np.array([r["onset_s"] - r["first_s"] for r in self.levels], dtype=float)
        row_of = {pair: j for j, pair in enumerate(self.pairs)}
        with PackWriter(
            partial,
            header,
            self.recipe_bytes,
            Listener(tracks.listener, tracks.orientation.astype(np.float32)),
            Cells(
                position=self.cells,
                kind=self.kind,
                lattice_index=np.full((self.cells.shape[0], 3), -1, dtype=np.int32),
                clearance_m=self.clearance.astype(np.float32),
                room=room,
                layers_y_m=(floor + recipe.heights.seated_m, floor + recipe.heights.standing_m),
            ),
            mirror=Mirror(
                signature=np.asarray(self.assets.signature, dtype=float),
                lead_s=self.assets.pack_lead_s,
                alignment_gain=self.assets.pack_gain,
                lowcut_hz=40.0,
                lowcut_order=8,
                tail_from_s=settings.render.tail_from_s,
                tail_bursts=settings.render.tail_bursts,
                tail_gain_db=tuple(float(v) for v in settings.parameters.tail_gain_db),
                histogram_bin_s=rays.bin_s,
                histogram_order=rays.order,
                receiver_radius_m=rays.receiver_radius_m,
                settings_json=json.dumps(settings.record(), sort_keys=True, default=str),
            ),
            crossover=self.crossover,
            air=Air(atmosphere, True),
            directivity={name: models[name] for name in named if name in models},
        ) as writer:
            for number, (name, track) in enumerate(tracks.sources.items(), start=1):
                self.journal.set_status(source=name, job=f"{number}/{len(tracks.sources)}")
                chosen = self.low[name]
                steps = tracks.steps
                pair = np.full((steps, 2, 2), -1, dtype=np.int32)
                local: dict[int, int] = {}
                for step in np.flatnonzero(track.audible):
                    for a in range(2):
                        for b in range(2):
                            position, cell = int(track.slot[step, a]), int(chosen.cell[step, b])
                            if position >= 0 and cell >= 0:
                                j = row_of[(position, cell)]
                                pair[step, a, b] = local.setdefault(j, len(local))
                mine = sorted(local, key=lambda j: local[j])
                ir = np.zeros((len(mine), header.channels, header.low_samples), dtype=np.float32)
                for row, j in enumerate(mine):
                    ir[row] = pair_low(
                        self.engine.cache.read(self.pair_key[j]),
                        self.crossover,
                        atmosphere,
                        sound_speed_m_s=settings.sound_speed_m_s,
                        lead_s=self.assets.pack_lead_s,
                        unit_at_1m=FIELD_UNIT_AT_1M,
                    )[0]
                high_gain_db, onset_s = step_levels(
                    audible=track.audible,
                    pair=pair,
                    position_weight=track.weight,
                    cell=chosen.cell,
                    mode=chosen.mode,
                    listener=tracks.listener,
                    cells=self.cells,
                    seam_db_of=seam[mine],
                    trail_s_of=trail[mine],
                    early=self.early[name],
                    alignment_gain=self.assets.pack_gain,
                )
                source = recipe.source(name)
                writer.add_source(
                    Source(
                        id=name,
                        kind=source.kind,
                        subtype=source.subtype or "",
                        position=track.position,
                        yaw_deg=track.yaw_deg.astype(np.float32),
                        audible=track.audible,
                        early=Early(**self.early[name].pack()),
                        tail=Tail(**self.tail[name]),
                        low=Low(
                            ir=ir,
                            pair_position=tracks.positions[
                                [self.pairs[j][0] for j in mine]
                            ].reshape(-1, 3),
                            pair_cell=np.array([self.pairs[j][1] for j in mine], dtype=np.int32),
                            pair_key=np.array([self.pair_key[j] for j in mine], dtype="S64"),
                            seam_db=seam[mine].astype(np.float32),
                            onset_s=onset[mine],
                            pair=pair,
                            position_weight=track.weight.astype(np.float32),
                            cell=chosen.cell,
                            mode=chosen.mode,
                        ),
                        level=Level(high_gain_db=high_gain_db, onset_s=onset_s),
                        directivity_model=source.directivity.model,
                        directivity_enabled=bool(source.directivity.enabled),
                        gain_db=float(source.gain_db),
                        tail_seed=tail_seed(recipe.seed, name),
                    )
                )
        partial.replace(target)
        self.report["pack"] = {"path": str(target), "bytes": target.stat().st_size}
        return target

    def check(self) -> None:
        """The pack read back whole; the engine on the host against the engine on the card."""
        with read_pack(self.out / "pack.h5", deep=True) as pack:
            self.report["read_back"] = {
                "sources": len(pack.sources),
                "steps": pack.header.steps,
                "cells": int(pack.cells.position.shape[0]),
            }
        self.report["render"] = render_check(self.out / "pack.h5", seconds=CHECK_SECONDS)
        if self.xp is not np and self.tracks.sources:
            name, track = next(iter(self.tracks.sources.items()))
            heard = np.flatnonzero(track.audible)[:400]
            if heard.size:
                host = trace_early(self.ms, track.position[heard], self.tracks.listener[heard])
                card = trace_early(
                    self.ms, track.position[heard], self.tracks.listener[heard], xp=self.xp
                )
                a, b = host.pack(), card.pack()
                same = all(
                    np.array_equal(a[k], b[k]) for k in ("offsets", "path_id", "order", "kind")
                )
                self.report["early_host_against_card"] = {
                    "source": name,
                    "steps": int(heard.size),
                    "same_paths": bool(same),
                    "delay_s": float(np.abs(a["delay_s"] - b["delay_s"]).max()) if same else None,
                    "gain": float(np.abs(a["gain"] - b["gain"]).max()) if same else None,
                }
        self.journal.say(f"check: render {self.report['render']}")

    # ---- the run ---------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Every stage; ``campaign.done`` or ``campaign.failed`` at the end."""
        journal = self.journal
        for marker in ("campaign.done", "campaign.failed"):
            (self.out / marker).unlink(missing_ok=True)
        try:
            journal.say(
                f"trace {self.recipe.dwelling.name}: {self.tracks.steps} steps,"
                f" {len(self.tracks.sources)} source(s), {self.asked.shape[0]} cell(s),"
                f" profile {self.profile.record()}"
            )
            self.check_assets()
            assert self.engine is not None
            centres = self.engine.place()
            journal.stage("assign", lambda: self.assign(centres))
            self.solve()
            journal.stage("paths", self.paths)
            journal.stage("rays", self.rays)
            journal.stage("level", self.level)
            journal.stage("write", self.write)
            journal.stage("check", self.check)
            self.report.update(
                {
                    "dwelling": self.recipe.dwelling.name,
                    "recipe_sha256": hashlib.sha256(self.recipe_bytes).hexdigest(),
                    "device": journal.status.get("device", {}),
                    "timings_s": dict(journal.timings),
                    "total_s": round(time.time() - journal.started, 1),
                    "pairs_engine": getattr(self.engine, "report", {}),
                }
            )
            (self.out / "trace_report.json").write_text(
                json.dumps(self.report, indent=1, default=str)
            )
            journal.set_status(stage="done")
            (self.out / "campaign.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            journal.say(f"trace complete in {(time.time() - journal.started) / 3600:.2f} h")
            return self.report
        except BaseException as error:
            journal.set_status(stage="failed", error=repr(error)[:500])
            (self.out / "campaign.failed").write_text(repr(error)[:2000])
            journal.say(f"trace FAILED: {error!r}"[:600])
            raise


def _end_db(omni: np.ndarray) -> float:
    """The energy of a response's last 50 ms over that of the whole, dB; -200 when silent."""
    samples = int(round(0.05 * LOW_RATE_HZ))
    whole = float(np.sum(np.asarray(omni, dtype=float) ** 2))
    if whole <= 0.0:
        return -200.0
    last = float(np.sum(np.asarray(omni[-samples:], dtype=float) ** 2))
    return round(float(10.0 * np.log10(max(last / whole, 1e-20))), 1)


def _spread(values: np.ndarray, digits: int = 3) -> dict[str, float] | None:
    if not values.size:
        return None
    return {
        "p10": round(float(np.percentile(values, 10)), digits),
        "median": round(float(np.median(values)), digits),
        "p90": round(float(np.percentile(values, 90)), digits),
        "worst": round(float(values[np.argmax(np.abs(values))]), digits),
    }


def render_check(path: Path, *, seconds: float | None = None) -> dict[str, Any]:
    """The pack rendered by the signal engine on ``numpy`` and, with a card, on ``cupy``.

    Dry noise at every source, the first ``seconds`` of the scene, block by
    block. Without a card the render is only shown to be finite and not
    silent; with one the two must agree to :data:`RENDER_TOLERANCE` of the
    peak (V4 of the plan).
    """
    from reverberate.compute import cuda_available
    from reverberate.render.engine import Engine

    with read_pack(path) as pack:
        h = pack.header
        samples = h.samples
        if seconds is not None:
            samples = min(samples, int(round(seconds / h.step_s)) * h.step_samples)
        dry = {
            name: np.random.default_rng(source.tail_seed % 2**32).standard_normal(samples)
            for name, source in pack.sources.items()
        }
        host = Engine(pack, dry, gpu=False)
        card = Engine(pack, dry, gpu=True) if cuda_available() else None
        block = int(round(CHECK_BLOCK_S / h.step_s)) * h.step_samples
        peak = 0.0
        worst = 0.0
        finite = True
        seconds_host = seconds_card = 0.0
        for start in range(0, samples, block):
            stop = min(start + block, samples)
            t0 = time.time()
            a = host.render(start, stop)
            seconds_host += time.time() - t0
            finite = finite and bool(np.all(np.isfinite(a)))
            peak = max(peak, float(np.abs(a).max()))
            if card is not None:
                t0 = time.time()
                b = to_numpy(card.render(start, stop))
                seconds_card += time.time() - t0
                worst = max(worst, float(np.abs(a - b).max()))
    record: dict[str, Any] = {
        "seconds": samples / h.sample_rate_hz,
        "sources": len(dry),
        "finite": finite,
        "peak": peak,
        "host_s": round(seconds_host, 2),
        "card": card is not None,
    }
    if card is not None:
        record["card_s"] = round(seconds_card, 2)
        record["max_over_peak"] = worst / peak if peak > 0.0 else None
        record["tolerance"] = RENDER_TOLERANCE
        record["passed"] = bool(peak > 0.0 and worst <= RENDER_TOLERANCE * peak)
    return record


def run_trace(
    bundle: Path,
    out: Path,
    *,
    pffdtd_dir: Path = Path("/root/pffdtd"),
    devices: str | None = None,
    gpu: bool | None = None,
    solvers: int | None = None,
    engine: PairsEngine | None = None,
) -> dict[str, Any]:
    return Trace(
        bundle=bundle,
        out=out,
        engine=engine,
        gpu=gpu,
        pffdtd_dir=pffdtd_dir,
        card_devices=devices,
        solvers=solvers,
    ).run()
