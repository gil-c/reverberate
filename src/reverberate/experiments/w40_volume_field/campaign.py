"""One campaign end to end, resumed from whatever exists on disk.

``plan.json``, the comms files, ``solve.json`` (the card's instance and its
completed bands), ``encode_boxes.json``, the encoded shards and the field
file are the state. A stage whose output exists is skipped, a failed stage
is retried after a pause, and every line of ``campaign.log`` names the
stage, the time, the elapsed hours and the credit, so a morning read tells
what happened without the transcript.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reverberate.experiments.w40_volume_field import machines
from reverberate.experiments.w40_volume_field.assemble import assemble_field
from reverberate.experiments.w40_volume_field.encode import encode_sharded, rent_encode_boxes
from reverberate.experiments.w40_volume_field.plan import (
    COMMS_NAME,
    bands_of_source,
    plan_field,
    prepare_field,
    slug,
)
from reverberate.experiments.w40_volume_field.solve import solve_on_card

DURATIONS_S = {"low": 1.2, "mid": 0.4, "high": 0.15}

#: The three grids of a storey, by band.
FMAX_HZ = {"low": 1000.0, "mid": 4000.0, "high": 8000.0}

#: The export's name for the whole storey.
STOREY_SCENE = "apartment_full"


def export_scene(hssd_root: Path, scene_id: str, models: Path) -> Path:
    """The storey's mesh, exported once per dwelling; the room named is the largest."""
    from reverberate.experiments.scene_export import export
    from reverberate.experiments.w40_volume_field.plan import free_floor

    if (models / "manifest.json").is_file():
        return models
    _, _, rooms = free_floor(hssd_root, scene_id)
    largest = max(rooms, key=lambda room: room.area_m2)
    export(hssd_root, scene_id, largest.regions[0], models)
    return models


def storey_keys(models: Path, fmax_hz: dict[str, float] = FMAX_HZ) -> dict[str, str]:
    """The cache key each band's grid has, from the export alone."""
    from reverberate.experiments.run import scene_spec

    return {band: scene_spec(models, STOREY_SCENE, fmax)[0].key for band, fmax in fmax_hz.items()}


def grids_cached(keys: dict[str, str]) -> bool:
    from reverberate.experiments.run import entry_from_key

    try:
        return all((entry_from_key(key).path / "manifest.json").is_file() for key in keys.values())
    except (FileNotFoundError, KeyError):
        return False


def voxelise_storey(
    models: Path, scratch: Path, *, hours: float, say: Callable[[str], None]
) -> None:
    """The three grids on one CPU rental, installed in the cache and published.

    Runs ``scripts/remote_chain.py`` as a subprocess: it sizes and rents the
    machine, fetches every grid and installs it as a cache entry. Its log is
    streamed line by line.
    """
    command = [
        sys.executable,
        str(Path("scripts/remote_chain.py")),
        "--models",
        str(models),
        "--scene",
        STOREY_SCENE,
        "--fmax",
        *[f"{f:g}" for f in FMAX_HZ.values()],
        "--hours",
        f"{hours:g}",
        "--out",
        str(scratch),
        "--nprocs",
        "12",
        "--min-ram-gb",
        "64",
        "--min-cpu-ghz",
        "3.5",
        "--max-dph",
        "0.6",
        "--yes",
    ]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if line.strip() and "DEF=" not in line and not line.startswith(" ["):
                say(f"  vox | {line.rstrip()[:160]}")
        if process.wait() != 0:
            raise RuntimeError(f"remote_chain exited {process.returncode}")


#: Nodes per side of a drawn cube in the coarse tier, the rooms the reader is
#: not standing in; the room stood in is always the grid itself. The viewer
#: keeps every room's coarse tier resident and holds about 2 M quads of them
#: (16 M for the one fine room): the storey of hssd_0076 at 8 kHz is about
#: 2e8 boundary nodes and 6.5 M native quads, halving the cube side divides
#: the quads by about four (vox_view's table), so 4 nodes a cube leaves
#: 0.4 M resident; at 4 kHz 2 nodes a cube leaves the same; at 1 kHz the
#: grid is small enough to be resident whole. A reader auditing an object
#: walks into its room and sees the grid.
COARSE_SPAN = {"low": 1, "mid": 2, "high": 4}


def build_audit(
    keys: dict[str, str],
    hssd_root: Path,
    scene_id: str,
    audit: Path,
    say: Callable[[str], None],
) -> None:
    """The audit view of every grid, tiered by room as ADR 0007 says.

    Built here, from the cache entries, once they are home: it re-reads a
    grid room by room, so it needs no rental, and the whole storey at 8 kHz
    takes minutes. The room the reader stands in is drawn at the grid's own
    step, without loss; every other room at ``COARSE_SPAN[band]`` nodes a
    cube, and the payload says so under ``aggregation``.
    """
    from reverberate.experiments.audit_view import build

    for band, key in keys.items():
        target = audit / f"{FMAX_HZ[band]:g}"
        if (target / "voxels" / "rooms.json").is_file():
            continue
        started = time.time()
        report = build(key, hssd_root, scene_id, target, coarse_span=COARSE_SPAN[band])
        say(
            f"audit {FMAX_HZ[band]:g} Hz: {report.get('rooms', '?')} rooms"
            f" in {(time.time() - started) / 60:.1f} min"
        )


def audit_meshes(out: Path, audit: Path, fmax_hz: dict[str, float] = FMAX_HZ) -> dict[str, str]:
    """``walk.json``'s ``meshes``: one tiered payload per band that exists, relative to the run."""
    meshes: dict[str, str] = {}
    for fmax in fmax_hz.values():
        payload = audit / f"{fmax:g}" / "voxels"
        if (payload / "rooms.json").is_file():
            meshes[f"{fmax:g}"] = str(payload.relative_to(out))
    return meshes


class Campaign:
    """The state machine; each stage is a method so a test can drive them with fakes."""

    def __init__(
        self, out: Path, client: Any, *, attempts: int = 3, pause_s: float = 120.0
    ) -> None:
        self.out = out
        self.client = client
        self.attempts = attempts
        self.pause_s = pause_s
        self.started = time.time()
        self.log_path = out / "campaign.log"

    def say(self, message: str) -> None:
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} +{(time.time() - self.started) / 3600:.2f} h"
            f" credit {machines.credit_of(self.client):.2f} USD | {message}"
        )
        print(line, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")

    def stage(self, name: str, work: Callable[[], Any]) -> Any:
        for attempt in range(1, self.attempts + 1):
            self.say(f"{name}: start, attempt {attempt}")
            try:
                result = work()
            except (Exception, SystemExit) as error:  # noqa: BLE001 - retried, then reported
                self.say(f"{name}: FAILED {str(error)[:300]}")
                if attempt == self.attempts:
                    raise
                time.sleep(self.pause_s)
                continue
            self.say(f"{name}: done")
            return result
        return None

    # ---- state readers ----------------------------------------------------------------

    def plan(self) -> dict[str, Any]:
        return dict(json.loads((self.out / "plan.json").read_text()))

    def jobs(self) -> list[tuple[dict[str, Any], str]]:
        plan = self.plan()
        return [(s, b) for s in plan["sources"] for b in bands_of_source(plan, s)]

    def comms_ready(self) -> bool:
        return all(
            (self.out / s["name"] / slug(b) / "comms" / COMMS_NAME).is_file()
            for s, b in self.jobs()
        )

    def solve_state(self) -> dict[str, Any]:
        """``solve.json`` as it stands, emptied when its instance no longer exists.

        Read again before every attempt: a retry after a failure that came
        after the rental must go back to the same host, not rent a second one.
        """
        path = self.out / "solve.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        if (
            record.get("instance") is not None
            and self.client.instance(int(record["instance"])) is None
        ):
            self.say(f"solve: instance {record['instance']} is gone; its pressure with it")
            return {}
        return record

    def encoded_ready(self) -> bool:
        return all(
            (self.out / "encoded" / f"{s['name']}__{slug(b)}.h5").is_file() for s, b in self.jobs()
        )

    def field_ready(self) -> bool:
        return all(
            (self.out / "field" / f"{s['name']}.h5").is_file() for s in self.plan()["sources"]
        )


def campaign(
    out: Path,
    *,
    hssd_root: Path,
    scene_id: str,
    sources_file: Path,
    models: Path,
    ram_gb: float,
    pitch_m: float,
    solve_hours: float,
    solve_max_dph: float,
    slice_ram_gb: float,
    encode_hours: float,
    encode_max_dph: float,
    min_cores: int,
    min_cpu_ghz: float,
    shards: int,
    attempts: int,
    keys: dict[str, str] | None = None,
    voxelise_hours: float = 4.0,
) -> None:
    """Export, voxelise, plan, prepare, solve, encode and assemble; resume from what exists.

    ``keys`` names the storey's grids when they are already in the cache;
    otherwise the export and the voxelisation run first, on a CPU rental.
    The audit view of every grid is then built on this machine from the cache
    entries, and ``walk.json`` points the viewer at them under ``meshes``
    with the scene they belong to. The encode boxes are
    rented while the card solves its last band, so the card leaves within
    minutes of its last shrink.
    """
    from reverberate import auth
    from reverberate.gpu import vast

    out.mkdir(parents=True, exist_ok=True)
    auth.inject(["VASTAI_API_KEY"])
    run = Campaign(out, vast.VastClient(timeout=60), attempts=attempts)
    say = run.say
    audit = out / "audit"

    if keys is None:
        run.stage("export", lambda: export_scene(hssd_root, scene_id, models))
        keys = storey_keys(models)
        say(f"grids: {keys}")
        if not grids_cached(keys):
            run.stage(
                "voxelise",
                lambda: voxelise_storey(models, out / "vox", hours=voxelise_hours, say=say),
            )
    the_keys = keys
    if len(audit_meshes(out, audit)) < len(the_keys):
        run.stage("audit", lambda: build_audit(the_keys, hssd_root, scene_id, audit, say))
    meshes = audit_meshes(out, audit)
    if not (out / "plan.json").is_file():
        run.stage(
            "plan",
            lambda: plan_field(
                out,
                hssd_root=hssd_root,
                scene_id=scene_id,
                sources=json.loads(sources_file.read_text()),
                storey_keys={"low": keys["low"], "mid": keys["mid"]},
                room_keys={},
                durations_s=DURATIONS_S,
                height_m=1.70,
                ram_gb=ram_gb,
                pitch_m=pitch_m,
                high_key=keys["high"],
            ),
        )
    plan = run.plan()
    say(f"plan: {len(plan['points'])} points at {plan['pitch_m']} m, bands {sorted(plan['bands'])}")
    if not run.comms_ready():
        run.stage("prepare", lambda: prepare_field(out))

    boxes: list[tuple[Any, int]] = []
    boxes_lock = threading.Lock()

    def rent_boxes_early() -> None:
        """Rent the encode boxes once the card is on its last band."""
        while True:
            time.sleep(60)
            state = run.solve_state()
            if state.get("complete") or (
                state.get("in_progress") and len(state.get("remaining", [])) <= 1
            ):
                break
        with boxes_lock:
            boxes.extend(
                run.stage(
                    "rent encode boxes",
                    lambda: rent_encode_boxes(
                        out,
                        shards=shards,
                        hours=encode_hours,
                        max_dph=encode_max_dph,
                        min_cores=min_cores,
                        min_cpu_ghz=min_cpu_ghz,
                        yes=True,
                        say=say,
                    ),
                )
            )

    if not run.encoded_ready() and not run.field_ready():
        if not run.solve_state().get("complete"):
            early = threading.Thread(target=rent_boxes_early, daemon=True)
            early.start()
            run.stage(
                "solve",
                lambda: solve_on_card(
                    out,
                    hours=solve_hours,
                    max_dph=solve_max_dph,
                    yes=True,
                    instance=run.solve_state().get("instance"),
                    slice_ram_gb=slice_ram_gb,
                    say=say,
                ),
            )
            early.join(timeout=encode_hours * 3600)
        instance = int(run.solve_state()["instance"])
        run.stage(
            "encode",
            lambda: encode_sharded(
                out,
                gpu_instance=instance,
                shards=shards,
                hours=encode_hours,
                max_dph=encode_max_dph,
                min_cores=min_cores,
                min_cpu_ghz=min_cpu_ghz,
                yes=True,
                boxes=boxes or None,
                say=say,
            ),
        )
    if not run.field_ready():
        run.stage("assemble", lambda: assemble_field(out, workers=4, meshes=meshes or None))
    say(f"campaign complete: {[str(out / 'field' / f'{s['name']}.h5') for s in plan['sources']]}")
