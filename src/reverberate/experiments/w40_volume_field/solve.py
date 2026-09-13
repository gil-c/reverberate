"""Every solve of a campaign on one card, the pressure shrunk and kept on its host.

Nothing is encoded here: the card's host has few slow cores. The instance is
left up with the float32 pressure under :data:`PRESSURE_DIR`, for
:mod:`.encode` to push to the encode boxes and then destroy.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.experiments.engine import write_record
from reverberate.experiments.run import entry_from_key
from reverberate.experiments.w40_volume_field import machines, remote
from reverberate.experiments.w40_volume_field.plan import COMMS_NAME, bands_of_source, slug
from reverberate.wave.comms import write_comms

#: The engine's output shrunk to float32 on the card's host, one file per
#: source and band, under this directory; the encode boxes pull from it.
PRESSURE_DIR = "/root/pressure"

#: The image the engine is built in; sm_80 and sm_90 cards.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

#: Measured on 50451826: the engine needs 2.2 x its output in host RAM when it
#: writes (cgroup peak at the limit, oom_kill, on a 125 GB output with 194 GB).
RAM_PER_OUTPUT = 2.2

#: VRAM per grid node and the engine's fixed overhead, measured on A100 80 GB.
VRAM_PER_NODE_B = 9.027
VRAM_FIXED_GB = 2.13


def slices_for(output_bytes: float, ram_gb: float) -> int:
    """How many consecutive receiver slices a band needs to fit ``ram_gb``."""
    return max(1, int(math.ceil(RAM_PER_OUTPUT * output_bytes / 1e9 / max(ram_gb - 8.0, 1.0))))


def sizing(
    plan: dict[str, Any], jobs: list[tuple[dict[str, Any], str]], *, slice_ram_gb: float
) -> dict[str, Any]:
    """VRAM, RAM, disk and card hours for these solves, with the slices per band."""
    parts = {
        b: slices_for(plan["bands"][b]["cost"]["sim_outs_bytes"], slice_ram_gb) for _, b in jobs
    }
    part_gb = max(plan["bands"][b]["cost"]["sim_outs_bytes"] / 1e9 / parts[b] for _, b in jobs)
    total_gb = sum(plan["bands"][b]["cost"]["sim_outs_bytes"] / 2e9 for _, b in jobs)
    return {
        "parts": parts,
        "vram_gb": max(
            int(plan["bands"][b]["cost"]["grid_points"]) * VRAM_PER_NODE_B / 1e9 + VRAM_FIXED_GB
            for _, b in jobs
        ),
        "ram_gb": RAM_PER_OUTPUT * part_gb + 8.0,
        "disk_gb": int(part_gb * 1.5 + total_gb + 60.0),
        "gpu_hours": sum(plan["bands"][b]["cost"]["estimated_gpu_s"] * parts[b] for _, b in jobs)
        / 3600.0,
        "pressure_gb": total_gb,
    }


def solve_on_card(
    out: Path,
    *,
    hours: float,
    max_dph: float,
    yes: bool,
    sources: list[str] | None = None,
    instance: int | None = None,
    slice_ram_gb: float = 180.0,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Solve every band of every source on one host with enough VRAM in total.

    A band whose output does not fit ``slice_ram_gb`` under the engine's 2.2x
    rule is solved as consecutive slices of its receiver rows, each a full
    solve of the storey, and the slices are merged on the host in row order.
    The card's time multiplies by the slices; the listening grid does not
    shrink. ``solve.json`` carries the instance from the moment it is rented,
    so a retry resumes on the same host.
    """
    from reverberate import auth
    from reverberate.gpu import vast
    from reverberate.wave import engine_inputs
    from reverberate.wave.remote import run_on, upload, watch_engine
    from reverberate.wave.remote_voxelise import REMOTE_VENV, provision

    plan = json.loads((out / "plan.json").read_text())
    wanted = [s for s in plan["sources"] if sources is None or s["name"] in sources]
    jobs = [(s, b) for s in wanted for b in bands_of_source(plan, s)]
    size = sizing(plan, jobs, slice_ram_gb=slice_ram_gb)
    parts = size["parts"]
    say(
        f"{len(jobs)} bands in {sum(parts.values())} solves: {size['vram_gb']:.0f} GB VRAM, "
        f"{size['ram_gb']:.0f} GB RAM, {size['disk_gb']} GB disk,"
        f" {size['gpu_hours']:.2f} h on an A100"
    )

    auth.inject(["VASTAI_API_KEY"])
    client = vast.VastClient(timeout=60)
    identity = vast.account_identity(client)
    if instance is None:
        offers: list[Any] = []
        for count in (1, 2, 4):
            offers += client.search(
                vast.search_query(
                    gpu_name="",
                    num_gpus=count,
                    min_disk_gb=size["disk_gb"],
                    min_gpu_ram_gb=40,
                    min_reliability=0.95,
                ),
                limit=400,
            )
        good = machines.card_hosts(
            offers, vram_gb=size["vram_gb"], ram_gb=size["ram_gb"], max_dph=max_dph
        )
        if not good:
            raise SystemExit(
                f"no A100/H100/H200 host under {max_dph} USD/h with {size['vram_gb']:.0f} GB VRAM,"
                f" {size['ram_gb']:.0f} GB RAM and {size['disk_gb']} GB disk"
            )
        for o in good[:5]:
            say(
                f"  {o.id} {o.num_gpus}x {o.gpu_name} {o.dph_total:.3f} USD/h,"
                f" {o.ram_gb:.0f} GB RAM,"
                f" {o.disk_gb:.0f} GB disk, {o.location}"
            )
        say(
            f"  cap {hours:g} h -> at most"
            f" {vast.estimate_cost_usd(good[0].dph_total, hours):.2f} USD"
        )
        if not yes:
            say("nothing rented: pass --yes")
            return {}
        machine, instance_id = machines.rent_one(
            client,
            identity,
            good,
            hours=hours,
            disk_gb=size["disk_gb"],
            image=IMAGE,
            remaining_usd=good[0].dph_total * min(hours, 3.0),
            say=say,
        )
    else:
        instance_id = instance
        machine = vast.wait_for_ssh(client, instance, identity)

    previous = (
        json.loads((out / "solve.json").read_text()) if (out / "solve.json").is_file() else {}
    )
    solves: list[dict[str, Any]] = (
        previous.get("solves", []) if previous.get("instance") == instance_id else []
    )

    def record(**extra: Any) -> None:
        write_record(
            out,
            "solve.json",
            {"instance": instance_id, "pressure_dir": PRESSURE_DIR, "solves": solves, **extra},
        )

    record()
    say(f"engine built in {provision(machine, Path('scripts/build_pffdtd.sh')) / 60:.1f} min")
    remote.install(machine)
    run_on(machine, f"mkdir -p {PRESSURE_DIR}", what="mkdir pressure")
    uploaded: set[str] = set()

    def exists(path: str) -> bool:
        return (
            run_on(machine, f"test -f {path} && echo yes || echo no", what="check").strip() == "yes"
        )

    for source, band in jobs:
        spec = plan["bands"][band]
        entry = entry_from_key(spec["cache_key"])
        cache_dir = f"/root/cache_{spec['cache_key']}"
        pressure = f"{PRESSURE_DIR}/{source['name']}__{slug(band)}.h5"
        if exists(pressure):
            say(f"{source['name']} {band}: pressure already on the host")
            continue
        record(in_progress=band, remaining=[b for _, b in jobs[jobs.index((source, band)) :]])
        positions = np.load(out / f"array_positions_{slug(band)}.npy")
        rows = [r for r in spec["rows"] if r is not None]
        cuts = (
            [0]
            + [rows[len(rows) * k // parts[band]][0] for k in range(1, parts[band])]
            + [positions.shape[0]]
        )
        part_files: list[str] = []
        timings = {"upload_s": 0.0, "engine_s": 0.0, "shrink_s": 0.0}
        for k in range(parts[band]):
            suffix = "" if parts[band] == 1 else f"_part{k}"
            remote_dir = f"/root/job_{source['name']}_{slug(band)}{suffix}"
            target = (
                pressure
                if parts[band] == 1
                else f"{PRESSURE_DIR}/{source['name']}__{slug(band)}.part{k}.h5"
            )
            part_files.append(target)
            if parts[band] > 1 and exists(target):
                say(f"{source['name']} {band} part {k}: already on the host")
                continue
            t0 = time.time()
            if parts[band] == 1:
                comms = out / source["name"] / slug(band) / "comms" / COMMS_NAME
            else:
                comms_dir = out / source["name"] / f"{slug(band)}_part{k}" / "comms"
                comms_dir.mkdir(parents=True, exist_ok=True)
                comms = write_comms(
                    entry.path,
                    np.asarray(source["position"], dtype=float),
                    positions[cuts[k] : cuts[k + 1]],
                    spec["samples"] / spec["sample_rate_hz"],
                    diff_source=True,
                    out_path=comms_dir / COMMS_NAME,
                    interpolation="nearest",
                )
                say(f"{source['name']} {band} part {k}: rows {cuts[k]}:{cuts[k + 1]}")
            if spec["cache_key"] not in uploaded:
                upload(
                    machine,
                    [f for f in engine_inputs(entry, comms) if f.name != COMMS_NAME],
                    cache_dir,
                )
                uploaded.add(spec["cache_key"])
            run_on(
                machine,
                f"mkdir -p {remote_dir} && ln -f {cache_dir}/*.h5 {remote_dir}/",
                what="link cache",
            )
            upload(machine, [comms], remote_dir)
            timings["upload_s"] += time.time() - t0
            t0 = time.time()
            log = watch_engine(
                machine,
                remote_dir,
                timeout=hours * 3600,
                poll_s=60.0,
                on_progress=lambda p: say(
                    f"    engine {p.percent if p.percent is not None else '?'}%"
                    f" {p.output_bytes / 1e9:.1f} GB"
                ),
            )
            timings["engine_s"] += time.time() - t0
            log_dir = out / source["name"] / f"{slug(band)}{suffix}"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "engine.log").write_text(log)
            if "sim_outs.h5" not in run_on(machine, f"ls {remote_dir}", what="ls job"):
                raise RuntimeError(
                    f"{source['name']} {band}{suffix}: the engine left no sim_outs.h5"
                    " (out of host RAM?)"
                )
            t0 = time.time()
            run_on(
                machine,
                f"cd {remote_dir} && {REMOTE_VENV} {remote.tool('shrink')} sim_outs.h5 {target}"
                f" && rm -rf {remote_dir}",
                what="shrink",
                timeout=3600,
            )
            timings["shrink_s"] += time.time() - t0
        if parts[band] > 1:
            t0 = time.time()
            run_on(
                machine,
                f"{REMOTE_VENV} {remote.tool('merge')} {' '.join(part_files)} {pressure}"
                f" && rm -f {' '.join(part_files)}",
                what="merge parts",
                timeout=3600,
            )
            timings["shrink_s"] += time.time() - t0
        size_text = run_on(machine, f"stat -c %s {pressure}", what="stat").strip()
        expected = int(spec["cost"]["sim_outs_bytes"] / 2)
        if not size_text.isdigit() or int(size_text) < 0.95 * expected:
            raise RuntimeError(f"{pressure} is {size_text} bytes, expected about {expected}")
        solves.append(
            {
                "source": source["name"],
                "band": band,
                "parts": parts[band],
                **{k: round(v) for k, v in timings.items()},
                "pressure": pressure,
                "bytes": int(size_text),
            }
        )
        record()
        say(
            f"{source['name']} {band}: {parts[band]} solve(s),"
            f" engine {timings['engine_s'] / 60:.1f} min,"
            f" shrunk {timings['shrink_s'] / 60:.1f} min"
        )
    record(complete=True)
    say(
        f"KEEPING {instance_id} up with the pressure under {PRESSURE_DIR};"
        f" run `encode --gpu-instance {instance_id}`"
    )
    return {"instance": instance_id, "solves": solves}
