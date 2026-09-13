"""Encode the pressure on cheap fast-core boxes, one shard of points each.

The card's host cuts every band into shards by point ranges, pushes each
shard to its box through that box's own ssh proxy, and is destroyed once
every shard is verified. Each box runs :mod:`.remote.child_encode` on as
many workers as its memory holds, and only the encoded signals come home.
"""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.w40_volume_field import machines, remote
from reverberate.experiments.w40_volume_field.plan import (
    COMMS_NAME,
    DELIVERY_RATE_HZ,
    bands_of_source,
    slug,
)
from reverberate.experiments.w40_volume_field.solve import IMAGE, PRESSURE_DIR

#: Where a box keeps its shards and its jobs.
BOX_PRESSURE = "/root/pressure"
BOX_RUN = "/root/run"


def shard_bounds(
    rows: list[Any], shards: int
) -> tuple[list[tuple[int, int]], list[list[list[int] | None]]]:
    """Point ranges cut into ``shards`` runs of contiguous rows, balanced by rows.

    Returns the row bounds of each shard and, per shard, the plan's ``rows``
    list with every point outside the shard set to ``None`` and the rows of
    the points inside shifted to the shard file's own origin.
    """
    placed = [(i, r) for i, r in enumerate(rows) if r is not None]
    total = placed[-1][1][1] if placed else 0
    bounds: list[tuple[int, int]] = []
    per_shard: list[list[list[int] | None]] = []
    target = total / max(shards, 1)
    start_row = 0
    cursor = 0
    for k in range(shards):
        end_row = total if k == shards - 1 else start_row
        while cursor < len(placed) and (
            k == shards - 1 or placed[cursor][1][1] - start_row < target
        ):
            end_row = placed[cursor][1][1]
            cursor += 1
        bounds.append((start_row, end_row))
        shifted: list[list[int] | None] = [None] * len(rows)
        for i, r in placed:
            if start_row <= r[0] and r[1] <= end_row:
                shifted[i] = [r[0] - start_row, r[1] - start_row]
        per_shard.append(shifted)
        start_row = end_row
    return bounds, per_shard


def encode_job(
    plan: dict[str, Any],
    band: str,
    source: dict[str, Any],
    job_dir: str,
    *,
    shard: int,
    shards: int,
    workers: int,
) -> dict[str, Any]:
    """The JSON the child encoder reads on stdin, for one shard of one band."""
    spec = plan["bands"][band]
    bounds, per_shard = shard_bounds(spec["rows"], shards)
    return {
        "remote_dir": job_dir,
        "rows": per_shard[shard],
        "row_offset": bounds[shard][0],
        "centres": spec["centres"],
        "grid_step_m": spec["grid_step_m"],
        "fmax_hz": spec["fmax_hz"],
        "sample_rate_hz": spec["sample_rate_hz"],
        "sound_speed_m_s": plan["sound_speed_m_s"],
        "delivery_rate_hz": DELIVERY_RATE_HZ,
        "lowcut_hz": plan["lowcut_hz"],
        "lowcut_order": plan["lowcut_order"],
        "order": plan["encoder"]["order"],
        "fit_order": plan["encoder"]["fit_order"],
        "workers": workers,
        "pressure": f"{BOX_PRESSURE}/{source['name']}__{slug(band)}.shard{shard}.h5",
        "positions": f"{BOX_RUN}/array_positions_{slug(band)}.npy",
        "comms": f"{job_dir}/comms/{COMMS_NAME}",
        "output": f"{job_dir}/encoded.h5",
        "band": band,
        "source": source["name"],
    }


def merge_encoded(parts: list[Path], merged: Path) -> None:
    """One encoded file from the shards of a band, points in plan order."""
    signals, index, centres = [], [], []
    rate = order = None
    attrs: dict[str, Any] = {}
    for part in parts:
        with h5py.File(part, "r") as handle:
            if handle["signals"].shape[0] == 0:
                continue
            signals.append(np.asarray(handle["signals"][...]))
            index.append(np.asarray(handle["point_index"][...]))
            centres.append(np.asarray(handle["centres"][...]))
            rate, order = float(handle.attrs["sample_rate_hz"]), int(handle.attrs["order"])
            attrs = {k: handle.attrs[k] for k in ("band", "source")}
    stacked = np.concatenate(signals)
    points = np.concatenate(index)
    order_of = np.argsort(points)
    with h5py.File(merged, "w") as handle:
        handle.create_dataset("signals", data=stacked[order_of])
        handle.create_dataset("point_index", data=points[order_of])
        handle.create_dataset("centres", data=np.concatenate(centres)[order_of])
        handle.attrs["sample_rate_hz"] = rate
        handle.attrs["order"] = order
        for k, v in attrs.items():
            handle.attrs[k] = v


def rent_encode_boxes(
    out: Path,
    *,
    shards: int,
    hours: float,
    max_dph: float,
    min_cores: int,
    min_cpu_ghz: float,
    yes: bool,
    say: Callable[[str], None] = print,
) -> list[tuple[Any, int]]:
    """Rent and provision ``shards`` boxes; widen the search by steps when thin.

    Done before the last solve ends when the campaign drives it, so the card
    leaves within minutes of its last shrink instead of billing while boxes
    are found and built (13 minutes at 01:49 on 2026-09-13).
    """
    from reverberate import auth
    from reverberate.gpu import vast
    from reverberate.wave.remote import run_on
    from reverberate.wave.remote_voxelise import provision

    plan = json.loads((out / "plan.json").read_text())
    jobs = [(s, b) for s in plan["sources"] for b in bands_of_source(plan, s)]
    pressure_gb = sum(plan["bands"][b]["cost"]["sim_outs_bytes"] / 2e9 for _, b in jobs)
    disk_gb = int(pressure_gb / shards + 40)
    auth.inject(["VASTAI_API_KEY"])
    client = vast.VastClient(timeout=60)
    identity = vast.account_identity(client)
    good: list[Any] = []
    for step in machines.widening_steps(min_cores, min_cpu_ghz, max_dph):
        offers = client.search(
            vast.search_query(
                gpu_name="",
                min_disk_gb=disk_gb,
                min_reliability=0.95,
                min_cpu_cores=step.min_cores,
                min_cpu_ghz=step.min_cpu_ghz,
                min_inet_down_mbps=300,
            ),
            limit=500,
        )
        good = machines.encode_boxes(offers, max_dph=step.max_dph)
        if len(good) >= shards:
            break
        say(
            f"  only {len(good)} boxes with {step.min_cores} cores at {step.min_cpu_ghz} GHz"
            f" under {step.max_dph} USD/h; widening"
        )
    if len(good) < shards:
        raise SystemExit(f"only {len(good)} boxes after widening; wanted {shards}")
    for o in good[: shards + 2]:
        say(
            f"  {o.id} {o.cpu_cores:.0f} vCPU at {o.cpu_ghz:.1f} GHz, {o.ram_gb:.0f} GB RAM,"
            f" {o.dph_total:.3f} USD/h, {o.location}"
        )
    say(
        f"  {pressure_gb:.0f} GB over {shards} boxes; cap {hours:g} h -> at most"
        f" {sum(vast.estimate_cost_usd(o.dph_total, hours) for o in good[:shards]):.2f} USD"
    )
    if not yes:
        say("nothing rented: pass --yes")
        return []
    boxes: list[tuple[Any, int]] = []
    for _ in range(shards):
        boxes.append(
            machines.rent_one(
                client, identity, good, hours=hours, disk_gb=disk_gb, image=IMAGE, say=say
            )
        )

    def build(box: Any) -> None:
        provision(box, Path("scripts/build_pffdtd.sh"))
        remote.install(box)
        run_on(box, f"mkdir -p {BOX_RUN} {BOX_PRESSURE}", what="mkdir")

    with ThreadPoolExecutor(max_workers=shards) as pool:
        list(pool.map(build, [b for b, _ in boxes]))
    json.dump([i for _, i in boxes], (out / "encode_boxes.json").open("w"))
    return boxes


def push_shards(
    out: Path,
    gpu: Any,
    gpu_instance: int,
    boxes: list[tuple[Any, int]],
    *,
    client: Any,
    hours: float,
    say: Callable[[str], None] = print,
) -> None:
    """Cut every band on the card's host and push each shard to its box.

    The card pushes rather than the boxes pulling: on 2026-09-13 four pulls
    into the card's proxy stalled at once, while pushes through each box's
    own proxy ran at 22 MB/s in total. A key made on the card is attached to
    every box through the API; no account secret leaves the laptop. Each
    box's shard is verified by size before the card is destroyed.
    """
    from reverberate.gpu import vast
    from reverberate.wave.remote import run_on
    from reverberate.wave.remote_voxelise import REMOTE_VENV

    plan = json.loads((out / "plan.json").read_text())
    jobs = [(s, b) for s in plan["sources"] for b in bands_of_source(plan, s)]
    shards = len(boxes)
    t0 = time.time()
    for source, band in jobs:
        bounds, _ = shard_bounds(plan["bands"][band]["rows"], shards)
        base = f"{PRESSURE_DIR}/{source['name']}__{slug(band)}"
        spec = {"bounds": bounds, "pattern": base + ".shard{k}.h5"}
        run_on(
            gpu,
            f"test -f {base}.shard{shards - 1}.h5 || {REMOTE_VENV} {remote.tool('split')}"
            f" {base}.h5 {shlex.quote(json.dumps(spec))}",
            what=f"split {source['name']} {band}",
            timeout=3600,
        )
    say(f"pressure cut into {shards} shards on the host in {(time.time() - t0) / 60:.1f} min")
    public = run_on(
        gpu,
        "test -f /root/.ssh/push || ssh-keygen -q -t ed25519 -N '' -f /root/.ssh/push;"
        " cat /root/.ssh/push.pub",
        what="push key",
    ).strip()
    for _, box_id in boxes:
        client.request("POST", f"/instances/{box_id}/ssh/", {"ssh_key": public})
        time.sleep(3)
    time.sleep(20)
    expected: dict[str, int] = {}
    for line in (
        run_on(
            gpu,
            f"ls -l {PRESSURE_DIR}/*.shard*.h5 | awk '{{print $5, $9}}'",
            what="ls shards",
        )
        .strip()
        .splitlines()
    ):
        size, path = line.split()
        expected[Path(path).name] = int(size)
    for k, (box, _) in enumerate(boxes):
        files = " ".join(f"{PRESSURE_DIR}/{s['name']}__{slug(b)}.shard{k}.h5" for s, b in jobs)
        machines.launch(
            gpu,
            "/root",
            f"push{k}",
            machines.resumable_transfer(
                files,
                BOX_PRESSURE + "/",
                port=box.port,
                host=box.host,
                key="/root/.ssh/push",
                name=f"push {k}",
            ),
        )
    say(f"{shards} pushes launched from the card's host")
    t0 = time.time()
    pending = set(range(shards))
    while pending:
        time.sleep(90)
        done = run_on(gpu, "ls /root/push*.done 2>/dev/null | tr '\\n' ' '", what="poll pushes")
        for k in sorted(pending):
            if f"push{k}.done" not in done:
                continue
            box = boxes[k][0]
            got = (
                run_on(
                    box,
                    f"ls -l {BOX_PRESSURE}/*.h5 | awk '{{print $5, $9}}'",
                    what="ls box",
                )
                .strip()
                .splitlines()
            )
            have = {Path(p).name: int(s) for s, p in (line.split() for line in got)}
            names = [f"{s['name']}__{slug(b)}.shard{k}.h5" for s, b in jobs]
            if all(have.get(n) == expected.get(n) for n in names):
                pending.discard(k)
                say(f"shard {k}: on its box and verified ({(time.time() - t0) / 60:.0f} min)")
            else:
                raise RuntimeError(f"shard {k}: sizes differ from the card's, {have} vs {expected}")
        if time.time() - t0 > hours * 3600:
            raise TimeoutError("the pushes passed their budget; the card stays up")
    say(f"every shard verified on its box; destroying the card's host {gpu_instance}")
    vast.teardown(client, gpu_instance)


def encode_on_box(
    out: Path,
    box: Any,
    box_id: int,
    shard: int,
    shards: int,
    *,
    hours: float,
    say: Callable[[str], None] = print,
) -> None:
    """Every band of the shard on one box, each band retried once, then the box is torn down."""
    from reverberate.gpu import vast
    from reverberate.wave.remote import run_on
    from reverberate.wave.remote_voxelise import REMOTE_VENV, rsync

    plan = json.loads((out / "plan.json").read_text())
    jobs = [(s, b) for s in plan["sources"] for b in bands_of_source(plan, s)]
    rsync(
        box,
        [str(out / "plan.json"), *[str(p) for p in out.glob("array_positions_*.npy")]],
        BOX_RUN + "/",
        download=False,
    )
    cores = int(run_on(box, "nproc", what="nproc").strip())
    workers = machines.workers_for(cores, machines.cgroup_memory_gb(box))
    say(f"shard {shard}: {workers} workers on {cores} cores")
    for source, band in jobs:
        done = out / "encoded" / f"{source['name']}__{slug(band)}.shard{shard}.h5"
        if done.is_file():
            continue
        job_dir = f"{BOX_RUN}/{source['name']}/{slug(band)}"
        run_on(box, f"mkdir -p {job_dir}/comms", what="mkdir job")
        rsync(
            box,
            [str(out / source["name"] / slug(band) / "comms" / COMMS_NAME)],
            f"{job_dir}/comms/{COMMS_NAME}",
            download=False,
        )
        job = encode_job(plan, band, source, job_dir, shard=shard, shards=shards, workers=workers)
        local_job = out / source["name"] / slug(band) / f"job.shard{shard}.json"
        local_job.write_text(json.dumps(job))
        rsync(box, [str(local_job)], f"{job_dir}/job.json", download=False)
        for attempt in (1, 2):
            t1 = time.time()
            machines.launch(
                box,
                job_dir,
                "encode",
                f"PYTHONPATH={remote.REMOTE_SRC} {REMOTE_VENV} {remote.tool('child_encode')}"
                " < job.json",
            )
            try:
                log = machines.wait(box, job_dir, "encode", timeout=hours * 3600)
            except RuntimeError as failure:
                say(f"shard {shard} {band}: attempt {attempt} failed: {str(failure)[-200:]}")
                if attempt == 2:
                    raise
                workers = max(1, workers // 2)
                job["workers"] = workers
                local_job.write_text(json.dumps(job))
                rsync(box, [str(local_job)], f"{job_dir}/job.json", download=False)
                continue
            say(
                f"shard {shard} {band}: {log.strip().splitlines()[-1][:90]}"
                f" ({(time.time() - t1) / 60:.1f} min)"
            )
            break
        done.parent.mkdir(exist_ok=True)
        rsync(box, [f"{job_dir}/encoded.h5"], str(done), download=True)
    say(f"shard {shard}: every band home; tearing down {box_id}")
    vast.teardown(client_of(box), box_id)


def client_of(_box: Any) -> Any:
    from reverberate import auth
    from reverberate.gpu import vast

    auth.inject(["VASTAI_API_KEY"])
    return vast.VastClient(timeout=60)


def encode_sharded(
    out: Path,
    *,
    gpu_instance: int,
    shards: int,
    hours: float,
    max_dph: float,
    min_cores: int,
    min_cpu_ghz: float,
    yes: bool,
    boxes: list[tuple[Any, int]] | None = None,
    say: Callable[[str], None] = print,
) -> list[Path]:
    """Push every shard off the card, encode each on its box, merge the shards.

    ``boxes`` may come from :func:`rent_encode_boxes` called earlier by the
    campaign; otherwise they are rented here. A shard that fails is retried on
    its own box with half the workers; a shard that fails twice leaves its
    box up with the pressure and is reported, the others still complete.
    """
    from reverberate import auth
    from reverberate.gpu import vast
    from reverberate.wave.remote import Machine

    auth.inject(["VASTAI_API_KEY"])
    client = vast.VastClient(timeout=60)
    identity = vast.account_identity(client)
    gpu_inst = client.instance(gpu_instance)
    if gpu_inst is None:
        raise SystemExit(f"instance {gpu_instance} does not exist; the pressure is gone with it")
    gpu = Machine.from_instance(gpu_inst, identity)
    machines.rearm_watchdog(gpu_instance, hours + 1.0, say=say)
    if boxes is None:
        boxes = rent_encode_boxes(
            out,
            shards=shards,
            hours=hours,
            max_dph=max_dph,
            min_cores=min_cores,
            min_cpu_ghz=min_cpu_ghz,
            yes=yes,
            say=say,
        )
        if not boxes:
            return []
    push_shards(out, gpu, gpu_instance, boxes, client=client, hours=hours, say=say)
    failures: list[int] = []
    with ThreadPoolExecutor(max_workers=len(boxes)) as pool:
        futures = {
            pool.submit(encode_on_box, out, box, box_id, k, len(boxes), hours=hours, say=say): k
            for k, (box, box_id) in enumerate(boxes)
        }
        for future, k in futures.items():
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - one shard failing must not hide the others
                failures.append(k)
                say(f"shard {k} FAILED: {str(error)[:300]}")
    if failures:
        raise SystemExit(
            f"shards {failures} failed; their boxes stay up with the pressure:"
            f" {[boxes[k][1] for k in failures]}"
        )
    plan = json.loads((out / "plan.json").read_text())
    for source in plan["sources"]:
        for band in bands_of_source(plan, source):
            parts = [
                out / "encoded" / f"{source['name']}__{slug(band)}.shard{k}.h5"
                for k in range(len(boxes))
            ]
            merge_encoded(parts, out / "encoded" / f"{source['name']}__{slug(band)}.h5")
            say(f"{source['name']} {band}: {len(parts)} shards merged")
    from reverberate.experiments.w40_volume_field.assemble import assemble_field

    return assemble_field(out, sources=[s["name"] for s in plan["sources"]], workers=4)
