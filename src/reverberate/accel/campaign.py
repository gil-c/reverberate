"""The campaign on the machine that holds the card, resumed from what is on disk.

One process, one machine, the stages in order: the three grids voxelised on
the card and installed in this machine's cache; the audit view of each
grid; the arrays placed and the plan written; each band solved by the
engine, in receiver slices when the host's RAM asks for it, and each slice
encoded on the card before the next is solved, so the pressure never leaves
the disk it was written to and is deleted once it is signals; the field
assembled. Every stage leaves its output where the next stage and a rerun
look for it, so the driver skips what exists and continues.

``status.json`` beside the log is rewritten at every step: the stage, the
band, the slice, the percentage, the elapsed time and an estimate of the
rest, which is what :mod:`reverberate.gpu.onebox` reads every five minutes
from the laptop. ``campaign.done`` or ``campaign.failed`` ends it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.accel.backend import device_report, xp_for
from reverberate.accel.solve import (
    host_memory_gb,
    output_sample_bytes,
    slices_for,
    solve_slices,
)

__all__ = ["Campaign", "run_campaign", "slice_rows"]


def slice_rows(rows: list[list[int] | None], start: int, stop: int) -> list[list[int] | None]:
    """The plan's rows restricted to ``[start, stop)`` and shifted to the slice's origin."""
    out: list[list[int] | None] = []
    for row in rows:
        if row is not None and start <= row[0] and row[1] <= stop:
            out.append([row[0] - start, row[1] - start])
        else:
            out.append(None)
    return out


@dataclass(frozen=True)
class SolveJob:
    """One band of one source: what the engine and the encoder are told."""

    name: str
    band: str
    source: dict[str, Any]
    spec: dict[str, Any]


@dataclass
class Campaign:
    """State on disk: the bundle, the run directory, the log and the status."""

    bundle: Path
    out: Path
    pffdtd_dir: Path
    devices: str | None = None
    gpu: bool | None = None
    workers: int | None = None
    #: Bands to solve, ``None`` for every band of the plan; a flow test solves one.
    solve_bands: tuple[str, ...] | None = None
    #: Whether to assemble the field; only when every band is solved.
    assemble_field: bool = True
    #: Whether to run the geometric mirror beside each assembled field (ADR 0014):
    #: the card phase and the host phase on this machine, the field never moving.
    mirror_field: bool = True
    #: A calibration json (``mirror/calibration/<key>.json``) to render the mirror with.
    mirror_parameters: Path | None = None
    #: Points of each band encoded again on the host's CPU path, in the
    #: background, and compared with the card's numbers: the card's own
    #: evidence, per campaign and per card. Zero disables it.
    selfcheck_points: int = 2
    started: float = field(default_factory=time.time)
    spec: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    selfchecks: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bundle = Path(self.bundle)
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.spec = json.loads((self.bundle / "campaign.json").read_text())
        self.xp = xp_for(self.gpu)
        self.status = {
            "stage": "start",
            "started": self.started,
            "device": device_report(),
            "host_ram_gb": round(host_memory_gb(), 1),
            "cpus": os.cpu_count(),
        }

    # ---- logging ----------------------------------------------------------------------

    def say(self, message: str) -> None:
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} +{(time.time() - self.started) / 3600:.2f} h"
            f" | {message}"
        )
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

    # ---- the bundle's content --------------------------------------------------------------

    @property
    def models(self) -> Path:
        return self.bundle / str(self.spec["models"])

    @property
    def model_json(self) -> Path:
        return self.bundle / str(self.spec["model_json"])

    @property
    def materials(self) -> Path:
        return self.bundle / str(self.spec["materials"])

    @property
    def keys(self) -> dict[str, str]:
        return {band: str(spec["cache_key"]) for band, spec in self.spec["bands"].items()}

    @property
    def fmax_hz(self) -> dict[str, float]:
        return {band: float(spec["fmax_hz"]) for band, spec in self.spec["bands"].items()}

    @property
    def durations_s(self) -> dict[str, float]:
        return {band: float(spec["duration_s"]) for band, spec in self.spec["bands"].items()}

    def rooms(self) -> list[Any]:
        from reverberate.geometry.rooms import rooms_from_record

        return rooms_from_record(json.loads((self.bundle / "rooms.json").read_text()))

    # ---- stages ---------------------------------------------------------------------------

    def voxelise(self) -> dict[str, Any]:
        """Every band's grid on the card, installed under the key the laptop computed."""
        from reverberate.accel.voxelise import voxelise_scene
        from reverberate.experiments.run import scene_spec
        from reverberate.wave.remote_voxelise import install_entry
        from reverberate.wave.voxelise import entry_for

        reports: dict[str, Any] = {}
        for band, fmax in self.fmax_hz.items():
            scene, model_json, _ = scene_spec(self.models, self.spec["storey_scene"], fmax)
            if scene.key != self.keys[band]:
                raise RuntimeError(
                    f"{band}: this machine computes key {scene.key} for the grid the laptop"
                    f" keyed {self.keys[band]}; the bundle and the code disagree"
                )
            entry = entry_for(scene)
            if entry.complete:
                self.say(f"voxelise {band}: entry {scene.key} already installed")
                reports[band] = {"cached": True}
                continue
            self.set_status(band=band, fmax_hz=fmax)
            staging = self.out / "vox" / f"{band}.partial"
            if staging.exists():
                shutil.rmtree(staging)
            t0 = time.time()
            report = voxelise_scene(
                model_json,
                staging,
                mat_folder=scene.mat_folder,
                mat_files=dict(scene.mat_files),
                fmax=fmax,
                ppw=scene.ppw,
                nh=int(scene.nh or 0),
                tc=scene.tc,
                rh=scene.rh,
                xp=self.xp,
                say=lambda m, band=band: self.say(f"  vox {band} | {m}"),
            )
            record = report.record()
            record["computed_on"] = "gpu"
            installed = install_entry(scene, staging, record, time.time() - t0)
            self.say(
                f"voxelise {band}: {record['boundary_nodes']} boundary nodes in"
                f" {(time.time() - t0) / 60:.1f} min -> {installed.path}"
            )
            reports[band] = record
        return reports

    def audit(self) -> dict[str, str]:
        """The tiered audit view of every grid, from the rooms the bundle carries."""
        from reverberate.experiments.audit_view import build
        from reverberate.experiments.w40_volume_field.storey import COARSE_SPAN, audit_meshes

        audit = self.out / "audit"
        rooms = self.rooms()
        for band, key in self.keys.items():
            target = audit / f"{self.fmax_hz[band]:g}"
            if (target / "voxels" / "rooms.json").is_file():
                continue
            self.set_status(band=band)
            t0 = time.time()
            report = build(
                key,
                None,
                self.spec["scene_id"],
                target,
                coarse_span=COARSE_SPAN.get(band, 1),
                partition_rooms=rooms,
            )
            self.say(
                f"audit {self.fmax_hz[band]:g} Hz: {len(report.get('rooms', []))} rooms in"
                f" {(time.time() - t0) / 60:.1f} min"
            )
        return audit_meshes(self.out, audit, self.fmax_hz)

    def plan(self) -> dict[str, Any]:
        """The arrays on every grid and ``plan.json``, from the bundle's points."""
        from reverberate.experiments.w40_volume_field.plan import plan_arrays

        if (self.out / "plan.json").is_file():
            return dict(json.loads((self.out / "plan.json").read_text()))
        points = np.load(self.bundle / "points.npy")
        labels = [str(v) for v in self.spec["labels"]]
        return plan_arrays(
            self.out,
            area=float(self.spec["free_floor_m2"]),
            rooms=None,
            scene_id=str(self.spec["scene_id"]),
            sources=list(self.spec["sources"]),
            storey_keys={"low": self.keys["low"], "mid": self.keys["mid"]},
            room_keys={},
            durations_s=self.durations_s,
            height_m=float(self.spec["height_m"]),
            ram_gb=float(self.spec["ram_gb"]),
            pitch_m=float(self.spec["pitch_m"]),
            order=int(self.spec["order"]),
            fit_order=int(self.spec["fit_order"]),
            high_key=self.keys.get("high"),
            points_and_labels=(points, labels),
        )

    def solve_and_encode(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Each band of each source: the engine in slices, the encoder on each slice."""
        from reverberate.accel.encode import merge_encoded
        from reverberate.experiments.run import entry_from_key
        from reverberate.experiments.w40_volume_field.plan import bands_of_source, slug

        ram_gb = host_memory_gb()
        records: dict[str, Any] = {}
        jobs = [
            SolveJob(name=f"{s['name']}__{slug(b)}", band=b, source=s, spec=plan["bands"][b])
            for s in plan["sources"]
            for b in bands_of_source(plan, s)
            if self.solve_bands is None or b in self.solve_bands
        ]
        sample_bytes = output_sample_bytes(self.pffdtd_dir)
        self.say(f"solve: the engine writes {sample_bytes} bytes a sample")
        for number, job in enumerate(jobs, start=1):
            merged = self.out / "encoded" / f"{job.name}.h5"
            if merged.is_file():
                self.say(f"solve {job.name}: encoded already")
                continue
            entry = entry_from_key(job.spec["cache_key"])
            positions = np.load(self.out / f"array_positions_{slug(job.band)}.npy")
            # The engine runs one step more than the plan's samples and keeps
            # every one; the plan's own figure assumes upstream's float64.
            output_bytes = (
                float(job.spec["cost"]["receivers"])
                * (float(job.spec["samples"]) + 1.0)
                * sample_bytes
            )
            parts_expected = slices_for(output_bytes, ram_gb)
            self.set_status(
                band=job.band,
                source=job.source["name"],
                job=f"{number}/{len(jobs)}",
                slices=parts_expected,
                estimated_engine_s=float(job.spec["cost"]["estimated_gpu_s"]) * parts_expected,
            )
            t0 = time.time()
            self.release_card()
            slices = solve_slices(
                job_root=self.out / "jobs" / job.name,
                entry_path=entry.path,
                source_position=np.asarray(job.source["position"], dtype=float),
                positions=positions,
                rows=job.spec["rows"],
                duration_s=float(job.spec["samples"]) / float(job.spec["sample_rate_hz"]),
                output_bytes=output_bytes,
                ram_gb=ram_gb,
                pffdtd_dir=self.pffdtd_dir,
                devices=self.devices,
                consume=lambda k, start, stop, sim_outs, comms, job=job, plan=plan, pos=positions: (
                    self.encode_slice(job, plan, pos, k, start, stop, sim_outs, comms)
                ),
                say=lambda m, name=job.name: self.say(f"  solve {name} | {m}"),
                grid_points=float(job.spec["cost"]["grid_points"]),
                already_done=lambda k, start, stop, job=job: self.part_path(job, k).is_file(),
            )
            parts = [self.part_path(job, int(record["slice"])) for record in slices]
            merge_encoded(parts, merged)
            for part in parts:
                part.unlink()
            shutil.rmtree(self.out / "jobs" / job.name, ignore_errors=True)
            records[job.name] = {
                "slices": slices,
                "total_s": round(time.time() - t0, 1),
                "engine_s": round(sum(s.get("engine_s", 0.0) for s in slices), 1),
                "encode_s": round(sum(s.get("consume_s", 0.0) for s in slices), 1),
            }
            self.say(
                f"solve {job.name}: {len(slices)} slice(s), engine"
                f" {records[job.name]['engine_s'] / 60:.1f} min, encode"
                f" {records[job.name]['encode_s'] / 60:.1f} min"
            )
        return records

    def part_path(self, job: SolveJob, k: int) -> Path:
        """Where slice ``k`` of a band's encoding lands before the merge."""
        return self.out / "encoded" / f"{job.name}.part{k}.h5"

    def encode_slice(
        self,
        job: SolveJob,
        plan: dict[str, Any],
        positions: np.ndarray,
        k: int,
        start: int,
        stop: int,
        sim_outs: Path,
        comms: Path,
    ) -> dict[str, Any]:
        """One slice's pressure to signals on the card, then the pressure is deleted."""
        from reverberate.accel.encode import encode_band

        part = self.part_path(job, k)
        self.set_status(slice=k, stage_detail="encode")
        report = encode_band(
            plan,
            job.band,
            job.source,
            pressure=sim_outs,
            comms=comms,
            positions=positions,
            output=part,
            xp=self.xp,
            row_offset=start,
            rows=slice_rows(job.spec["rows"], start, stop),
            say=lambda m: self.say(f"    encode {job.name} | {m}"),
        )
        if k == 0 and self.selfcheck_points > 0:
            self.launch_selfcheck(
                job.name, job.band, job.source, sim_outs, comms, start, stop, part
            )
        sim_outs.unlink()
        self.release_card()
        self.set_status(stage_detail="engine")
        return report

    def release_card(self) -> None:
        """Give the card back before the engine runs: cupy's pools keep what they grew to.

        On 51006261 the encoder's pool held 32 GB of the first card and the
        next solve asserted out of memory before it had read its grid.
        """
        if self.xp is np:
            return
        self.xp.get_default_memory_pool().free_all_blocks()
        self.xp.get_default_pinned_memory_pool().free_all_blocks()
        self.xp.fft.config.get_plan_cache().clear()

    def launch_selfcheck(
        self,
        name: str,
        band: str,
        source: dict[str, Any],
        sim_outs: Path,
        comms: Path,
        start: int,
        stop: int,
        gpu_part: Path,
    ) -> None:
        """Keep the pressure of a few points and encode them on the CPU path, detached."""
        import subprocess
        import sys

        import h5py

        from reverberate.experiments.w40_volume_field.plan import slug

        plan = json.loads((self.out / "plan.json").read_text())
        rows = slice_rows(plan["bands"][band]["rows"], start, stop)
        chosen = [i for i, r in enumerate(rows) if r is not None][: self.selfcheck_points]
        if not chosen:
            return
        folder = self.out / "selfcheck"
        folder.mkdir(exist_ok=True)
        sample = folder / f"{name}.sample.h5"
        sample_rows: list[list[int] | None] = [None] * len(rows)
        cursor = 0
        with h5py.File(sim_outs, "r") as source_file, h5py.File(sample, "w") as target:
            blocks = []
            for index in chosen:
                a, b = rows[index]  # type: ignore[misc]
                blocks.append(np.asarray(source_file["u_out"][a:b]))
                sample_rows[index] = [cursor, cursor + (b - a)]
                cursor += b - a
            target.create_dataset("u_out", data=np.concatenate(blocks))
        # The comms rows of the chosen points, in the same order.
        with (
            h5py.File(comms, "r") as source_file,
            h5py.File(folder / f"{name}.comms.h5", "w") as target,
        ):
            alpha = np.asarray(source_file["out_alpha"][...])
            pieces = [alpha[rows[i][0] : rows[i][1]] for i in chosen]  # type: ignore[index]
            target.create_dataset("out_alpha", data=np.concatenate(pieces))
            target.create_dataset("diff", data=np.asarray(source_file["diff"]))
        positions = np.load(self.out / f"array_positions_{slug(band)}.npy")
        plan_rows = plan["bands"][band]["rows"]
        np.save(
            folder / f"{name}.positions.npy",
            np.concatenate([positions[plan_rows[i][0] : plan_rows[i][1]] for i in chosen]),
        )
        job = {
            "plan": str(self.out / "plan.json"),
            "band": band,
            "source": source,
            "pressure": str(sample),
            "comms": str(folder / f"{name}.comms.h5"),
            "positions": str(folder / f"{name}.positions.npy"),
            "rows": sample_rows,
            # The merged file of the band, which outlives the part the slice
            # wrote; the check waits for it, since the CPU path is the slower.
            "gpu_encoded": str(self.out / "encoded" / f"{name}.h5"),
            "output": str(folder / f"{name}.json"),
            "cpu_encoded": str(folder / f"{name}.cpu.h5"),
        }
        job_path = folder / f"{name}.job.json"
        job_path.write_text(json.dumps(job))
        log = (folder / f"{name}.log").open("w")
        process = subprocess.Popen(
            [sys.executable, "-m", "reverberate.accel", "selfcheck", "--job", str(job_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "REVERBERATE_NO_GPU": "1", "OMP_NUM_THREADS": "4"},
        )
        self.selfchecks.append((name, process))
        self.say(f"  selfcheck {name}: {len(chosen)} points on the CPU path, pid {process.pid}")

    def wait_selfchecks(self, timeout_s: float = 3600.0) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, process in self.selfchecks:
            try:
                process.wait(timeout=timeout_s)
            except Exception:  # noqa: BLE001 - a check that hangs is reported, not waited for
                process.kill()
                results[name] = {"error": "timed out"}
                continue
            path = self.out / "selfcheck" / f"{name}.json"
            results[name] = (
                json.loads(path.read_text())
                if path.is_file()
                else {"error": f"exit {process.returncode}"}
            )
            summary = results[name].get("max_over_peak", {}).get("max")
            self.say(f"  selfcheck {name}: max |diff| / peak {summary}")
        return results

    def assemble(self, meshes: dict[str, str]) -> list[Path]:
        from reverberate.experiments.w40_volume_field.assemble import assemble_field

        # A point's assembly holds three bands, their tails and its filters in
        # float64: about a gigabyte a worker. Every core but one, within that.
        by_ram = int(max(1.0, (host_memory_gb() - 8.0) / 1.5))
        workers = self.workers or max(1, min((os.cpu_count() or 2) - 1, by_ram))
        self.say(f"assemble: {workers} workers")
        return assemble_field(self.out, workers=workers, meshes=meshes or None)

    def mirror(self, fields: list[Path]) -> list[dict[str, Any]]:
        """The geometric mirror of every assembled field, both phases here; the reports."""
        from reverberate.mirror.calibrate import Parameters
        from reverberate.mirror.stage import MirrorSettings, run_mirror

        parameters = Parameters()
        if self.mirror_parameters is not None:
            record = json.loads(Path(self.mirror_parameters).read_text())
            parameters = Parameters.from_record(record.get("parameters", record))
        workers = self.workers or max(1, (os.cpu_count() or 2) - 1)
        settings = MirrorSettings(workers=workers, parameters=parameters)
        reports = []
        names = {Path(f).stem for f in fields}
        for source in self.spec["sources"]:
            name = str(source["name"])
            if name not in names:
                continue
            report = run_mirror(
                self.out,
                models=self.models,
                source={"name": name, "position": [float(v) for v in source["position"]]},
                settings=settings,
                phase="all",
                say=self.say,
            )
            summary = report.get("summary") or {}
            self.say(f"mirror {name}: {json.dumps(summary.get('medians', summary))[:300]}")
            reports.append({k: v for k, v in report.items() if k != "settings"})
        return reports

    def run(self) -> dict[str, Any]:
        """Every stage in order; ``campaign.done`` or ``campaign.failed`` at the end."""
        for marker in ("campaign.done", "campaign.failed"):
            (self.out / marker).unlink(missing_ok=True)
        try:
            self.say(
                f"campaign {self.spec.get('dwelling')} on {self.status['device'].get('gpu')},"
                f" {self.status['host_ram_gb']} GB RAM, {self.status['cpus']} cpus"
            )
            vox = self.stage("voxelise", self.voxelise)
            meshes = self.stage("audit", self.audit)
            plan = self.stage("plan", self.plan)
            self.say(f"plan: {len(plan['points'])} points, bands {sorted(plan['bands'])}")
            solves = self.stage("solve", lambda: self.solve_and_encode(plan))
            fields = (
                self.stage("assemble", lambda: self.assemble(meshes))
                if self.assemble_field and self.solve_bands is None
                else []
            )
            mirrors = (
                self.stage("mirror", lambda: self.mirror(fields))
                if self.mirror_field and fields
                else []
            )
            selfchecks = self.wait_selfchecks()
            report = {
                "dwelling": self.spec.get("dwelling"),
                "device": self.status["device"],
                "host_ram_gb": self.status["host_ram_gb"],
                "cpus": self.status["cpus"],
                "timings_s": self.timings,
                "total_s": round(time.time() - self.started, 1),
                "voxelise": vox,
                "solves": solves,
                "fields": [str(f) for f in fields],
                "mirrors": mirrors,
                "meshes": meshes,
                "selfcheck": selfchecks,
            }
            (self.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
            self.set_status(stage="done")
            (self.out / "campaign.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            self.say(f"campaign complete in {(time.time() - self.started) / 3600:.2f} h")
            return report
        except BaseException as error:
            self.set_status(stage="failed", error=repr(error)[:500])
            (self.out / "campaign.failed").write_text(repr(error)[:2000])
            self.say(f"campaign FAILED: {error!r}"[:600])
            raise


def run_campaign(
    bundle: Path,
    out: Path,
    *,
    pffdtd_dir: Path,
    devices: str | None = None,
    gpu: bool | None = None,
    workers: int | None = None,
    solve_bands: tuple[str, ...] | None = None,
    selfcheck_points: int = 2,
    mirror_field: bool = True,
    mirror_parameters: Path | None = None,
) -> dict[str, Any]:
    return Campaign(
        bundle=bundle,
        out=out,
        pffdtd_dir=pffdtd_dir,
        devices=devices,
        gpu=gpu,
        workers=workers,
        solve_bands=solve_bands,
        selfcheck_points=selfcheck_points,
        mirror_field=mirror_field,
        mirror_parameters=mirror_parameters,
    ).run()
