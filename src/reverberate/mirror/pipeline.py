"""The mirror of one source over a lattice: trace, render, align, write.

:func:`trace` computes every receiver's paths and the rays' histogram;
:func:`render` turns them into one response per point; :func:`run` does both
beside a wave field, puts the result on the field's clock and scale and
writes it in the field's format. Every step takes :class:`Devices`: the
host's cores, one card or several.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.compute import Devices
from reverberate.mirror.diffract import diffracted_paths
from reverberate.mirror.direct import direct_energy, measure_signature
from reverberate.mirror.engine import histogram_on_devices, paths_on_devices
from reverberate.mirror.files import (
    align_to_reference,
    lattice_of,
    read_omni,
    write_field,
    write_paths,
)
from reverberate.mirror.geometry import (
    DerivedScene,
    GeometryRules,
    derive,
    load_derived,
    write_derived,
)
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, occluder_grid
from reverberate.mirror.parameters import Parameters, apply_parameters, image_scene, regain
from reverberate.mirror.rays import Histogram, RaySettings
from reverberate.mirror.render import RenderSettings, render_point
from reverberate.spatial.encode import Ambisonic

__all__ = ["MirrorSettings", "Traced", "derive_scene", "render", "run", "trace"]


@dataclass(frozen=True)
class MirrorSettings:
    """Every choice of a mirror field, in one record."""

    rules: GeometryRules = field(default_factory=GeometryRules)
    ism: IsmSettings = field(default_factory=IsmSettings)
    rays: RaySettings = field(default_factory=RaySettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    parameters: Parameters = field(default_factory=Parameters)
    sound_speed_m_s: float = 343.2
    seed: int = 0
    #: Points with a direct path the source signature is read on.
    signature_points: int = 48

    def pinned(self) -> MirrorSettings:
        """These settings with the rays and tail the calibration was fitted under."""
        fitted = self.parameters.rendered_with
        if not fitted:
            return self
        rays = RaySettings(**{**self.rays.record(), **fitted.get("rays", {})})
        tail = {k: v for k, v in fitted.get("render", {}).items() if k.startswith("tail_")}
        return replace(self, rays=rays, render=replace(self.render, **tail))

    def record(self) -> dict[str, Any]:
        return {
            "rules": self.rules.record(),
            "ism": self.ism.record(),
            "rays": self.rays.record(),
            "render": self.render.record(),
            "parameters": self.parameters.record(),
            "sound_speed_m_s": self.sound_speed_m_s,
            "seed": self.seed,
            "signature_points": self.signature_points,
        }


def derive_scene(
    models: Path, cache: Path, rules: GeometryRules, seed: int = 0, say: Any = None
) -> DerivedScene:
    """The derived geometry of ``models/apartment_full.json``; ``cache`` when its rules match."""
    if Path(cache).with_suffix(".json").is_file():
        found = load_derived(cache)
        if found.rules == rules:
            return found
    manifest_path = Path(models) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    derived = derive(
        Path(models) / "apartment_full.json", rules=rules, manifest=manifest, seed=seed, say=say
    )
    write_derived(derived, cache)
    return derived


@dataclass
class Traced:
    """What the cards compute: every receiver's paths and the rays' histogram."""

    paths: list[Paths]
    histogram: Histogram
    record: dict[str, Any]


def trace(
    catalogue: DerivedScene,
    source: np.ndarray,
    positions: np.ndarray,
    settings: MirrorSettings,
    devices: Devices | None = None,
    say: Any = None,
) -> Traced:
    """The image tree, every receiver's paths and the histogram of the rays."""
    devices = devices or Devices.detect()
    source = np.asarray(source, dtype=float).reshape(3)
    scene = apply_parameters(catalogue, settings.parameters)
    region = (positions.min(axis=0) - 0.5, positions.max(axis=0) + 0.5)
    ism = replace(settings.ism, sound_speed_m_s=settings.sound_speed_m_s)
    rays = replace(settings.rays, sound_speed_m_s=settings.sound_speed_m_s)
    tree = grow_tree(scene, source, ism, region=region)
    grid = occluder_grid(scene, rays.cell_m)
    every = paths_on_devices(scene, tree, positions, ism, devices=devices, grid=grid, say=say)
    if settings.parameters.image_absorption_scale is not None:
        images = image_scene(catalogue, settings.parameters)
        every = [regain(p, images) for p in every]
    histogram = histogram_on_devices(
        scene, source, positions, rays, devices=devices, grid=grid, say=say
    )
    return Traced(
        every,
        histogram,
        {
            "images": tree.count,
            "paths_median": float(np.median([p.count for p in every])),
            "without_direct": int(sum(1 for p in every if not np.any(p.order == 0))),
            "hits_per_receiver_median": float(np.median(histogram.hits.sum(axis=1))),
        },
    )


def render(
    traced: Traced,
    catalogue: DerivedScene,
    source: np.ndarray,
    positions: np.ndarray,
    settings: MirrorSettings,
    *,
    rate: float,
    order: int,
    scratch: Path,
    signature: np.ndarray | None = None,
    devices: Devices | None = None,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Every point's response to ``scratch/<index>.npy``; the direct energies and the record.

    Points with a direct path first: their tail scales, as a median, serve
    the points without one, which also get their diffracted onset.
    """
    devices = devices or Devices.detect()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    source = np.asarray(source, dtype=float).reshape(3)
    every = traced.paths
    render_settings = replace(settings.render, order=order, sample_rate_hz=rate)
    c = settings.sound_speed_m_s
    with_direct = [i for i, p in enumerate(every) if bool(np.any(p.order == 0))]
    without = [i for i, p in enumerate(every) if not bool(np.any(p.order == 0))]
    onsets: dict[int, Paths] = {}
    record: dict[str, Any] = {}
    if without:
        onsets, record["diffraction"] = diffracted_paths(
            catalogue, source, positions, without, sound_speed_m_s=c
        )

    def point(index: int, fallback: Any, xp: Any) -> tuple[int, float, dict[str, Any]]:
        response, point_record = render_point(
            index,
            every[index],
            traced.histogram,
            render_settings,
            tail_gain_db=np.asarray(settings.parameters.tail_gain_db, dtype=float),
            receiver_radius_m=settings.rays.receiver_radius_m,
            sound_speed_m_s=c,
            seed=settings.seed + index,
            fallback=fallback,
            signature=signature,
            xp=xp,
        )
        np.save(scratch / f"{index}.npy", response.signals.astype(np.float32))
        return index, direct_energy(response.signals, rate), point_record

    def batch(indices: list[int], fallback_of: Any) -> list[tuple[int, float, dict[str, Any]]]:
        def work(card: int, share: np.ndarray) -> list[tuple[int, float, dict[str, Any]]]:
            if card < 0:
                return [point(indices[k], fallback_of(indices[k]), np) for k in share]
            import cupy

            with cupy.cuda.Device(card):
                return [point(indices[k], fallback_of(indices[k]), cupy) for k in share]

        return [r for part in devices.map(work, devices.split(len(indices))) for r in part]

    energies: dict[int, float] = {}
    scales = []
    for index, energy, point_record in batch(with_direct, lambda _: None):
        energies[index] = energy
        tail = point_record.get("tail")
        if isinstance(tail, dict) and "scale_per_band" in tail:
            scales.append(np.asarray(tail["scale_per_band"], dtype=float))
    fallback_scale = np.median(np.stack(scales), axis=0) if scales else None
    record["tail_scale_fallback"] = None if fallback_scale is None else fallback_scale.tolist()

    def fallback_of(index: int) -> Any:
        if fallback_scale is None:
            return None
        straight = float(np.linalg.norm(positions[index] - source))
        return (fallback_scale, straight / c, onsets.get(index))

    for index, energy, _ in batch(without, fallback_of):
        energies[index] = energy
    record["points_without_direct"] = len(without)
    return energies, record


class _Aligned(Mapping[int, Ambisonic]):
    """The rendered responses on disk, shifted onto the reference's clock on access."""

    def __init__(self, scratch: Path, count: int, rate: float, order: int, lead: int) -> None:
        self.scratch, self.count, self.rate, self.order, self.lead = (
            scratch,
            count,
            rate,
            order,
            lead,
        )

    def __getitem__(self, index: int) -> Ambisonic:
        path = self.scratch / f"{index}.npy"
        if not path.is_file():
            raise KeyError(index)
        signals = np.load(path).astype(np.float64)
        shifted = np.zeros_like(signals)
        if self.lead >= 0:
            shifted[:, self.lead :] = signals[:, : signals.shape[1] - self.lead]
        else:
            shifted[:, : self.lead] = signals[:, -self.lead :]
        return Ambisonic(shifted, self.rate, self.order, np.zeros(3))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.count))

    def __len__(self) -> int:
        return self.count


def run(
    run_dir: Path,
    source: str,
    position: np.ndarray,
    *,
    models: Path,
    settings: MirrorSettings | None = None,
    devices: Devices | None = None,
    out: str = "field_mirror",
    say: Any = print,
) -> dict[str, Any]:
    """The mirror field of ``source`` beside its wave field ``field/<source>.h5``.

    Writes ``<out>/<source>.h5`` on the wave field's clock and scale, and in
    ``mirror/``: the derived scene, the paths of every point, the source
    signature and the report. Returns the report.
    """
    run_dir = Path(run_dir)
    mirror = run_dir / "mirror"
    mirror.mkdir(parents=True, exist_ok=True)
    settings = (settings or MirrorSettings()).pinned()
    devices = devices or Devices.detect()
    position = np.asarray(position, dtype=float).reshape(3)
    reference = run_dir / "field" / f"{source}.h5"
    report: dict[str, Any] = {
        "source": source,
        "devices": devices.record(),
        "settings": settings.record(),
        "timings_s": {},
    }

    def timed(label: str, work: Any) -> Any:
        started = time.time()
        result = work()
        report["timings_s"][label] = round(time.time() - started, 1)
        say(f"mirror {source} | {label}: {time.time() - started:.1f} s")
        return result

    positions, rate, order = lattice_of(run_dir, source)
    catalogue = timed(
        "derive", lambda: derive_scene(models, mirror / "scene", settings.rules, settings.seed)
    )
    report["scene"] = {"key": catalogue.key, "summary": catalogue.summary()}
    traced = timed("trace", lambda: trace(catalogue, position, positions, settings, devices, say))
    report["trace"] = traced.record
    write_paths(traced.paths, mirror / f"paths_{source}.npz")
    with_direct = [i for i, p in enumerate(traced.paths) if bool(np.any(p.order == 0))]
    step = max(1, len(with_direct) // max(1, settings.signature_points))
    omni, _ = read_omni(reference, with_direct[::step])
    taps, report["signature"] = measure_signature(omni, rate)
    np.save(mirror / f"signature_{source}.npy", taps)
    scratch = mirror / f"render_{source}"
    shutil.rmtree(scratch, ignore_errors=True)
    energies, report["render"] = timed(
        "render",
        lambda: render(
            traced,
            catalogue,
            position,
            positions,
            settings,
            rate=rate,
            order=order,
            scratch=scratch,
            signature=taps,
            devices=devices,
        ),
    )
    # Aligned on the points with a direct path: elsewhere the first arrival is
    # diffracted or reflected, and its level says nothing of the scale.
    direct = set(with_direct)
    alignment = align_to_reference(
        reference,
        {i: e for i, e in energies.items() if i in direct},
        sound_speed_m_s=settings.sound_speed_m_s,
    )
    report["alignment"] = alignment.record()
    lead = int(round(alignment.lead_s * rate))
    timed(
        "write",
        lambda: write_field(
            run_dir / out / f"{source}.h5",
            reference,
            _Aligned(scratch, len(traced.paths), rate, order, lead),
            provenance={"scene_key": catalogue.key, "settings": settings.record()},
            gain=alignment.gain,
        ),
    )
    shutil.rmtree(scratch, ignore_errors=True)
    (mirror / f"report_{source}.json").write_text(json.dumps(report, indent=1, default=str))
    return report
