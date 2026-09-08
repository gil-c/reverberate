"""Show and play what entered and left the wave solver, for one run.

This module is the *data* half of the acoustic view. It reduces a rendered run
to one JSON payload plus its audio, and the apartment browser renders it as the
third mode alongside colour and label. There is one application and one
apartment selector; a run is simply another way of looking at the apartment it
was simulated in, reached from the same place as every other view, and the
camera does not move when the mode changes.

The geometry described here is **the serialised surface list the solver read**,
``model_json``, and nothing else. Not the authored HSSD instances, not a
prettier collider, not a re-derived mesh. That is roadmap constraint 9: the
viewer and the solver read the same file, so a picture and a response can be
proved to be of the same object by comparing one digest. The acoustic mode that
stood here before coloured authored instances by a ``pyroomacoustics``
absorption palette, which describes a ray solver this project no longer uses; it
has been removed rather than kept beside its replacement, because two modes
called acoustic showing different geometries is worse than one.

Everything the page needs is computed here, at build time, and embedded as
plain JSON: the envelopes, the energy decay curves, the measures. The browser
draws and plays; it does no signal processing. That keeps the analysis in the
tested Python path rather than duplicated in JavaScript where nothing checks it.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import metrics
from reverberate.store import ObjectStore

__all__ = [
    "SPECTROGRAM_BINS",
    "SPECTROGRAM_FRAMES",
    "RunRef",
    "absorption_colour",
    "RunView",
    "build_site",
    "decay_curve_points",
    "discover_runs",
    "envelope",
    "model_json_of",
    "model_materials",
    "run_scene",
    "spectrogram",
    "surface_groups",
]

#: How many points an envelope or decay curve is reduced to before it is
#: embedded. A 72 kHz response is 108,000 samples per receiver and twelve of
#: them would be a 10 MB page for a plot a few hundred pixels wide. 512 is
#: comfortably more than the pixels available and small enough to inline.
PLOT_POINTS = 512

#: Spectrogram size. Chosen so one image is 32 KB before base64 and twelve of
#: them add about half a megabyte to the page, which is the most that is worth
#: paying to avoid doing an FFT in the browser.
SPECTROGRAM_BINS = 128
SPECTROGRAM_FRAMES = 256

#: Decibels below the loudest bin that the spectrogram floor sits at. Wider than
#: this and the late tail is a black rectangle; narrower and the direct sound is
#: the only thing with any contrast.
SPECTROGRAM_RANGE_DB = 70.0


def envelope(signal: np.ndarray, points: int = PLOT_POINTS) -> list[list[float]]:
    """Min and max per bucket, so a waveform keeps its shape when reduced.

    Decimating by sampling would alias a 72 kHz response into whatever the
    stride happened to hit, and the result would look quieter than it is.
    Taking both extremes of each bucket preserves the outline the eye reads as
    the waveform.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size == 0:
        return []
    points = max(1, min(int(points), signal.size))
    edges = np.linspace(0, signal.size, points + 1, dtype=int)
    return [
        [float(np.min(chunk)), float(np.max(chunk))]
        for start, stop in zip(edges[:-1], edges[1:], strict=True)
        if (chunk := signal[start : max(stop, start + 1)]).size
    ]


def decay_curve_points(
    signal: np.ndarray, sample_rate_hz: float, points: int = PLOT_POINTS
) -> dict[str, list[float]]:
    """Broadband Schroeder decay, in decibels against seconds.

    Reduced by taking the value at evenly spaced samples rather than an
    average: the curve is already monotonic and smooth, so a point on it is
    representative in a way a point on a raw waveform is not.
    """
    curve = metrics.energy_decay_curve(np.asarray(signal, dtype=float).ravel())
    if curve.size == 0:
        return {"seconds": [], "db": []}
    index = np.linspace(0, curve.size - 1, min(points, curve.size), dtype=int)
    values = curve[index]
    # -inf is the tail after the integration reaches zero energy. It plots as a
    # break in the line rather than a value, so it is clamped to the floor the
    # axis shows instead of being silently dropped.
    values = np.where(np.isfinite(values), values, -120.0)
    return {
        "seconds": [round(float(value), 6) for value in index / float(sample_rate_hz)],
        "db": [round(float(value), 3) for value in values],
    }


def spectrogram(
    signal: np.ndarray,
    sample_rate_hz: float,
    bins: int = SPECTROGRAM_BINS,
    frames: int = SPECTROGRAM_FRAMES,
    max_hz: float | None = None,
) -> dict[str, Any]:
    """A short time Fourier magnitude, in decibels, quantised to bytes.

    Returned as base64 bytes rather than numbers because a float per pixel
    would be six times the size for a picture that is only ever going to be
    drawn into a canvas. Level 0 is the floor at ``SPECTROGRAM_RANGE_DB`` below
    the loudest bin and level 255 is that loudest bin, so the scale is relative
    to this response and two responses are not comparable by eye. That is
    stated on the page rather than left for the reader to assume either way.

    ``max_hz`` bounds the vertical axis. Left unset it is the Nyquist rate,
    which for a 48 kHz delivery file is 24 kHz: since the solver only ran to
    4 kHz that would spend five sixths of the picture drawing the empty band
    above the low pass. Setting it a little above the simulated ``fmax`` keeps
    the roll off visible, which is worth seeing, without the dead space.

    Rows run low frequency first, which is the opposite of how it is drawn; the
    page flips it so that low frequencies are at the bottom where a reader
    expects them.
    """
    import base64

    signal = np.asarray(signal, dtype=float).ravel()
    nyquist = float(sample_rate_hz) / 2.0
    ceiling = min(float(max_hz), nyquist) if max_hz else nyquist
    # Enough FFT rows that `bins` of them cover the requested ceiling.
    rows = max(int(np.ceil(bins * nyquist / ceiling)), bins)
    window_length = 2 * rows
    if signal.size < window_length:
        signal = np.pad(signal, (0, window_length - signal.size))

    starts = np.linspace(0, signal.size - window_length, frames, dtype=int)
    window = np.hanning(window_length)
    columns = np.abs(np.fft.rfft(signal[starts[:, None] + np.arange(window_length)] * window))
    magnitude = columns[:, :bins].T

    peak = float(np.max(magnitude))
    if peak <= 0:
        data = np.zeros((bins, frames), dtype=np.uint8)
    else:
        with np.errstate(divide="ignore"):
            decibels = 20.0 * np.log10(np.maximum(magnitude, 1e-12) / peak)
        scaled = (decibels + SPECTROGRAM_RANGE_DB) / SPECTROGRAM_RANGE_DB
        data = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)

    return {
        "bins": bins,
        "frames": frames,
        "max_hz": round(bins * nyquist / rows, 1),
        "seconds": round(signal.size / float(sample_rate_hz), 4),
        "range_db": SPECTROGRAM_RANGE_DB,
        "data": base64.b64encode(data.tobytes()).decode("ascii"),
    }


# The octave centres of the material table, from experiments.scene_export.BANDS.
# Repeated rather than imported because that module pulls in the mesh stack, and
# a test pins the two lists together so the copy cannot drift.
MATERIAL_BANDS_HZ = (16.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
COLOUR_BAND_HZ = 1000.0

# Reflective through to absorbent. Grey is deliberately not on the ramp: the
# exporter writes grey for every group, so a grey surface on screen means the
# absorption was not found rather than that it was found to be middling.
# Stopped often enough that no leg of it passes through grey, which a straight
# red to blue interpolation does at exactly the absorptions a soft furnishing
# has.
_ABSORPTION_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (170, 52, 44)),
    (0.2, (214, 140, 56)),
    (0.45, (222, 200, 74)),
    (0.7, (110, 180, 92)),
    (1.0, (56, 120, 200)),
)
UNKNOWN_COLOUR = (128, 128, 128)


def absorption_colour(alpha: float) -> list[int]:
    """Position an absorption coefficient on the reflective to absorbent ramp."""
    alpha = min(max(float(alpha), 0.0), 1.0)
    for (lo, low), (hi, high) in zip(_ABSORPTION_RAMP, _ABSORPTION_RAMP[1:], strict=False):
        if alpha <= hi:
            t = (alpha - lo) / (hi - lo)
            return [round(a + (b - a) * t) for a, b in zip(low, high, strict=True)]
    return list(_ABSORPTION_RAMP[-1][1])


def surface_groups(
    model: dict[str, Any],
    materials: Mapping[str, Sequence[float]] | None = None,
    geometry: bool = True,
) -> list[dict[str, Any]]:
    """The solver's own surface list, one entry per material group.

    ``model_json`` stores each group as flat triangle indices into its own
    point list, which is what PFFDTD reads. It is handed to the browser in that
    same shape so nothing is re-derived on the way.

    ``geometry`` may be turned off, and for the whole flat it must be. The
    triangles reach the browser as JSON, and this scene's exported model is
    192 MB of them: a page that hides the mesh behind a tiered grid would still
    make a reader download and parse a quarter of a gigabyte to see nothing
    drawn from it. Without it the group keeps its label, its colour and its
    triangle count, so the legend and the material table are unchanged.

    Its ``color`` field is not passed through. The exporter writes ``[128, 128,
    128]`` into every group because PFFDTD ignores it, so honouring it renders
    thirteen materials as one flat grey. The colour here is the material's own
    absorption at 1 kHz, which is the quantity the view exists to show.
    """
    band = MATERIAL_BANDS_HZ.index(COLOUR_BAND_HZ)
    materials = materials or {}
    groups = []
    for label, group in sorted(model.get("mats_hash", {}).items()):
        # Only counted when the geometry is off. Building the arrays to throw
        # them away is minutes on this flat: 3.2 M triangles across 51 groups,
        # each a Python list numpy has to walk element by element.
        count = len(group["tris"])
        points = np.asarray(group["pts"], dtype=float) if geometry else np.zeros((0, 3))
        triangles = np.asarray(group["tris"], dtype=int) if geometry else np.zeros((count, 3), int)
        coefficients = materials.get(label)
        alpha = float(coefficients[band]) if coefficients is not None else None
        groups.append(
            {
                "label": label,
                "positions": (
                    [round(float(value), 5) for value in points.ravel()] if geometry else []
                ),
                "indices": [int(value) for value in triangles.ravel()] if geometry else [],
                "colour": absorption_colour(alpha) if alpha is not None else list(UNKNOWN_COLOUR),
                "absorption": None if alpha is None else round(alpha, 3),
                "triangles": int(triangles.shape[0]) if triangles.ndim > 1 else 0,
                "sides": group.get("sides"),
            }
        )
    return groups


def model_materials(model_json: Path) -> dict[str, list[float]]:
    """The absorption table exported beside the model, or nothing if absent.

    The model itself carries only labels. The manifest written next to it by the
    export carries the coefficients those labels stand for, which is the only
    place the two are tied together.
    """
    manifest = Path(model_json).parent / "manifest.json"
    if not manifest.is_file():
        return {}
    materials = json.loads(manifest.read_text()).get("materials", {})
    return {str(k): [float(v) for v in values] for k, values in materials.items()}


@dataclass(frozen=True)
class RunView:
    """What was built, so the caller can report it rather than guess."""

    groups: int
    triangles: int
    samples: int
    audio_files: int
    path: Path

    def summary(self) -> str:
        return (
            f"{self.groups} surface groups, {self.triangles} triangles, "
            f"{self.samples} samples, {self.audio_files} audio files, at {self.path}"
        )


def _sample_rows(
    run_dir: Path, report: dict[str, Any], audio_names: set[str]
) -> list[dict[str, Any]]:
    """One row per source and receiver pair, with its plots and its audio."""
    from reverberate.response import read_raw

    rows: list[dict[str, Any]] = []
    for entry in report["sources"]:
        index = int(entry["source_index"])
        response = read_raw(run_dir / "responses" / f"source{index}.h5")
        rate = float(response.sample_rate_hz)
        # A little above the simulated band, so the low pass edge is visible
        # and the empty decades above it are not drawn.
        ceiling = float(response.provenance.fmax_hz) * 1.5
        measures = {int(row["receiver"]): row for row in entry["measures"]}
        for receiver in range(response.ir.shape[0]):
            signal = response.ir[receiver]
            name = f"source{index}_receiver{receiver}_wet.wav"
            rows.append(
                {
                    "id": f"s{index}r{receiver}",
                    "label": f"source {index} to receiver {receiver}",
                    "source_index": index,
                    "receiver_index": receiver,
                    "sample_rate_hz": rate,
                    "seconds": round(signal.size / rate, 4),
                    "peak": round(float(np.max(np.abs(signal))), 6),
                    "envelope": envelope(signal),
                    "decay": decay_curve_points(signal, rate),
                    "spectrogram": spectrogram(signal, rate, max_hz=ceiling),
                    "measures": measures.get(receiver),
                    "wet_audio": f"audio/{name}" if name in audio_names else None,
                }
            )
    return rows


def _room_geometry(report: dict[str, Any]) -> dict[str, Any] | None:
    """The room as the solver's boundary realised it, whichever key holds it.

    Two report shapes put different things under ``room``: a point run puts the
    geometry there, a spatial run puts the room's name and the geometry under
    ``room_geometry``. Choosing by truthiness handed the panel a string and it
    asked the string for a volume.
    """
    for key in ("room_geometry", "room"):
        value = report.get(key)
        if isinstance(value, dict):
            return value
    return None


def is_spatial(report: Mapping[str, Any]) -> bool:
    """Whether this run is an ambisonic one rather than a set of point receivers.

    The two produce different artefacts and neither is a special case of the
    other: a point run writes one response per receiver and one wet file per
    pair, a spatial run writes one ambisonic response about one listening point
    and one pair of ears per head and head orientation.
    """
    return "binaural_decodes" in report


def _spatial_rows(
    run_dir: Path, report: dict[str, Any], audio_names: set[str]
) -> list[dict[str, Any]]:
    """One row per head and head orientation, plus the ambisonic response itself.

    The plots come from the left ear of each binaural response, because that is
    the signal the file plays; the ambisonic row plots its own W channel, which
    is the pressure at the listening point.
    """
    import sofar

    rate = float(report["sample_rate_hz"])
    ceiling = float(report["encoder"].get("max_frequency_hz") or 16000.0) * 1.5
    rows: list[dict[str, Any]] = []

    ambisonic = run_dir / "responses" / "ambisonic.sofa"
    if ambisonic.is_file():
        signals = np.asarray(sofar.read_sofa(str(ambisonic)).Data_IR)[0]
        rows.append(
            {
                "id": "ambisonic",
                "label": "ambisonic, omnidirectional channel",
                "source_index": 0,
                "receiver_index": 0,
                "yaw_deg": 0.0,
                "sample_rate_hz": rate,
                "seconds": round(signals.shape[1] / rate, 4),
                "peak": round(float(np.max(np.abs(signals[0]))), 6),
                "envelope": envelope(signals[0]),
                "decay": decay_curve_points(signals[0], rate),
                "spectrogram": spectrogram(signals[0], rate, max_hz=ceiling),
                "measures": report.get("omnidirectional"),
                "wet_audio": (
                    "audio/ambisonic_acn_sn3d.wav"
                    if "ambisonic_acn_sn3d.wav" in audio_names
                    else None
                ),
                "note": "64 channels in ambiX; the plots are the W channel alone",
            }
        )

    for head, block in report["binaural_decodes"].items():
        path = run_dir / "responses" / f"binaural_{head}.sofa"
        if not path.is_file():
            continue
        sofa = sofar.read_sofa(str(path))
        responses = np.asarray(sofa.Data_IR)
        views = np.asarray(sofa.ListenerView, dtype=float)
        for index in range(responses.shape[0]):
            yaw = float(np.degrees(np.arctan2(views[index][1], views[index][0])))
            left = responses[index, 0]
            name = f"binaural_{head}_yaw{int(round(yaw))}.wav"
            rows.append(
                {
                    "id": f"{head}_yaw{int(round(yaw))}",
                    "label": f"{head.replace('_', ' ')}, head at {yaw:+.0f} degrees",
                    "source_index": 0,
                    "receiver_index": 0,
                    "yaw_deg": yaw,
                    "sample_rate_hz": rate,
                    "seconds": round(left.size / rate, 4),
                    "peak": round(float(np.max(np.abs(responses[index]))), 6),
                    "envelope": envelope(left),
                    "decay": decay_curve_points(left, rate),
                    "spectrogram": spectrogram(left, rate, max_hz=ceiling),
                    "measures": block["measures"].get(f"yaw_{int(round(yaw))}"),
                    "wet_audio": f"audio/{name}" if name in audio_names else None,
                    "note": block["decoder"].get("head"),
                }
            )
    return rows


@dataclass(frozen=True)
class RunRef:
    """Where a rendered run is, and which apartment it belongs to."""

    name: str
    scene_id: str
    room: str
    path: Path


def run_scene(run_dir: Path) -> RunRef:
    """Which apartment and room a run was simulated in.

    Read from ``plan.json`` rather than inferred from the mesh, because the plan
    is what chose the room; the mesh is only its consequence.
    """
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    return RunRef(
        name=run_dir.name,
        scene_id=str(plan["scene_id"]),
        room=str(plan["room"]),
        path=run_dir,
    )


#: Keys :func:`build_site` dereferences without a default **for a run of placed
#: points**. A report missing any of them cannot be drawn, so
#: :func:`discover_runs` refuses it there rather than letting the builder raise
#: halfway through the collection and take every other run down with it.
POINT_RUN_REPORT_KEYS = (
    "run",
    "scene_sha256",
    "cache_key",
    "room",
    "theory",
    "placement",
    "binaural_note",
    "dry_voice",
    "sources",
    "model_json",
)

#: The same list for a **spatial** run, which is a different shape and not a
#: shorter version of the one above. It has one listening point, an ambisonic
#: expansion about it and a pair of ears per head and orientation, so it holds
#: no ``placement``, no ``sources`` and no ``scene_sha256``, and inventing them
#: would put receivers on the page that the run never simulated. Checking it
#: against the point keys would drop a good run out of the viewer without a
#: word, which is the quiet version of the crash the guard exists to stop.
SPATIAL_RUN_REPORT_KEYS = (
    "run",
    "cache_key",
    "room",
    "theory",
    "model_json",
    "array",
    "encoder",
    "binaural_decodes",
    "sample_rate_hz",
)


def report_is_drawable(record: Mapping[str, Any]) -> bool:
    """Whether a report holds every key **its own shape** is read for.

    The shape is decided by :func:`is_spatial`, which reads what the report
    holds rather than what the run was called.
    """
    keys = SPATIAL_RUN_REPORT_KEYS if is_spatial(record) else POINT_RUN_REPORT_KEYS
    return all(key in record for key in keys)


def discover_runs(runs_root: Path) -> list[RunRef]:
    """Every rendered run under ``runs_root``, newest name last.

    A run counts as rendered only if it has the plan that names the scene, the
    report the payload is built from, **and** every field that report is read
    for, as listed in :data:`POINT_RUN_REPORT_KEYS` or, for the other shape,
    :data:`SPATIAL_RUN_REPORT_KEYS`. A half-finished run directory is skipped
    rather than offered and then failing to open.

    That last condition is not belt and braces. The runs directory is shared
    between sessions, and a report written for something other than a solve, a
    cost study or a domain census, has a plan and a report and none of the
    placement a page needs. Checking only that the two files exist let one such
    directory raise ``KeyError`` inside the builder's loop and take down the
    whole viewer, every other run with it.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    # Pointing at a single run directory is the easy mistake, and globbing one
    # level down would answer it with an empty selector and no explanation.
    if (runs_root / "plan.json").is_file():
        runs_root = runs_root.parent
    found: list[RunRef] = []
    for plan in sorted(runs_root.glob("*/plan.json")):
        report = plan.parent / "report.json"
        if not report.is_file():
            continue
        try:
            if not report_is_drawable(json.loads(report.read_text())):
                continue
            found.append(run_scene(plan.parent))
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    return found


def _voxel_cache_dir(
    key: str, recorded_root: object, run: str, store: ObjectStore | None
) -> Path | None:
    """Where this run's grid is, looking in the three places it can be.

    In order: this machine's own cache, the root the report happens to name,
    then ``store``. The order is not arbitrary -- the recorded root is the
    least trustworthy of the three, because it is an absolute path in whatever
    tree wrote the report, and W29's pointed into a worktree that no longer
    exists. The key is content addressed, so any copy found under it is the
    same grid.

    ``store`` is passed in rather than opened here, the way
    :func:`reverberate.wave.voxelise.voxelise` takes one: building a page must
    not reach the network because a caller forgot it could.
    """
    from reverberate.wave.voxelise import cache_root

    candidates = [cache_root() / key]
    if recorded_root:
        candidates.append(Path(str(recorded_root)) / key)
    for candidate in candidates:
        if (candidate / "vox_out.h5").is_file():
            return candidate

    from reverberate.wave.vox_store import fetch_entry

    if store is None:
        print(f"{run}: no grid at {candidates[0]} and no store to look further in")
        return None
    print(f"{run}: no grid on this machine, pulling {key} from the store")
    try:
        fetched = fetch_entry(store, key)
    except Exception as error:  # noqa: BLE001 - a store that will not answer is not a build failure
        print(f"{run}: the store could not deliver {key} ({error}); the page keeps the triangles")
        return None
    if fetched is None:
        print(f"{run}: the store holds no grid {key} either; the page keeps the triangles")
        return None
    return fetched.path


def _write_voxels(
    report: dict[str, Any], target: Path, store: ObjectStore | None = None
) -> dict[str, Any] | None:
    """Thin the run's voxelisation into the page, from wherever the grid is.

    Optional on purpose, and never silent about it. The cache is content
    addressed and sized in terabytes, so it is the first thing pruned; a run
    page that stopped building because a grid was collected would be a worse
    trade than a run page without the grid view.

    What is *not* an acceptable trade is falling back without saying so. The
    page then draws the exported triangles under a mode that promises the
    solver's own grid, and the two look enough alike that the substitution
    reads as a rendering choice rather than as a missing file.
    """
    key = report.get("cache_key")
    run = str(report.get("run") or "run")
    if not key:
        print(f"{run}: the report names no cache key, so the page keeps the triangles")
        return None
    cache_dir = _voxel_cache_dir(str(key), report.get("cache_root"), run, store)
    if cache_dir is None:
        return None
    from reverberate.viz.vox_view import TARGET_CUBES, read_surface, write_voxel_payload

    # The labels come from the voxelisation's own manifest, and nowhere else.
    # A node carries a material *index*, and the index is a position in the
    # list the voxeliser was handed: the room's thirteen, not the apartment's
    # fifty-two. Reading them from the model would name every node wrongly, and
    # reading them from the report -- which has no such key -- produced an
    # empty list that merely looked like a material list.
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    labels = sorted(manifest.get("materials") or {})
    # How finely this run asks to be drawn. The block size is picked to fit a
    # budget of blocks, and the budget -- not the grid -- is what decides what
    # a reader sees: this flat comes out at 16.34 mm blocks at 4, 8 *and*
    # 16 kHz under the default, because its surface area is the same at all
    # three. A grid published to be looked at rather than solved on can afford
    # more, and says so here rather than having it guessed.
    budget = int(report.get("viewer_cubes") or TARGET_CUBES)
    return write_voxel_payload(read_surface(cache_dir, budget), labels, target)


#: Where a run keeps a tiered audit payload, if it has one. Written by
#: :mod:`reverberate.experiments.audit_view`, one directory per room.
AUDIT_DIR = "voxels"


def _link_audit(
    run_dir: Path, target: Path, store: ObjectStore | None = None
) -> dict[str, Any] | None:
    """Expose a run's tiered audit payload, if it built one, without copying it.

    Linked rather than copied for the same reason the scenes are: the whole
    flat at 16 kHz is about 1.7 GB of quads across twelve rooms, and the site
    is a temporary directory that would otherwise hold a second copy of it for
    as long as the viewer runs.

    Pulled from the store when this checkout has none, which is the same order
    the grid follows above: local, then remote, then say so. A payload is
    derived and could be rebuilt instead, but rebuilding is thirteen minutes of
    one core and fetching is a download, so a reader who only wants to look
    should not have to compute.
    """
    source = run_dir / AUDIT_DIR
    index = source / "rooms.json"
    if not index.is_file() and store is not None:
        from reverberate.viz.payload_store import fetch_payload

        run = run_dir.name
        try:
            if fetch_payload(store, run, source) is not None:
                print(f"{run}: no audit payload on this machine, pulled it from the store")
        except Exception as error:  # noqa: BLE001 - a store that will not answer is not a failure
            print(f"{run}: the store could not deliver the audit payload ({error})")
    if not index.is_file():
        return None
    link = target / AUDIT_DIR
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(source.resolve(), target_is_directory=True)
    record: dict[str, Any] = json.loads(index.read_text())
    record["dir"] = AUDIT_DIR
    return record


def model_json_of(run_dir: Path, report: dict[str, Any]) -> Path:
    """Where the run's exported model actually is.

    Reports written before the path was resolved carry a relative one, which
    only opens from the directory it was written in. The viewer is started
    from wherever the reader happens to be -- a run configuration, a shell, a
    different checkout -- so it is resolved here against the places it could
    sensibly be, and the failure names every one of them rather than the last.
    """
    stored = Path(str(report["model_json"]))
    if stored.is_absolute() and stored.is_file():
        return stored
    tried = [stored]
    # The runs root, then its parent: a relative path in a report is relative
    # to the tree the run lives in, and "data/runs/x/models/y.json" is written
    # from the checkout root, two levels above the run directory.
    for root in (Path.cwd(), run_dir, run_dir.parent, run_dir.parent.parent.parent):
        candidate = root / stored
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{run_dir.name}: the exported model named in report.json is not at any "
        "of " + ", ".join(str(path) for path in tried)
    )


def build_site(run_dir: Path, target: Path, store: ObjectStore | None = None) -> RunView:
    """Write one run's payload and audio into ``target``.

    Reads only what the run already published: ``report.json`` for the room and
    the measures, ``responses/`` for the signals, ``audio/`` for the playback.
    It does not re-run the solver, re-measure, or reach the network, so the page
    is exactly the artefacts that were published and not a second opinion.
    """
    run_dir, target = Path(run_dir), Path(target)
    report = json.loads((run_dir / "report.json").read_text())
    model_json = model_json_of(run_dir, report)
    model = json.loads(model_json.read_text())

    target.mkdir(parents=True, exist_ok=True)
    audio_names: set[str] = set()
    source_audio = run_dir / "audio"
    if source_audio.is_dir():
        shutil.copytree(source_audio, target / "audio", dirs_exist_ok=True)
        audio_names = {path.name for path in source_audio.glob("*.wav")}

    audit = _link_audit(run_dir, target, store)
    groups = surface_groups(model, model_materials(model_json), geometry=audit is None)
    scene = run_scene(run_dir)
    spatial = is_spatial(report)
    if spatial:
        # A spatial run has one listening point rather than a placement, and its
        # geometry lives in the plan beside the report. Read from there rather
        # than reshaped into a placement it does not have: inventing one would
        # put six receivers on the page that the run never simulated.
        plan = json.loads((run_dir / "plan.json").read_text())
        centre = list(report["array"]["centre"])
        sources = [{"index": 0, "position": list(plan["source"]), "archetype": "source"}]
        receivers = [{"index": 0, "position": centre, "archetype": "listening point"}]
        receivers += [
            {"index": index + 1, "position": list(position)}
            for index, position in enumerate(plan.get("extra_receivers", []))
        ]
        samples = _spatial_rows(run_dir, report, audio_names)
    else:
        placement = report["placement"]
        sources = [
            {"index": index, "position": entry["position"], "archetype": entry.get("archetype")}
            for index, entry in enumerate(placement["sources"])
        ]
        receivers = [
            {"index": index, "position": entry["position"]}
            for index, entry in enumerate(placement["receivers"])
        ]
        samples = _sample_rows(run_dir, report, audio_names)

    payload = {
        "run": report["run"],
        "scene_id": scene.scene_id,
        "room_name": scene.room,
        "scene_sha256": report.get("scene_sha256") or report.get("geometry_sha256", ""),
        "cache_key": report["cache_key"],
        "spatial": spatial,
        # The ball the field was expanded about, so a reader can see what the
        # encoder actually looked at rather than a point that stands for it.
        "array": report.get("array"),
        "encoder": report.get("encoder"),
        "conditioning": report.get("conditioning"),
        "centre_identity": report.get("centre_identity"),
        "direction_of_arrival": report.get("direction_of_arrival"),
        "heads": report.get("heads"),
        "licence_conflict": report.get("licence_conflict"),
        "air_absorption": report.get("air_absorption"),
        # A spatial run's report uses "room" for the room's *name* and
        # "room_geometry" for the boundary the solver realised; a point run uses
        # "room" for the geometry itself. Picking by truthiness handed the
        # panel the string "bedroom.001" and it asked it for a volume.
        "room": _room_geometry(report),
        "theory": report["theory"],
        "theory_shell_only": report.get("theory_shell_only"),
        "theory_note": report.get("theory_note"),
        "measured_anomaly": report.get("measured_anomaly"),
        # What the solver sealed off. Drawn, not merely recorded: sealing stops
        # the simulation carrying sound through a region, and the whole reason
        # the census exists is that this must be visible rather than inferred.
        "sealed": report.get("sealed"),
        # A run with a tiered audit payload does not also build the single
        # one: they draw the same grid, and the tiered one is the only form a
        # whole flat at 16 kHz exists in.
        "audit": audit,
        "voxels": None if audit else _write_voxels(report, target, store),
        "band_note": report.get("band_note"),
        "low_cut_hz": report.get("low_cut_hz"),
        "binaural_note": report.get("binaural_note"),
        "omissions": report.get("omissions", []),
        "dry_voice": report.get("dry_voice"),
        "dry_audio": "audio/dry_voice.wav" if "dry_voice.wav" in audio_names else None,
        "groups": groups,
        "sources": sources,
        "receivers": receivers,
        "samples": samples,
    }
    (target / "run.json").write_text(json.dumps(payload) + "\n")
    return RunView(
        groups=len(groups),
        triangles=sum(int(group["triangles"]) for group in groups),
        samples=len(payload["samples"]),
        audio_files=len(audio_names),
        path=target,
    )
