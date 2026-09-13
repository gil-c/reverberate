"""From encoded bands to the field: one HDF5 per source, and ``walk.json``.

Per point the three bands are levelled, continued and assembled exactly as
``w38_ambisonic_bands`` assembles one point; the tail beyond the solved
window is synthesised from the source room's own absorption. The one thing
the field does that a single point never had to: a point without a low band
array of its own borrows its nearest neighbour's, levelled onto its own mid
band, and says so in ``/low_borrowed``.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.audio import Atmosphere
from reverberate.experiments.engine import write_record
from reverberate.experiments.run import entry_from_key
from reverberate.experiments.w40_volume_field.plan import bands_of_source, slug
from reverberate.spatial.bands import (
    BandSolve,
    assemble,
    continue_from,
    decay_from_bands,
    extend,
    extend_spectrum,
    level_gain,
)
from reverberate.spatial.encode import Ambisonic

__all__ = ["assemble_field", "level_borrowed_low"]


def level_borrowed_low(
    low: np.ndarray, mid: np.ndarray, rate_hz: float, low_fmax_hz: float, mid_fmax_hz: float
) -> np.ndarray:
    """A neighbour's low band put at the level this point's own would have.

    A borrowed low band sits at its own point's level, up to 5 dB from this
    point's (a wall, a doorway); the pair check refused one such point on
    hssd_0076. The two solves' source bandwidths predict the low/mid ratio,
    so the borrowed band is scaled until the ratio measured on the calibration
    octaves equals that prediction, and the check then passes by construction.
    """
    from reverberate import bands as band_split
    from reverberate.spatial.bands import calibration_bands_hz

    rate = int(round(rate_hz))
    n = min(low.shape[-1], mid.shape[-1])
    measured = band_split.level_ratio(
        np.asarray(low[0, :n], dtype=float),
        np.asarray(mid[0, :n], dtype=float),
        rate,
        calibration_hz=calibration_bands_hz(low_fmax_hz, rate),
    )
    return np.asarray(low * (measured / (low_fmax_hz / mid_fmax_hz)), dtype=low.dtype)


def _read_encoded(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    with h5py.File(path, "r") as handle:
        signals = np.asarray(handle["signals"][...])
        points = np.asarray(handle["point_index"][...])
        centres = np.asarray(handle["centres"][...])
        rate = float(handle.attrs["sample_rate_hz"])
        order = int(handle.attrs["order"])
    return signals, points, centres, rate, order


def _assemble_point(args: tuple[Any, ...]) -> tuple[int, np.ndarray, dict[str, Any]]:
    (
        index,
        low_sig,
        mid_sig,
        high_sig,
        centres,
        fmaxes,
        steps,
        rate,
        order,
        absorption,
        atmosphere,
        c,
        seed,
        ceiling_hz,
        scale,
    ) = args
    solves = {
        "low": BandSolve(
            "low",
            Ambisonic(low_sig.astype(float), rate, order, centres["low"]),
            fmaxes["low"],
            steps["low"],
        ),
        "mid": BandSolve(
            "mid",
            Ambisonic(mid_sig.astype(float), rate, order, centres["mid"]),
            fmaxes["mid"],
            steps["mid"],
        ),
    }
    has_high = high_sig is not None
    if has_high:
        solves["high"] = BandSolve(
            "high",
            Ambisonic(high_sig.astype(float), rate, order, centres["high"]),
            fmaxes["high"],
            steps["high"],
        )
    total_s = solves["low"].ambisonic.duration_s
    decay, decay_record = decay_from_bands(
        solves["low"],
        solves["mid"],
        mean_absorption=absorption,
        atmosphere=atmosphere,
        sound_speed_m_s=c,
    )
    record: dict[str, Any] = {"has_high": has_high, "decay_s": [round(float(v), 4) for v in decay]}
    if has_high:
        # The high band is solved in the room alone, with its doorways sealed;
        # the mid band on the whole storey lets energy leave through them. At
        # the far end of the merged living room, 6 m and more from the source,
        # the two disagree by 8 to 10 dB and the level check refuses the pair.
        # That is roadmap W23's doorway question showing up per point. Such a
        # point keeps its mid band and gets its top synthesised, and says so.
        try:
            gain_mid, _ = level_gain(solves["mid"], solves["high"])
            record["gain_mid"] = float(gain_mid)
        except ValueError as refusal:
            has_high = False
            record["has_high"] = False
            record["high_dropped"] = str(refusal)[:160]
    if has_high:
        solves["high"], _ = continue_from(
            solves["high"],
            solves["mid"],
            gain=gain_mid,
            t60_s=decay,
            atmosphere=atmosphere,
            sound_speed_m_s=c,
        )
        for k, name in enumerate(("mid", "high")):
            solves[name], _ = extend(
                solves[name],
                total_s,
                calibration=solves["mid"],
                mean_absorption=absorption,
                atmosphere=atmosphere,
                sound_speed_m_s=c,
                seed=seed + 1000 * k + index,
                t60_s=decay,
            )
        assembled, _ = assemble(solves["low"], solves["mid"], solves["high"])
        solved_to = fmaxes["high"]
    else:
        solves["mid"], _ = extend(
            solves["mid"],
            total_s,
            calibration=solves["mid"],
            mean_absorption=absorption,
            atmosphere=atmosphere,
            sound_speed_m_s=c,
            seed=seed + index,
            t60_s=decay,
        )
        # No high band here: the mid band carries the top, and the seam sits
        # where the mid band's own edge would be crossed over in a three band
        # run, so the synthesised octaves start from the same place.
        top = BandSolve("top", solves["mid"].ambisonic, fmaxes["mid"] * 1.0001, steps["mid"])
        assembled, _ = assemble(
            solves["low"],
            solves["mid"],
            top,
            crossovers_hz=(0.8 * fmaxes["low"], 0.8 * fmaxes["mid"]),
        )
        solved_to = 0.8 * fmaxes["mid"]
    assembled, extension = extend_spectrum(
        assembled,
        fmax_hz=solved_to,
        t60_s=decay,
        atmosphere=atmosphere,
        sound_speed_m_s=c,
        ceiling_hz=ceiling_hz,
    )
    record["solved_to_hz"] = solved_to
    record["extension_db"] = extension.get("energy_added_db")
    record["scale"] = float(scale)
    samples = int(round(total_s * rate))
    return index, (scale * assembled.signals[:, :samples]).astype(np.float32), record


def assemble_field(
    out: Path,
    *,
    sources: list[str] | None = None,
    workers: int = 4,
    ceiling_hz: float | None = None,
    seed: int = 20260910,
    meshes: dict[str, str] | None = None,
) -> list[Path]:
    """One HDF5 per source: ``/ir[point, channel, sample]`` at 48 kHz, order 7.

    The format is ``docs/formats/ambisonic-field.md``. ``walk.json`` beside
    the fields is the viewer's index; ``meshes`` names viewer meshes when the
    caller has some.
    """
    from reverberate.experiments.w20_render import room_geometry
    from reverberate.metrics import band_centres
    from reverberate.tail import mean_absorption_of

    plan = json.loads((out / "plan.json").read_text())
    atmosphere = Atmosphere()
    c = float(plan["sound_speed_m_s"])
    points = np.asarray(plan["points"], dtype=float)
    written = []
    for source in plan["sources"]:
        if sources is not None and source["name"] not in sources:
            continue
        encoded: dict[str, Any] = {}
        for band in bands_of_source(plan, source):
            path = out / "encoded" / f"{source['name']}__{slug(band)}.h5"
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing; solve first")
            encoded[band] = _read_encoded(path)
        rate = encoded["low"][3]
        order = encoded["low"][4]
        high_band = next((b for b in encoded if b.startswith("high")), None)
        fmaxes = {"low": plan["bands"]["low"]["fmax_hz"], "mid": plan["bands"]["mid"]["fmax_hz"]}
        steps = {
            "low": plan["bands"]["low"]["grid_step_m"],
            "mid": plan["bands"]["mid"]["grid_step_m"],
        }
        if high_band:
            fmaxes["high"] = plan["bands"][high_band]["fmax_hz"]
            steps["high"] = plan["bands"][high_band]["grid_step_m"]
        # Absorption for the tail from the source room's own boundary, the
        # solver's, rather than an assumption.
        absorption_note = "assumed flat 0.2"
        bands_count = len(band_centres(int(round(rate))))
        absorption = np.full(bands_count, 0.2)
        if high_band:
            entry = entry_from_key(plan["bands"][high_band]["cache_key"])
            model = Path(str(entry.manifest.get("model_json", "")))
            if model.is_file() and (entry.path / "vox_out.h5").is_file():
                geometry = room_geometry(entry.path, model, sound_speed_m_s=343.0)
                absorption = mean_absorption_of({"room": geometry.record()}, bands_count)
                absorption_note = "the solver's own boundary of the source room"
        rows_by_band = {
            band: {int(p): k for k, p in enumerate(encoded[band][1])} for band in encoded
        }
        # The low band is encoded on a lattice twice as coarse; every point
        # takes the nearest encoded low band point, itself when it has one.
        low_indices = np.asarray(sorted(rows_by_band["low"]), dtype=int)
        low_positions = points[low_indices]
        common = sorted(rows_by_band["mid"])
        tasks: list[tuple[Any, ...]] = []
        borrowed_low: set[int] = set()
        for index in common:
            nearest = int(
                low_indices[np.argmin(np.linalg.norm(low_positions - points[index], axis=1))]
            )
            low_sig = encoded["low"][0][rows_by_band["low"][nearest]]
            mid_sig = encoded["mid"][0][rows_by_band["mid"][index]]
            if nearest != index:
                low_sig = level_borrowed_low(low_sig, mid_sig, rate, fmaxes["low"], fmaxes["mid"])
                borrowed_low.add(index)
            high_sig = None
            centres = {
                "low": encoded["low"][2][rows_by_band["low"][nearest]],
                "mid": encoded["mid"][2][rows_by_band["mid"][index]],
            }
            if high_band and index in rows_by_band[high_band]:
                high_sig = encoded[high_band][0][rows_by_band[high_band][index]]
                centres["high"] = encoded[high_band][2][rows_by_band[high_band][index]]
            tasks.append(
                (
                    index,
                    low_sig,
                    mid_sig,
                    high_sig,
                    centres,
                    fmaxes,
                    steps,
                    rate,
                    order,
                    absorption,
                    atmosphere,
                    c,
                    seed,
                    ceiling_hz,
                )
            )
        if borrowed_low:
            print(
                f"{source['name']}: {len(borrowed_low)} points borrow a neighbour's low band,"
                " levelled to their own mid band",
                flush=True,
            )
        # Two passes. The three band assembly takes the high band as its
        # reference scale and levels the mid band on it; a point without a
        # high band would otherwise sit at the mid band's own scale, 3.8 dB
        # louder on W39's point, and a listener crossing the room's boundary
        # would hear a step. The points with a high band go first, and the
        # median of their mid gain is applied to the rest.
        with_high = [t + (1.0,) for t in tasks if t[3] is not None]
        without = [t for t in tasks if t[3] is None]
        samples = None
        results: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, signals, record in pool.map(_assemble_point, with_high, chunksize=4):
                results[index] = (signals, record)
                samples = signals.shape[1]
                if len(results) % 25 == 0:
                    print(f"  {source['name']}: {len(results)}/{len(tasks)} points assembled")
            gains = [r["gain_mid"] for _, r in results.values() if "gain_mid" in r]
            scale = float(np.median(gains)) if gains else 1.0
            print(
                f"  {source['name']}: mid on high gain, median over {len(gains)} points:"
                f" {20 * np.log10(scale):.2f} dB"
            )
            without = [t + (scale,) for t in without]
            for index, signals, record in pool.map(_assemble_point, without, chunksize=4):
                results[index] = (signals, record)
                samples = signals.shape[1]
                if len(results) % 25 == 0:
                    print(f"  {source['name']}: {len(results)}/{len(tasks)} points assembled")
        assert samples is not None
        samples = min(signals.shape[1] for signals, _ in results.values())
        # Exactly the low band's duration at the delivery rate, 57 600 samples for
        # 1.2 s: resampling from the grid rate leaves a few samples over.
        samples = min(samples, int(round(plan["bands"]["low"]["duration_s"] * rate)))
        indices = sorted(results)
        path = out / "field" / f"{source['name']}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            ir = handle.create_dataset(
                "ir",
                shape=(len(indices), (order + 1) ** 2, samples),
                dtype=np.float32,
                chunks=(1, (order + 1) ** 2, samples),
            )
            for row, index in enumerate(indices):
                ir[row] = results[index][0][:, :samples]
            handle.create_dataset("positions", data=points[indices])
            handle.create_dataset("point_index", data=np.asarray(indices))
            origin = points.min(axis=0)
            cells = np.rint(
                (points[indices] - origin) / np.array([plan["pitch_m"], 1.0, plan["pitch_m"]])
            ).astype(np.int32)
            cells[:, 1] = 0
            handle.create_dataset("cell_index", data=cells)
            handle.attrs["grid_origin_m"] = origin
            handle.attrs["grid_step_m"] = np.array([plan["pitch_m"], 0.0, plan["pitch_m"]])
            handle.attrs["grid_shape"] = np.array(
                [int(cells[:, 0].max()) + 1, 1, int(cells[:, 2].max()) + 1]
            )
            handle.create_dataset(
                "rooms", data=np.asarray([plan["rooms"][i] for i in indices], dtype="S")
            )
            handle.create_dataset(
                "has_high", data=np.asarray([results[i][1]["has_high"] for i in indices])
            )
            handle.create_dataset(
                "solved_to_hz", data=np.asarray([results[i][1]["solved_to_hz"] for i in indices])
            )
            handle.create_dataset(
                "high_dropped",
                data=np.asarray([bool(results[i][1].get("high_dropped")) for i in indices]),
            )
            handle.attrs["high_dropped_count"] = int(
                sum(bool(results[i][1].get("high_dropped")) for i in indices)
            )
            handle.create_dataset(
                "low_borrowed", data=np.asarray([i in borrowed_low for i in indices])
            )
            handle.attrs["low_borrowed_count"] = len(borrowed_low)
            handle.attrs["mid_on_high_gain"] = scale
            handle.create_dataset(
                "direct_path_m",
                data=np.linalg.norm(
                    points[indices] - np.asarray(source["position"], dtype=float), axis=1
                ),
            )
            handle.attrs["sample_rate_hz"] = rate
            handle.attrs["order"] = order
            handle.attrs["ordering"] = "ACN"
            handle.attrs["normalisation"] = "N3D"
            handle.attrs["axes"] = (
                "x front, y left, z up; scene coordinates are y-up (x, height, z)"
            )
            handle.attrs["source_position"] = np.asarray(source["position"], dtype=float)
            handle.attrs["source_id"] = source["name"]
            handle.attrs["source_name"] = source.get("label", source["name"])
            handle.attrs["directivity"] = "omni"
            handle.attrs["gain"] = 1.0
            handle.attrs["dwelling"] = plan.get("dwelling", "")
            handle.attrs["source_room"] = source["room"]
            handle.attrs["scene_id"] = plan["scene_id"]
            handle.attrs["absorption_for_tail"] = absorption_note
            handle.attrs["provenance_json"] = json.dumps(
                {
                    "run": plan["run"],
                    "trick": plan["trick"],
                    "bands": {b: plan["bands"][b]["cache_key"] for b in encoded},
                }
            )
        print(
            f"{source['name']}: {len(indices)} points -> {path}"
            f" ({path.stat().st_size / 1e9:.2f} GB)"
        )
        written.append(path)
    walk = {
        "dwelling": plan.get("dwelling", ""),
        "scene_id": plan["scene_id"],
        "height_m": plan["height_m"],
        "pitch_m": plan["pitch_m"],
        "sources": [
            {
                "id": s["name"],
                "name": s.get("label", s["name"]),
                "room": s["room"],
                "position": s["position"],
                "directivity": "omni",
                "field": f"field/{s['name']}.h5",
            }
            for s in plan["sources"]
            if (out / "field" / f"{s['name']}.h5").is_file()
        ],
        # The grids the field was solved on, by band, and the viewer payloads
        # the campaign built from them, with the scene they belong to: the
        # app refuses a mesh of another scene, and the first campaign's
        # hard-coded paths would have drawn hssd_0002's walls around hssd_0076.
        "voxel_cache_keys": {b: plan["bands"][b]["cache_key"] for b in plan["bands"]},
        "meshes": meshes or {},
        "meshes_scene_id": plan["scene_id"] if meshes else None,
    }
    write_record(out, "walk.json", walk)
    return written
