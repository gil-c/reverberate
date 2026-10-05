"""The low band pairs of a dwelling: many source positions, few receivers, one machine.

A field (ADR 0015) is one source heard at every point of a storey. A scene
(ADR 0016) is the opposite: two thousand source positions, stations and rail
samples every 8 cm, each heard at the few listening cells the listener is
near while that source sounds. This module is the campaign for it:

- **the bundle** (:func:`prepare_pairs_bundle`) holds the storey's export, the
  source positions, the cells and which cells each position is heard at,
  with the one grid's cache key. It is a ``campaign.json`` like a field's,
  so :mod:`reverberate.gpu.onebox` sizes, rents, launches, watches and
  fetches it unchanged;
- **the campaign** (:class:`PairsCampaign`) voxelises that one grid, stands an
  order 7 array at every cell, and for each source position solves once,
  encodes the cells it is heard at on the card, and writes each response in
  the cache form of :mod:`reverberate.spatial.lowband`: 4 kHz, 4800 samples,
  1.23 MB. No mid or high grid, no mirror, no field;
- **the cache** (:class:`PairCache`) is addressed by
  :func:`reverberate.spatial.lowband.pair_key`. A pair already there is not
  solved again, a source position whose pairs are all there is not solved
  at all, and the trace stage reads it by key.

**Which side is solved.** A source is a point; a listening cell is an array
of about a thousand nodes whose 64 channels are a fit. By reciprocity one
could solve from the cells instead, but a cell's channel is a filtered sum
over its nodes, so that is one solve per channel: ``64 C`` solves for ``C``
cells against ``S`` for ``S`` source positions. The first recipe has 1636
audible positions and 895 cells along the listener's path. The source side
is solved.

**What it costs**, from the low band of the line campaign of 2026-10-03
(``docs/open-questions/low-band-pairs-cost.md``): the stencil is a fixed
price per source position and everything else is per cell, so
:func:`estimate` is ``S`` times the first plus the pairs times the second.
Three things of a field's campaign were wrong for this use and are not done
here: the encoder is prepared once for the campaign and not once per solve
(104 s each), the encoding is at 4 kHz and not at 48 kHz, and the air and
the masks are left to the trace, which knows the recipe.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.accel import dsp
from reverberate.accel.campaign import Campaign
from reverberate.accel.encode import BandEncoder, encode_point, geometry_key, prepare_band
from reverberate.accel.solve import host_memory_gb, output_sample_bytes, solve_slices
from reverberate.compute import to_numpy
from reverberate.spatial.array import ArrayDesign
from reverberate.spatial.encode import EncoderSettings
from reverberate.spatial.lowband import LOW_DURATION_S, LOW_RATE_HZ, pair_key, solve_fmax_hz

__all__ = [
    "CACHE_LEVERS",
    "COMPACT_PAIR_BYTES",
    "KIND",
    "REFERENCE_FMAX_HZ",
    "SOLVER",
    "PairCache",
    "PairEncoder",
    "PairsCampaign",
    "cost_record",
    "encoder_record",
    "estimate",
    "install_pairs",
    "prepare_pairs_bundle",
    "run_pairs",
    "source_positions",
]

#: What a bundle's ``campaign.json`` says it is, and the command line reads.
KIND = "low-band-pairs"

#: The engine's name in a pair's key and in a pack's provenance. A change of
#: anything between the grid and the cache form changes it.
SOLVER = "pffdtd-cuda+reverberate.accel.pairs/1"

#: A solve's level goes as its ``fmax`` (:mod:`reverberate.bands`), and a
#: field is on the scale of its 8 kHz solve, which is the scale the mirror is
#: aligned to. A pair is put on it by ``fmax / 8000``: the predicted ratio,
#: which two fields measured to 0.07 dB (0.5039 and 0.5016 for 4 kHz on 8).
REFERENCE_FMAX_HZ = 8000.0

#: The array stood at a cell, as a field's: its outer radius is this or twelve
#: grid steps, whichever is larger (0.26 m on the grid to 1500 Hz, 0.39 m on
#: the grid to 1 kHz), and its ball must be free air. A cell whose ball is
#: not is tried a little to the side, and the centre it got is recorded.
OUTER_RADIUS_M = 0.16
SHIFTS_M = (0.05, 0.10)
LOWCUT_HZ, LOWCUT_ORDER = 40.0, 8

#: Measured on 2 x A100 80 GB, 1.74 USD/h, instance 54075335, 2026-10-03, by
#: fitting the three solves of the line campaign: grid node updates a second,
#: seconds per receiver node and step in the engine's copy and write, seconds
#: per receiver node in ``write_comms``, seconds per fitted bin and cell in
#: the encoder, and the encoder's preparation for the 2400 bins of that band.
UPDATES_PER_S = 1.354e11
OUTPUT_S_PER_SAMPLE = 13.25e-9
COMMS_S_PER_NODE = 20.3e-6
ENCODE_S_PER_BIN = 0.267 / 2400.0
PREPARE_S = 104.4
NODES_PER_CELL = 984
#: The low grid of hssd_0076 at 1 kHz, from which another ``fmax`` is scaled.
GRID_NODES_AT_1K = 19_520_600
STEPS_AT_1K = 21_846


def encoder_record(order: int, fit_order: int, fmax_hz: float) -> dict[str, Any]:
    """What of the encoding enters a pair's key."""
    return {
        "order": int(order),
        "fit_order": int(fit_order),
        "fmax_hz": float(fmax_hz),
        "lowcut_hz": LOWCUT_HZ,
        "lowcut_order": LOWCUT_ORDER,
        "outer_radius_m": OUTER_RADIUS_M,
        "shifts_m": list(SHIFTS_M),
        "rate_hz": LOW_RATE_HZ,
        "reference_fmax_hz": REFERENCE_FMAX_HZ,
    }


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


#: The compact form a cache may keep its pairs in (:mod:`reverberate.render.compact`):
#: every bin of the transform, in 16 bits with a scale a channel and 100 Hz. **Every
#: bin**, up to the cache form's own 2 kHz, and not up to the solve's 1500 Hz: a cached
#: pair holds 45 dB under its energy above 1500 Hz in the median and 32 dB in the worst
#: of forty pairs of the first scene, and the pack's response is cut from it in time
#: (the onset's window) before its masks, so what is dropped there comes back under
#: 1414 Hz: -46 dB in the worst third octave of the stored response with the bins cut at
#: 1500 Hz, -83 dB with all of them, on sixty pairs, and no onset moved (2026-10-05,
#: ``docs/open-questions/low-band-compact.md``).
CACHE_LEVERS = "bins,int16"
#: A pair's file in that form, against 1 228 928 as samples: 2400 bins of 64 channels,
#: two numbers of 16 bits each, and the scales.
COMPACT_PAIR_BYTES = 621_446
_PLAIN, _COMPACT = ".npy", ".npz"


@dataclass
class PairCache:
    """Responses by pair key under one directory, a grid's key at a time.

    ``<root>/<voxel_low_key>/<kk>/<key>.npy`` is one response,
    ``[channel, 4800]`` float32 in the cache form, and
    ``<root>/<voxel_low_key>/index.jsonl`` one line a pair: its key, the two
    positions, the centre the array really stood at, and how it was made.

    **A pair may be kept compact**: ``<key>.npz`` in place of ``<key>.npy``,
    the bins of its transform as :func:`reverberate.render.compact.encode`
    keeps them (:data:`CACHE_LEVERS`), half the bytes. ``levers`` is the
    form this cache *writes*; it *reads* either, pair by pair, so a cache
    may hold both and a pair is the same pair in both: its key does not
    say the form, and the compact one is within -80 dB of the other where
    a pack reads it.
    """

    root: Path
    voxel_low_key: str
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: The levers new pairs are written with, as text; ``None`` writes the samples.
    levers: str | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.levers is not None:
            self.levers = _cache_levers(self.levers)

    @classmethod
    def local(cls, voxel_low_key: str) -> PairCache:
        """This machine's cache, ``<data root>/cache/low-pairs``."""
        from reverberate.settings import data_root

        return cls(data_root() / "cache" / "low-pairs", voxel_low_key)

    @property
    def directory(self) -> Path:
        return self.root / self.voxel_low_key

    def path(self, key: str) -> Path:
        """The pair's file: the one that is there, else where this cache would write it."""
        stem = self.directory / key[:2] / key
        compact, plain = stem.with_suffix(_COMPACT), stem.with_suffix(_PLAIN)
        if compact.is_file():
            return compact
        if plain.is_file():
            return plain
        return compact if self.levers else plain

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def read(self, key: str) -> np.ndarray:
        """One response, ``[channel, sample]`` float32; a missing pair is a ``KeyError``."""
        path = self.path(key)
        if not path.is_file():
            raise KeyError(f"no pair {key} under {self.directory}")
        if path.suffix == _COMPACT:
            return _read_compact(path)
        return np.asarray(np.load(path))

    def first_channel(self, key: str) -> np.ndarray:
        """Channel 0 of a pair, ``[1, sample]``, without the 63 others being read or made.

        Of the samples the file is mapped and 19 kB of it read; of the
        compact form the first degree's bins alone are transformed.
        """
        path = self.path(key)
        if not path.is_file():
            raise KeyError(f"no pair {key} under {self.directory}")
        if path.suffix == _COMPACT:
            return _read_compact(path, first_only=True)
        return np.array(np.load(path, mmap_mode="r")[:1])

    def write(self, key: str, response: np.ndarray, record: dict[str, Any]) -> Path:
        """The response under its key, and its line in the index; the file appears whole."""
        stem = self.directory / key[:2] / key
        stem.parent.mkdir(parents=True, exist_ok=True)
        if self.levers:
            path = stem.with_suffix(_COMPACT)
            partial = stem.with_suffix(".partial" + _COMPACT)
            _write_compact(partial, np.asarray(response, dtype=np.float32), self.levers)
        else:
            path = stem.with_suffix(_PLAIN)
            partial = stem.with_suffix(".partial" + _PLAIN)
            np.save(partial, np.asarray(response, dtype=np.float32))
        partial.replace(path)
        self._index(key, record)
        return path

    def adopt(self, key: str, source: Path, record: dict[str, Any]) -> Path:
        """A pair's file from another cache, in the form it is in; its line in the index."""
        source = Path(source)
        path = (self.directory / key[:2] / key).with_suffix(source.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial" + source.suffix)
        shutil.copyfile(source, partial)
        partial.replace(path)
        self._index(key, record)
        return path

    def _index(self, key: str, record: dict[str, Any]) -> None:
        with self._lock, (self.directory / "index.jsonl").open("a") as handle:
            handle.write(json.dumps({"key": key, **record}, sort_keys=True) + "\n")

    def records(self) -> dict[str, dict[str, Any]]:
        """The index, by key; a pair written twice keeps its last line."""
        index = self.directory / "index.jsonl"
        found: dict[str, dict[str, Any]] = {}
        if index.is_file():
            for line in index.read_text().splitlines():
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # An index copied while its campaign wrote it ends in half a line.
                        continue
                    found[str(record["key"])] = record
        return found

    def publish(self, store: Any) -> list[str]:
        """Every pair the store lacks, then the index; returns the keys sent."""
        sent = []
        for key in sorted(self.records()):
            if not self.has(key):
                continue
            # Under the name of the form it is kept in; a pair the store has in the other
            # form is the same pair and is not sent again.
            names = [f"{remote_prefix(self.voxel_low_key)}{key}{suffix}" for suffix in _FORMS]
            if not any(store.exists(name) for name in names):
                path = self.path(key)
                store.put_file(f"{remote_prefix(self.voxel_low_key)}{key}{path.suffix}", path)
                sent.append(key)
        index = self.directory / "index.jsonl"
        if index.is_file():
            store.put_file(f"{remote_prefix(self.voxel_low_key)}index.jsonl", index)
        return sent

    def fetch(self, store: Any, key: str) -> bool:
        """One pair from the store into this cache; whether the store had it."""
        for suffix in _FORMS:
            remote = f"{remote_prefix(self.voxel_low_key)}{key}{suffix}"
            if store.exists(remote):
                target = (self.directory / key[:2] / key).with_suffix(suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                store.get_file(remote, target)
                return True
        return False


_FORMS = (_COMPACT, _PLAIN)


def _cache_levers(text: str) -> str | None:
    """The levers a cache may be written with, as text; ``None`` for none.

    The bins, in 32 bits or in 16. A degree let go or a response cut where
    it has decayed is a choice of one pack's listening, and a cache serves
    every pack of its dwelling.
    """
    from reverberate.render.compact import Levers

    levers = Levers.parse(text)
    if levers.off:
        return None
    if levers.degree_db or levers.decay_db:
        raise ValueError(
            "a pair cache keeps every degree whole: the degree and the decay are a pack's levers"
        )
    return "bins,int16" if levers.sample == "int16" else "bins"


def _write_compact(path: Path, response: np.ndarray, levers: str) -> None:
    from reverberate.render.compact import FORMAT, Levers, encode

    made = encode(
        response,
        Levers.parse(levers),
        rate_hz=LOW_RATE_HZ,
        top_hz=LOW_RATE_HZ / 2.0,
        sound_speed_m_s=343.0,
    )
    meta = {
        "format": FORMAT,
        "levers": levers,
        "rate_hz": LOW_RATE_HZ,
        "top_hz": LOW_RATE_HZ / 2.0,
        "samples": int(response.shape[-1]),
    }
    with path.open("wb") as handle:
        np.savez(
            handle,
            samples=made.samples,
            scale=made.scale,
            data=made.data,
            meta=np.array(json.dumps(meta, sort_keys=True)),
        )


def _read_compact(path: Path, *, first_only: bool = False) -> np.ndarray:
    """A compact pair as ``[channel, sample]`` float32; of ``first_only``, channel 0 alone."""
    from reverberate.render.compact import FORMAT, Encoded, _bins, decode

    with np.load(path) as held:
        meta = json.loads(str(held["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path.name} is {meta.get('format')!r}; this reader knows {FORMAT!r}")
        lengths, scale, data = held["samples"], held["scale"], held["data"]
    rate, top = float(meta["rate_hz"]), float(meta["top_hz"])
    if first_only:
        first, stop = _bins(int(lengths[0]), 0.0, top, rate)
        lengths, scale, data = lengths[:1], scale[:1], data[: (stop - first) * 2]
    return decode(
        Encoded(samples=lengths, scale=scale, data=data),
        first_hz=np.zeros(int(lengths.shape[0])),
        top_hz=top,
        rate_hz=rate,
        samples=int(meta["samples"]),
    )


def remote_prefix(voxel_low_key: str) -> str:
    """Where a grid's pairs live in the store, relative to the project prefix."""
    return f"low-pairs/{voxel_low_key}/"


def install_pairs(pulled: Path, *, publish: bool = False) -> dict[str, Any]:
    """The ``pairs`` directory a campaign brought home, into this machine's cache.

    ``pulled`` holds one directory per grid key, as :class:`PairCache` lays
    it out. Pairs already here are left alone; with ``publish`` the cache is
    then sent to the shared store.
    """
    installed: list[str] = []
    published: list[str] = []
    damaged: list[str] = []
    for source in sorted(p for p in Path(pulled).iterdir() if p.is_dir()):
        arrived = PairCache(Path(pulled), source.name)
        cache = PairCache.local(source.name)
        for key, record in arrived.records().items():
            if arrived.has(key) and not cache.has(key):
                # A transfer that was cut leaves part of its last file under the file's
                # name: a response that does not read whole is not a pair.
                try:
                    response = arrived.read(key)
                except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile):
                    damaged.append(key)
                    continue
                if response.ndim != 2 or not np.all(np.isfinite(response)):
                    damaged.append(key)
                    continue
                # In the form it came in: a compact pair is not grown back on the laptop.
                cache.adopt(key, arrived.path(key), {k: v for k, v in record.items() if k != "key"})
                installed.append(key)
        if publish:
            from reverberate.store import shared_store

            store = shared_store()
            if store is not None:
                published.extend(cache.publish(store))
    return {"installed": installed, "published": published, "damaged": damaged}


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------


def source_positions(recipe: Any, *, audible_only: bool = True) -> dict[str, Any]:
    """Every position the low band is solved from for a recipe, in whole millimetres.

    ``recipe`` is a :class:`reverberate.scenes.Recipe`, or its JSON tree. The
    positions are those of :func:`reverberate.scenes.low_band_source_positions`,
    the recipe's own kinematics: the stations where a source dwells, the
    samples of the rails it travels and of the vertical rail of a seat it
    rises on. **Only the positions a source is audible at** are given, the
    two solved samples either side of it at every audible step: a wave solve
    is what a campaign pays for, and a rail walked in silence needs none.
    With ``audible_only`` false every position the sources pass through is
    given, which fills the dwelling's cache for the rails whole.

    Returns ``positions`` as ``[position, 3]`` in metres, each once, and
    ``by_source``: the rows each source reads.
    """
    from reverberate.scenes import Recipe, low_band_source_positions

    held = recipe if isinstance(recipe, Recipe) else Recipe.from_dict(recipe)
    found = low_band_source_positions(held, audible_only=audible_only)
    return {
        "positions": found.positions,
        "by_source": {name: list(rows) for name, rows in found.by_source.items()},
    }


def estimate(
    sources: int,
    pairs: int,
    *,
    fmax_hz: float,
    duration_s: float = LOW_DURATION_S,
    rate_usd_per_hour: float = 1.74,
    grid_nodes: float | None = None,
) -> dict[str, Any]:
    """Seconds and USD of a campaign of ``sources`` solves and ``pairs`` responses.

    The measured terms are the constants of this module, taken on 2 x A100
    at 1.74 USD/h; another card needs its own. ``grid_nodes`` defaults to
    the storey of hssd_0076 scaled as the cube of ``fmax_hz``.
    """
    scale = fmax_hz / 1000.0
    nodes = float(grid_nodes) if grid_nodes is not None else GRID_NODES_AT_1K * scale**3
    steps = STEPS_AT_1K * scale * duration_s / LOW_DURATION_S
    stencil = nodes * steps / UPDATES_PER_S
    bins = 2.0 * duration_s * fmax_hz
    per_cell = {
        "engine_output_s": NODES_PER_CELL * steps * OUTPUT_S_PER_SAMPLE,
        "comms_s": NODES_PER_CELL * COMMS_S_PER_NODE,
        "encode_s": bins * ENCODE_S_PER_BIN,
    }
    cell_s = float(sum(per_cell.values()))
    prepare = PREPARE_S * bins / 2400.0
    seconds = prepare + sources * stencil + pairs * cell_s
    return {
        "fmax_hz": fmax_hz,
        "grid_nodes": nodes,
        "steps": steps,
        "vram_gb": nodes * 9.027 / 1e9 + 2.13,
        "stencil_s_per_source": round(stencil, 2),
        "per_cell_s": {k: round(v, 3) for k, v in per_cell.items()},
        "cell_s": round(cell_s, 3),
        "prepare_s": round(prepare, 1),
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600.0, 3),
        "billed_rate_usd_per_hour": rate_usd_per_hour,
        "usd": round(seconds / 3600.0 * rate_usd_per_hour, 2),
        "usd_per_source": round(stencil / 3600.0 * rate_usd_per_hour, 4),
        "usd_per_pair": round(cell_s / 3600.0 * rate_usd_per_hour, 5),
        "cache_gb": round(pairs * 64 * duration_s * LOW_RATE_HZ * 4 / 1e9, 2),
    }


def prepare_pairs_bundle(
    bundle: Path,
    *,
    scene_id: str,
    sources: np.ndarray,
    cells: np.ndarray,
    heard_at: list[list[int]] | None = None,
    models_from: Path | None = None,
    hssd_root: Path | None = None,
    fmax_hz: float | None = None,
    duration_s: float = LOW_DURATION_S,
    order: int = 7,
    fit_order: int = 10,
    ppw: float = 10.5,
    ram_gb: float = 120.0,
) -> dict[str, Any]:
    """The export, the positions, the cells and the pairs into ``bundle``; ``campaign.json``.

    ``sources`` is ``[position, 3]`` (:func:`source_positions` of a recipe,
    or any list) and ``cells`` ``[cell, 3]``, both in the scene's frame.
    ``heard_at[s]`` lists the cells source position ``s`` is heard at; left
    out, every position is heard at every cell. ``fmax_hz`` defaults to
    :func:`reverberate.spatial.lowband.solve_fmax_hz`, 1500 Hz: the grid a
    crossover at 1 kHz needs when the wave solver carries its low side alone.

    The export is taken from ``models_from`` (an earlier export's directory,
    as :func:`reverberate.accel.bundle.prepare_bundle` takes it) or made from
    the HSSD download at ``hssd_root``.
    """
    from reverberate.accel.bundle import STOREY_SCENE, export_models
    from reverberate.experiments.run import scene_spec

    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    fmax = float(fmax_hz) if fmax_hz is not None else solve_fmax_hz()
    sources = np.asarray(sources, dtype=float).reshape(-1, 3)
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    if heard_at is None:
        heard_at = [list(range(cells.shape[0]))] * sources.shape[0]
    if len(heard_at) != sources.shape[0]:
        raise ValueError(f"{len(heard_at)} lists of cells for {sources.shape[0]} source positions")
    for row, listed in enumerate(heard_at):
        if any(not 0 <= int(c) < cells.shape[0] for c in listed):
            raise ValueError(f"source position {row} names a cell the bundle does not hold")
    models = export_models(bundle, scene_id=scene_id, hssd_root=hssd_root, models_from=models_from)
    scene, _, _ = scene_spec(models, STOREY_SCENE, fmax)
    if scene.ppw != ppw:
        raise ValueError(f"the export's spec uses {scene.ppw} points per wavelength, not {ppw}")
    np.save(bundle / "sources.npy", sources)
    np.save(bundle / "cells.npy", cells)
    heard = [sorted({int(c) for c in listed}) for listed in heard_at]
    (bundle / "heard_at.json").write_text(json.dumps(heard))
    pairs = sum(len(listed) for listed in heard)
    campaign = {
        "kind": KIND,
        "scene_id": scene_id,
        "dwelling": _dwelling_of(scene_id),
        "models": str(models.relative_to(bundle)),
        "model_json": str((models / f"{STOREY_SCENE}.json").relative_to(bundle)),
        "storey_scene": STOREY_SCENE,
        "materials": str((bundle / "models" / "materials").relative_to(bundle)),
        "bands": {
            "low": {
                "fmax_hz": fmax,
                "duration_s": duration_s,
                "cache_key": scene.key,
                "nh": int(scene.nh or 0),
            }
        },
        "ppw": ppw,
        "tc": 20.0,
        "rh": 50.0,
        "ram_gb": ram_gb,
        "order": order,
        "fit_order": fit_order,
        # What the rental sizes the host's RAM and disk by: the most cells one solve writes.
        "points": max((len(listed) for listed in heard), default=0),
        "source_positions": int(sources.shape[0]),
        "cells": int(cells.shape[0]),
        "pairs": int(pairs),
        "solver": SOLVER,
        "encoder": encoder_record(order, fit_order, fmax),
        "estimate": estimate(
            int(sources.shape[0]), int(pairs), fmax_hz=fmax, duration_s=duration_s
        ),
    }
    (bundle / "campaign.json").write_text(json.dumps(campaign, indent=1))
    return campaign


def _dwelling_of(scene_id: str) -> str:
    from reverberate.experiments.w40_volume_field.plan import dwelling_of

    return dwelling_of(scene_id)


# --------------------------------------------------------------------------
# the encoding of one cell
# --------------------------------------------------------------------------


@dataclass
class PairEncoder:
    """From the engine's node rows to one cell's response in the cache form.

    The filters are a field's (:class:`reverberate.accel.encode.BandPipeline`)
    up to the resampling, which goes to 4 kHz, and the fit runs there on 4800
    samples. The air is not applied and the crossover is not taken: both
    belong to the recipe, and the trace applies them
    (:func:`reverberate.spatial.lowband.to_stored`).
    """

    grid_rate_hz: float
    grid_step_m: float
    sound_speed_m_s: float
    fmax_hz: float
    settings: EncoderSettings
    xp: Any
    samples: int
    scale: float
    encoders: dict[str, BandEncoder] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def encoder_for(self, offsets: np.ndarray) -> BandEncoder:
        """The preparation for this geometry: once for the campaign, not once per solve."""
        key = geometry_key(offsets)
        if key not in self.encoders:
            self.encoders[key] = prepare_band(
                offsets,
                grid_step_m=self.grid_step_m,
                sound_speed_m_s=self.sound_speed_m_s,
                settings=self.settings,
                samples=self.samples,
                sample_rate_hz=LOW_RATE_HZ,
                xp=self.xp,
            )
        return self.encoders[key]

    def cell(
        self, u_out: Any, out_alpha: Any, offsets: np.ndarray, *, differentiated: bool
    ) -> np.ndarray:
        """One cell's order 7 response, ``[channel, sample]`` float32 at 4 kHz."""
        xp = self.xp
        with self._lock:
            signals = dsp.reduce_nodes(
                xp.asarray(u_out, dtype=xp.float64), xp.asarray(out_alpha), xp
            )
            lowcut = dsp.lowcut_sos(
                self.grid_rate_hz, LOWCUT_HZ, LOWCUT_ORDER, differentiated=differentiated
            )
            signals = dsp.sosfilt(lowcut, signals, xp)
            signals = dsp.sosfiltfilt(dsp.lowpass_sos(self.grid_rate_hz, self.fmax_hz), signals, xp)
            signals = dsp.resample(signals, self.grid_rate_hz, LOW_RATE_HZ, xp)
            # The engine runs a step over its window; the cache form is the window.
            block = xp.zeros((signals.shape[0], self.samples), dtype=xp.float64)
            kept = min(self.samples, int(signals.shape[1]))
            block[:, :kept] = signals[:, :kept]
            encoded = encode_point(self.encoder_for(offsets), block, xp)
            return np.asarray(to_numpy(encoded * self.scale), dtype=np.float32)


# --------------------------------------------------------------------------
# the campaign
# --------------------------------------------------------------------------


@dataclass
class PairsCampaign(Campaign):
    """The pairs of a bundle, solved and cached on the machine that holds the card."""

    #: Source positions solved at once, each by its own engine process; with
    #: several cards each worker keeps to one. ``None`` is one per card.
    solvers: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.spec.get("kind") != KIND:
            raise ValueError(f"{self.bundle} is not a bundle of low band pairs")
        self.sources = np.load(self.bundle / "sources.npy")
        self.cells = np.load(self.bundle / "cells.npy")
        self.heard_at: list[list[int]] = json.loads((self.bundle / "heard_at.json").read_text())
        self.cache = PairCache(self.out / "pairs", self.keys["low"])
        self._status_lock = threading.Lock()

    # ---- what a pair is called ---------------------------------------------------------

    @property
    def encoder_settings(self) -> EncoderSettings:
        return EncoderSettings(
            order=int(self.spec["order"]),
            fit_order=int(self.spec["fit_order"]),
            max_frequency_hz=self.fmax_hz["low"],
        )

    def key_of(self, source: int, cell: int) -> str:
        return pair_key(
            self.keys["low"],
            self.sources[source],
            self.cells[cell],
            encoder=dict(self.spec["encoder"]),
            solver=str(self.spec["solver"]),
            window_s=self.durations_s["low"],
        )

    # ---- stages --------------------------------------------------------------------------

    def place(self) -> dict[str, Any]:
        """An array at every cell on the low grid; a cell with no free ball near it has none."""
        from reverberate.experiments.run import entry_from_key
        from reverberate.experiments.w38_ambisonic_bands import outer_radius_for
        from reverberate.experiments.w40_volume_field.plan import place_arrays
        from reverberate.wave.comms import load_grid

        entry = entry_from_key(self.keys["low"])
        grid = load_grid(entry.path)
        radius = outer_radius_for(float(grid.h), OUTER_RADIUS_M)
        designs, _ = place_arrays(
            self.cells,
            entry.path,
            settings=self.encoder_settings,
            outer_radius_m=radius,
            shifts_m=SHIFTS_M,
        )
        refused = [i for i, design in enumerate(designs) if design is None]
        record = {
            "cells": int(self.cells.shape[0]),
            "refused": refused,
            "grid_step_m": float(grid.h),
            "outer_radius_m": radius,
            "sample_rate_hz": 1.0 / float(grid.Ts),
            "nodes_per_cell": float(np.mean([d.count for d in designs if d is not None] or [0])),
            # The expansion is about a node of the grid, up to half a cell's
            # diagonal from the cell asked for: the centre a translation starts from.
            "centres": [None if d is None else [float(v) for v in d.centre] for d in designs],
        }
        # Whole or not at all: the workers of a trace each place the arrays, and say the same.
        partial = self.out / f"pairs_plan.{os.getpid()}.partial.json"
        partial.write_text(json.dumps(record, indent=1))
        partial.replace(self.out / "pairs_plan.json")
        self.designs: list[ArrayDesign | None] = designs
        self.entry_path = entry.path
        self.encoder = PairEncoder(
            grid_rate_hz=1.0 / float(grid.Ts),
            grid_step_m=float(grid.h),
            sound_speed_m_s=self.sound_speed_m_s(),
            fmax_hz=self.fmax_hz["low"],
            settings=self.encoder_settings,
            xp=self.xp,
            samples=round(self.durations_s["low"] * LOW_RATE_HZ),
            scale=self.fmax_hz["low"] / REFERENCE_FMAX_HZ,
        )
        if refused:
            self.say(f"place: {len(refused)} cell(s) hold no array, a surface in the ball")
            self.say(f"place: refused {refused[:20]}")
        return record

    def sound_speed_m_s(self) -> float:
        from reverberate.experiments.run import sound_speed

        return float(sound_speed(float(self.spec["tc"])))

    def todo(self, source: int) -> list[int]:
        """The cells of a source position still to be solved: placed, and not in the cache."""
        return [
            cell
            for cell in self.heard_at[source]
            if self.designs[cell] is not None and not self.cache.has(self.key_of(source, cell))
        ]

    def solve_source(self, source: int, devices: str | None, ram_gb: float) -> dict[str, Any]:
        """One solve from one position, its cells encoded and cached as each slice lands."""
        cells = self.todo(source)
        if not cells:
            return {"source": source, "cells": 0, "skipped": True}
        designs = [d for d in (self.designs[cell] for cell in cells) if d is not None]
        positions = np.concatenate([d.positions for d in designs])
        counts = np.concatenate([[0], np.cumsum([d.count for d in designs])])
        rows: list[list[int] | None] = [
            [int(counts[i]), int(counts[i + 1])] for i in range(len(cells))
        ]
        steps = self.durations_s["low"] * self.encoder.grid_rate_hz
        output_bytes = float(positions.shape[0]) * (steps + 1.0) * self.sample_bytes
        job_root = self.out / "jobs" / f"source_{source:05d}"
        started = time.time()

        def consume(k: int, start: int, stop: int, sim_outs: Path, comms: Path) -> dict[str, Any]:
            t0 = time.time()
            with h5py.File(comms, "r") as handle:
                out_alpha = np.asarray(handle["out_alpha"][...], dtype=np.float64)
                differentiated = bool(np.asarray(handle["diff"]).item())
            done = 0
            with h5py.File(sim_outs, "r") as handle:
                for row, cell, design in zip(rows, cells, designs, strict=True):
                    if row is None or row[0] < start or row[1] > stop:
                        continue
                    a, b = row[0] - start, row[1] - start
                    block = np.asarray(handle["u_out"][a:b]).astype(np.float32).astype(np.float64)
                    response = self.encoder.cell(
                        block,
                        out_alpha[a:b],
                        design.positions - design.centre,
                        differentiated=differentiated,
                    )
                    self.cache.write(
                        self.key_of(source, cell),
                        response,
                        {
                            "source_m": [float(v) for v in self.sources[source]],
                            "cell_m": [float(v) for v in self.cells[cell]],
                            "centre_m": [float(v) for v in design.centre],
                            "fmax_hz": self.fmax_hz["low"],
                            "scale": self.encoder.scale,
                            "solver": str(self.spec["solver"]),
                        },
                    )
                    done += 1
            Path(sim_outs).unlink()
            return {"cells": done, "encode_s": round(time.time() - t0, 2)}

        slices = solve_slices(
            job_root=job_root,
            entry_path=self.entry_path,
            source_position=np.asarray(self.sources[source], dtype=float),
            positions=positions,
            rows=rows,
            duration_s=self.durations_s["low"],
            output_bytes=output_bytes,
            ram_gb=ram_gb,
            pffdtd_dir=self.pffdtd_dir,
            devices=devices,
            consume=consume,
            say=lambda m: self.say(f"  source {source} | {m}"),
        )
        shutil.rmtree(job_root, ignore_errors=True)
        return {
            "source": source,
            "cells": len(cells),
            "receivers": int(positions.shape[0]),
            "comms_s": round(sum(s.get("comms_s", 0.0) for s in slices), 2),
            "engine_s": round(sum(s.get("engine_s", 0.0) for s in slices), 2),
            "encode_s": round(sum(s.get("consume_s", 0.0) for s in slices), 2),
            "total_s": round(time.time() - started, 2),
        }

    def solve(self) -> list[dict[str, Any]]:
        """Every source position that still has a pair to make, ``solvers`` at a time."""
        self.sample_bytes = output_sample_bytes(self.pffdtd_dir)
        cards = self.devices.split(",") if self.devices else self.card_names()
        solvers = self.solvers or max(1, len(cards))
        wanted = [s for s in range(self.sources.shape[0]) if self.todo(s)]
        cached = self.sources.shape[0] - len(wanted)
        self.say(
            f"solve: {len(wanted)} source position(s) to solve, {cached} wholly cached,"
            f" {solvers} at a time"
        )
        ram_gb = host_memory_gb() / solvers
        records: list[dict[str, Any]] = []
        started = time.time()

        def work(job: tuple[int, int]) -> dict[str, Any]:
            number, source = job
            devices = cards[number % len(cards)] if cards and solvers > 1 else self.devices
            record = self.solve_source(source, devices, ram_gb)
            with self._status_lock:
                records.append(record)
                elapsed = time.time() - started
                self.set_status(
                    job=f"{len(records)}/{len(wanted)}",
                    pairs=sum(int(r["cells"]) for r in records),
                    remaining_s=round(elapsed / len(records) * (len(wanted) - len(records)), 1),
                )
            self.say(
                f"source {source}: {record['cells']} cell(s), engine {record.get('engine_s')} s,"
                f" encode {record.get('encode_s')} s"
            )
            return record

        if solvers > 1:
            with ThreadPoolExecutor(max_workers=solvers) as pool:
                list(pool.map(work, enumerate(wanted)))
        else:
            for job in enumerate(wanted):
                work(job)
        return sorted(records, key=lambda r: int(r["source"]))

    def card_names(self) -> list[str]:
        """The visible cards' indices as the engine is told them; empty without a card."""
        count = int(self.status["device"].get("devices") or 0) if self.xp is not np else 0
        return [str(i) for i in range(count)]

    def run(self) -> dict[str, Any]:
        """Voxelise, place, solve; ``campaign.done`` or ``campaign.failed`` at the end."""
        for marker in ("campaign.done", "campaign.failed"):
            (self.out / marker).unlink(missing_ok=True)
        try:
            self.say(
                f"pairs {self.spec.get('dwelling')}: {self.sources.shape[0]} source position(s),"
                f" {self.cells.shape[0]} cell(s), {self.spec.get('pairs')} pair(s), grid to"
                f" {self.fmax_hz['low']:g} Hz on {self.status['device'].get('gpu')}"
            )
            vox = self.stage("voxelise", self.voxelise)
            placed = self.stage("plan", self.place)
            solves = self.stage("solve", self.solve)
            report = {
                "kind": KIND,
                "dwelling": self.spec.get("dwelling"),
                "device": self.status["device"],
                "host_ram_gb": self.status["host_ram_gb"],
                "cpus": self.status["cpus"],
                "timings_s": self.timings,
                "total_s": round(time.time() - self.started, 1),
                "voxelise": vox,
                "placed": {k: v for k, v in placed.items() if k != "centres"},
                "solves": solves,
                "pairs_solved": sum(int(r["cells"]) for r in solves),
                "pairs_in_cache": len(self.cache.records()),
                "prepare_s": round(sum(e.prepare_s for e in self.encoder.encoders.values()), 1),
                "solver": self.spec.get("solver"),
            }
            (self.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
            self.set_status(stage="done")
            (self.out / "campaign.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            self.say(f"pairs complete in {(time.time() - self.started) / 3600:.2f} h")
            return report
        except BaseException as error:
            self.set_status(stage="failed", error=repr(error)[:500])
            (self.out / "campaign.failed").write_text(repr(error)[:2000])
            self.say(f"pairs FAILED: {error!r}"[:600])
            raise


def run_pairs(
    bundle: Path,
    out: Path,
    *,
    pffdtd_dir: Path,
    devices: str | None = None,
    gpu: bool | None = None,
    solvers: int | None = None,
) -> dict[str, Any]:
    return PairsCampaign(
        bundle=bundle, out=out, pffdtd_dir=pffdtd_dir, devices=devices, gpu=gpu, solvers=solvers
    ).run()


def cost_record(
    report: dict[str, Any], *, billed_rate_usd_per_hour: float, instance: int | str
) -> dict[str, Any]:
    """A pack's cost record for the ``low`` stage, from a campaign's report and its rate.

    The machine does not know what it is billed; the laptop, which rented
    it, does. A cost without its rate is not written.
    """
    seconds = float(report["total_s"])
    device = report.get("device", {})
    return {
        "stage": "low",
        "seconds": seconds,
        "card": device.get("gpu"),
        "cards": device.get("devices"),
        "billed_rate_usd_per_hour": float(billed_rate_usd_per_hour),
        "usd": round(seconds / 3600.0 * float(billed_rate_usd_per_hour), 4),
        "instance": instance,
    }
