"""What the page's scene view asks the server in order to be heard.

The scene view replays a recipe without sound (:mod:`reverberate.viz.scene_api`).
Once the recipe is traced into a pack (``docs/formats/scene-pack.md``) the
page plays it: this module finds the packs, has the signal engine render
them (:mod:`reverberate.viz.audit_stems`) and hands the page the engine's
order 7 samples. The page turns them by the head and decodes them to two
ears, as it does for a solver's field; it renders nothing.

Under ``/api/audit/``:

- ``GET  packs?recipe_sha256=<hex>``: the packs this server knows, those of
  that recipe marked. A pack is any ``*.h5`` under the recipes folder (so a
  pack dropped beside its recipe is found) or under a folder or path given
  at start.
- ``GET  tracks?pack=<id>``: where everything is over time, read off the
  pack itself, in the shape of ``/api/scene/tracks``: a pack is then shown
  even when its recipe is not on this disk.
- ``POST status``: ``{pack, directivity, cursor, audible, background}`` says
  where the listener is and answers with what is rendered, per source, as
  runs of chunks, with the render's rate. The page asks twice a second: a
  page that stops asking stops the render.
- ``GET  chunk?pack=&directivity=&index=&sources=a,b&wait_ms=``: the mix of
  the chosen sources over one chunk, raw float32 frames of 64 channels. It
  waits for the render up to ``wait_ms`` and answers 503 beyond.
- ``GET  checksum?pack=&directivity=&index=&sources=``: the SHA-256 of the
  engine's samples for that chunk, taken as they left the engine and before
  they were written. ``fresh=1`` takes it from an engine made for the
  question, past the cache.
- ``GET  arrivals?pack=&source=&from=&to=``: the selected source's arrivals
  at the listener over a run of steps, from the pack's ``early`` table, and
  the direction its tail's energy comes from.

``directivity`` is ``pack`` (each source as the pack says), ``on`` or ``off``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.render.engine import Engine
from reverberate.render.pack import SCHEMA as PACK_SCHEMA
from reverberate.render.pack import ScenePack
from reverberate.render.variant import label_of
from reverberate.render.variant import summary as variant_summary
from reverberate.spatial.sh import real_sh, scene_to_ambisonic
from reverberate.viz.audit_dry import dry_track
from reverberate.viz.audit_stems import AuditSettings, Session, StemService
from reverberate.viz.scene_api import SceneError

__all__ = ["PREFIX", "AuditService", "Binary", "pack_tracks"]

#: The path every endpoint lives under.
PREFIX = "api/audit"

#: The step the tracks of a pack are sampled at for the page, seconds.
TRACK_STEP_S = 0.1
#: The longest a chunk request waits for its render, and the most steps of
#: arrivals one request returns.
MAX_WAIT_MS = 60_000
MAX_ARRIVAL_STEPS = 200
#: The directions the tail's energy is given at: elevations by azimuths.
TAIL_GRID = (13, 24)

_DIRECTIVITY = {"pack": None, "on": True, "off": False}


@dataclass(frozen=True)
class Binary:
    """An answer that is bytes and not JSON."""

    payload: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def scene_digest(f: h5py.File) -> str:
    """The digest of where the head and every source are at every step: one scene's.

    Two packs of one recipe have it in common, and so have two packs whose
    recipes differ by their rails' pitch alone, which changes where the low
    band is solved and nothing of the movements: the page lets the owner
    switch between such packs at one instant.
    """
    held = hashlib.sha256()
    held.update(f"{int(f.attrs['steps'])} {float(f.attrs['step_s'])!r}".encode())
    held.update(np.ascontiguousarray(f["listener/position"][...], dtype="<f8").tobytes())
    for name in sorted(f["sources"]):
        held.update(name.encode())
        position = f["sources"][name]["position"][...]
        held.update(np.ascontiguousarray(position, dtype="<f8").tobytes())
    return held.hexdigest()


def describe_pack(path: Path) -> dict[str, Any] | None:
    """What a pack says of itself, read off its attributes; ``None`` if it is not one."""
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            if _text(a.get("schema", "")) != PACK_SCHEMA:
                return None
            models = set(f["directivity"]) if "directivity" in f else set()
            sources = [
                {
                    "id": name,
                    "kind": _text(group.attrs["kind"]),
                    "directivity_model": _text(group.attrs["directivity_model"]),
                    "directivity_enabled": bool(group.attrs["directivity_enabled"]),
                }
                for name, group in f["sources"].items()
            ]
            text = a.get("provenance_json", "{}")
            try:
                provenance = dict(json.loads(_text(text)))
            except ValueError:
                provenance = {}
            told = variant_summary(path, provenance)
            return {
                "id": hashlib.sha256(str(path).encode()).hexdigest()[:12],
                # A variant's pack is called by its variant; any other, as it lies.
                "name": told["name"] or label_of(path),
                "variant": told,
                "scene_sha256": scene_digest(f),
                "path": str(path),
                "bytes": path.stat().st_size,
                "profile": _text(a["profile"]),
                "recipe_sha256": _text(a["recipe_sha256"]),
                "dwelling": _text(a["dwelling"]),
                "scene_id": _text(a["scene_id"]),
                "duration_s": float(a["duration_s"]),
                "steps": int(a["steps"]),
                "step_s": float(a["step_s"]),
                "sample_rate_hz": float(a["sample_rate_hz"]),
                "order": int(a["order"]),
                "has_low": bool(a["has_low"]),
                "has_tail": bool(a["has_tail"]),
                "sources": sources,
                # The switch is offered when it would change something.
                "directivity_switch": any(
                    s["directivity_model"] in models and s["directivity_model"] != "omni"
                    for s in sources
                ),
            }
    except (OSError, KeyError, ValueError):
        return None


def _column(values: np.ndarray, decimals: int) -> list[float]:
    rounded: list[float] = np.round(np.asarray(values, dtype=float), decimals).tolist()
    return rounded


def pack_tracks(session: Session, step_s: float = TRACK_STEP_S) -> dict[str, Any]:
    """Every source and the listener on a time grid, as the pack holds them.

    The shape of :func:`reverberate.viz.scene_api.tracks`, without the
    recipe's segments: a pack says where a source is, not what it is doing.
    ``activity`` is when the source is fed audio, which is when it sounds.
    """
    pack = session.pack
    h = pack.header
    stride = max(1, int(round(step_s / h.step_s)))
    rows = np.arange(0, h.steps, stride)
    if rows[-1] != h.steps - 1:
        rows = np.append(rows, h.steps - 1)
    sources = []
    for name, source in pack.sources.items():
        spans: list[list[float]] = []
        for piece in session.plans[name].source.get("activity", []):
            if spans and abs(spans[-1][1] - piece["start_s"]) < 1e-9:
                spans[-1][1] = float(piece["end_s"])
            else:
                spans.append([float(piece["start_s"]), float(piece["end_s"])])
        sources.append(
            {
                "id": name,
                "kind": source.kind,
                "subtype": source.subtype,
                "x": _column(source.position[rows, 0], 3),
                "y": _column(source.position[rows, 1], 3),
                "z": _column(source.position[rows, 2], 3),
                "yaw_deg": _column(source.yaw_deg[rows], 2),
                "movement": [],
                "activity": spans,
            }
        )
    head, turn = pack.listener.position, pack.listener.orientation
    return {
        "recipe_sha256": h.recipe_sha256,
        "duration_s": (h.steps - 1) * h.step_s,
        "step_s": stride * h.step_s,
        "t": _column(rows * h.step_s, 3),
        "sources": sources,
        "listener": {
            "x": _column(head[rows, 0], 3),
            "y": _column(head[rows, 1], 3),
            "z": _column(head[rows, 2], 3),
            "yaw_deg": _column(turn[rows, 0], 2),
            "pitch_deg": _column(turn[rows, 1], 2),
            "roll_deg": _column(turn[rows, 2], 2),
            "movement": [],
        },
    }


def _tail_grid() -> np.ndarray:
    """Unit vectors of the scene's frame on a grid of elevations by azimuths, ``[el, az, 3]``."""
    rows, columns = TAIL_GRID
    el = np.linspace(-0.5 * np.pi, 0.5 * np.pi, rows)[:, None]
    az = (2.0 * np.pi * np.arange(columns) / columns)[None, :]
    return np.stack(
        [np.cos(el) * np.cos(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.sin(az)], axis=-1
    )


class AuditService:
    """The audit's endpoints over the packs of some folders and one stem cache."""

    def __init__(
        self, stems: StemService, folders: Iterable[Path] = (), packs: Iterable[Path] = ()
    ) -> None:
        self.stems = stems
        self.folders = [Path(folder) for folder in folders]
        self.given = [Path(path).resolve() for path in packs]
        self._known: dict[str, tuple[tuple[int, int], dict[str, Any] | None]] = {}
        self._moments: dict[tuple[str, str, int], np.ndarray] = {}
        grid = _tail_grid()
        self._grid = grid
        self._grid_sh = real_sh(3, scene_to_ambisonic(grid.reshape(-1, 3)))

    # -- packs --------------------------------------------------------------

    def packs(self) -> list[dict[str, Any]]:
        """Every pack under the folders and every pack given by path, by name."""
        paths = list(self.given)
        for folder in self.folders:
            if folder.is_dir():
                paths += sorted(p.resolve() for p in folder.rglob("*.h5"))
        listed = []
        for path in dict.fromkeys(paths):
            try:
                stat = path.stat()
            except OSError:
                continue
            stamp = (stat.st_size, stat.st_mtime_ns)
            kept = self._known.get(str(path))
            if kept is None or kept[0] != stamp:
                kept = (stamp, describe_pack(path))
                self._known[str(path)] = kept
            if kept[1] is not None:
                listed.append(kept[1])
        return listed

    def _session(self, query: Mapping[str, Any]) -> Session:
        wanted = query.get("pack")
        found = next((p for p in self.packs() if p["id"] == wanted), None)
        if found is None:
            raise SceneError(404, f"no pack {wanted!r}")
        switch = query.get("directivity", "pack")
        if switch not in _DIRECTIVITY:
            raise SceneError(400, "directivity is pack, on or off")
        try:
            return self.stems.open(Path(found["path"]), AuditSettings(_DIRECTIVITY[switch]))
        except (OSError, ValueError) as error:
            raise SceneError(422, f"{found['name']}: {error}") from error

    @staticmethod
    def _sources(session: Session, given: Any) -> list[str]:
        names = given if isinstance(given, list) else [n for n in str(given or "").split(",") if n]
        unknown = [name for name in names if name not in session.stems]
        if unknown:
            raise SceneError(400, f"the pack holds no source {', '.join(map(str, unknown))}")
        return [name for name in session.order if name in names]

    @staticmethod
    def _whole(query: Mapping[str, Any], name: str, low: int, high: int) -> int:
        try:
            value = int(query[name])
        except (KeyError, TypeError, ValueError) as error:
            raise SceneError(400, f"{name} is not a whole number") from error
        if not low <= value < high:
            raise SceneError(400, f"{name} = {value} is outside {low} to {high - 1}")
        return value

    # -- the sound ------------------------------------------------------------

    def status(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise SceneError(400, "the body is not an object")
        session = self._session(body)
        cursor = body.get("cursor")
        self.stems.want(
            session,
            cursor=None if cursor is None else self._whole(body, "cursor", 0, session.chunks),
            audible=None if "audible" not in body else self._sources(session, body["audible"]),
            background=body.get("background"),
        )
        h = session.pack.header
        return {
            **self.stems.status(session),
            "samples": h.samples,
            "sample_rate_hz": h.sample_rate_hz,
            "channels": h.channels,
            "order": h.order,
            "step_samples": h.step_samples,
        }

    def chunk(self, query: Mapping[str, Any]) -> Binary:
        session = self._session(query)
        index = self._whole(query, "index", 0, session.chunks)
        sources = self._sources(session, query.get("sources"))
        wait_ms = min(MAX_WAIT_MS, max(0, int(query.get("wait_ms", 0) or 0)))
        # Asking for a chunk is saying where the listener is.
        self.stems.want(session, cursor=index)
        try:
            ready = self.stems.wait(session, sources, index, wait_ms / 1000.0)
        except RuntimeError as error:
            raise SceneError(500, f"the engine failed: {error}") from error
        if not ready:
            raise SceneError(503, f"chunk {index} is not rendered yet")
        frames = session.mix(sources, index)
        start, _ = next(iter(session.stems.values())).span(index)
        return Binary(
            frames.tobytes(),
            {
                "X-Audit-Start": str(start),
                "X-Audit-Frames": str(frames.shape[0]),
                "X-Audit-Channels": str(frames.shape[1]),
                "X-Audit-Sources": ",".join(sources),
                "X-Audit-Levels": json.dumps(session.levels(sources, index)),
            },
        )

    def checksum(self, query: Mapping[str, Any]) -> dict[str, Any]:
        """The digest of the engine's samples for a chunk, and of each source's alone."""
        session = self._session(query)
        index = self._whole(query, "index", 0, session.chunks)
        sources = self._sources(session, query.get("sources"))
        start, stop = next(iter(session.stems.values())).span(index)
        answer: dict[str, Any] = {"index": index, "start": start, "frames": stop - start}
        if str(query.get("fresh", "0")) == "1":
            # Past the cache: an engine of its own, the stems as it yields them.
            stems = {}
            for name in sources:
                engine = Engine(
                    session.pack,
                    {name: dry_track(session.plans[name], session.pack.header.sample_rate_hz)},
                    settings=session.settings.render(),
                )
                stems[name] = np.ascontiguousarray(engine.stem(name, start, stop).T.astype("<f4"))
            per_source = {n: hashlib.sha256(s.tobytes()).hexdigest() for n, s in stems.items()}
            if len(sources) == 1:
                mixed = stems[sources[0]]
            else:
                total = np.zeros((stop - start, session.pack.header.channels))
                for name in sources:
                    total += stems[name]
                mixed = total.astype("<f4")
            answer["from"] = "an engine made for this request"
        else:
            if not session.ready(sources, index):
                raise SceneError(503, f"chunk {index} is not rendered yet")
            per_source = {n: str(session.stems[n].records[index]["sha256"]) for n in sources}
            mixed = session.mix(sources, index)
            answer["from"] = "the engine's samples when the chunk was rendered"
        answer["sources"] = per_source
        # One source: the engine's own digest. Several: that of their sum, as `chunk` sums them.
        answer["sha256"] = (
            per_source[sources[0]]
            if len(sources) == 1
            else hashlib.sha256(mixed.tobytes()).hexdigest()
        )
        return answer

    # -- what is drawn in the volume ------------------------------------------

    def _tail_moments(self, session: Session, source_id: str, row: int) -> np.ndarray:
        """A histogram's directional moments summed over its bins and bands: 16 numbers."""
        name = (session.pack_sha256, source_id, row)
        if name not in self._moments:
            tail = session.pack.sources[source_id].tail
            assert tail is not None
            moments = np.asarray(tail.moments[row], dtype=float)
            self._moments[name] = moments.reshape(-1, moments.shape[-1]).sum(axis=0)
        return self._moments[name]

    def _tail_energy(self, session: Session, source_id: str, step: int) -> list[float] | None:
        """Where the tail's energy comes from at a step, on the grid, its largest value one."""
        tail = session.pack.sources[source_id].tail
        if tail is None or int(tail.hist[step, 0, 0]) < 0:
            return None
        position, cell = float(tail.position_weight[step]), float(tail.cell_weight[step])
        total = np.zeros(self._grid_sh.shape[1])
        for a, wa in ((0, 1.0 - position), (1, position)):
            for b, wb in ((0, 1.0 - cell), (1, cell)):
                row = int(tail.hist[step, a, b])
                if row >= 0 and wa * wb > 0.0:
                    total += wa * wb * self._tail_moments(session, source_id, row)
        energy = np.maximum(self._grid_sh @ total, 0.0)
        peak = float(energy.max())
        return _column(energy / peak, 3) if peak > 0.0 else None

    def arrivals(self, query: Mapping[str, Any]) -> dict[str, Any]:
        session = self._session(query)
        pack: ScenePack = session.pack
        named = self._sources(session, query.get("source"))
        if len(named) != 1:
            raise SceneError(400, "source names one source")
        source_id = named[0]
        steps = pack.header.steps
        first = self._whole(query, "from", 0, steps)
        # As many steps as asked, to the most one answer holds and to the scene's end.
        asked = self._whole(query, "to", first + 1, 1 << 31)
        last = min(asked, first + MAX_ARRIVAL_STEPS, steps)
        source = pack.sources[source_id]
        listed = []
        for step in range(first, last):
            rows = source.early.rows(step)
            gain = np.asarray(source.early.gain[rows], dtype=float)
            listed.append(
                {
                    "listener": _column(pack.listener.position[step], 3),
                    "direction": _column(np.asarray(source.early.arrival[rows]).ravel(), 4),
                    "delay_s": _column(source.early.delay_s[rows], 6),
                    # One number per arrival: its gain over the bands, in power.
                    "gain": _column(np.sqrt(np.mean(gain**2, axis=1)) if gain.size else gain, 6),
                    "order": [int(v) for v in source.early.order[rows]],
                    "kind": [int(v) for v in source.early.kind[rows]],
                    "tail": self._tail_energy(session, source_id, step),
                }
            )
        return {
            "source": source_id,
            "from": first,
            "to": last,
            "step_s": pack.header.step_s,
            "sound_speed_m_s": pack.header.sound_speed_m_s,
            "tail_grid": {
                "elevations": TAIL_GRID[0],
                "azimuths": TAIL_GRID[1],
                "directions": _column(self._grid.ravel(), 4),
            },
            "steps": listed,
        }

    # -- routing ------------------------------------------------------------

    def handle(
        self, method: str, parts: list[str], query: Mapping[str, Any], body: Any = None
    ) -> Any:
        """Answer ``parts``, the path below :data:`PREFIX`; raises :class:`SceneError`."""
        match (method, parts):
            case ("GET", ["packs"]):
                wanted = query.get("recipe_sha256")
                listed = self.packs()
                if listed:
                    # A page is looking at packs: the workers are up before it presses play.
                    self.stems.start()
                # A pack of the recipe on show, or of the same scene under another recipe.
                scenes = {p["scene_sha256"] for p in listed if p["recipe_sha256"] == wanted}
                return [
                    {
                        **pack,
                        "matches": pack["recipe_sha256"] == wanted,
                        "same_scene": pack["scene_sha256"] in scenes,
                    }
                    for pack in listed
                ]
            case ("GET", ["tracks"]):
                return pack_tracks(self._session(query))
            case ("POST", ["status"]):
                return self.status(body)
            case ("GET", ["chunk"]):
                return self.chunk(query)
            case ("GET", ["checksum"]):
                return self.checksum(query)
            case ("GET", ["arrivals"]):
                return self.arrivals(query)
        raise SceneError(404, f"no such endpoint: {method} /{PREFIX}/{'/'.join(parts)}")
