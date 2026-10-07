"""The routes the components ask: what a player, a sonogram, a track list and a test need.

An application hands its :class:`~reverberate.viz.parts.media.Library` to
:class:`Media` and has, under ``api/``:

- ``GET  items``: what there is to play.
- ``GET  frames?item=&start=&count=[&balance=]``: float32 frames,
  ``[frame][channel]``, the item's stems summed under a balance. ``item``
  may be a hidden name of the blind test under way (``blind:X``).
- ``GET  sonogram?item=[&less=]``: a sonogram as float32 ``[column][band]``,
  what it is in the ``X-Sonogram`` header; with ``less``, that item's
  sonogram taken from it, in dB.
- ``GET  levels?item=``: each stem's level every 50 ms, for a light and a
  meter.
- ``GET  balance`` and ``POST balance``: the track list's faders; a POST
  saves them and answers with the version a player asks its frames under.
  Where the application keeps the file for a button (``autosave=False``), a
  POST only changes what is heard and ``POST balance/save`` writes it.
- ``GET  blind``, ``POST blind/start``, ``POST blind/answer``,
  ``POST blind/stop``: a blind test, one at a time.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.viz.parts import balance as balances
from reverberate.viz.parts.blind import HIDDEN, BlindTest
from reverberate.viz.parts.media import Library, colour_of, kind_of
from reverberate.viz.parts.server import AppServer, Binary, HttpError, Request
from reverberate.viz.parts.sonogram import Sonogram, difference, levels_db, sonogram

__all__ = ["Media"]

#: Frames a request may ask for: ten seconds of order 7 are 123 MB.
MAX_FRAMES = 480_000
#: Hop of a level track, seconds.
LEVEL_HOP_S = 0.05


class Media:
    """A library behind the components' routes, with its balance and its blind test."""

    def __init__(
        self,
        library: Library,
        results: Path,
        *,
        context: Mapping[str, Any] | None = None,
        tracks: Sequence[Mapping[str, Any]] | None = None,
        balance_path: Path | None = None,
        autosave: bool = True,
        saved_with: Callable[[Mapping[str, Mapping[str, Any]]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.library = library
        #: Where a balance and the blind tests' results are written.
        self.results = Path(results)
        self.context = dict(context or {})
        #: What the application knows of a source, ``id`` and any of ``label``, ``kind``,
        #: ``colour``; a source it does not name is told by its name alone.
        self.tracks = {str(row["id"]): dict(row) for row in tracks or ()}
        self._balance_path = None if balance_path is None else Path(balance_path)
        #: Whether every change of a fader is written at once; else ``balance/save`` writes.
        self.autosave = bool(autosave)
        #: What is added to the context of a balance when it is written, from the balance.
        self.saved_with = saved_with
        self._dirty = False
        self.blind: BlindTest | None = None
        self._lock = threading.Lock()
        self._sonograms: OrderedDict[str, Sonogram] = OrderedDict()
        self._levels: dict[str, dict[str, list[float]]] = {}
        names: list[str] = []
        for item in library.describe():
            names += [stem for stem in item["stems"] if stem not in names]
        self.sources = names
        self._balance = balances.load(self.balance_path, names)
        self._versions: OrderedDict[int, dict[str, float]] = OrderedDict(
            {0: balances.gains(self._balance)}
        )

    @property
    def balance_path(self) -> Path:
        return self._balance_path or self.results / "balance.json"

    def mount(self, server: AppServer, prefix: str = "api") -> None:
        server.route("GET", f"{prefix}/items", lambda request: self.library.describe())
        server.route("GET", f"{prefix}/frames", self.frames)
        server.route("GET", f"{prefix}/sonogram", self.sonogram)
        server.route("GET", f"{prefix}/levels", self.levels)
        server.route("GET", f"{prefix}/balance", lambda request: self.balance())
        server.route("POST", f"{prefix}/balance", self.set_balance)
        server.route("GET", f"{prefix}/blind", lambda request: self.blind_state())
        server.route("POST", f"{prefix}/blind", self.blind_step)

    # --- what is played -----------------------------------------------------------------

    def _name(self, name: str) -> str:
        if not name.startswith(HIDDEN):
            return name
        if self.blind is None:
            raise HttpError(409, "no blind test is under way")
        return self.blind.resolve(name[len(HIDDEN) :])

    def _gains(self, request: Request) -> dict[str, float] | None:
        if "balance" not in request.query:
            return None
        version = int(request.number("balance"))
        with self._lock:
            if version not in self._versions:
                raise HttpError(409, f"balance {version} is no longer held: ask the present one")
            return self._versions[version]

    def frames(self, request: Request) -> Binary:
        start, count = int(request.number("start")), int(request.number("count"))
        if count > MAX_FRAMES:
            raise HttpError(413, f"a request is of {MAX_FRAMES} frames at most")
        data = self.library.frames(
            self._name(request.text("item")), start, count, self._gains(request)
        )
        headers = {"X-Frames": str(data.shape[0]), "X-Channels": str(data.shape[1])}
        return Binary(np.ascontiguousarray(data, dtype="<f4").tobytes(), headers=headers)

    # --- what is looked at ---------------------------------------------------------------

    def _sonogram(self, name: str) -> Sonogram:
        with self._lock:
            if name in self._sonograms:
                self._sonograms.move_to_end(name)
                return self._sonograms[name]
        item = self.library.item(name)
        made = sonogram(self.library.mono(name), item.rate)
        with self._lock:
            self._sonograms[name] = made
            while len(self._sonograms) > 12:
                self._sonograms.popitem(last=False)
        return made

    def sonogram(self, request: Request) -> Binary:
        if self.blind is not None and not self.blind.finished:
            raise HttpError(409, "a blind test is under way: nothing is shown")
        name = request.text("item")
        made = self._sonogram(name)
        told: dict[str, Any] = {**made.describe(), "item": name, "unit": "dB re full scale"}
        table = made.levels_db
        if "less" in request.query:
            other = self._sonogram(request.query["less"])
            try:
                table = difference(made, other)
            except ValueError as error:
                raise HttpError(400, str(error)) from error
            told.update(less=request.query["less"], unit="dB, the item less the other")
        return Binary(
            np.ascontiguousarray(table, dtype="<f4").tobytes(),
            headers={"X-Sonogram": json.dumps(told)},
        )

    def levels(self, request: Request) -> dict[str, Any]:
        name = request.text("item")
        item = self.library.item(name)
        with self._lock:
            held = self._levels.get(name)
        if held is None:
            held = {
                stem.id: np.round(
                    levels_db(self.library.mono(name, stem=stem.id), item.rate, hop_s=LEVEL_HOP_S),
                    1,
                ).tolist()
                for stem in item.stems
            }
            with self._lock:
                self._levels[name] = held
        return {"item": name, "hop_s": LEVEL_HOP_S, "unit": "dB re full scale", "stems": held}

    # --- the track list --------------------------------------------------------------------

    def balance(self) -> dict[str, Any]:
        with self._lock:
            version = next(reversed(self._versions))
            counts = {"voice": 0, "noise": 0}
            tracks = []
            for name in self.sources:
                known = self.tracks.get(name, {})
                kind = str(known.get("kind") or kind_of(name))
                tracks.append(
                    {
                        "id": name,
                        "label": known.get("label") or name,
                        "kind": kind,
                        "colour": known.get("colour") or colour_of(counts[kind], kind),
                        **self._balance[name],
                    }
                )
                counts[kind] += 1
            return {
                "version": version,
                "tracks": tracks,
                "fader_db": list(balances.FADER_DB),
                "saved": str(self.balance_path) if self.balance_path.is_file() else None,
                "file": str(self.balance_path),
                "unsaved": self._dirty,
            }

    def gains_now(self) -> dict[str, float]:
        """The factor of each source under the balance as it stands."""
        with self._lock:
            return dict(self._versions[next(reversed(self._versions))])

    def _write(self) -> None:
        context = dict(self.context)
        if self.saved_with is not None:
            context.update(self.saved_with(self._balance))
        balances.save(self.balance_path, self._balance, context)
        self._dirty = False

    def set_balance(self, request: Request) -> dict[str, Any]:
        if request.parts == ("save",):
            self._write()
            return self.balance()
        if request.parts:
            raise HttpError(404, f"a balance is not told {request.parts[0]!r}")
        cleaned = balances.clean(request.body, self.sources)
        with self._lock:
            self._balance = cleaned
            version = next(reversed(self._versions)) + 1
            self._versions[version] = balances.gains(cleaned)
            while len(self._versions) > 16:
                self._versions.popitem(last=False)
            self._dirty = True
        if self.autosave:
            self._write()
        return self.balance()

    # --- the blind test --------------------------------------------------------------------

    def blind_state(self) -> dict[str, Any]:
        return {"kind": None} if self.blind is None else self.blind.state()

    def blind_step(self, request: Request) -> dict[str, Any]:
        action = request.parts[0] if request.parts else ""
        body = request.body if isinstance(request.body, Mapping) else {}
        if action == "start":
            items = body.get("items")
            if not isinstance(items, Mapping):
                raise HttpError(400, "a test is started with {kind, items: {name: item}, trials}")
            for item in items.values():
                self.library.item(str(item))
            if self.blind is not None and not self.blind.finished:
                self.blind.stop()
            self.blind = BlindTest(
                str(body.get("kind")),
                {str(k): str(v) for k, v in items.items()},
                int(body.get("trials") or 0),
                self.results,
                seed=body.get("seed"),
                context={**self.context, **dict(body.get("context") or {})},
            )
            return self.blind.state()
        if self.blind is None:
            raise HttpError(409, "no blind test is under way")
        if action == "answer":
            return self.blind.answer(body.get("answer"))
        if action == "stop":
            return self.blind.stop()
        raise HttpError(404, f"a blind test is not told {action!r}")
