"""A blind test, the key held on this side: ABX to tell two apart, a ranking to prefer.

Switching freely between two renders says what differs. Whether the
difference is heard at all, or is the name on the button, is another
question, and a blind test answers it with a number.

- **ABX.** A and B are known; X is one of the two, drawn for each trial.
  The listener says which. Under the hypothesis that he cannot tell, each
  answer is right with a probability of one half, and the score of ``k``
  right out of ``n`` has the probability :func:`chance` of being reached or
  passed by guessing.
- **Ranking.** Every variant under a name that says nothing (``X1`` ...),
  in an order drawn for each trial; the listener ranks them, the one he
  prefers first. Told at the end: each variant's mean rank and how often it
  came first, and the probability that one variant comes first that often
  when the order is chance, which is ``1 / N`` a trial. It is multiplied by
  the number of variants, since the one read is the one that did best.

The page plays a hidden name through :meth:`BlindTest.resolve` and is never
told what it stands for before the test is over. Nothing is said after a
trial: a listener told he was right learns the answer, not the difference.
The result is a JSON file (:data:`SCHEMA`), written when the last trial is
answered, or when the test is stopped, with what was answered until then.
"""

from __future__ import annotations

import json
import math
import secrets
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.viz.parts.server import HttpError

__all__ = ["SCHEMA", "BlindTest", "chance"]

SCHEMA = "reverberate.apps.blind"
#: The name a hidden item is asked for under, before what the page calls it.
HIDDEN = "blind:"


def chance(right: int, trials: int, probability: float = 0.5) -> float:
    """The probability of ``right`` or more out of ``trials`` when each is ``probability``."""
    if not 0 <= right <= trials:
        raise ValueError(f"{right} out of {trials} is not a score")
    return float(
        min(
            sum(
                math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
                for k in range(right, trials + 1)
            ),
            1.0,
        )
    )


class BlindTest:
    """One test from its first trial to its file."""

    def __init__(
        self,
        kind: str,
        items: Mapping[str, str],
        trials: int,
        results: Path,
        *,
        seed: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """``items`` is what is compared, by name, each the item played for it.

        For ``"abx"`` exactly two, A then B. ``results`` is the folder the
        result is written in; ``context`` goes into the file as given.
        """
        if kind not in ("abx", "rank"):
            raise HttpError(400, f"a blind test is abx or rank, not {kind!r}")
        if kind == "abx" and len(items) != 2:
            raise HttpError(400, "an ABX test is of two")
        if len(items) < 2:
            raise HttpError(400, "there is nothing to compare one item with")
        if not 1 <= int(trials) <= 200:
            raise HttpError(400, "a test has between 1 and 200 trials")
        self.kind, self.items, self.trials = kind, dict(items), int(trials)
        self.results = Path(results)
        self.context = dict(context or {})
        self.seed = secrets.randbits(63) if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.started = datetime.now(UTC)
        self.done: list[dict[str, Any]] = []
        self.saved: Path | None = None
        self._stopped = False
        self._hidden: dict[str, str] = {}
        self._since = time.monotonic()
        self._draw()

    @property
    def finished(self) -> bool:
        return self._stopped or len(self.done) >= self.trials

    def _draw(self) -> None:
        names = list(self.items)
        if self.kind == "abx":
            self._hidden = {"X": names[int(self._rng.integers(2))]}
        else:
            order = self._rng.permutation(len(names))
            self._hidden = {f"X{i + 1}": names[int(j)] for i, j in enumerate(order)}
        self._since = time.monotonic()

    def state(self) -> dict[str, Any]:
        """What the page may know: where the test is, and the names it plays."""
        names = list(self.items)
        known = dict(zip(("A", "B"), names, strict=True)) if self.kind == "abx" else {}
        told: dict[str, Any] = {
            "kind": self.kind,
            "trials": self.trials,
            "trial": min(len(self.done) + 1, self.trials),
            "answered": len(self.done),
            "finished": self.finished,
            "known": known,
            "hidden": sorted(self._hidden),
        }
        if self.finished:
            told["result"] = self.result()
            told["trials_done"] = self.done
            told["saved"] = None if self.saved is None else str(self.saved)
        return told

    def resolve(self, name: str) -> str:
        """The item behind a name the page plays: ``A``, ``B`` or a hidden one."""
        if self.finished:
            raise HttpError(409, "the test is over")
        names = list(self.items)
        if self.kind == "abx" and name in ("A", "B"):
            return self.items[names[("A", "B").index(name)]]
        if name not in self._hidden:
            raise HttpError(404, f"no hidden item named {name!r}")
        return self.items[self._hidden[name]]

    def answer(self, given: Any) -> dict[str, Any]:
        """Take the trial's answer: ``"A"`` or ``"B"``, or the hidden names, the preferred first."""
        if self.finished:
            raise HttpError(409, "the test is over")
        names = list(self.items)
        trial: dict[str, Any] = {
            "trial": len(self.done) + 1,
            "seconds": round(time.monotonic() - self._since, 2),
        }
        if self.kind == "abx":
            if given not in ("A", "B"):
                raise HttpError(400, "the answer to an ABX trial is A or B")
            said = names[("A", "B").index(given)]
            trial.update(x=self._hidden["X"], answer=said, right=said == self._hidden["X"])
        else:
            if not isinstance(given, list) or sorted(given) != sorted(self._hidden):
                raise HttpError(400, "a ranking names every hidden item once, the preferred first")
            trial.update(
                shown=dict(self._hidden), ranking=[self._hidden[str(name)] for name in given]
            )
        self.done.append(trial)
        if self.finished:
            self.save()
        else:
            self._draw()
        return self.state()

    def stop(self) -> dict[str, Any]:
        """End the test where it is; what was answered is written."""
        if not self.finished:
            self._stopped = True
            self.save()
        return self.state()

    def result(self) -> dict[str, Any]:
        """The score and its probability by chance, over the trials answered."""
        count = len(self.done)
        if self.kind == "abx":
            right = sum(bool(t["right"]) for t in self.done)
            return {
                "answered": count,
                "right": right,
                "chance": chance(right, count) if count else None,
                "says": "the probability of this score or a better one by guessing",
            }
        names = list(self.items)
        firsts = {name: sum(t["ranking"][0] == name for t in self.done) for name in names}
        ranks = {name: _mean([t["ranking"].index(name) + 1 for t in self.done]) for name in names}
        best = max(firsts.values(), default=0)
        alone = chance(best, count, 1.0 / len(names)) if count else None
        return {
            "answered": count,
            "first": firsts,
            "mean_rank": ranks,
            "chance": None if alone is None else min(alone * len(names), 1.0),
            "says": "the probability that some variant comes first this often when the order is"
            " chance: one variant's, times the number of variants",
        }

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": 1,
            "kind": self.kind,
            "started": self.started.isoformat(timespec="seconds"),
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "complete": len(self.done) >= self.trials,
            "trials_planned": self.trials,
            "seed": self.seed,
            "items": self.items,
            "context": self.context,
            "trials": self.done,
            "result": self.result(),
        }

    def save(self) -> Path:
        self.results.mkdir(parents=True, exist_ok=True)
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        path = self.results / f"blind-{self.kind}-{stamp}.json"
        count = 2
        while path.exists():
            path = self.results / f"blind-{self.kind}-{stamp}-{count}.json"
            count += 1
        path.write_text(json.dumps(self.record(), indent=1) + "\n", encoding="utf-8")
        self.saved = path
        return path


def _mean(values: Sequence[int]) -> float | None:
    return round(float(np.mean(values)), 3) if values else None
