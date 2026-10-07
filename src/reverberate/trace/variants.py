"""A named set of cost variants of one scene, to be judged by ear on one excerpt.

The reference trace of a scene is the Cartesian grid at 10.5 points per
wavelength (which a trace takes only when told since 2026-10-07: the
command of every variant names its grid), rails every 8 cm read from two
positions, a low response of 1.2 s and 100 000 rays. Each of those may be
made cheaper by an option that changes the result; a **variant** is a name
and the options it takes. A **set** is a small JSON file of them
(``trace/sets/listening_v1.json``):

.. code-block:: json

    {"set": "listening_v1",
     "variants": [{"name": "reference", "says": "...", "flags": {}},
                  {"name": "grid-7.2", "says": "...", "flags": {"low_ppw": 7.2}}]}

The flags a variant may take are :data:`FLAGS`: ``low_ppw``, ``low_scheme``
and ``low_seconds`` (the wave grid and how long it is solved for),
``rail_positions`` and ``rail_pitch_m`` (how a moving source's positions
are read and how far apart they are solved), ``rays`` (a tail site's
count). The first variant of a set is its reference.

:func:`price` gives, for every variant, the predicted cost of the **whole
scene** (the saving is what the owner decides on) and of the **excerpt** he
listens to, and the ``trace rent --smoke`` command that makes the excerpt's
pack. Nothing here rents a machine.

**A variant of another rail pitch is another recipe.** The solved
positions of a rail are the recipe's (``rails[].pitch_m``, rule 4). The
variant's recipe is the reference's with that one number changed on every
rail and in its ``generator`` block: the stations, the rails' points, the
movements, the clips and the seed are the reference's, so the scene heard
is the same one. :func:`recipe_at_pitch` says what changed.

**The window.** :func:`listening_windows` proposes where the variants will
differ most: where a source speaks while it walks (the rail's reading),
where a walking voice passes close to the head (the direct sound of a
moving source, where an interpolation error is least masked), and where a
source is heard from another room (no direct path: the low band and the
tail carry it alone).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.pack import STEP_S
from reverberate.render.variant import VARIANT_FILE, read_variant
from reverberate.scenes import Recipe, save_recipe
from reverberate.scenes.recipe import canonical_bytes, recipe_sha256
from reverberate.trace.plan import (
    RAYS_MEASURED,
    REFERENCE_PPW,
    Profile,
    estimate,
    make_plan,
    tracks_of,
)

__all__ = [
    "DEFAULT_SET",
    "FLAGS",
    "VARIANT_FILE",
    "Variant",
    "Window",
    "command",
    "listening_windows",
    "load_set",
    "price",
    "read_variant",
    "recipe_at_pitch",
    "table",
]

#: The set the first scene is judged with.
DEFAULT_SET = Path(__file__).parent / "sets" / "listening_v1.json"
#: What a variant may set, and the reference's value of each.
FLAGS: dict[str, Any] = {
    "low_ppw": None,
    "low_scheme": "cartesian",
    "low_seconds": None,
    "rail_positions": 2,
    "rail_pitch_m": None,
    "rays": None,
}
#: A walking voice this close to the head is passing it.
PASSING_M = 1.5


@dataclass(frozen=True)
class Variant:
    """One named choice of options."""

    name: str
    says: str = ""
    flags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.flags) - set(FLAGS))
        if unknown:
            raise ValueError(
                f"variant {self.name!r} sets {', '.join(unknown)}; a variant sets "
                + ", ".join(FLAGS)
            )
        if not self.name or any(c in self.name for c in "/\\ \t"):
            raise ValueError(f"a variant's name is a file's name, not {self.name!r}")

    def get(self, flag: str) -> Any:
        return self.flags.get(flag, FLAGS[flag])

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "says": self.says, "flags": dict(self.flags)}


def load_set(path: Path | None = None) -> tuple[str, list[Variant]]:
    """A set's name and its variants, the reference first."""
    tree = json.loads(Path(path or DEFAULT_SET).read_text())
    variants = [
        Variant(str(v["name"]), str(v.get("says", "")), dict(v.get("flags", {})))
        for v in tree["variants"]
    ]
    names = [v.name for v in variants]
    if not variants or len(set(names)) != len(names):
        raise ValueError("a set names one variant or more, each once")
    if variants[0].flags:
        raise ValueError("the first variant of a set is the reference: it sets nothing")
    return str(tree.get("set", Path(path or DEFAULT_SET).stem)), variants


def recipe_at_pitch(recipe: Recipe, pitch_m: float) -> tuple[Recipe, dict[str, Any]]:
    """``recipe`` with its rails solved every ``pitch_m``, and what changed.

    The same scene: every rail keeps its points and every source its
    movements; only where along a rail the low band is solved changes, and
    with it the recipe's identity.
    """
    tree = json.loads(canonical_bytes(recipe))
    was = sorted({float(rail["pitch_m"]) for rail in tree["rails"]})
    for rail in tree["rails"]:
        rail["pitch_m"] = float(pitch_m)
    generator = tree.get("generator")
    if generator and "rails" in dict(generator.get("parameters", {})):
        generator["parameters"]["rails"]["pitch_m"] = float(pitch_m)
    changed = Recipe.from_dict(tree)
    return changed, {
        "rails": len(tree["rails"]),
        "pitch_m": {"was": was[0] if len(was) == 1 else was, "is": float(pitch_m)},
        "recipe_sha256": {"was": recipe_sha256(recipe), "is": recipe_sha256(changed)},
    }


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A stretch of the scene and what makes it worth listening to, in seconds of each."""

    start_s: float
    seconds: float
    #: Seconds a source is audible while it walks or rises, summed over the sources.
    walking_s: float
    #: Seconds a walking, audible voice is within :data:`PASSING_M` of the head.
    passing_s: float
    #: Seconds a source is audible from another room than the listener's.
    other_room_s: float

    @property
    def score(self) -> float:
        """The three, each counted up to a quarter of the window, so that one alone does not
        win; and a twentieth of what is over, so that of two full windows the fuller does."""
        cap = 0.25 * self.seconds
        parts = (self.walking_s, self.passing_s, self.other_room_s)
        return float(sum(min(v, cap) + 0.05 * max(v - cap, 0.0) for v in parts))

    def says(self) -> str:
        return (
            f"{self.start_s:g} s: {self.walking_s:.1f} s of a source audible on the move,"
            f" {self.passing_s:.1f} s of it within {PASSING_M} m of the head,"
            f" {self.other_room_s:.1f} s of a source audible from another room"
        )


def _rooms(recipe: Recipe, points: np.ndarray) -> np.ndarray:
    """The room of the station nearest each point, as a pack names a cell's (ADR 0010)."""
    stations = np.asarray([s.position for s in recipe.stations], dtype=float).reshape(-1, 3)
    names = np.asarray([s.room for s in recipe.stations])
    flat = np.asarray(points, dtype=float)[:, [0, 2]]
    near = np.linalg.norm(flat[:, None, :] - stations[None, :, [0, 2]], axis=2).argmin(axis=1)
    return np.asarray(names[near])


def listening_windows(
    recipe: Recipe, seconds: float, *, every_s: float = 5.0, best: int = 3
) -> list[Window]:
    """The ``best`` windows of ``seconds`` to judge variants on, the best first.

    Scored on the whole scene's tracks, a window every ``every_s``; of two
    windows that overlap by more than half only the better is kept.
    """
    whole = tracks_of(recipe)
    span = int(round(seconds / STEP_S))
    stride = max(1, int(round(every_s / STEP_S)))
    head_room = _rooms(recipe, whole.listener)
    walking = np.zeros(whole.steps)
    passing = np.zeros(whole.steps)
    other = np.zeros(whole.steps)
    for track in whole.sources.values():
        moving = track.audible & track.moving
        near = np.linalg.norm(track.position - whole.listener, axis=1) < PASSING_M
        walking += moving
        passing += moving & near
        other += track.audible & (_rooms(recipe, track.position) != head_room)
    sums = [np.concatenate([[0.0], np.cumsum(v)]) for v in (walking, passing, other)]
    found = []
    for first in range(0, max(whole.steps - span, 1), stride):
        last = min(first + span, whole.steps)
        a, b, c = (float(s[last] - s[first]) * STEP_S for s in sums)
        found.append(Window(first * STEP_S, float(seconds), a, b, c))
    found.sort(key=lambda w: (-w.score, w.start_s))
    kept: list[Window] = []
    for window in found:
        if all(abs(window.start_s - k.start_s) >= 0.5 * seconds for k in kept):
            kept.append(window)
        if len(kept) == best:
            break
    return kept


# --------------------------------------------------------------------------
# the prices and the commands
# --------------------------------------------------------------------------


def command(
    variant: Variant,
    recipe_path: Path,
    home: Path,
    *,
    start_s: float,
    seconds: float,
    sources: int,
    passed: list[str],
) -> str:
    """The ``trace rent --smoke`` line that makes the variant's excerpt; ``passed`` as given."""
    words = [
        "python -m reverberate.trace rent",
        f"--recipe {shlex.quote(str(recipe_path))}",
        f"--home {shlex.quote(str(home))}",
        f"--smoke {seconds:g} --smoke-start {start_s:g} --smoke-sources {sources}",
    ]
    if variant.get("low_scheme") != FLAGS["low_scheme"]:
        words.append(f"--low-scheme {variant.get('low_scheme')}")
    # A variant that names no grid is on the reference's, which the command line no
    # longer takes unless told (``trace.plan.LOW_PPW``): it is said by name.
    words.append(f"--low-ppw {float(variant.get('low_ppw') or REFERENCE_PPW):g}")
    if variant.get("low_seconds") is not None:
        words.append(f"--low-seconds {float(variant.get('low_seconds')):g}")
    if int(variant.get("rail_positions")) != FLAGS["rail_positions"]:
        words.append(f"--rail-positions {int(variant.get('rail_positions'))}")
    if variant.get("rays") is not None:
        words.append(f"--rays {int(variant.get('rays'))}")
    return " ".join([*words, *(shlex.quote(word) for word in passed)])


def _cost(priced: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "usd": float(priced["total_usd"]),
        "seconds": float(priced["total_s"]),
        "low_usd": float(priced["usd"]["low"]),
        "rays_usd": float(priced["usd"]["rays"]),
        "source_positions": int(record["source_positions"]),
        "solves": int(priced["solves"]),
        "pairs": int(priced.get("pairs", record["pairs"])),
        "rate_usd_per_hour": float(priced["billed_rate_usd_per_hour"]),
        "measured_on": str(priced["measured_on"]),
    }


def price(
    variants: list[Variant],
    recipe: Recipe,
    recipe_path: Path,
    triangles: np.ndarray,
    home: Path,
    *,
    start_s: float,
    seconds: float,
    sources: int = 3,
    rate_usd_per_hour: float,
    rays: int = RAYS_MEASURED,
    passed: list[str] | None = None,
    write: bool = True,
) -> list[dict[str, Any]]:
    """Every variant's record: its recipe, its two predicted costs, its command.

    ``rays`` is the mirror's own count, which a variant without the flag
    keeps. With ``write`` each variant's directory under ``home`` receives
    its recipe, where it is not the reference's, and :data:`VARIANT_FILE`;
    the command names that directory as the run's home, so that the pack
    comes back beside the file the audit reads.
    """
    home = Path(home)
    plans: dict[tuple[float | None, int, bool, float | None], dict[str, Any]] = {}
    recipes: dict[float | None, tuple[Recipe, dict[str, Any] | None]] = {None: (recipe, None)}
    records = []
    for variant in variants:
        pitch = variant.get("rail_pitch_m")
        pitch = None if pitch is None else float(pitch)
        if pitch not in recipes:
            recipes[pitch] = recipe_at_pitch(recipe, pitch)  # type: ignore[arg-type]
        mine, changed = recipes[pitch]
        positions = int(variant.get("rail_positions"))
        costs = {}
        for part, profile in (
            ("scene", Profile(rail_positions=positions)),
            (
                "excerpt",
                Profile(
                    seconds=seconds, sources=sources, start_s=start_s, rail_positions=positions
                ),
            ),
        ):
            # The pairs are counted on the grid's nodes: another grid, another count.
            ppw = variant.get("low_ppw")
            ppw = None if ppw is None else float(ppw)
            key = (pitch, positions, part == "scene", ppw)
            if key not in plans:
                plans[key] = make_plan(mine, triangles, profile, low_ppw=ppw).record
            cast = variant.get("rays")
            priced = estimate(
                plans[key],
                rate_usd_per_hour=rate_usd_per_hour,
                fetch_pairs=part == "scene",
                low_ppw=variant.get("low_ppw"),
                low_seconds=variant.get("low_seconds"),
                rays=None if cast is None and rays == RAYS_MEASURED else int(cast or rays),
            )
            costs[part] = _cost(priced, plans[key])
        directory = home / variant.name
        path = Path(recipe_path) if changed is None else directory / "recipe.json"
        record = {
            **variant.record(),
            "recipe": str(path),
            "recipe_sha256": recipe_sha256(mine),
            "recipe_changed": changed,
            "window": {"start_s": float(start_s), "seconds": float(seconds), "sources": sources},
            "predicted": costs,
            # Filled by the audit from the pack's own cost records once it is home.
            "measured": None,
            "home": str(directory),
            "command": command(
                variant,
                path,
                directory,
                start_s=start_s,
                seconds=seconds,
                sources=sources,
                passed=list(passed or []),
            ),
        }
        if write:
            directory.mkdir(parents=True, exist_ok=True)
            if changed is not None:
                save_recipe(mine, path)
            (directory / VARIANT_FILE).write_text(json.dumps(record, indent=1))
        records.append(record)
    return records


def table(records: list[dict[str, Any]]) -> str:
    """The set as a person reads it: what each variant saves on the whole scene, then the lines."""
    reference = records[0]["predicted"]["scene"]["usd"]
    lines = [
        f"{'variant':<16} {'whole scene':>12} {'saved':>7} {'positions':>9} {'solves':>7}"
        f" {'pairs':>7} {'excerpt':>9}   flags",
    ]
    for record in records:
        scene, excerpt = record["predicted"]["scene"], record["predicted"]["excerpt"]
        saved = 100.0 * (1.0 - scene["usd"] / reference) if reference > 0 else 0.0
        flags = ", ".join(f"{k}={v}" for k, v in record["flags"].items()) or "the reference"
        lines.append(
            f"{record['name']:<16} {scene['usd']:>8.2f} USD {saved:>6.0f}%"
            f" {scene['source_positions']:>9} {scene['solves']:>7} {scene['pairs']:>7}"
            f" {excerpt['usd']:>5.2f} USD   {flags}"
        )
    rate = records[0]["predicted"]["scene"]
    lines.append(
        f"  at {rate['rate_usd_per_hour']:g} USD/h, the low band as measured on"
        f" {rate['measured_on']}: one card's machine-seconds, not an offer's wall time"
    )
    for record in records:
        changed = record["recipe_changed"]
        if changed is not None:
            lines.append(
                f"  {record['name']}: another recipe, {record['recipe']}: {changed['rails']}"
                f" rails solved every {changed['pitch_m']['is']:g} m and not"
                f" {changed['pitch_m']['was']}; the same stations, movements and clips;"
                f" recipe_sha256 {changed['recipe_sha256']['is'][:16]}"
            )
    lines.append("")
    for record in records:
        lines.append(f"# {record['name']}: {record['says']}".rstrip(": "))
        lines.append(record["command"])
    return "\n".join(lines)
