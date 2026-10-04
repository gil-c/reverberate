"""A synthetic pack to listen to before a traced one exists.

``python -m reverberate.viz.audit_demo <out.h5>`` writes a pack the audit
plays, built by the engine's own synthetic profiles
(:mod:`reverberate.render.pack`, :mod:`reverberate.render.benchmark`):

- ``--profile free`` (the default): the free field. Three sources stand
  round a listener who walks a line and turns the head; each is one arrival,
  so what is heard is where each source is, and nothing of a room. This is
  the pack to check directions, the head and the mix by ear.
- ``--profile density``: random tables of the density of a traced pack, with
  a low band and a tail. It sounds of nothing; it costs what a traced pack
  costs to render, and takes its place on disk (about 150 MB a second of
  scene and source), so it is the pack to measure the render with.

Each source is given activity for about half the scene, so that silence,
births and deaths are exercised. Nothing here is a room.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.benchmark import density_pack
from reverberate.render.pack import (
    Listener,
    ScenePack,
    synthetic_free_field,
    validate,
    write_pack,
)

__all__ = ["demo_pack", "main"]

_KINDS = ("near_voice", "far_voice", "noise")


def _recipe(duration_s: float, names: list[str], seed: int) -> bytes:
    """A recipe's bytes holding what the engine reads, each source active in turns."""
    rng = np.random.default_rng(seed)
    sources = []
    for index, name in enumerate(names):
        activity: list[dict[str, Any]] = []
        # The first source speaks from the start, so pressing play is heard at once.
        t = 0.0 if index == 0 else float(rng.uniform(1.0, min(4.0, 0.5 * duration_s)))
        while t < duration_s - 1.0:
            end = min(duration_s, t + float(rng.uniform(4.0, 9.0)))
            activity.append(
                {
                    "start_s": round(t, 2),
                    "end_s": round(end, 2),
                    "clip": {"library": "placeholder", "name": f"{name}-{len(activity)}"},
                    "clip_offset_s": 0.0,
                    "gain_db": 0.0,
                }
            )
            t = end + float(rng.uniform(3.0, 8.0))
        sources.append({"id": name, "gain_db": 0.0, "activity": activity})
    recipe = {
        "schema": "reverberate.scene-recipe",
        "duration_s": float(duration_s),
        "sources": sources,
    }
    text = json.dumps(
        recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def _with_recipe(pack: ScenePack, recipe: bytes) -> ScenePack:
    digest = hashlib.sha256(recipe).hexdigest()
    provenance = {**pack.header.provenance, "recipe_sha256": digest}
    header = replace(pack.header, recipe_sha256=digest, provenance=provenance)
    return replace(pack, header=header, recipe=recipe)


def demo_pack(
    *,
    profile: str = "free",
    duration_s: float = 60.0,
    sources: int = 3,
    origin: tuple[float, float, float] = (0.0, 1.5, 0.0),
    seed: int = 0,
) -> ScenePack:
    """The pack the module describes."""
    names = [f"s{i + 1}" for i in range(sources)]
    recipe = _recipe(duration_s, names, seed)
    if profile == "density":
        made = density_pack(duration_s=duration_s, moving=True, sources=sources, seed=seed)
        kinds = {
            name: replace(source, kind=_KINDS[i % 3])
            for i, (name, source) in enumerate(made.sources.items())
        }
        pack = _with_recipe(replace(made, sources=kinds), recipe)
    elif profile == "free":
        start = np.asarray(origin, dtype=float)
        end = start + np.array([0.05 * duration_s, 0.0, 0.0])  # a slow walk along +x
        middle = 0.5 * (start + end)
        packs = []
        for index, name in enumerate(names):
            angle = 2.0 * np.pi * index / sources + 0.5
            radius = 2.0 + 0.5 * index
            place = middle + radius * np.array([np.cos(angle), 0.0, np.sin(angle)])
            place[1] = start[1] + (0.0, -0.4, 0.5)[index % 3]
            packs.append(
                synthetic_free_field(
                    level="A",
                    source=place,
                    listener_start=start,
                    listener_end=end,
                    duration_s=duration_s,
                    source_id=name,
                    seed=seed,
                )
            )
        first = packs[0]
        steps = first.header.steps
        t = np.arange(steps) * first.header.step_s
        # The head looks along the walk and sweeps 60 degrees either side of it.
        orientation = np.zeros((steps, 3), dtype=np.float32)
        orientation[:, 0] = 60.0 * np.sin(2.0 * np.pi * t / 20.0)
        pack = replace(
            first,
            listener=Listener(first.listener.position, orientation),
            sources={
                name: replace(p.sources[name], kind=_KINDS[i % 3])
                for i, (name, p) in enumerate(zip(names, packs, strict=True))
            },
        )
        pack = _with_recipe(pack, recipe)
    else:
        raise ValueError(f"profile is 'free' or 'density', got {profile!r}")
    validate(pack)
    return pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="the pack to write, an .h5")
    parser.add_argument("--profile", choices=("free", "density"), default="free")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--origin", type=float, nargs=3, default=(0.0, 1.5, 0.0))
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    pack = demo_pack(
        profile=arguments.profile,
        duration_s=arguments.seconds,
        sources=arguments.sources,
        origin=tuple(arguments.origin),
        seed=arguments.seed,
    )
    write_pack(arguments.target, pack)
    size = arguments.target.stat().st_size
    print(f"wrote {arguments.target}: {len(pack.sources)} sources, {size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
